//! Acoustic checkpoint modules and CachedNAR's invariant AR prefill.
use super::*;
use crate::nar::{check_cancelled, finite, Chunk};

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default)]
pub struct NarConfig {
    pub latent_dim: usize,
    pub max_latent_frames: usize,
    pub timestep_shift: f64,
    pub latent_type: String,
}
impl Default for NarConfig {
    fn default() -> Self {
        Self {
            latent_dim: 64,
            max_latent_frames: 24576,
            timestep_shift: 1.,
            latent_type: "vae".into(),
        }
    }
}

#[derive(Debug)]
pub struct TimestepEmbedder {
    first: Linear,
    second: Linear,
    freqs: Tensor,
}
impl TimestepEmbedder {
    fn load(hidden: usize, vb: VarBuilder<'_>) -> Result<Self> {
        let freqs =
            ((Tensor::arange(0f32, 128f32, vb.device())? * -10000f64.ln())? / 128.)?.exp()?;
        Ok(Self {
            first: candle_nn::linear(256, hidden, vb.pp("mlp.0"))?,
            second: candle_nn::linear(hidden, hidden, vb.pp("mlp.2"))?,
            freqs,
        })
    }
    pub fn forward(&self, t: &Tensor) -> Result<Tensor> {
        let args = t
            .to_dtype(DType::F32)?
            .unsqueeze(D::Minus1)?
            .broadcast_mul(&self.freqs)?;
        let emb = Tensor::cat(&[args.cos()?, args.sin()?], D::Minus1)?
            .to_dtype(self.first.weight().dtype())?;
        let x = linear_forward(&self.first, &emb)?;
        let x = x.to_dtype(DType::F32)?.silu()?.to_dtype(emb.dtype())?;
        Ok(linear_forward(&self.second, &x)?)
    }
}

#[derive(Debug)]
pub struct AudioPositionEmbedding {
    pe: Tensor,
}
impl AudioPositionEmbedding {
    fn load(frames: usize, hidden: usize, vb: VarBuilder<'_>) -> Result<Self> {
        // This non-learnable buffer is serialized in the released checkpoint.
        // Load its actual values rather than regenerating with a different exp/sin.
        Ok(Self {
            pe: vb.get((frames, hidden), "pe")?,
        })
    }
    pub fn forward(&self, positions: &Tensor) -> Result<Tensor> {
        Ok(self.pe.index_select(positions, 0)?)
    }
}

#[derive(Debug)]
struct NarLayer {
    nar_input_layernorm: RMSNorm,
    nar_self_attn: Attention,
    nar_pre_mlp_layernorm: RMSNorm,
    nar_mlp: MLP,
}
#[derive(Debug)]
pub(super) struct NarWeights {
    config: NarConfig,
    layers: Vec<NarLayer>,
    vae2llm: Linear,
    llm2vae: Linear,
    time_embedder: TimestepEmbedder,
    latent_pos_embed: AudioPositionEmbedding,
}
impl YuE2ForCausalLM {
    /// Keep AR-only loading available; explicitly opt into the acoustic tensors.
    pub fn load_with_nar(
        config: YuE2Config,
        nar_config: NarConfig,
        vb: VarBuilder<'_>,
    ) -> Result<Self> {
        ensure!(
            nar_config.latent_dim == 64
                && nar_config.max_latent_frames > 0
                && nar_config.latent_type == "vae",
            "Expected VAE acoustic architecture with 64 channels"
        );
        ensure!(
            nar_config.timestep_shift.is_finite() && nar_config.timestep_shift > 0.,
            "Invalid timestep shift"
        );
        let mut model = Self::load(config, vb.clone())?;
        let c = &model.config;
        let layers = (0..c.num_hidden_layers)
            .map(|i| {
                let v = vb.pp(format!("model.layers.{i}"));
                Ok(NarLayer {
                    nar_input_layernorm: RMSNorm::load(
                        c.hidden_size,
                        c.rms_norm_eps,
                        v.pp("nar_input_layernorm"),
                    )?,
                    nar_self_attn: Attention::load(c, v.pp("nar_self_attn"))?,
                    nar_pre_mlp_layernorm: RMSNorm::load(
                        c.hidden_size,
                        c.rms_norm_eps,
                        v.pp("nar_pre_mlp_layernorm"),
                    )?,
                    nar_mlp: MLP::load(c, v.pp("nar_mlp"))?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        model.nar = Some(NarWeights {
            vae2llm: candle_nn::linear(64, c.hidden_size, vb.pp("vae2llm"))?,
            llm2vae: candle_nn::linear(c.hidden_size, 64, vb.pp("llm2vae"))?,
            time_embedder: TimestepEmbedder::load(c.hidden_size, vb.pp("time_embedder"))?,
            latent_pos_embed: AudioPositionEmbedding::load(
                nar_config.max_latent_frames,
                c.hidden_size,
                vb.pp("latent_pos_embed"),
            )?,
            config: nar_config,
            layers,
        });
        Ok(model)
    }

    /// Memory-map the full immutable AR + NAR checkpoint, without copying files.
    ///
    /// # Safety
    /// model.safetensors must remain unchanged for the returned model's lifetime.
    pub unsafe fn from_pretrained_with_nar(
        directory: impl AsRef<Path>,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let directory = directory.as_ref();
        let data = std::fs::read(directory.join("config.json"))?;
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[directory.join("model.safetensors")],
                dtype,
                device,
            )?
        };
        Self::load_with_nar(
            serde_json::from_slice(&data)?,
            serde_json::from_slice(&data)?,
            vb,
        )
    }
}

/// A single acoustic chunk, with one invariant AR key/value pair per layer.
pub struct CachedNAR<'a> {
    model: &'a YuE2ForCausalLM,
    chunk: &'a Chunk,
    weights: &'a NarWeights,
    cache: Vec<(Tensor, Tensor)>,
    cos: Tensor,
    sin: Tensor,
    pos_emb: Tensor,
    nar_length: usize,
    query_chunk_size: usize,
}
impl<'a> CachedNAR<'a> {
    pub fn new(
        model: &'a YuE2ForCausalLM,
        chunk: &'a Chunk,
        query_chunk_size: usize,
    ) -> Result<Self> {
        let weights = model
            .nar
            .as_ref()
            .context("Load the model with NAR weights first")?;
        let (frames, dim) = chunk.noise.dims2()?;
        ensure!(
            frames > 0 && dim == 64,
            "Expected nonempty acoustic noise [frames,64]"
        );
        finite(&chunk.noise)?;
        let ar_length = chunk.ar_tokens.len();
        let nar_length = frames.checked_add(2).context("NAR length overflow")?;
        ensure!(
            ar_length > 0
                && chunk
                    .ar_tokens
                    .iter()
                    .all(|&t| (t as usize) < model.config.vocab_size),
            "AR prefix empty or outside vocabulary"
        );
        ensure!(
            ar_length
                .checked_add(nar_length)
                .is_some_and(|n| n <= model.config.max_position_embeddings),
            "Original acoustic chunk exceeds model context"
        );
        ensure!(query_chunk_size > 0, "Query tile must be positive");
        let positions = Tensor::arange(
            ar_length as u32,
            (ar_length + nar_length) as u32,
            model.device(),
        )?
        .unsqueeze(0)?;
        let (cos, sin) = model.model.rotary_emb.forward(&positions)?;
        let local: Vec<u32> = (0..nar_length)
            .map(|i| i.min(weights.config.max_latent_frames - 1) as u32)
            .collect();
        let pos_emb = weights
            .latent_pos_embed
            .forward(&Tensor::new(local, model.device())?)?
            .unsqueeze(0)?;
        let mut engine = Self {
            model,
            chunk,
            weights,
            cache: vec![],
            cos,
            sin,
            pos_emb,
            nar_length,
            query_chunk_size,
        };
        engine.prefill()?;
        Ok(engine)
    }

    fn attention(
        &self,
        attn: &Attention,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        causal: bool,
    ) -> Result<Tensor> {
        let out = attn.sdpa_tiled(
            &q.transpose(1, 2)?,
            &k.transpose(1, 2)?,
            &v.transpose(1, 2)?,
            causal.then_some(0),
            self.query_chunk_size,
        )?;
        Ok(out.transpose(1, 2)?.contiguous()?.flatten_from(2)?)
    }
    fn prefill(&mut self) -> Result<()> {
        let backbone = &self.model.model;
        let ar_length = self.chunk.ar_tokens.len();
        let visible = if self.chunk.nar_cond_end == 0 {
            ar_length
        } else {
            self.chunk.nar_cond_end.min(ar_length)
        };
        let ids =
            Tensor::new(self.chunk.ar_tokens.as_slice(), self.model.device())?.unsqueeze(0)?;
        let positions =
            Tensor::arange(0u32, ar_length as u32, self.model.device())?.unsqueeze(0)?;
        let (cos, sin) = backbone.rotary_emb.forward(&positions)?;
        let mut x = backbone.embed_tokens.forward(&ids)?;
        for layer in &backbone.layers {
            let (q, k, v) = layer.self_attn.project_qkv(
                &layer.input_layernorm.forward(&x)?,
                &backbone.rotary_emb,
                &cos,
                &sin,
            )?;
            let cached = if visible == ar_length {
                (k.clone(), v.clone())
            } else {
                (
                    k.narrow(1, 0, visible)?.copy()?,
                    v.narrow(1, 0, visible)?.copy()?,
                )
            };
            self.cache.push(cached);
            let h = self.attention(&layer.self_attn, &q, &k, &v, true)?;
            x = (&x + linear_forward(&layer.self_attn.o_proj, &h)?)?;
            x = (&x
                + layer
                    .mlp
                    .forward(&layer.post_attention_layernorm.forward(&x)?)?)?;
        }
        Ok(())
    }

    pub fn velocity(&self, state: &Tensor, raw_t: f64) -> Result<Tensor> {
        ensure!(
            state.shape() == self.chunk.noise.shape(),
            "ODE state shape changed"
        );
        ensure!(raw_t.is_finite(), "Non-finite timestep");
        let dtype = self.model.dtype();
        let zero = Tensor::zeros((1, 64), dtype, self.model.device())?;
        let x_nar = Tensor::cat(&[&zero, state, &zero], 0)?.unsqueeze(0)?;
        let raw = Tensor::new(raw_t as f32, self.model.device())?.to_dtype(dtype)?;
        let t_sig = candle_nn::ops::sigmoid(&raw.to_dtype(DType::F32)?)?.to_dtype(dtype)?;
        let shift = self.weights.config.timestep_shift;
        let numerator = scalar_mul(&t_sig, shift)?;
        let denominator =
            (scalar_mul(&t_sig, shift - 1.)?.to_dtype(DType::F32)? + 1.)?.to_dtype(dtype)?;
        let shifted = (numerator.to_dtype(DType::F32)? / denominator.to_dtype(DType::F32)?)?
            .to_dtype(dtype)?;
        let time = self
            .weights
            .time_embedder
            .forward(&shifted.broadcast_as(self.nar_length)?.contiguous()?)?
            .unsqueeze(0)?;
        let mut x = (linear_forward(&self.weights.vae2llm, &x_nar)? + time)?;
        x = (x + &self.pos_emb)?;
        for (layer, (ar_k, ar_v)) in self.weights.layers.iter().zip(&self.cache) {
            let (q, k, v) = layer.nar_self_attn.project_qkv(
                &layer.nar_input_layernorm.forward(&x)?,
                &self.model.model.rotary_emb,
                &self.cos,
                &self.sin,
            )?;
            let k = Tensor::cat(&[ar_k, &k], 1)?;
            let v = Tensor::cat(&[ar_v, &v], 1)?;
            let h = self.attention(&layer.nar_self_attn, &q, &k, &v, false)?;
            x = (&x + linear_forward(&layer.nar_self_attn.o_proj, &h)?)?;
            x = (&x
                + layer
                    .nar_mlp
                    .forward(&layer.nar_pre_mlp_layernorm.forward(&x)?)?)?;
        }
        Ok(
            linear_forward(&self.weights.llm2vae, &self.model.model.norm.forward(&x)?)?
                .squeeze(0)?
                .narrow(0, 1, self.nar_length - 2)?,
        )
    }

    pub fn solve(
        &self,
        steps: usize,
        cancelled: Option<&dyn Fn() -> bool>,
        mut on_progress: Option<&mut dyn FnMut(usize, usize)>,
    ) -> Result<Tensor> {
        ensure!(steps > 0, "Steps must be positive");
        let mut state = self
            .chunk
            .noise
            .to_device(self.model.device())?
            .to_dtype(self.model.dtype())?;
        let dt = 1. / steps as f64;
        for step in 0..steps {
            check_cancelled(cancelled)?;
            let t = 1. - step as f64 * dt;
            let first = self.velocity(&state, raw_time(t))?;
            let mid = (&state - scalar_mul(&first, dt / 2.)?)?;
            check_cancelled(cancelled)?;
            state = (&state - scalar_mul(&self.velocity(&mid, raw_time(t - dt / 2.))?, dt)?)?;
            if let Some(callback) = on_progress.as_mut() {
                callback(step + 1, steps);
            }
        }
        let result = state.to_dtype(DType::F32)?.to_device(&Device::Cpu)?;
        finite(&result)?;
        Ok(result)
    }
}

fn raw_time(t: f64) -> f64 {
    (t / (1. - t)).ln().clamp(-20., 20.)
}
fn scalar_mul(x: &Tensor, scale: f64) -> Result<Tensor> {
    Ok((x.to_dtype(DType::F32)? * scale)?.to_dtype(x.dtype())?)
}

#[cfg(test)]
#[path = "nar_tests.rs"]
mod tests;
