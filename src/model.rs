//! yue2-infer 0.1.6 backbone with checkpoint-compatible names.
use anyhow::{ensure, Context, Result};
use candle_core::{DType, Device, Module, Tensor, D};
use candle_nn::{Embedding, Linear, VarBuilder};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[cfg(test)]
mod diagnostics;
pub(crate) mod nar;
pub use nar::{AudioPositionEmbedding, NarConfig, TimestepEmbedder};

fn linear_forward(linear: &Linear, x: &Tensor) -> candle_core::Result<Tensor> {
    if x.dtype() == DType::BF16 && (x.device().is_cpu() || linear.bias().is_some()) {
        // candle's CPU matmul has no BF16 implementation. Preserve BF16
        // operation boundaries around an FP32 accumulation on this backend.
        // Torch's biased addmm adds bias before its BF16 cast. Keep that single
        // rounding boundary for the four small acoustic auxiliary projections.
        let weight = linear.weight().to_dtype(DType::F32)?;
        let bias = linear.bias().map(|b| b.to_dtype(DType::F32)).transpose()?;
        return Linear::new(weight, bias)
            .forward(&x.to_dtype(DType::F32)?)?
            .to_dtype(DType::BF16);
    }
    linear.forward(x)
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct YuE2Config {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub max_position_embeddings: usize,
    #[serde(default)]
    pub tie_word_embeddings: bool,
}

impl YuE2Config {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.hidden_size > 0
                && self.num_hidden_layers > 0
                && self.intermediate_size > 0
                && self.vocab_size > 0
                && self.max_position_embeddings > 0,
            "Empty model dimensions"
        );
        ensure!(
            self.num_key_value_heads > 0
                && self.num_attention_heads > 0
                && self
                    .num_attention_heads
                    .is_multiple_of(self.num_key_value_heads),
            "Invalid GQA head count"
        );
        ensure!(
            self.head_dim > 0 && self.head_dim.is_multiple_of(2),
            "RoPE requires an even head_dim"
        );
        ensure!(
            self.rope_theta.is_finite()
                && self.rope_theta > 0.
                && self.rms_norm_eps.is_finite()
                && self.rms_norm_eps > 0.,
            "Invalid norm/RoPE config"
        );
        ensure!(
            !self.tie_word_embeddings,
            "Released checkpoint requires an untied lm_head"
        );
        Ok(())
    }
}

/// Resolve only local snapshots; no Hub client or download is used.
pub fn snapshot_dir(model_name: &str) -> Result<PathBuf> {
    ensure!(
        ["YuE2-3B", "YuE2-Vae"].contains(&model_name),
        "Unknown YuE2 model"
    );
    let home = std::env::var_os("HF_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|p| PathBuf::from(p).join("yue2/hf")))
        .context("Set HF_HOME or supply an explicit model directory")?;
    let repo = home
        .join("hub")
        .join(format!("models--m-a-p--{model_name}"));
    let main = repo.join("refs/main");
    if main.is_file() {
        let revision = std::fs::read_to_string(main)?;
        let revision = revision.trim();
        ensure!(
            !revision.is_empty() && revision.bytes().all(|b| b.is_ascii_hexdigit()),
            "Invalid snapshot revision"
        );
        let path = repo.join("snapshots").join(revision);
        ensure!(path.is_dir(), "Missing offline snapshot {}", path.display());
        return Ok(path);
    }
    let mut paths = std::fs::read_dir(repo.join("snapshots"))?
        .map(|entry| entry.map(|e| e.path()))
        .collect::<std::io::Result<Vec<_>>>()?;
    paths.retain(|p| p.is_dir());
    ensure!(
        paths.len() == 1,
        "Ambiguous offline snapshots; supply an explicit directory"
    );
    Ok(paths.remove(0))
}

#[derive(Debug)]
pub struct RMSNorm {
    weight: Tensor,
    eps: f64,
}

impl RMSNorm {
    pub fn load(dim: usize, eps: f64, vb: VarBuilder<'_>) -> Result<Self> {
        Ok(Self {
            weight: vb.get(dim, "weight")?,
            eps,
        })
    }
    pub fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        // Python casts the reciprocal RMS back BEFORE either BF16 multiply.
        // candle_nn's fused RMSNorm has different rounding semantics.
        let scale = (x.to_dtype(DType::F32)?.sqr()?.mean_keepdim(D::Minus1)? + self.eps)?
            .sqrt()?
            .recip()?
            .to_dtype(x.dtype())?;
        x.broadcast_mul(&scale)?.broadcast_mul(&self.weight)
    }
}

#[derive(Debug)]
pub struct RotaryEmbedding {
    inv_freq: Tensor,
    head_dim: usize,
}

impl RotaryEmbedding {
    pub fn new(head_dim: usize, base: f64, device: &Device) -> Result<Self> {
        ensure!(
            head_dim > 0 && head_dim.is_multiple_of(2) && base.is_finite() && base > 0.,
            "Invalid RoPE dimensions/base"
        );
        let inv: Vec<f32> = (0..head_dim)
            .step_by(2)
            .map(|i| 1. / (base as f32).powf(i as f32 / head_dim as f32))
            .collect();
        Ok(Self {
            inv_freq: Tensor::new(inv, device)?,
            head_dim,
        })
    }
    pub fn forward(&self, positions: &Tensor) -> Result<(Tensor, Tensor)> {
        let angles = positions
            .to_dtype(DType::F32)?
            .unsqueeze(D::Minus1)?
            .broadcast_mul(&self.inv_freq)?;
        Ok((angles.cos()?, angles.sin()?))
    }
    pub fn apply(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let half = self.head_dim / 2;
        let x1 = x.narrow(D::Minus1, 0, half)?;
        let x2 = x.narrow(D::Minus1, half, half)?;
        let cos = cos.to_dtype(x.dtype())?.unsqueeze(2)?;
        let sin = sin.to_dtype(x.dtype())?.unsqueeze(2)?;
        let a = (x1.broadcast_mul(&cos)? - x2.broadcast_mul(&sin)?)?;
        let b = (x2.broadcast_mul(&cos)? + x1.broadcast_mul(&sin)?)?;
        Ok(Tensor::cat(&[a, b], D::Minus1)?)
    }
}

/// Fixed-capacity, append-only cache; each update writes into an existing tensor.
/// Layout is [batch, kv_heads, capacity, head_dim], as in Python StaticKVCache.
#[derive(Debug)]
pub struct StaticKVCache {
    key_cache: Vec<Tensor>,
    value_cache: Vec<Tensor>,
    seen_tokens: usize,
    max_seq_len: usize,
    batch_size: usize,
    next_layer: usize,
}

impl StaticKVCache {
    pub fn new(
        config: &YuE2Config,
        batch_size: usize,
        max_seq_len: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        config.validate()?;
        ensure!(
            batch_size > 0 && max_seq_len > 0 && max_seq_len <= config.max_position_embeddings,
            "Invalid KV cache capacity"
        );
        let shape = (
            batch_size,
            config.num_key_value_heads,
            max_seq_len,
            config.head_dim,
        );
        let allocate = || {
            (0..config.num_hidden_layers)
                .map(|_| Tensor::zeros(shape, dtype, device))
                .collect::<candle_core::Result<Vec<_>>>()
        };
        Ok(Self {
            key_cache: allocate()?,
            value_cache: allocate()?,
            seen_tokens: 0,
            max_seq_len,
            batch_size,
            next_layer: 0,
        })
    }
    pub fn get_seq_length(&self) -> usize {
        self.seen_tokens
    }
    pub fn capacity(&self) -> usize {
        self.max_seq_len
    }
    pub fn reset(&mut self) {
        self.seen_tokens = 0;
        self.next_layer = 0;
    }
    fn update(
        &mut self,
        key: &Tensor,
        value: &Tensor,
        layer_idx: usize,
    ) -> Result<(Tensor, Tensor)> {
        ensure!(
            layer_idx == self.next_layer,
            "KV cache updates must follow layer order; reset after a failed forward"
        );
        let end = self.seen_tokens + key.dim(2)?;
        ensure!(
            end <= self.max_seq_len,
            "KV cache capacity {} exceeded by {end}; generation was not shortened",
            self.max_seq_len
        );
        self.key_cache[layer_idx].slice_set(&key.contiguous()?, 2, self.seen_tokens)?;
        self.value_cache[layer_idx].slice_set(&value.contiguous()?, 2, self.seen_tokens)?;
        self.next_layer += 1;
        if self.next_layer == self.key_cache.len() {
            self.seen_tokens = end;
            self.next_layer = 0;
        }
        Ok((
            self.key_cache[layer_idx].narrow(2, 0, end)?,
            self.value_cache[layer_idx].narrow(2, 0, end)?,
        ))
    }
}

#[derive(Debug)]
pub struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RMSNorm,
    k_norm: RMSNorm,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

impl Attention {
    fn load(c: &YuE2Config, vb: VarBuilder<'_>) -> Result<Self> {
        let linear = |out, name| candle_nn::linear_no_bias(c.hidden_size, out, vb.pp(name));
        Ok(Self {
            q_proj: linear(c.num_attention_heads * c.head_dim, "q_proj")?,
            k_proj: linear(c.num_key_value_heads * c.head_dim, "k_proj")?,
            v_proj: linear(c.num_key_value_heads * c.head_dim, "v_proj")?,
            o_proj: candle_nn::linear_no_bias(
                c.num_attention_heads * c.head_dim,
                c.hidden_size,
                vb.pp("o_proj"),
            )?,
            q_norm: RMSNorm::load(c.head_dim, c.rms_norm_eps, vb.pp("q_norm"))?,
            k_norm: RMSNorm::load(c.head_dim, c.rms_norm_eps, vb.pp("k_norm"))?,
            num_heads: c.num_attention_heads,
            num_kv_heads: c.num_key_value_heads,
            head_dim: c.head_dim,
        })
    }
    fn project_qkv(
        &self,
        x: &Tensor,
        rope: &RotaryEmbedding,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let (b, t, _) = x.dims3()?;
        let q = linear_forward(&self.q_proj, x)?.reshape((b, t, self.num_heads, self.head_dim))?;
        let k =
            linear_forward(&self.k_proj, x)?.reshape((b, t, self.num_kv_heads, self.head_dim))?;
        let v =
            linear_forward(&self.v_proj, x)?.reshape((b, t, self.num_kv_heads, self.head_dim))?;
        Ok((
            rope.apply(&self.q_norm.forward(&q)?, cos, sin)?,
            rope.apply(&self.k_norm.forward(&k)?, cos, sin)?,
            v,
        ))
    }
    fn repeat_kv(&self, x: &Tensor) -> Result<Tensor> {
        let (b, h, t, d) = x.dims4()?;
        let groups = self.num_heads / self.num_kv_heads;
        Ok(x.unsqueeze(2)?
            .expand((b, h, groups, t, d))?
            .contiguous()?
            .reshape((b, h * groups, t, d))?)
    }
    fn sdpa(&self, q: &Tensor, k: &Tensor, v: &Tensor, offset: usize) -> Result<Tensor> {
        self.sdpa_tiled(q, k, v, Some(offset), 128)
    }
    fn sdpa_tiled(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        offset: Option<usize>,
        tile: usize,
    ) -> Result<Tensor> {
        // Bound temporary score storage without changing the visible key set.
        // Accumulate scores and weighted values in FP32. Eager Torch CUDA SDPA
        // rounds the unnormalized softmax numerator to BF16 before its value
        // product, then divides the FP32 accumulator by the FP32 row sum. Keep
        // this rounding boundary (verified by the first-layer operation oracle).
        let (_, _, query_len, _) = q.dims4()?;
        let key_len = k.dim(2)?;
        let key = self.repeat_kv(k)?.to_dtype(DType::F32)?.t()?.contiguous()?;
        let value = self.repeat_kv(v)?.to_dtype(DType::F32)?.contiguous()?;
        let mut pieces = Vec::new();
        for start in (0..query_len).step_by(tile) {
            let count = (query_len - start).min(tile);
            let query = q
                .narrow(2, start, count)?
                .to_dtype(DType::F32)?
                .contiguous()?;
            let scores = (query.matmul(&key)? * (1. / (self.head_dim as f64).sqrt()))?;
            let scores = if let Some(offset) = offset {
                let mask: Vec<f32> = (0..count)
                    .flat_map(|i| {
                        (0..key_len).map(move |j| {
                            if j <= offset + start + i {
                                0.
                            } else {
                                f32::NEG_INFINITY
                            }
                        })
                    })
                    .collect();
                scores.broadcast_add(&Tensor::from_vec(mask, (count, key_len), q.device())?)?
            } else {
                scores
            };
            let numerator = scores
                .broadcast_sub(&scores.max_keepdim(D::Minus1)?)?
                .exp()?;
            let denominator = numerator.sum_keepdim(D::Minus1)?;
            let weights = numerator.to_dtype(q.dtype())?.to_dtype(DType::F32)?;
            pieces.push(
                weights
                    .matmul(&value)?
                    .broadcast_div(&denominator)?
                    .to_dtype(q.dtype())?,
            );
        }
        Ok(Tensor::cat(&pieces, 2)?)
    }
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        x: &Tensor,
        rope: &RotaryEmbedding,
        cos: &Tensor,
        sin: &Tensor,
        cache: Option<&mut StaticKVCache>,
        layer_idx: usize,
        offset: usize,
    ) -> Result<Tensor> {
        let (b, t, _) = x.dims3()?;
        let (q, k, v) = self.project_qkv(x, rope, cos, sin)?;
        let (q, k, v) = (q.transpose(1, 2)?, k.transpose(1, 2)?, v.transpose(1, 2)?);
        let (k, v) = match cache {
            Some(c) => c.update(&k, &v, layer_idx)?,
            None => (k, v),
        };
        let out = self.sdpa(&q, &k, &v, offset)?;
        Ok(linear_forward(
            &self.o_proj,
            &out.transpose(1, 2)?
                .contiguous()?
                .reshape((b, t, self.num_heads * self.head_dim))?,
        )?)
    }
}

#[derive(Debug)]
pub struct MLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl MLP {
    fn load(c: &YuE2Config, vb: VarBuilder<'_>) -> Result<Self> {
        Ok(Self {
            gate_proj: candle_nn::linear_no_bias(
                c.hidden_size,
                c.intermediate_size,
                vb.pp("gate_proj"),
            )?,
            up_proj: candle_nn::linear_no_bias(
                c.hidden_size,
                c.intermediate_size,
                vb.pp("up_proj"),
            )?,
            down_proj: candle_nn::linear_no_bias(
                c.intermediate_size,
                c.hidden_size,
                vb.pp("down_proj"),
            )?,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Torch SiLU evaluates in FP32 and rounds once to BF16. Candle's CUDA
        // BF16 SiLU rounds exp/add/div individually, which fails logits parity.
        let gate = linear_forward(&self.gate_proj, x)?;
        let gate = gate.to_dtype(DType::F32)?.silu()?.to_dtype(x.dtype())?;
        Ok(linear_forward(
            &self.down_proj,
            &(gate * linear_forward(&self.up_proj, x)?)?,
        )?)
    }
}

#[derive(Debug)]
pub struct DecoderLayer {
    input_layernorm: RMSNorm,
    self_attn: Attention,
    post_attention_layernorm: RMSNorm,
    mlp: MLP,
}

impl DecoderLayer {
    fn load(c: &YuE2Config, vb: VarBuilder<'_>) -> Result<Self> {
        Ok(Self {
            input_layernorm: RMSNorm::load(
                c.hidden_size,
                c.rms_norm_eps,
                vb.pp("input_layernorm"),
            )?,
            self_attn: Attention::load(c, vb.pp("self_attn"))?,
            post_attention_layernorm: RMSNorm::load(
                c.hidden_size,
                c.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
            mlp: MLP::load(c, vb.pp("mlp"))?,
        })
    }
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        x: &Tensor,
        rope: &RotaryEmbedding,
        cos: &Tensor,
        sin: &Tensor,
        cache: Option<&mut StaticKVCache>,
        layer_idx: usize,
        offset: usize,
    ) -> Result<Tensor> {
        let h = self.self_attn.forward(
            &self.input_layernorm.forward(x)?,
            rope,
            cos,
            sin,
            cache,
            layer_idx,
            offset,
        )?;
        let x = (x + h)?;
        Ok((&x
            + self
                .mlp
                .forward(&self.post_attention_layernorm.forward(&x)?)?)?)
    }
}

#[derive(Debug)]
pub struct Backbone {
    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    norm: RMSNorm,
    rotary_emb: RotaryEmbedding,
}

impl Backbone {
    fn load(c: &YuE2Config, vb: VarBuilder<'_>) -> Result<Self> {
        Ok(Self {
            embed_tokens: candle_nn::embedding(c.vocab_size, c.hidden_size, vb.pp("embed_tokens"))?,
            layers: (0..c.num_hidden_layers)
                .map(|i| DecoderLayer::load(c, vb.pp(format!("layers.{i}"))))
                .collect::<Result<_>>()?,
            norm: RMSNorm::load(c.hidden_size, c.rms_norm_eps, vb.pp("norm"))?,
            rotary_emb: RotaryEmbedding::new(c.head_dim, c.rope_theta, vb.device())?,
        })
    }
}

pub struct CausalLMOutput {
    pub logits: Tensor,
    /// Empty unless capture_hidden=true; names match ar_logits.safetensors.
    pub hidden_states: std::collections::BTreeMap<String, Tensor>,
}

#[derive(Debug)]
pub struct YuE2ForCausalLM {
    pub config: YuE2Config,
    model: Backbone,
    lm_head: Linear,
    nar: Option<nar::NarWeights>,
}

impl YuE2ForCausalLM {
    pub fn load(config: YuE2Config, vb: VarBuilder<'_>) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            model: Backbone::load(&config, vb.pp("model"))?,
            lm_head: candle_nn::linear_no_bias(
                config.hidden_size,
                config.vocab_size,
                vb.pp("lm_head"),
            )?,
            config,
            nar: None,
        })
    }
    /// Memory-map immutable checkpoint weights; only AR tensors are loaded.
    ///
    /// # Safety
    /// The caller must ensure model.safetensors is never modified or truncated
    /// for the lifetime of the returned model (including CPU mmap-backed tensors).
    pub unsafe fn from_pretrained(
        directory: impl AsRef<Path>,
        dtype: DType,
        device: &Device,
    ) -> Result<Self> {
        let directory = directory.as_ref();
        let config: YuE2Config =
            serde_json::from_slice(&std::fs::read(directory.join("config.json"))?)?;
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[directory.join("model.safetensors")],
                dtype,
                device,
            )?
        };
        Self::load(config, vb)
    }
    pub fn dtype(&self) -> DType {
        self.lm_head.weight().dtype()
    }
    pub fn device(&self) -> &Device {
        self.lm_head.weight().device()
    }

    /// Unpadded AR inference. Passing a cache appends the current tokens; passing
    /// None computes a full causal prefill. logits_to_keep=0 returns all positions.
    pub fn forward(
        &self,
        input_ids: &Tensor,
        mut cache: Option<&mut StaticKVCache>,
        logits_to_keep: usize,
        capture_hidden: bool,
    ) -> Result<CausalLMOutput> {
        let (b, t) = input_ids.dims2()?;
        ensure!(b > 0 && t > 0, "Input must contain at least one token");
        let offset = cache.as_ref().map_or(0, |c| c.seen_tokens);
        ensure!(
            t <= self.config.max_position_embeddings.saturating_sub(offset),
            "AR input exceeds model context"
        );
        if let Some(c) = cache.as_ref() {
            ensure!(
                c.batch_size == b && c.key_cache.len() == self.model.layers.len(),
                "KV cache/model shape mismatch"
            );
            ensure!(c.next_layer == 0, "Reset KV cache after a failed forward");
            ensure!(
                offset + t <= c.max_seq_len,
                "KV cache capacity {} exceeded by {}; generation was not shortened",
                c.max_seq_len,
                offset + t
            );
        }
        let positions =
            Tensor::arange(offset as u32, (offset + t) as u32, self.device())?.unsqueeze(0)?;
        let (cos, sin) = self.model.rotary_emb.forward(&positions)?;
        let mut x = self.model.embed_tokens.forward(input_ids)?;
        let mut hidden_states = std::collections::BTreeMap::new();
        if capture_hidden {
            hidden_states.insert("embedding".into(), x.clone());
        }
        for (i, layer) in self.model.layers.iter().enumerate() {
            x = layer.forward(
                &x,
                &self.model.rotary_emb,
                &cos,
                &sin,
                cache.as_deref_mut(),
                i,
                offset,
            )?;
            if capture_hidden && (i == 0 || i == 13) {
                hidden_states.insert(format!("layer.{i}"), x.clone());
            }
        }
        x = self.model.norm.forward(&x)?;
        if capture_hidden {
            hidden_states.insert("final_norm".into(), x.clone());
        }
        let kept = if logits_to_keep == 0 {
            t
        } else {
            logits_to_keep.min(t)
        };
        let logits = linear_forward(&self.lm_head, &x.narrow(1, t - kept, kept)?.contiguous()?)?;
        Ok(CausalLMOutput {
            logits,
            hidden_states,
        })
    }
}
