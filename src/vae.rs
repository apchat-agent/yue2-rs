//! FP32 Oobleck decoder, ported from the released modeling_vae.py.
//! Oobleck / SnakeBeta attribution and MIT terms: THIRD_PARTY_NOTICES.md.
use anyhow::{ensure, Context, Result};
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, VarBuilder};
use serde::Deserialize;
use std::path::Path;

#[cfg(test)]
mod tests;

#[derive(Clone, Debug, Deserialize)]
#[serde(default)]
pub struct DecoderConfig {
    pub out_channels: usize,
    pub channels: usize,
    pub latent_dim: usize,
    pub c_mults: Vec<usize>,
    pub strides: Vec<usize>,
    pub use_snake: bool,
    pub snake_type: String,
    pub antialias_activation: bool,
    pub use_nearest_upsample: bool,
    pub use_filter: bool,
    pub final_tanh: bool,
}

impl Default for DecoderConfig {
    fn default() -> Self {
        Self {
            out_channels: 2,
            channels: 64,
            latent_dim: 64,
            c_mults: vec![1, 2, 4, 8, 16, 32],
            strides: vec![2, 2, 4, 4, 5, 6],
            use_snake: true,
            snake_type: "vanilla".into(),
            antialias_activation: false,
            use_nearest_upsample: false,
            use_filter: false,
            final_tanh: false,
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(default)]
pub struct YuE2VAEConfig {
    pub decoder_config: DecoderConfig,
    pub sample_rate: usize,
    pub latent_dim: usize,
    pub downsampling_ratio: usize,
    pub audio_channels: usize,
    pub decode_core_frames: usize,
    pub decode_halo_frames: usize,
}

impl Default for YuE2VAEConfig {
    fn default() -> Self {
        Self {
            decoder_config: DecoderConfig::default(),
            sample_rate: 48000,
            latent_dim: 64,
            downsampling_ratio: 1920,
            audio_channels: 2,
            decode_core_frames: 1024,
            decode_halo_frames: 16,
        }
    }
}

impl YuE2VAEConfig {
    fn validate(&self) -> Result<()> {
        let c = &self.decoder_config;
        ensure!(
            !c.antialias_activation && !c.use_nearest_upsample && !c.use_filter,
            "Unsupported option for the released decoder"
        );
        ensure!(
            !c.use_snake || c.snake_type == "vanilla",
            "Expected vanilla SnakeBeta"
        );
        ensure!(
            c.channels > 0
                && c.latent_dim > 0
                && c.out_channels > 0
                && !c.c_mults.is_empty()
                && c.c_mults.iter().all(|&x| x > 0)
                && c.strides.len() == c.c_mults.len()
                && c.strides.iter().all(|&x| x > 0),
            "Invalid decoder dimensions/strides"
        );
        let ratio = c
            .strides
            .iter()
            .try_fold(1usize, |a, &b| a.checked_mul(b))
            .context("Decoder stride product overflow")?;
        ensure!(
            ratio == self.downsampling_ratio,
            "Decoder strides do not match downsampling_ratio"
        );
        ensure!(
            c.latent_dim == self.latent_dim && c.out_channels == self.audio_channels,
            "Decoder channels do not match VAE configuration"
        );
        ensure!(
            self.decode_core_frames > 0 && self.sample_rate > 0,
            "Invalid VAE core/sample rate"
        );
        for &m in &c.c_mults {
            c.channels
                .checked_mul(m)
                .context("Decoder channels overflow")?;
        }
        Ok(())
    }
}

// torch.nn.utils.weight_norm defaults to dim=0 for BOTH convolution kinds.
// Conv1d v=[out,in,k]; ConvTranspose1d v=[in,out,k], g=[v.dim(0),1,1].
// Fold once; no g/v tensors are retained by the decoder.
fn folded_weight(shape: (usize, usize, usize), vb: &VarBuilder<'_>) -> Result<Tensor> {
    let v = vb.get(shape, "weight_v")?;
    let g = vb.get((shape.0, 1, 1), "weight_g")?;
    let norm = v.sqr()?.sum_keepdim((1, 2))?.sqrt()?;
    Ok(v.broadcast_mul(&g.div(&norm)?)?.contiguous()?)
}

fn wn_conv1d(
    input: usize,
    output: usize,
    kernel: usize,
    dilation: usize,
    bias: bool,
    vb: VarBuilder<'_>,
) -> Result<Conv1d> {
    Ok(Conv1d::new(
        folded_weight((output, input, kernel), &vb)?,
        if bias {
            Some(vb.get(output, "bias")?)
        } else {
            None
        },
        Conv1dConfig {
            padding: dilation * (kernel - 1) / 2,
            dilation,
            ..Default::default()
        },
    ))
}

#[derive(Debug)]
pub struct SnakeBeta {
    alpha: Tensor,
    inverse_beta: Tensor,
}

impl SnakeBeta {
    fn load(channels: usize, vb: VarBuilder<'_>) -> Result<Self> {
        Ok(Self {
            alpha: vb
                .get(channels, "alpha")?
                .exp()?
                .reshape((1, channels, 1))?,
            inverse_beta: (vb.get(channels, "beta")?.exp()? + 1e-9)?
                .recip()?
                .reshape((1, channels, 1))?,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(x.add(
            &x.broadcast_mul(&self.alpha)?
                .sin()?
                .sqr()?
                .broadcast_mul(&self.inverse_beta)?,
        )?)
    }
}

#[derive(Debug)]
enum Activation {
    Snake(SnakeBeta),
    Elu,
}

impl Activation {
    fn load(channels: usize, snake: bool, vb: VarBuilder<'_>) -> Result<Self> {
        Ok(if snake {
            Self::Snake(SnakeBeta::load(channels, vb)?)
        } else {
            Self::Elu
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Snake(s) => s.forward(x),
            Self::Elu => Ok(x.elu(1.)?),
        }
    }
}

#[derive(Debug)]
struct WNConvTranspose1d {
    weight: Tensor,
    bias: Tensor,
    stride: usize,
}

impl WNConvTranspose1d {
    fn load(input: usize, output: usize, stride: usize, vb: VarBuilder<'_>) -> Result<Self> {
        Ok(Self {
            weight: folded_weight((input, output, 2 * stride), &vb)?,
            bias: vb.get(output, "bias")?.reshape((1, output, 1))?,
            stride,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Zero-padding dispatches candle's GEMM/col2im implementation. Cropping
        // ceil(stride/2) from both ends is exactly ConvTranspose1d's padding,
        // including the stride=5 stage's one-sample length loss.
        // Candle 0.11 CPU MatMul merges batches sharing the kernel. Its merge
        // needs contiguous [batch,time,channel] rows; an NCL allocation gives
        // wrong rows after batch 0. Keep the public tensor axes [B,C,T].
        let x = if x.device().is_cpu() {
            x.transpose(1, 2)?.contiguous()?.transpose(1, 2)?
        } else {
            x.contiguous()?
        };
        let y = x.conv_transpose1d(&self.weight, 0, 0, self.stride, 1, 1)?;
        let padding = self.stride.div_ceil(2);
        Ok(y.narrow(2, padding, y.dim(2)? - 2 * padding)?
            .broadcast_add(&self.bias)?)
    }
}

#[derive(Debug)]
pub struct ResidualUnit {
    first_activation: Activation,
    first: Conv1d,
    second_activation: Activation,
    second: Conv1d,
}

impl ResidualUnit {
    fn load(channels: usize, dilation: usize, snake: bool, vb: VarBuilder<'_>) -> Result<Self> {
        let vb = vb.pp("layers");
        Ok(Self {
            first_activation: Activation::load(channels, snake, vb.pp("0"))?,
            first: wn_conv1d(channels, channels, 7, dilation, true, vb.pp("1"))?,
            second_activation: Activation::load(channels, snake, vb.pp("2"))?,
            second: wn_conv1d(channels, channels, 1, 1, true, vb.pp("3"))?,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = self.first.forward(&self.first_activation.forward(x)?)?;
        Ok(self
            .second
            .forward(&self.second_activation.forward(&y)?)?
            .add(x)?)
    }
}

#[derive(Debug)]
pub struct DecoderBlock {
    activation: Activation,
    upsample: WNConvTranspose1d,
    residuals: Vec<ResidualUnit>,
}

impl DecoderBlock {
    fn load(
        input: usize,
        output: usize,
        stride: usize,
        snake: bool,
        vb: VarBuilder<'_>,
    ) -> Result<Self> {
        let vb = vb.pp("layers");
        Ok(Self {
            activation: Activation::load(input, snake, vb.pp("0"))?,
            upsample: WNConvTranspose1d::load(input, output, stride, vb.pp("1"))?,
            residuals: [1, 3, 9]
                .into_iter()
                .enumerate()
                .map(|(i, d)| ResidualUnit::load(output, d, snake, vb.pp((i + 2).to_string())))
                .collect::<Result<_>>()?,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = self.upsample.forward(&self.activation.forward(x)?)?;
        for unit in &self.residuals {
            x = unit.forward(&x)?;
        }
        Ok(x)
    }
}

#[derive(Debug)]
pub struct OobleckDecoder {
    input: Conv1d,
    blocks: Vec<DecoderBlock>,
    activation: Activation,
    output: Conv1d,
    final_tanh: bool,
}

/// Receives each complete block on the freely evolving decoder path.
pub type BlockObserver<'a> = &'a mut dyn FnMut(usize, &Tensor) -> Result<()>;

impl OobleckDecoder {
    fn load(c: &DecoderConfig, vb: VarBuilder<'_>) -> Result<Self> {
        let vb = vb.pp("layers");
        let mults = std::iter::once(1)
            .chain(c.c_mults.iter().copied())
            .collect::<Vec<_>>();
        let depth = c.c_mults.len();
        let mut blocks = Vec::with_capacity(depth);
        for i in (1..=depth).rev() {
            blocks.push(DecoderBlock::load(
                mults[i] * c.channels,
                mults[i - 1] * c.channels,
                c.strides[i - 1],
                c.use_snake,
                vb.pp((depth - i + 1).to_string()),
            )?);
        }
        Ok(Self {
            input: wn_conv1d(
                c.latent_dim,
                mults[depth] * c.channels,
                7,
                1,
                true,
                vb.pp("0"),
            )?,
            blocks,
            activation: Activation::load(c.channels, c.use_snake, vb.pp((depth + 1).to_string()))?,
            output: wn_conv1d(
                c.channels,
                c.out_channels,
                7,
                1,
                false,
                vb.pp((depth + 2).to_string()),
            )?,
            final_tanh: c.final_tanh,
        })
    }

    fn forward(&self, x: &Tensor, mut observer: Option<BlockObserver<'_>>) -> Result<Tensor> {
        let mut x = self.input.forward(x)?;
        for (index, block) in self.blocks.iter().enumerate() {
            x = block.forward(&x)?;
            if let Some(callback) = observer.as_deref_mut() {
                callback(index, &x)?;
            }
        }
        let audio = self.output.forward(&self.activation.forward(&x)?)?;
        Ok(if self.final_tanh {
            audio.tanh()?
        } else {
            audio
        })
    }
}

#[derive(Debug)]
pub struct YuE2VAE {
    config: YuE2VAEConfig,
    decoder: OobleckDecoder,
    device: Device,
}

impl YuE2VAE {
    pub fn load(config: YuE2VAEConfig, vb: VarBuilder<'_>) -> Result<Self> {
        config.validate()?;
        ensure!(vb.dtype() == DType::F32, "VAE requires FP32 weights");
        let decoder = OobleckDecoder::load(&config.decoder_config, vb.pp("decoder"))?;
        Ok(Self {
            config,
            decoder,
            device: vb.device().clone(),
        })
    }

    /// Only decoder tensors are loaded; checkpoint g/v are folded in memory.
    ///
    /// # Safety
    /// The checkpoint must not be modified while its memory map is in use.
    pub unsafe fn from_pretrained(directory: impl AsRef<Path>, device: &Device) -> Result<Self> {
        let directory = directory.as_ref();
        let config = serde_json::from_slice(&std::fs::read(directory.join("config.json"))?)?;
        let tensors =
            candle_core::safetensors::MmapedSafetensors::new(directory.join("model.safetensors"))?;
        for (name, tensor) in tensors.tensors() {
            if name.starts_with("decoder.") {
                ensure!(
                    DType::try_from(tensor.dtype())? == DType::F32,
                    "VAE export contains a non-FP32 decoder tensor: {name}"
                );
            }
        }
        let vb = VarBuilder::from_backend(Box::new(tensors), DType::F32, device.clone());
        Self::load(config, vb)
    }

    pub fn config(&self) -> &YuE2VAEConfig {
        &self.config
    }

    fn latent(&self, latent: &Tensor) -> Result<()> {
        let (batch, channels, frames) = latent
            .dims3()
            .context("Expected [B,latent_dim,T] latents")?;
        ensure!(
            batch > 0 && channels == self.config.latent_dim && frames > 0,
            "Expected nonempty [B,{},T] latents",
            self.config.latent_dim
        );
        ensure!(
            latent
                .to_dtype(DType::F32)?
                .to_device(&Device::Cpu)?
                .flatten_all()?
                .to_vec1::<f32>()?
                .iter()
                .all(|x| x.is_finite()),
            "VAE latents contain non-finite values"
        );
        Ok(())
    }

    /// Full unclipped FP32 [B,audio_channels,samples] waveform on the decoder device.
    pub fn decode(&self, latent: &Tensor) -> Result<Tensor> {
        self.decode_with_blocks(latent, None)
    }

    pub fn decode_with_blocks(
        &self,
        latent: &Tensor,
        observer: Option<BlockObserver<'_>>,
    ) -> Result<Tensor> {
        self.latent(latent)?;
        self.decoder.forward(
            &latent
                .to_device(&self.device)?
                .to_dtype(DType::F32)?
                .contiguous()?,
            observer,
        )
    }

    pub fn natural_output_length(&self, frames: usize) -> Result<usize> {
        ensure!(frames > 0, "frames must be positive");
        self.config
            .decoder_config
            .strides
            .iter()
            .rev()
            .try_fold(frames, |length, &stride| {
                length
                    .checked_mul(stride)
                    .and_then(|n| n.checked_sub(stride % 2))
                    .filter(|&n| n > 0)
                    .context("Invalid or overflowing VAE output length")
            })
    }

    pub fn required_halo(&self, core_frames: usize) -> Result<usize> {
        ensure!(core_frames > 0, "core_frames must be positive");
        let samples = core_frames
            .checked_mul(self.config.downsampling_ratio)
            .context("Core length overflow")?;
        // Inclusive dependency interval, traversed in reverse execution order.
        // Final Conv1d, then each block's residuals and transposed convolution,
        // then the initial Conv1d. Pointwise activations do not expand support.
        let (mut low, mut high) = (-3i128, samples as i128 - 1 + 3);
        for &stride in &self.config.decoder_config.strides {
            low -= 3 * (1 + 3 + 9);
            high += 3 * (1 + 3 + 9);
            let s = stride as i128;
            let p = stride.div_ceil(2) as i128;
            low = -(-(low + p - (2 * s - 1))).div_euclid(s);
            high = (high + p).div_euclid(s);
        }
        low -= 3;
        high += 3;
        Ok(usize::try_from(
            0.max(-low).max(high - core_frames as i128 + 1),
        )?)
    }

    /// Decode exact cropped cores with their receptive-field halos, returning
    /// CPU FP32 audio. The final tile retains the natural (shorter) end length.
    pub fn decode_tiled(
        &self,
        latent: &Tensor,
        core_frames: Option<usize>,
        halo_frames: Option<usize>,
        mut on_progress: Option<&mut dyn FnMut(usize, usize)>,
    ) -> Result<Tensor> {
        self.latent(latent)?;
        let core = core_frames.unwrap_or(self.config.decode_core_frames);
        let halo = halo_frames.unwrap_or(self.config.decode_halo_frames);
        let required = self.required_halo(core)?;
        ensure!(halo >= required, "halo_frames must be at least {required}");
        let frames = latent.dim(2)?;
        let ratio = self.config.downsampling_ratio;
        let total = self.natural_output_length(frames)?;
        let tiles = frames.div_ceil(core);
        let mut crops = Vec::with_capacity(tiles);
        for (index, start) in (0..frames).step_by(core).enumerate() {
            let end = start.saturating_add(core).min(frames);
            let left = start.saturating_sub(halo);
            let right = end.saturating_add(halo).min(frames);
            let tile = self.decode(&latent.narrow(2, left, right - left)?)?;
            let out_start = start * ratio;
            let out_end = end
                .checked_mul(ratio)
                .context("Tile length overflow")?
                .min(total);
            let crop_start = (start - left) * ratio;
            crops.push(
                tile.narrow(2, crop_start, out_end - out_start)?
                    .contiguous()?
                    .to_device(&Device::Cpu)?,
            );
            if let Some(callback) = on_progress.as_deref_mut() {
                callback(index + 1, tiles);
            }
        }
        Ok(Tensor::cat(&crops, 2)?)
    }
}
