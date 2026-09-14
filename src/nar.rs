//! Original context chunks and midpoint acoustic flow matching from nar.py.
pub use crate::model::nar::CachedNAR;
use crate::{
    model::YuE2ForCausalLM,
    protocol::{chunk_ranges, CODEC_OFFSET, CODEC_SIZE, CONTEXT, MUSIC_END},
};
use anyhow::{ensure, Result};
use candle_core::{Device, DeviceLocation, Tensor};
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};

#[derive(Clone, Debug)]
pub struct Chunk {
    pub ar_tokens: Vec<u32>,
    /// CPU FP32 [frames, 64]; fixture noise can be supplied directly.
    pub noise: Tensor,
    /// Zero exposes all AR keys; positive values restrict visibility.
    pub nar_cond_end: usize,
}

/// Own the noise seed independently of AR sampling. A fresh candle CUDA device
/// has its own cuRAND generator; never reseed or draw from the model's device
/// RNG (Phase 2 uses it). Transfer the one full-song FP32 draw to CPU before cuts.
/// CPU/Metal use candle's rand dependency because candle cannot seed CPU RNG.
pub fn song_chunks(
    prefix: &[u32],
    codec: &[u32],
    seed: u64,
    context: usize,
    device: &Device,
) -> Result<Vec<Chunk>> {
    ensure!(
        !prefix.is_empty() && !codec.is_empty(),
        "Prefix and codec must be nonempty"
    );
    ensure!(
        codec.iter().all(|&v| v < CODEC_SIZE),
        "Codec outside vocabulary"
    );
    ensure!(
        (1..=CONTEXT).contains(&context),
        "Context must be in 1..={CONTEXT}"
    );
    let ranges = chunk_ranges(codec.len(), prefix.len(), context)?;
    let noise = match device.location() {
        DeviceLocation::Cuda { gpu_id } => {
            let rng_device = Device::new_cuda(gpu_id)?;
            rng_device.set_seed(seed)?;
            Tensor::randn(0f32, 1f32, (codec.len(), 64), &rng_device)?.to_device(&Device::Cpu)?
        }
        _ => {
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            let values: Vec<f32> = (0..codec
                .len()
                .checked_mul(64)
                .ok_or_else(|| anyhow::anyhow!("Noise size overflow"))?)
                .map(|_| StandardNormal.sample(&mut rng))
                .collect();
            Tensor::from_vec(values, (codec.len(), 64), &Device::Cpu)?
        }
    };
    ranges
        .into_iter()
        .map(|(a, b)| {
            let mut ar_tokens = prefix.to_vec();
            ar_tokens.extend(codec[a..b].iter().map(|&v| v + CODEC_OFFSET));
            ar_tokens.push(MUSIC_END);
            Ok(Chunk {
                ar_tokens,
                noise: noise.narrow(0, a, b - a)?,
                nar_cond_end: 0,
            })
        })
        .collect()
}

pub struct SynthesisOptions<'a> {
    pub steps: usize,
    pub context: usize,
    pub query_chunk_size: usize,
    pub cancelled: Option<&'a dyn Fn() -> bool>,
    pub on_progress: Option<&'a mut dyn FnMut(usize, usize)>,
}

impl Default for SynthesisOptions<'_> {
    fn default() -> Self {
        Self {
            steps: 32,
            context: CONTEXT,
            query_chunk_size: 128,
            cancelled: None,
            on_progress: None,
        }
    }
}

/// Solve chunks serially, dropping each AR cache before the next prefill.
/// Returns CPU FP32 [frames, 64]. Model weights must include the NAR path.
pub fn synthesize(
    model: &YuE2ForCausalLM,
    prefix: &[u32],
    codec: &[u32],
    seed: u64,
    mut options: SynthesisOptions<'_>,
) -> Result<Tensor> {
    ensure!(
        options.steps > 0 && options.query_chunk_size > 0,
        "Steps and query tile must be positive"
    );
    check_cancelled(options.cancelled)?;
    let chunks = song_chunks(prefix, codec, seed, options.context, model.device())?;
    let total = options
        .steps
        .checked_mul(chunks.len())
        .ok_or_else(|| anyhow::anyhow!("Progress count overflow"))?;
    let mut output = Vec::with_capacity(chunks.len());
    for (i, chunk) in chunks.iter().enumerate() {
        check_cancelled(options.cancelled)?;
        let engine = CachedNAR::new(model, chunk, options.query_chunk_size)?;
        let mut progress = |completed, _| {
            if let Some(callback) = options.on_progress.as_mut() {
                callback(i * options.steps + completed, total);
            }
        };
        output.push(engine.solve(options.steps, options.cancelled, Some(&mut progress))?);
    }
    Ok(Tensor::cat(&output, 0)?)
}

pub(crate) fn check_cancelled(cancelled: Option<&dyn Fn() -> bool>) -> Result<()> {
    ensure!(
        !cancelled.is_some_and(|f| f()),
        "Cancelled during acoustic flow matching"
    );
    Ok(())
}

pub(crate) fn finite(t: &Tensor) -> Result<()> {
    ensure!(
        t.to_dtype(candle_core::DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?
            .iter()
            .all(|v| v.is_finite()),
        "Acoustic tensor contains non-finite values"
    );
    Ok(())
}
