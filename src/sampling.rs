//! Eager, request-local AR generation; port of yue2-infer 0.1.6 sampling.py.
use crate::{
    model::{StaticKVCache, YuE2ForCausalLM},
    protocol::*,
};
use anyhow::{bail, ensure, Result};
use candle_core::{DType, Device, Tensor};
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::{sync::Mutex, time::Instant};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Phase {
    Abc,
    Semantic,
}

impl Phase {
    pub fn end(self) -> u32 {
        match self {
            Self::Abc => ABC_END,
            Self::Semantic => MUSIC_END,
        }
    }
    fn allowed(self, token: u32) -> bool {
        token == self.end()
            || match self {
                Self::Abc => token < EOD,
                Self::Semantic => (CODEC_OFFSET..CODEC_OFFSET + CODEC_SIZE).contains(&token),
            }
    }
}

// Sampling arithmetic runs on the host: candle's CUDA full-vocabulary sort
// exceeds shared memory, and Tensor::cumsum builds a quadratic-size matrix.
// Round each historical BF16 operation, rather than silently upcasting cot=off.
fn rounded(value: f32, dtype: DType) -> f32 {
    match dtype {
        DType::BF16 => half::bf16::from_f32(value).to_f32(),
        DType::F16 => half::f16::from_f32(value).to_f32(),
        _ => value,
    }
}

pub fn window_penalty(
    scores: &mut [f32],
    recent_ids: &[u32],
    penalty: f64,
    dtype: DType,
) -> Result<()> {
    if penalty == 1. || recent_ids.is_empty() {
        return Ok(());
    }
    let mut frequencies = std::collections::BTreeMap::<u32, usize>::new();
    for &token in recent_ids {
        ensure!(
            (token as usize) < scores.len(),
            "History token outside vocabulary"
        );
        *frequencies.entry(token).or_default() += 1;
    }
    for (token, count) in frequencies {
        let alpha = rounded(rounded(penalty as f32, dtype).powf(count as f32), dtype);
        let score = &mut scores[token as usize];
        *score = rounded(
            if *score < 0. {
                *score * alpha
            } else {
                *score / alpha
            },
            dtype,
        );
    }
    Ok(())
}

fn softmax(scores: &[f32], dtype: DType) -> Vec<f32> {
    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = scores.iter().map(|s| (*s - max).exp()).collect();
    let total: f32 = exp.iter().sum();
    exp.into_iter().map(|p| rounded(p / total, dtype)).collect()
}

/// One full vocabulary row, returned as FP32 host values. In legacy mode these
/// values represent exactly the original dtype (normally BF16).
pub fn distribution(
    logits: &Tensor,
    sampling: &Sampling,
    history: &[u32],
    step: usize,
    phase: Phase,
    legacy_off: bool,
) -> Result<Vec<f32>> {
    sampling.validate()?;
    ensure!(
        logits.elem_count() == logits.dim(candle_core::D::Minus1)?,
        "Expected one logits row"
    );
    let dtype = if legacy_off {
        logits.dtype()
    } else {
        DType::F32
    };
    ensure!(
        matches!(dtype, DType::F32 | DType::BF16 | DType::F16),
        "Unsupported sampling dtype"
    );
    let mut scores = logits
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    ensure!(
        scores.len() > phase.end() as usize,
        "Vocabulary is missing the phase end token"
    );
    for (token, score) in scores.iter_mut().enumerate() {
        if !phase.allowed(token as u32)
            || (token == phase.end() as usize && step < sampling.min_tokens)
        {
            *score = f32::NEG_INFINITY;
        }
    }
    window_penalty(
        &mut scores,
        &history[history.len().saturating_sub(sampling.penalty_window)..],
        sampling.repetition_penalty,
        dtype,
    )?;
    ensure!(
        scores.iter().any(|s| s.is_finite())
            && scores
                .iter()
                .all(|s| s.is_finite() || *s == f32::NEG_INFINITY),
        "Invalid or empty sampling distribution"
    );
    if sampling.temperature == 0. {
        return Ok(scores);
    }
    if sampling.temperature != 1. {
        for score in &mut scores {
            *score = rounded(*score / sampling.temperature as f32, dtype);
        }
    }
    ensure!(
        scores.iter().any(|s| s.is_finite())
            && scores
                .iter()
                .all(|s| s.is_finite() || *s == f32::NEG_INFINITY),
        "Temperature produced invalid sampling scores"
    );
    let mut indices: Vec<usize> = scores
        .iter()
        .enumerate()
        .filter(|(_, s)| s.is_finite())
        .map(|(i, _)| i)
        .collect();
    // Keep every tie at the kth score, as scores < threshold does in Python.
    let k = sampling.top_k.min(indices.len());
    indices.select_nth_unstable_by(k - 1, |&a, &b| scores[b].total_cmp(&scores[a]));
    let threshold = scores[indices[k - 1]];
    for score in &mut scores {
        if *score < threshold {
            *score = f32::NEG_INFINITY;
        }
    }
    if sampling.top_p < 1. {
        indices.retain(|&i| scores[i].is_finite());
        indices.sort_unstable_by(|&a, &b| scores[b].total_cmp(&scores[a]).then(a.cmp(&b)));
        let values: Vec<f32> = indices.iter().map(|&i| scores[i]).collect();
        let probabilities = softmax(&values, dtype);
        let mut cumulative = 0f32;
        for (rank, (&index, &probability)) in indices.iter().zip(&probabilities).enumerate() {
            cumulative += probability;
            let preceding = rounded(rounded(cumulative, dtype) - probability, dtype);
            if rank >= if legacy_off { 3 } else { 1 }
                && preceding > rounded(sampling.top_p as f32, dtype)
            {
                scores[index] = f32::NEG_INFINITY;
            }
        }
    }
    Ok(scores)
}

fn sample(scores: &[f32], temperature: f64, dtype: DType, uniform: f32) -> Result<u32> {
    if temperature == 0. {
        return Ok(scores
            .iter()
            .enumerate()
            .fold((0, f32::NEG_INFINITY), |best, (i, &s)| {
                if s > best.1 {
                    (i, s)
                } else {
                    best
                }
            })
            .0 as u32);
    }
    let probabilities = softmax(scores, dtype);
    let total: f64 = probabilities.iter().map(|&p| f64::from(p)).sum();
    ensure!(
        total.is_finite() && total > 0.,
        "Invalid multinomial probabilities"
    );
    let threshold = f64::from(uniform) * total;
    let mut cumulative = 0.;
    let mut last = 0;
    for (i, p) in probabilities.into_iter().enumerate() {
        if p > 0. {
            last = i;
        }
        cumulative += f64::from(p);
        if cumulative > threshold {
            return Ok(i as u32);
        }
    }
    Ok(last as u32)
}

/// Capture each request's candle device RNG draws together so subsequent
/// generations cannot reset its stream. CPU candle 0.11 cannot be seeded;
/// there we use its rand dependency's seeded StdRng as an explicit fallback.
fn uniforms(device: &Device, seed: u64, count: usize) -> Result<Vec<f32>> {
    if device.is_cpu() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        return Ok((0..count).map(|_| rng.random::<f32>()).collect());
    }
    static RNG: Mutex<()> = Mutex::new(());
    let _lock = RNG
        .lock()
        .map_err(|_| anyhow::anyhow!("Sampling RNG lock poisoned"))?;
    device.set_seed(seed)?;
    Ok(Tensor::rand(0f32, 1f32, count, device)?.to_vec1()?)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Timing {
    pub seconds: f64,
    pub prefill_seconds: f64,
    pub ttft_seconds: Option<f64>,
    pub output_tokens: usize,
    pub content_tokens: usize,
    pub output_tps: f64,
    pub prefix_tokens: usize,
    pub cfg_branches: usize,
    pub execution: String,
    pub attention: String,
}

#[derive(Default)]
pub struct GenerationOptions<'p, 'a> {
    pub negative: Option<&'p [u32]>,
    pub cfg_scale: Option<f64>,
    pub legacy_off: bool,
    pub cancelled: Option<&'a dyn Fn() -> bool>,
    pub on_token: Option<&'a mut dyn FnMut(Phase, u32)>,
}

#[derive(Debug)]
pub struct TokenGeneration {
    pub tokens: Vec<u32>,
    pub timing: Timing,
    pub truncated: bool,
}

/// Torch multiplies the BF16 difference by an FP32 scalar, then rounds once.
/// Candle's BF16 affine casts the scalar too, so perform that multiply in FP32.
pub fn cfg_logits(conditional: &Tensor, unconditional: &Tensor, scale: f64) -> Result<Tensor> {
    let difference = (conditional - unconditional)?;
    let scaled = (difference.to_dtype(DType::F32)? * scale)?.to_dtype(conditional.dtype())?;
    Ok((unconditional + scaled)?)
}

pub fn generate_tokens(
    model: &YuE2ForCausalLM,
    prefix: &[u32],
    sampling: &Sampling,
    seed: u64,
    phase: Phase,
    mut options: GenerationOptions<'_, '_>,
) -> Result<TokenGeneration> {
    sampling.validate()?;
    let context = CONTEXT.min(model.config.max_position_embeddings);
    let valid_prefix = |ids: &[u32]| -> Result<()> {
        ensure!(!ids.is_empty(), "Empty generation prefix");
        ensure!(
            ids.len() <= context && sampling.max_tokens <= context - ids.len(),
            "Prefix + requested generation budget exceeds context; no implicit truncation"
        );
        ensure!(
            ids.iter()
                .all(|&id| (id as usize) < model.config.vocab_size),
            "Prefix token outside vocabulary"
        );
        Ok(())
    };
    valid_prefix(prefix)?;
    let cfg_scale = options.cfg_scale.unwrap_or(1.);
    ensure!(cfg_scale.is_finite(), "CFG scale must be finite");
    ensure!(
        cfg_scale == 1. || options.negative.is_some(),
        "CFG requires a negative prefix"
    );
    if let Some(negative) = options.negative {
        valid_prefix(negative)?;
    }
    let check_cancelled = || -> Result<()> {
        if options.cancelled.is_some_and(|cancel| cancel()) {
            bail!("Cancelled during {phase:?} generation");
        }
        Ok(())
    };
    check_cancelled()?;
    let random = if sampling.temperature == 0. {
        vec![]
    } else {
        uniforms(model.device(), seed, sampling.max_tokens)?
    };
    let prefill = |ids: &[u32]| -> Result<(Tensor, StaticKVCache)> {
        let mut cache = StaticKVCache::new(
            &model.config,
            1,
            ids.len() + sampling.max_tokens,
            model.dtype(),
            model.device(),
        )?;
        let logits = model
            .forward(
                &Tensor::new(ids, model.device())?.unsqueeze(0)?,
                Some(&mut cache),
                1,
                false,
            )?
            .logits
            .flatten_all()?;
        Ok((logits, cache))
    };
    model.device().synchronize()?;
    let start = Instant::now();
    let (mut conditional, mut positive_cache) = prefill(prefix)?;
    let mut negative_branch = if cfg_scale != 1. {
        Some(prefill(options.negative.expect("validated CFG prefix"))?)
    } else {
        None
    };
    model.device().synchronize()?;
    let prefill_seconds = start.elapsed().as_secs_f64();
    let (mut history, mut first, mut eos) = (Vec::new(), None, false);
    for step in 0..sampling.max_tokens {
        check_cancelled()?;
        // Tensor operations preserve subtraction/multiply/add BF16 boundaries.
        let logits = match &negative_branch {
            Some((unconditional, _)) => cfg_logits(&conditional, unconditional, cfg_scale)?,
            None => conditional.clone(),
        };
        let scores = distribution(&logits, sampling, &history, step, phase, options.legacy_off)?;
        let dtype = if options.legacy_off {
            logits.dtype()
        } else {
            DType::F32
        };
        let token = sample(
            &scores,
            sampling.temperature,
            dtype,
            random.get(step).copied().unwrap_or(0.),
        )?;
        first.get_or_insert_with(|| start.elapsed().as_secs_f64());
        if let Some(callback) = options.on_token.as_mut() {
            callback(phase, token);
        }
        if token == phase.end() {
            eos = true;
            break;
        }
        history.push(token);
        if step + 1 < sampling.max_tokens {
            let input = Tensor::new(&[[token]], model.device())?;
            conditional = model
                .forward(&input, Some(&mut positive_cache), 1, false)?
                .logits
                .flatten_all()?;
            if let Some((logits, cache)) = &mut negative_branch {
                *logits = model
                    .forward(&input, Some(cache), 1, false)?
                    .logits
                    .flatten_all()?;
            }
        }
    }
    model.device().synchronize()?;
    let seconds = start.elapsed().as_secs_f64();
    let count = history.len() + usize::from(eos);
    let timing = Timing {
        seconds,
        prefill_seconds,
        ttft_seconds: first,
        output_tokens: count,
        content_tokens: history.len(),
        output_tps: count as f64 / seconds,
        prefix_tokens: prefix.len(),
        cfg_branches: if cfg_scale == 1. { 1 } else { 2 },
        execution: "eager".into(),
        attention: "sdpa".into(),
    };
    Ok(TokenGeneration {
        tokens: history,
        timing,
        truncated: !eos,
    })
}
