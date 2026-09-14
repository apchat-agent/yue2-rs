//! Symbolic, semantic and acoustic stages of yue2-infer 0.1.6 pipeline.py.
use crate::{
    model::YuE2ForCausalLM,
    protocol::*,
    sampling::{generate_tokens, GenerationOptions, Phase},
    tokenizer::YuE2TextTokenizer,
};
use anyhow::{ensure, Result};
use candle_core::{DType, Device, Tensor};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    path::{Path, PathBuf},
    time::Instant,
};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SymbolicPlan {
    pub request: SongRequest,
    pub abc: Option<String>,
    pub abc_ids: Vec<u32>,
    pub prefix: Vec<u32>,
    #[serde(default = "empty_timing")]
    pub timing: Value,
    #[serde(default)]
    pub truncated: bool,
}
fn empty_timing() -> Value {
    json!({})
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SemanticResult {
    pub plan: SymbolicPlan,
    /// Codec indices in [0, CODEC_SIZE), after subtracting CODEC_OFFSET.
    pub tokens: Vec<u32>,
    pub timing: Value,
    pub truncated: bool,
}

#[derive(Default)]
pub struct Callbacks<'a> {
    pub cancelled: Option<&'a dyn Fn() -> bool>,
    pub on_token: Option<&'a mut dyn FnMut(Phase, u32)>,
}

/// Plan generation deliberately uses one branch, regardless of semantic CFG.
pub fn plan(
    model: &YuE2ForCausalLM,
    tokenizer: &YuE2TextTokenizer,
    request: &SongRequest,
    sampling: &Sampling,
    callbacks: Callbacks<'_>,
) -> Result<SymbolicPlan> {
    if let Some(plan) = external_plan(tokenizer, request)? {
        return Ok(plan);
    }
    let result = generate_tokens(
        model,
        &token_prefixes(request, tokenizer, None)?,
        sampling,
        request.seed,
        Phase::Abc,
        GenerationOptions {
            cancelled: callbacks.cancelled,
            on_token: callbacks.on_token,
            ..Default::default()
        },
    )?;
    let abc = tokenizer.decode(
        &result
            .tokens
            .iter()
            .map(|&v| i64::from(v))
            .collect::<Vec<_>>(),
    )?;
    let prefix = token_prefixes(request, tokenizer, Some(&result.tokens))?;
    Ok(SymbolicPlan {
        request: request.clone(),
        abc: Some(abc),
        abc_ids: result.tokens,
        prefix,
        timing: serde_json::to_value(result.timing)?,
        truncated: result.truncated,
    })
}

fn external_plan(
    tokenizer: &YuE2TextTokenizer,
    request: &SongRequest,
) -> Result<Option<SymbolicPlan>> {
    request.validate()?;
    if request.cot == "off" {
        return Ok(Some(SymbolicPlan {
            request: request.clone(),
            abc: None,
            abc_ids: vec![],
            prefix: token_prefixes(request, tokenizer, None)?,
            timing: empty_timing(),
            truncated: false,
        }));
    }
    if let Some(abc) = &request.abc {
        let ids = tokenizer.encode(abc);
        let prefix = token_prefixes(request, tokenizer, Some(&ids))?;
        let timing =
            json!({"seconds": 0., "output_tokens": 0, "external_prefix_tokens": ids.len()});
        return Ok(Some(SymbolicPlan {
            request: request.clone(),
            abc: Some(abc.clone()),
            abc_ids: ids,
            prefix,
            timing,
            truncated: false,
        }));
    }
    Ok(None)
}

pub fn generate_semantic(
    model: &YuE2ForCausalLM,
    tokenizer: &YuE2TextTokenizer,
    plan: &SymbolicPlan,
    sampling: &Sampling,
    callbacks: Callbacks<'_>,
) -> Result<SemanticResult> {
    let request = &plan.request;
    ensure!(
        token_prefixes(request, tokenizer, Some(&plan.abc_ids))? == plan.prefix,
        "Plan prefix disagrees with request/exact ABC IDs"
    );
    let negative = if request.guidance() != 1. {
        Some(negative_prefix(request, tokenizer, Some(&plan.abc_ids))?)
    } else {
        None
    };
    let result = generate_tokens(
        model,
        &plan.prefix,
        sampling,
        request.seed,
        Phase::Semantic,
        GenerationOptions {
            negative: negative.as_deref(),
            cfg_scale: Some(request.guidance()),
            legacy_off: request.cot == "off",
            cancelled: callbacks.cancelled,
            on_token: callbacks.on_token,
        },
    )?;
    ensure!(
        result
            .tokens
            .iter()
            .all(|t| (CODEC_OFFSET..CODEC_OFFSET + CODEC_SIZE).contains(t)),
        "Generated token outside codec vocabulary"
    );
    Ok(SemanticResult {
        plan: plan.clone(),
        tokens: result
            .tokens
            .into_iter()
            .map(|t| t - CODEC_OFFSET)
            .collect(),
        timing: serde_json::to_value(result.timing)?,
        truncated: result.truncated,
    })
}

/// Acoustic stage with exact retained plan validation and request-owned noise.
pub fn synthesize(
    model: &YuE2ForCausalLM,
    tokenizer: &YuE2TextTokenizer,
    semantic: &SemanticResult,
    options: crate::nar::SynthesisOptions<'_>,
) -> Result<candle_core::Tensor> {
    ensure!(
        token_prefixes(
            &semantic.plan.request,
            tokenizer,
            Some(&semantic.plan.abc_ids)
        )? == semantic.plan.prefix,
        "Semantic result does not retain the request's exact prefix"
    );
    crate::nar::synthesize(
        model,
        &semantic.plan.prefix,
        &semantic.tokens,
        semantic.plan.request.seed,
        options,
    )
}

/// Decode CPU FP32 [T,64] or [1,64,T] into clipped CPU FP32 [samples,2].
pub fn decode(vae: &crate::vae::YuE2VAE, latents: &Tensor) -> Result<Tensor> {
    let z = match latents.dims() {
        [_, 64] => latents.t()?.unsqueeze(0)?,
        [1, 64, _] => latents.clone(),
        _ => anyhow::bail!("Expected latents [T,64] or [1,64,T]"),
    }
    .to_dtype(DType::F32)?;
    // Python pipeline explicitly selects core=1024 and halo=16.
    let audio = vae.decode_tiled(&z, Some(1024), Some(16), None)?;
    crate::nar::finite(&audio)?;
    Ok(audio.squeeze(0)?.clamp(-1f32, 1f32)?.t()?.contiguous()?)
}

pub struct SongResult {
    pub audio: Tensor,
    pub sample_rate: usize,
    pub semantic: SemanticResult,
    pub latents: Tensor,
    pub config: Value,
    pub weights: Value,
    pub timing: Value,
    pub request_identity: String,
}

/// Lazy checkpoint-native pipeline. Drops the backbone before VAE allocation.
/// Models are never converted or copied on disk; directories must stay immutable.
pub struct YuE2Pipeline {
    model_dir: PathBuf,
    vae_dir: PathBuf,
    device: Device,
    tokenizer: YuE2TextTokenizer,
    model: Option<YuE2ForCausalLM>,
    generation_config: GenerationConfig,
    weights: Value,
    load_timing: Value,
    runtime_sha256: String,
    decoder_release: Value,
    pub progress: bool,
}

impl YuE2Pipeline {
    /// # Safety
    /// Both checkpoint files must remain unchanged while this pipeline is alive.
    pub unsafe fn from_pretrained(
        model_dir: impl AsRef<Path>,
        vae_dir: impl AsRef<Path>,
        device: Device,
        generation_config: GenerationConfig,
    ) -> Result<Self> {
        generation_config.validate()?;
        let start = Instant::now();
        let model_dir = model_dir.as_ref().to_path_buf();
        let vae_dir = vae_dir.as_ref().to_path_buf();
        let tokenizer = YuE2TextTokenizer::new(model_dir.join("qwen.tiktoken"))?;
        let weights = json!({"mot": crate::storage::model_identity(&model_dir)?,
            "vae": crate::storage::model_identity(&vae_dir)?});
        let vae_config: Value =
            serde_json::from_slice(&std::fs::read(vae_dir.join("config.json"))?)?;
        let runtime_sha256 = crate::storage::sha256_file(std::env::current_exe()?)?;
        Ok(Self {
            model_dir,
            vae_dir,
            device,
            tokenizer,
            model: None,
            generation_config,
            weights,
            load_timing: json!({"resolve_and_integrity_seconds": start.elapsed().as_secs_f64()}),
            runtime_sha256,
            decoder_release: vae_config["release_variant"].clone(),
            progress: false,
        })
    }

    fn load_model(&mut self) -> Result<()> {
        if self.model.is_none() {
            self.status("Loading model");
            let start = Instant::now();
            let dtype = if self.device.is_cuda() {
                DType::BF16
            } else {
                DType::F32
            };
            // SAFETY: caller promises immutable checkpoint files for pipeline lifetime.
            self.model = Some(unsafe {
                YuE2ForCausalLM::from_pretrained_with_nar(&self.model_dir, dtype, &self.device)?
            });
            self.device.synchronize()?;
            self.load_timing["mot_load_seconds"] = json!(start.elapsed().as_secs_f64());
        }
        Ok(())
    }

    fn status(&self, text: &str) {
        if self.progress {
            eprintln!("{text}");
        }
    }

    pub fn plan(&mut self, request: &SongRequest) -> Result<SymbolicPlan> {
        if let Some(plan) = external_plan(&self.tokenizer, request)? {
            return Ok(plan);
        }
        self.load_model()?;
        self.status("Planning score");
        let mut count = 0;
        let mut on_token = |_, _| {
            count += 1;
            if count % 250 == 0 {
                eprintln!("Planning score: {count} tokens");
            }
        };
        plan(
            self.model.as_ref().expect("loaded model"),
            &self.tokenizer,
            request,
            &self.generation_config.abc,
            Callbacks {
                on_token: self.progress.then_some(&mut on_token),
                ..Default::default()
            },
        )
    }

    pub fn generate_semantic(&mut self, plan: &SymbolicPlan) -> Result<SemanticResult> {
        self.load_model()?;
        self.status("Generating song");
        let mut count = 0;
        let mut on_token = |_, _| {
            count += 1;
            if count % 250 == 0 {
                eprintln!("Generating song: {count} tokens");
            }
        };
        generate_semantic(
            self.model.as_ref().expect("loaded model"),
            &self.tokenizer,
            plan,
            &self.generation_config.semantic,
            Callbacks {
                on_token: self.progress.then_some(&mut on_token),
                ..Default::default()
            },
        )
    }

    pub fn synthesize(&mut self, semantic: &SemanticResult) -> Result<Tensor> {
        self.load_model()?;
        self.status("Synthesizing audio");
        let mut on_progress = |done, total| {
            if done % 8 == 0 {
                eprintln!("Synthesizing audio: {done}/{total} steps");
            }
        };
        synthesize(
            self.model.as_ref().expect("loaded model"),
            &self.tokenizer,
            semantic,
            crate::nar::SynthesisOptions {
                steps: self.generation_config.ode_steps,
                context: self.generation_config.context,
                on_progress: self.progress.then_some(&mut on_progress),
                ..Default::default()
            },
        )
    }

    pub fn decode(&mut self, latents: &Tensor) -> Result<Tensor> {
        self.status("Decoding audio");
        // Python moves AR to CPU before loading the VAE. Releasing it also keeps
        // the next request lazy, without retaining a second host weight allocation.
        self.model = None;
        let start = Instant::now();
        // SAFETY: caller promises immutable checkpoint files for pipeline lifetime.
        let vae = unsafe { crate::vae::YuE2VAE::from_pretrained(&self.vae_dir, &self.device)? };
        self.device.synchronize()?;
        self.load_timing["vae_load_seconds"] = json!(start.elapsed().as_secs_f64());
        decode(&vae, latents)
    }

    pub fn effective_config(&self, request: &SongRequest) -> Result<Value> {
        request.validate()?;
        let config = self.generation_config.to_dict()?;
        let defaults = GenerationConfig::default().to_dict()?;
        let mut overrides = serde_json::Map::new();
        for (key, value) in config.as_object().expect("config object") {
            if defaults[key] != *value {
                overrides.insert(key.clone(), value.clone());
            }
        }
        if request.guidance() != if request.cot == "off" { 1.01 } else { 1.0 } {
            overrides.insert("cfg_scale".into(), json!(request.guidance()));
        }
        Ok(
            json!({"generation": config, "overrides": overrides, "cot": request.cot,
            "cfg_scale": request.guidance(),
            "cfg_negative": if request.cot == "off" { "instruction_only" } else { "same_instruction_and_exact_abc" },
            "backend": "candle-eager", "quantization": "none",
            "model_dtype": if self.device.is_cuda() { "bfloat16" } else { "float32" },
            "vae_dtype": "float32", "vae_decode": "halo_crop", "vae_core_frames": 1024, "vae_halo_frames": 16,
            "device": format!("{:?}", self.device.location()), "offload_ar": false,
            "runtime_sha256": self.runtime_sha256, "decoder_release": self.decoder_release,
            "validation_status": "unvalidated"}),
        )
    }

    pub fn generate(&mut self, request: &SongRequest) -> Result<SongResult> {
        let config = self.effective_config(request)?;
        let request_identity = crate::storage::identity(&json!({"request": request.to_dict()?,
            "config": config, "weights": self.weights}))?;
        self.device.synchronize()?;
        let start = Instant::now();
        let plan = self.plan(request)?;
        let semantic = self.generate_semantic(&plan)?;
        let nar_start = Instant::now();
        let latents = self.synthesize(&semantic)?;
        self.device.synchronize()?;
        let nar_seconds = nar_start.elapsed().as_secs_f64();
        let vae_start = Instant::now();
        let audio = self.decode(&latents)?;
        self.device.synchronize()?;
        let timing = json!({"abc": plan.timing, "semantic": semantic.timing,
            "nar_seconds": nar_seconds, "vae_seconds": vae_start.elapsed().as_secs_f64(),
            "load": self.load_timing, "e2e_seconds": start.elapsed().as_secs_f64()});
        Ok(SongResult {
            audio,
            sample_rate: 48000,
            semantic,
            latents,
            config,
            weights: self.weights.clone(),
            timing,
            request_identity,
        })
    }
}
