//! Symbolic, semantic and acoustic stages of yue2-infer 0.1.6 pipeline.py.
use crate::{
    model::YuE2ForCausalLM,
    protocol::*,
    sampling::{generate_tokens, GenerationOptions, Phase},
    tokenizer::YuE2TextTokenizer,
};
use anyhow::{ensure, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

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
    request.validate()?;
    if request.cot == "off" {
        return Ok(SymbolicPlan {
            request: request.clone(),
            abc: None,
            abc_ids: vec![],
            prefix: token_prefixes(request, tokenizer, None)?,
            timing: empty_timing(),
            truncated: false,
        });
    }
    if let Some(abc) = &request.abc {
        let ids = tokenizer.encode(abc);
        let prefix = token_prefixes(request, tokenizer, Some(&ids))?;
        let timing =
            json!({"seconds": 0., "output_tokens": 0, "external_prefix_tokens": ids.len()});
        return Ok(SymbolicPlan {
            request: request.clone(),
            abc: Some(abc.clone()),
            abc_ids: ids,
            prefix,
            timing,
            truncated: false,
        });
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
