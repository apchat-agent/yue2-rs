use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use std::cell::Cell;
use yue2::{
    model::{YuE2Config, YuE2ForCausalLM},
    protocol::*,
    sampling::{distribution, generate_tokens, GenerationOptions, Phase},
};

// A real backbone with equal logits: greedy semantic chooses EOS as soon as
// permitted. Before min_tokens it must choose CODEC_OFFSET. No checkpoint needed.
fn model() -> Result<YuE2ForCausalLM> {
    YuE2ForCausalLM::load(
        YuE2Config {
            hidden_size: 2,
            num_hidden_layers: 1,
            num_attention_heads: 1,
            num_key_value_heads: 1,
            head_dim: 2,
            intermediate_size: 2,
            vocab_size: VOCAB_SIZE,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.,
            max_position_embeddings: 32,
            tie_word_embeddings: false,
        },
        VarBuilder::zeros(DType::F32, &Device::Cpu),
    )
}

#[test]
fn eos_minimum_budget_and_callback_accounting() -> Result<()> {
    let model = model()?;
    let mut settings = Sampling {
        temperature: 0.,
        min_tokens: 2,
        max_tokens: 3,
        ..Sampling::default()
    };
    let mut observed = vec![];
    let mut callback = |phase, token| observed.push((phase, token));
    let result = generate_tokens(
        &model,
        &[EOD, MUSIC_START],
        &settings,
        831001,
        Phase::Semantic,
        GenerationOptions {
            on_token: Some(&mut callback),
            ..Default::default()
        },
    )?;
    assert_eq!(result.tokens, [CODEC_OFFSET, CODEC_OFFSET]);
    assert!(!result.truncated);
    assert_eq!(
        observed,
        [
            (Phase::Semantic, CODEC_OFFSET),
            (Phase::Semantic, CODEC_OFFSET),
            (Phase::Semantic, MUSIC_END)
        ]
    );
    assert_eq!(result.timing.output_tokens, 3);
    assert_eq!(result.timing.content_tokens, 2);
    assert_eq!(result.timing.cfg_branches, 1);
    settings.max_tokens = 2;
    let capped = generate_tokens(
        &model,
        &[EOD],
        &settings,
        831001,
        Phase::Semantic,
        GenerationOptions::default(),
    )?;
    assert_eq!(capped.tokens, result.tokens);
    assert!(capped.truncated);
    assert_eq!(capped.timing.output_tokens, 2);
    settings.min_tokens = 0;
    let ended = generate_tokens(
        &model,
        &[EOD],
        &settings,
        831001,
        Phase::Semantic,
        GenerationOptions::default(),
    )?;
    assert!(ended.tokens.is_empty() && !ended.truncated);
    assert_eq!(ended.timing.output_tokens, 1);
    assert!(ended.timing.ttft_seconds.is_some());
    Ok(())
}

#[test]
fn context_cfg_and_cancellation_are_explicit() -> Result<()> {
    let model = model()?;
    let settings = Sampling {
        temperature: 0.,
        min_tokens: 2,
        max_tokens: 2,
        ..Sampling::default()
    };
    let run = |prefix: &[u32], options| {
        generate_tokens(&model, prefix, &settings, 10, Phase::Semantic, options)
    };
    assert!(run(&[EOD; 31], GenerationOptions::default()).is_err());
    assert!(run(&[], GenerationOptions::default()).is_err());
    assert!(run(&[VOCAB_SIZE as u32], GenerationOptions::default()).is_err());
    assert!(run(
        &[EOD],
        GenerationOptions {
            cfg_scale: Some(1.01),
            ..Default::default()
        }
    )
    .is_err());
    assert!(run(
        &[EOD],
        GenerationOptions {
            negative: Some(&[EOD; 31]),
            ..Default::default()
        }
    )
    .is_err());
    let output = run(
        &[EOD, MUSIC_START],
        GenerationOptions {
            negative: Some(&[EOD]),
            cfg_scale: Some(1.01),
            legacy_off: true,
            ..Default::default()
        },
    )?;
    assert!(output.truncated);
    assert_eq!(output.tokens, [CODEC_OFFSET, CODEC_OFFSET]);
    assert_eq!(output.timing.cfg_branches, 2);
    assert!(run(
        &[EOD],
        GenerationOptions {
            cancelled: Some(&|| true),
            ..Default::default()
        }
    )
    .unwrap_err()
    .to_string()
    .contains("Cancelled"));
    let stopped = Cell::new(false);
    let cancel = || stopped.get();
    let mut callback = |_, _| stopped.set(true);
    let err = run(
        &[EOD],
        GenerationOptions {
            cancelled: Some(&cancel),
            on_token: Some(&mut callback),
            ..Default::default()
        },
    )
    .unwrap_err();
    assert!(err.to_string().contains("Cancelled"));
    Ok(())
}

#[test]
fn seeded_cpu_sampling_resets_per_request() -> Result<()> {
    let model = model()?;
    let settings = Sampling {
        temperature: 1.,
        top_p: 1.,
        min_tokens: 4,
        max_tokens: 4,
        ..Sampling::abc_default()
    };
    let run = |seed| {
        generate_tokens(
            &model,
            &[EOD],
            &settings,
            seed,
            Phase::Abc,
            GenerationOptions::default(),
        )
    };
    let first = run(831001)?;
    let other = run(42)?;
    let repeated = run(831001)?;
    assert_eq!(first.tokens, repeated.tokens);
    assert_ne!(first.tokens, other.tokens);
    assert!(first.truncated && first.tokens.iter().all(|&t| t < EOD));
    Ok(())
}

#[test]
fn masks_ties_and_top_p_crossing() -> Result<()> {
    let mut values = vec![f32::NEG_INFINITY; VOCAB_SIZE];
    values[1] = 2.;
    values[2] = 2.;
    values[3] = 1.;
    values[4] = 0.;
    values[CODEC_OFFSET as usize] = 99.;
    let logits = Tensor::new(values.as_slice(), &Device::Cpu)?;
    let mut settings = Sampling {
        temperature: 1.,
        top_p: 1.,
        top_k: 1,
        min_tokens: 0,
        ..Sampling::abc_default()
    };
    let scores = distribution(&logits, &settings, &[], 0, Phase::Abc, false)?;
    assert_eq!(scores.iter().filter(|s| s.is_finite()).count(), 2);
    settings.top_k = 4;
    settings.top_p = 0.5;
    values[1] = 3f32.ln();
    values[2] = 2f32.ln();
    values[3] = 0.;
    values[4] = f32::NEG_INFINITY;
    let scores = distribution(
        &Tensor::new(values, &Device::Cpu)?,
        &settings,
        &[],
        0,
        Phase::Abc,
        false,
    )?;
    // Exactly .5 of the mass precedes token 2: Python's strict > keeps it.
    assert!(scores[1].is_finite() && scores[2].is_finite() && !scores[3].is_finite());
    Ok(())
}
