//! Phase 2 gates. All generation uses the immutable reference seed/settings.
use anyhow::{ensure, Context, Result};
use candle_core::{DType, Device};
use serde_json::Value;
use std::{path::PathBuf, process::Command};
use yue2::{
    model::YuE2ForCausalLM,
    pipeline::{self, Callbacks, SymbolicPlan},
    protocol::*,
    sampling::{distribution, Phase},
    tokenizer::YuE2TextTokenizer,
};

fn fixtures() -> Result<Option<PathBuf>> {
    let Some(path) = std::env::var_os("YUE2_FIXTURES") else {
        eprintln!("SKIP: YUE2_FIXTURES is unset");
        return Ok(None);
    };
    let mut root = PathBuf::from(path);
    if !root.join("manifest.json").is_file() {
        root = root.join("first-song");
    }
    Ok(Some(root))
}

#[test]
#[ignore = "requires YUE2_FIXTURES and the local tokenizer; tiny CPU model"]
fn pipeline_modes_and_truncation() -> Result<()> {
    let Some(root) = fixtures()? else {
        return Ok(());
    };
    let metadata: Value = serde_json::from_slice(&std::fs::read(root.join("sampling.json"))?)?;
    let reference: SymbolicPlan = serde_json::from_value(metadata["reference_plan"].clone())?;
    let tokenizer =
        YuE2TextTokenizer::new(yue2::model::snapshot_dir("YuE2-3B")?.join("qwen.tiktoken"))?;
    let model = YuE2ForCausalLM::load(
        yue2::model::YuE2Config {
            hidden_size: 2,
            num_hidden_layers: 1,
            num_attention_heads: 1,
            num_key_value_heads: 1,
            head_dim: 2,
            intermediate_size: 2,
            vocab_size: VOCAB_SIZE,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.,
            max_position_embeddings: CONTEXT,
            tie_word_embeddings: false,
        },
        candle_nn::VarBuilder::zeros(DType::F32, &Device::Cpu),
    )?;
    let mut settings = Sampling {
        temperature: 0.,
        min_tokens: 2,
        max_tokens: 2,
        ..Sampling::default()
    };
    let truncated = pipeline::plan(
        &model,
        &tokenizer,
        &reference.request,
        &settings,
        Callbacks::default(),
    )?;
    assert!(truncated.truncated);
    assert_eq!(truncated.abc_ids, [0, 0]);
    assert!(truncated.prefix.ends_with(&[0, 0, ABC_END, MUSIC_START]));
    let semantic = pipeline::generate_semantic(
        &model,
        &tokenizer,
        &truncated,
        &settings,
        Callbacks::default(),
    )?;
    assert!(semantic.plan.truncated && semantic.truncated);
    assert_eq!(semantic.tokens, [0, 0]);
    settings.min_tokens = 0;
    let semantic = pipeline::generate_semantic(
        &model,
        &tokenizer,
        &truncated,
        &settings,
        Callbacks::default(),
    )?;
    assert!(semantic.plan.truncated && !semantic.truncated && semantic.tokens.is_empty());
    let mut corrupted = reference.clone();
    corrupted.prefix.pop();
    assert!(pipeline::generate_semantic(
        &model,
        &tokenizer,
        &corrupted,
        &settings,
        Callbacks::default()
    )
    .is_err());
    let mut request = reference.request.clone();
    request.abc = reference.abc.clone();
    let mut observed = vec![];
    let mut callback = |phase, token| observed.push((phase, token));
    let external = pipeline::plan(
        &model,
        &tokenizer,
        &request,
        &settings,
        Callbacks {
            on_token: Some(&mut callback),
            cancelled: Some(&|| true),
        },
    )?;
    assert!(observed.is_empty() && !external.truncated);
    assert_eq!(external.abc, reference.abc);
    assert_eq!(
        external.abc_ids,
        tokenizer.encode(request.abc.as_ref().unwrap())
    );
    assert_eq!(external.timing["output_tokens"], 0);
    request.abc = None;
    request.cot = "off".into();
    let off = pipeline::plan(
        &model,
        &tokenizer,
        &request,
        &settings,
        Callbacks::default(),
    )?;
    assert!(off.abc.is_none() && off.abc_ids.is_empty() && !off.truncated);
    assert!(off.prefix.ends_with(&[ABC_START, ABC_END, MUSIC_START]));
    let semantic =
        pipeline::generate_semantic(&model, &tokenizer, &off, &settings, Callbacks::default())?;
    assert!(semantic.tokens.is_empty() && !semantic.truncated);
    assert_eq!(semantic.timing["cfg_branches"], 2);
    println!("Pipeline controls PASS: provided ABC, cot=off CFG, exact prefix validation, independent plan/semantic truncation and codec subtraction");
    Ok(())
}

#[test]
#[ignore = "requires YUE2_FIXTURES and --phase2-sampling oracles"]
fn sampling_python_oracle() -> Result<()> {
    let Some(root) = fixtures()? else {
        return Ok(());
    };
    let metadata: Value = serde_json::from_slice(&std::fs::read(root.join("sampling.json"))?)?;
    let device = if std::env::var("YUE2_TEST_DEVICE").as_deref() == Ok("cuda") {
        ensure!(
            std::env::var("CUDA_VISIBLE_DEVICES").as_deref() == Ok("0"),
            "Only GPU 0 authorized"
        );
        Device::new_cuda(0)?
    } else {
        Device::Cpu
    };
    let tensors = candle_core::safetensors::load(root.join("sampling.safetensors"), &device)?;
    for case in metadata["cases"].as_array().context("Missing cases")? {
        let stem = case["stem"].as_str().unwrap();
        let settings: Sampling = serde_json::from_value(case["sampling"].clone())?;
        let phase: Phase = serde_json::from_value(case["phase"].clone())?;
        let history: Vec<u32> = serde_json::from_value(case["history"].clone())?;
        let legacy = case["legacy_off"].as_bool().unwrap();
        let actual = distribution(
            &tensors[&format!("{stem}.logits")],
            &settings,
            &history,
            case["step"].as_u64().unwrap() as usize,
            phase,
            legacy,
        )?;
        let expected = tensors[&format!("{stem}.scores")]
            .flatten_all()?
            .to_vec1::<f32>()?;
        let mut max = 0f32;
        for (i, (&a, &e)) in actual.iter().zip(&expected).enumerate() {
            assert_eq!(a.is_finite(), e.is_finite(), "{stem} token {i}: {a} vs {e}");
            if e.is_finite() {
                max = max.max((a - e).abs());
            }
        }
        println!("Sampling {stem} {phase:?} legacy={legacy}: exact support; max_abs={max}");
        assert!(max <= if legacy { 0. } else { 2e-5 }, "{stem}: {max}");
    }
    let c = &tensors["cfg.conditional"];
    let u = &tensors["cfg.unconditional"];
    let actual = yue2::sampling::cfg_logits(c, u, 1.01)?
        .to_dtype(DType::F32)?
        .to_vec1::<f32>()?;
    assert_eq!(
        actual,
        tensors["cfg.logits"]
            .to_dtype(DType::F32)?
            .to_vec1::<f32>()?
    );
    println!("BF16 CFG subtraction/multiply/add: 1024/1024 exact");
    Ok(())
}

#[test]
#[ignore = "requires YUE2_FIXTURES, local 3B checkpoint; GPU 0 for P2c"]
fn p2_full_generation_gates() -> Result<()> {
    let Some(root) = fixtures()? else {
        return Ok(());
    };
    let metadata: Value = serde_json::from_slice(&std::fs::read(root.join("sampling.json"))?)?;
    let reference_plan: SymbolicPlan = serde_json::from_value(metadata["reference_plan"].clone())?;
    let config = GenerationConfig::from_dict(&metadata["generation"])?;
    let device = match std::env::var("YUE2_TEST_DEVICE")
        .as_deref()
        .unwrap_or("cpu")
    {
        "cpu" => Device::Cpu,
        "cuda" => {
            ensure!(
                std::env::var("CUDA_VISIBLE_DEVICES").as_deref() == Ok("0"),
                "Only GPU 0 authorized"
            );
            Device::new_cuda(0)?
        }
        other => anyhow::bail!("Unknown test device {other}"),
    };
    let model_dir = match std::env::var_os("YUE2_MODEL_DIR") {
        Some(p) => PathBuf::from(p),
        None => yue2::model::snapshot_dir("YuE2-3B")?,
    };
    let tokenizer = YuE2TextTokenizer::new(model_dir.join("qwen.tiktoken"))?;
    // SAFETY: reference checkpoint files are read-only throughout this task.
    let model = unsafe { YuE2ForCausalLM::from_pretrained(model_dir, DType::BF16, &device)? };
    let output = root.join("p2");
    std::fs::create_dir_all(&output)?;
    let mut count = 0;
    let mut on_token = |phase, _| {
        count += 1;
        if count % 250 == 0 {
            println!("{phase:?}: {count} output tokens");
        }
    };
    let plan = pipeline::plan(
        &model,
        &tokenizer,
        &reference_plan.request,
        &config.abc,
        Callbacks {
            on_token: Some(&mut on_token),
            ..Default::default()
        },
    )?;
    std::fs::write(
        output.join("score.abc"),
        plan.abc.as_ref().context("Missing ABC")?,
    )?;
    std::fs::write(output.join("plan.json"), serde_json::to_vec_pretty(&plan)?)?;
    let plan_ratio = plan.abc_ids.len() as f64 / reference_plan.abc_ids.len() as f64;
    println!(
        "P2a ABC tokens={}/{} ratio={plan_ratio:.9}; truncated={}",
        plan.abc_ids.len(),
        reference_plan.abc_ids.len(),
        plan.truncated
    );
    println!("P2c plan timing: {}", plan.timing);
    // Run every gate even if the ABC parser fails. Structure is advisory.
    let parser = Command::new(std::env::var_os("YUE2_PYTHON").unwrap_or_else(|| "python3".into()))
        .arg(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tools/check_phase2.py"
        ))
        .arg(&output)
        .env("PYTHONDONTWRITEBYTECODE", "1")
        .status()?;
    count = 0;
    let mut on_token = |phase, _| {
        count += 1;
        if count % 250 == 0 {
            println!("{phase:?}: {count} output tokens");
        }
    };
    let semantic = pipeline::generate_semantic(
        &model,
        &tokenizer,
        &reference_plan,
        &config.semantic,
        Callbacks {
            on_token: Some(&mut on_token),
            ..Default::default()
        },
    )?;
    std::fs::write(
        output.join("semantic.json"),
        serde_json::to_vec_pretty(&semantic)?,
    )?;
    let tensors = candle_core::safetensors::load(root.join("tokens.safetensors"), &Device::Cpu)?;
    let reference_count = tensors["semantic_ids"].elem_count();
    let semantic_ratio = semantic.tokens.len() as f64 / reference_count as f64;
    println!("P2b semantic tokens={}/{reference_count} ratio={semantic_ratio:.9}; codec min={} max={}; truncated={}",
        semantic.tokens.len(), semantic.tokens.iter().min().unwrap_or(&0), semantic.tokens.iter().max().unwrap_or(&0), semantic.truncated);
    println!("P2c semantic timing: {}", semantic.timing);
    assert!(
        (0.8..=1.2).contains(&semantic_ratio),
        "P2b semantic count gate"
    );
    assert!(semantic.tokens.iter().all(|&t| t < CODEC_SIZE));
    assert_eq!(semantic.plan.prefix, reference_plan.prefix);
    assert_eq!(
        semantic.truncated,
        semantic.timing["output_tokens"].as_u64() == Some(semantic.tokens.len() as u64)
    );
    println!("P2b PASS; codec range, exact Python plan, length and EOS accounting valid");
    assert!(parser.success(), "P2a ABC does not parse");
    println!(
        "P2a advisory length target within 30%: {}",
        (0.7..=1.3).contains(&plan_ratio)
    );
    println!(
        "P2a/P2b PASS; P2c measured on {device:?}; seed={}",
        reference_plan.request.seed
    );
    Ok(())
}
