//! P3 uses the existing immutable P0 noise, never a newly sampled parity input.
use anyhow::{ensure, Context, Result};
use candle_core::{DType, Device, Tensor};
use std::path::PathBuf;
use yue2::{
    model::YuE2ForCausalLM,
    nar::{song_chunks, CachedNAR, Chunk},
};

fn metrics(a: &Tensor, b: &Tensor) -> Result<(f64, f64)> {
    ensure!(a.shape() == b.shape(), "Shape mismatch");
    let a = a.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let b = b.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let (mut max, mut dot, mut aa, mut bb) = (0f64, 0., 0., 0.);
    for (a, b) in a.into_iter().zip(b) {
        ensure!(a.is_finite() && b.is_finite(), "Nonfinite parity value");
        let (a, b) = (a as f64, b as f64);
        max = max.max((a - b).abs());
        dot += a * b;
        aa += a * a;
        bb += b * b;
    }
    Ok((max, dot / (aa * bb).sqrt()))
}

#[test]
#[ignore = "requires YUE2_FIXTURES, 3B checkpoint and the existing P3b FP32 Python control"]
fn p3a_fp32_paired_latents() -> Result<()> {
    paired_latents(DType::F32)
}

#[test]
#[ignore = "requires YUE2_FIXTURES, 3B checkpoint and CUDA GPU 0"]
fn p3b_bf16_latents() -> Result<()> {
    paired_latents(DType::BF16)
}

fn paired_latents(dtype: DType) -> Result<()> {
    let Some(root) = std::env::var_os("YUE2_FIXTURES") else {
        eprintln!("SKIP: YUE2_FIXTURES is unset");
        return Ok(());
    };
    let mut root = PathBuf::from(root);
    if !root.join("manifest.json").is_file() {
        root = root.join("first-song");
    }
    let device = if std::env::var("YUE2_TEST_DEVICE").as_deref() == Ok("cuda") {
        ensure!(
            std::env::var("CUDA_VISIBLE_DEVICES").as_deref() == Ok("0"),
            "Only GPU 0 authorized"
        );
        Device::new_cuda(0)?
    } else {
        Device::Cpu
    };
    if dtype == DType::BF16 {
        ensure!(device.is_cuda(), "P3(b) requires production CUDA BF16");
    }
    let tier = if dtype == DType::F32 {
        "P3(a) FP32"
    } else {
        "P3(b) BF16"
    };
    let tensors = candle_core::safetensors::load(root.join("nar.safetensors"), &Device::Cpu)?;
    let metadata: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("nar.json"))?)?;
    let steps = metadata["steps"].as_u64().context("Missing steps")? as usize;
    ensure!(
        steps == 32 && metadata["method"] == "midpoint",
        "P3 requires 32 midpoint steps"
    );
    // Reuse the paired FP32 Python path established in P3b, preserving P0.
    let fp32 = if dtype == DType::F32 {
        Some(candle_core::safetensors::load(
            root.join("p3b-python.safetensors"),
            &Device::Cpu,
        )?)
    } else {
        report_bf16_stages(&root)?;
        None
    };
    let directory = std::env::var_os("YUE2_MODEL_DIR")
        .map(PathBuf::from)
        .map(Ok)
        .unwrap_or_else(|| yue2::model::snapshot_dir("YuE2-3B"))?;
    // SAFETY: local checkpoint is immutable throughout the run.
    let model = unsafe { YuE2ForCausalLM::from_pretrained_with_nar(directory, dtype, &device)? };
    let mut passed = true;
    let mut outputs = Vec::new();
    for index in 0..2 {
        let stem = format!("two_chunk.{index}");
        let ids = tensors[&format!("{stem}.ar_tokens")]
            .to_dtype(DType::U32)?
            .flatten_all()?
            .to_vec1::<u32>()?;
        let chunk = Chunk {
            ar_tokens: ids,
            noise: tensors[&format!("{stem}.noise")].clone(),
            nar_cond_end: 0,
        };
        let engine = CachedNAR::new(&model, &chunk, 128)?;
        if dtype == DType::BF16 && std::env::var_os("YUE2_NAR_DIAGNOSTIC").is_some() {
            for step in [0, 1, 16, 31] {
                let key = format!("{stem}.step.{step:02}");
                let raw = tensors[&format!("{key}.raw_t")].to_scalar::<f64>()?;
                let v =
                    engine.velocity(&tensors[&format!("{key}.state")].to_device(&device)?, raw)?;
                let (max, cos) = metrics(&v, &tensors[&format!("{key}.first")])?;
                println!("P3 velocity {key}: max_abs={max:.12} cosine={cos:.12}");
            }
        }
        let start = std::time::Instant::now();
        let mut progress = |done, total| {
            if done % 8 == 0 {
                println!("{tier} {stem} step {done}/{total}");
            }
        };
        let actual = engine.solve(steps, None, Some(&mut progress))?;
        let reference = match &fp32 {
            Some(control) => &control[&format!("control.fp32.chunk.{index}")],
            None => &tensors[&format!("{stem}.latents")],
        };
        let (max, cos) = metrics(&actual, reference)?;
        println!(
            "{tier} {stem}: frames={} max_abs={max:.12} cosine={cos:.12} seconds={:.6}",
            actual.dim(0)?,
            start.elapsed().as_secs_f64()
        );
        passed &= if dtype == DType::F32 {
            max <= 1e-3 && cos >= 0.999999
        } else {
            cos >= 0.999 // max_abs and stage diagnostics are advisory in tier (b).
        };
        outputs.push((stem, actual));
    }
    std::fs::create_dir_all(root.join("p3"))?;
    candle_core::safetensors::save(
        &outputs
            .into_iter()
            .collect::<std::collections::HashMap<_, _>>(),
        root.join(if dtype == DType::F32 {
            "p3/latents-fp32.safetensors"
        } else {
            "p3/latents.safetensors"
        }),
    )?;
    ensure!(
        passed,
        "{tier}: both chunks must satisfy the TASK.md gate (FP32 max_abs <= 1e-3 and cosine >= 0.999999; BF16 cosine >= 0.999)"
    );
    Ok(())
}

// These native-stage measurements are the retained P3b investigation, not
// additional pass/fail criteria. This test reruns both complete trajectories.
fn report_bf16_stages(root: &std::path::Path) -> Result<()> {
    let path = root.join("p3b-comparison.json");
    let report: serde_json::Value = serde_json::from_slice(&std::fs::read(&path)?)?;
    let rows = &report["rows"];
    println!(
        "P3(b) native-stage diagnostic source: {} (retained P3b trace)",
        path.display()
    );
    for name in [
        "noise",
        "state",
        "rope.inv_freq",
        "ar.cos",
        "ar.embedding.output",
        "ar.layer.00.norm.output",
        "ar.layer.00.q_proj.output",
        "ar.layer.00.q",
        "audio.output",
        "time.freqs",
        "time.output",
        "vae2llm.output",
        "injected",
        "nar.layer.00.output",
        "final_norm.output",
        "velocity",
        "step.00.mid",
        "step.00.next",
        "step.00.raw_t",
        "latents",
    ] {
        let row = &rows[name];
        println!(
            "P3(b) stage {name}: Python={} Rust={} max_abs={}",
            row["python_dtype"], row["rust_dtype"], row["max_abs"]
        );
    }
    let first = report["first_over_1e-3"]
        .as_str()
        .context("Missing first divergent stage")?;
    println!("P3(b) first divergent operation above 1e-3: {first} (AR layer 0 Q after RoPE), max_abs={}; earlier nonzero FP32 precursor rope.inv_freq max_abs={}",
        rows[first]["max_abs"], rows["rope.inv_freq"]["max_abs"]);
    Ok(())
}

#[test]
#[ignore = "requires YUE2_FIXTURES and CUDA GPU 0 for RNG isolation"]
fn nar_cuda_noise_owns_seed() -> Result<()> {
    if std::env::var_os("YUE2_FIXTURES").is_none() {
        eprintln!("SKIP: YUE2_FIXTURES is unset");
        return Ok(());
    }
    if std::env::var("YUE2_TEST_DEVICE").as_deref() != Ok("cuda") {
        eprintln!("SKIP: CUDA RNG test");
        return Ok(());
    }
    ensure!(
        std::env::var("CUDA_VISIBLE_DEVICES").as_deref() == Ok("0"),
        "Only GPU 0 authorized"
    );
    let device = Device::new_cuda(0)?;
    let draw = || -> Result<Vec<f32>> { Ok(Tensor::rand(0f32, 1f32, 128, &device)?.to_vec1()?) };
    device.set_seed(17)?;
    let expected = draw()?;
    device.set_seed(17)?;
    let a = song_chunks(&[1, 2], &[3; 9], 831001, 23, &device)?;
    assert_eq!(expected, draw()?, "NAR must not advance or reseed AR RNG");
    // Same operations as Phase 2's request RNG reset and capture.
    device.set_seed(456)?;
    draw()?;
    let b = song_chunks(&[1, 2], &[3; 9], 831001, 13, &device)?;
    let flatten = |chunks: &[Chunk]| -> Result<Vec<f32>> {
        Ok(
            Tensor::cat(&chunks.iter().map(|c| &c.noise).collect::<Vec<_>>(), 0)?
                .flatten_all()?
                .to_vec1()?,
        )
    };
    assert_eq!(
        flatten(&a)?,
        flatten(&b)?,
        "AR reseeding and chunk cuts must not change NAR noise"
    );
    let c = song_chunks(&[1, 2], &[3; 9], 831002, 23, &device)?;
    assert_ne!(flatten(&a)?, flatten(&c)?);
    println!("NAR CUDA RNG PASS: 576/576 values repeat after AR reseed; context-independent cuts; different seed differs; AR stream unchanged");
    Ok(())
}
