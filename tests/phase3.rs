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
#[ignore = "requires YUE2_FIXTURES and 3B checkpoint; GPU 0 for the BF16 P3 gate"]
fn p3_dumped_noise_latents() -> Result<()> {
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
    let tensors = candle_core::safetensors::load(root.join("nar.safetensors"), &Device::Cpu)?;
    let metadata: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("nar.json"))?)?;
    let steps = metadata["steps"].as_u64().context("Missing steps")? as usize;
    let directory = std::env::var_os("YUE2_MODEL_DIR")
        .map(PathBuf::from)
        .map(Ok)
        .unwrap_or_else(|| yue2::model::snapshot_dir("YuE2-3B"))?;
    // SAFETY: local checkpoint is immutable throughout the run.
    let model =
        unsafe { YuE2ForCausalLM::from_pretrained_with_nar(directory, DType::BF16, &device)? };
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
        if std::env::var_os("YUE2_NAR_DIAGNOSTIC").is_some() {
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
                println!("P3 {stem} step {done}/{total}");
            }
        };
        let actual = engine.solve(steps, None, Some(&mut progress))?;
        let (max, cos) = metrics(&actual, &tensors[&format!("{stem}.latents")])?;
        println!(
            "P3 {stem}: frames={} max_abs={max:.12} cosine={cos:.12} seconds={:.6}",
            actual.dim(0)?,
            start.elapsed().as_secs_f64()
        );
        passed &= max <= 1e-2 && cos >= 0.999;
        outputs.push((stem, actual));
    }
    std::fs::create_dir_all(root.join("p3"))?;
    candle_core::safetensors::save(
        &outputs
            .into_iter()
            .collect::<std::collections::HashMap<_, _>>(),
        root.join("p3/latents.safetensors"),
    )?;
    ensure!(
        passed,
        "P3 requires max_abs <= 0.01 and cosine >= 0.999 for BOTH chunks"
    );
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
