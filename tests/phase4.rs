//! P4 compares the freely evolving FP32 decoder with the immutable P0 oracle.
use anyhow::{ensure, Result};
use candle_core::{DType, Device, Tensor};
use std::{collections::HashMap, path::PathBuf};
use yue2::vae::YuE2VAE;

fn fixture() -> Result<Option<(PathBuf, Device)>> {
    let Some(root) = std::env::var_os("YUE2_FIXTURES") else {
        eprintln!("SKIP: YUE2_FIXTURES is unset");
        return Ok(None);
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
    Ok(Some((root, device)))
}

fn model(device: &Device) -> Result<YuE2VAE> {
    let directory = std::env::var_os("YUE2_VAE_DIR")
        .map(PathBuf::from)
        .map(Ok)
        .unwrap_or_else(|| yue2::model::snapshot_dir("YuE2-Vae"))?;
    // SAFETY: the local checkpoint is immutable throughout this run.
    unsafe { YuE2VAE::from_pretrained(directory, device) }
}

fn metrics(actual: &Tensor, reference: &Tensor) -> Result<(f64, f64)> {
    ensure!(
        actual.shape() == reference.shape(),
        "VAE parity shape mismatch"
    );
    ensure!(
        actual.dtype() == DType::F32 && reference.dtype() == DType::F32,
        "P4 requires FP32"
    );
    let a = actual
        .to_device(&Device::Cpu)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let b = reference
        .to_device(&Device::Cpu)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let (mut max, mut signal, mut error) = (0f64, 0f64, 0f64);
    for (a, b) in a.into_iter().zip(b) {
        ensure!(a.is_finite() && b.is_finite(), "Nonfinite VAE parity value");
        let diff = a as f64 - b as f64;
        max = max.max(diff.abs());
        signal += (b as f64).powi(2);
        error += diff * diff;
    }
    Ok((max, 10. * (signal / error).log10()))
}

#[test]
#[ignore = "requires YUE2_FIXTURES and the FP32 VAE checkpoint"]
fn p4a_decoder_blocks() -> Result<()> {
    let Some((root, device)) = fixture()? else {
        return Ok(());
    };
    let reference = candle_core::safetensors::load(root.join("vae.safetensors"), &Device::Cpu)?;
    let model = model(&device)?;
    let (mut passed, mut count) = (true, 0);
    let start = std::time::Instant::now();
    let mut observe = |index, actual: &Tensor| {
        let (max, _) = metrics(actual, &reference[&format!("block.{index}")])?;
        println!(
            "P4a block.{index}: shape={:?} dtype={:?} max_abs={max:.12}",
            actual.dims(),
            actual.dtype()
        );
        passed &= max <= 1e-3;
        count += 1;
        Ok(())
    };
    let audio = model.decode_with_blocks(&reference["slice"], Some(&mut observe))?;
    let (max, snr) = metrics(&audio, &reference["slice_audio"])?;
    println!(
        "P4a slice audio (reported): shape={:?} max_abs={max:.12} snr_db={snr:.6} seconds={:.6}",
        audio.dims(),
        start.elapsed().as_secs_f64()
    );
    ensure!(
        count == 6 && passed,
        "P4a requires all six DecoderBlock outputs max_abs <= 1e-3"
    );
    Ok(())
}

#[test]
#[ignore = "requires YUE2_FIXTURES and the FP32 VAE checkpoint"]
fn p4b_reference_audio() -> Result<()> {
    let Some((root, device)) = fixture()? else {
        return Ok(());
    };
    let reference = candle_core::safetensors::load(root.join("vae.safetensors"), &Device::Cpu)?;
    let model = model(&device)?;
    let latent = reference["latents"].t()?.unsqueeze(0)?.contiguous()?;
    ensure!(model.config().sample_rate == 48000, "P4 requires 48 kHz");
    println!(
        "P4b config: frames={} core=1024 halo=16 required_halo={} FP32, unclipped",
        latent.dim(2)?,
        model.required_halo(1024)?
    );
    let start = std::time::Instant::now();
    let mut progress = |done, total| println!("P4b tile {done}/{total}");
    let audio = model.decode_tiled(&latent, Some(1024), Some(16), Some(&mut progress))?;
    ensure!(
        audio.dims() == [1, 2, model.natural_output_length(latent.dim(2)?)?],
        "Wrong full audio length"
    );
    ensure!(
        audio
            .flatten_all()?
            .to_vec1::<f32>()?
            .iter()
            .all(|x| x.is_finite()),
        "Nonfinite full audio"
    );
    let (max, snr) = metrics(&audio.narrow(2, 0, 4 * 48000)?, &reference["audio"])?;
    println!("P4b first 4 seconds: samples=192000 channels=2 max_abs={max:.12} snr_db={snr:.6}");
    println!(
        "P4b full audio: shape={:?} seconds={:.6}",
        audio.dims(),
        start.elapsed().as_secs_f64()
    );
    std::fs::create_dir_all(root.join("p4"))?;
    candle_core::safetensors::save(
        &HashMap::from([("audio".to_string(), audio)]),
        root.join("p4/audio.safetensors"),
    )?;
    ensure!(
        snr >= 40.,
        "P4b requires SNR >= 40 dB over the first 4 seconds"
    );
    Ok(())
}
