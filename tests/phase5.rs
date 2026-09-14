//! Real-waveform native storage check, independent of sampled E2E output.
use anyhow::{ensure, Result};
use candle_core::Device;
use std::path::PathBuf;

#[test]
#[ignore = "requires YUE2_FIXTURES and the Rust P4 decoded waveform"]
fn p5_native_audio_storage() -> Result<()> {
    let Some(root) = std::env::var_os("YUE2_FIXTURES") else {
        eprintln!("SKIP: YUE2_FIXTURES is unset");
        return Ok(());
    };
    let mut root = PathBuf::from(root);
    if !root.join("manifest.json").is_file() {
        root = root.join("first-song");
    }
    let output = std::env::var_os("YUE2_P5_AUDIO_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("runs/p5-audio"));
    yue2::storage::ensure_empty_directory(&output)?;
    let tensors = candle_core::safetensors::load(root.join("p4/audio.safetensors"), &Device::Cpu)?;
    let audio = tensors["audio"]
        .squeeze(0)?
        .clamp(-1f32, 1f32)?
        .t()?
        .contiguous()?;
    ensure!(
        audio.dims() == [2849216, 2],
        "Expected full first-song waveform"
    );
    for extension in ["wav", "flac"] {
        yue2::storage::save_audio(output.join(format!("audio.{extension}")), &audio, 48000)?;
    }
    println!(
        "P5 native audio written: FP32 [2849216,2], 48000 Hz, FLOAT WAV / PCM_24 FLAC, {}",
        output.display()
    );
    Ok(())
}
