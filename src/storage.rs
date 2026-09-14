//! NumPy, audio and hashed artifact protocol from yue2-infer storage/pipeline.py.
use crate::pipeline::{SongResult, SymbolicPlan};
use anyhow::{ensure, Context, Result};
use candle_core::{Device, Tensor};
use flacenc::{component::BitRepr, error::Verify};
use serde::Serialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File},
    io::{BufReader, BufWriter, Read, Write},
    path::{Path, PathBuf},
};

pub fn sha256_file(path: impl AsRef<Path>) -> Result<String> {
    let mut input = BufReader::new(File::open(path)?);
    let mut hash = Sha256::new();
    let mut buffer = vec![0; 1024 * 1024];
    loop {
        let count = input.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hash.update(&buffer[..count]);
    }
    Ok(format!("{:x}", hash.finalize()))
}

/// Python's sorted compact UTF-8 JSON, including its float exponent spelling.
pub fn identity(value: &Value) -> Result<String> {
    fn canonical(value: &Value, out: &mut String) -> Result<()> {
        match value {
            Value::Number(n) if n.is_f64() => {
                let number = n.as_f64().context("Invalid float")?;
                if number != 0. && !(1e-4..1e16).contains(&number.abs()) {
                    let text = format!("{number:e}");
                    let (mantissa, exponent) = text.split_once('e').context("Missing exponent")?;
                    let exponent: i32 = exponent.parse()?;
                    out.push_str(&format!("{mantissa}e{exponent:+03}"));
                } else {
                    let text = number.to_string();
                    out.push_str(&text);
                    if !text.contains('.') {
                        out.push_str(".0");
                    }
                }
            }
            Value::Array(values) => {
                out.push('[');
                for (index, value) in values.iter().enumerate() {
                    if index != 0 {
                        out.push(',');
                    }
                    canonical(value, out)?;
                }
                out.push(']');
            }
            Value::Object(values) => {
                out.push('{');
                // Explicit sorting also works if a downstream crate enables
                // serde_json's preserve_order feature through feature unification.
                let sorted: std::collections::BTreeMap<_, _> = values.iter().collect();
                for (index, (key, value)) in sorted.into_iter().enumerate() {
                    if index != 0 {
                        out.push(',');
                    }
                    out.push_str(&serde_json::to_string(key)?);
                    out.push(':');
                    canonical(value, out)?;
                }
                out.push('}');
            }
            _ => out.push_str(&serde_json::to_string(value)?),
        }
        Ok(())
    }
    let mut text = String::new();
    canonical(value, &mut text)?;
    Ok(format!("{:x}", Sha256::digest(text.as_bytes())))
}

pub fn write_json(path: impl AsRef<Path>, value: &impl Serialize) -> Result<()> {
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let temporary = path.with_extension(format!("json.{}.tmp", std::process::id()));
    let mut text = serde_json::to_vec_pretty(value)?;
    text.push(b'\n');
    fs::write(&temporary, text)?;
    fs::rename(temporary, path)?;
    Ok(())
}

pub fn model_identity(directory: &Path) -> Result<Value> {
    // The checkpoint-native loader currently accepts one model.safetensors.
    let path = directory.join("model.safetensors");
    let files = json!({"model.safetensors": {"sha256": sha256_file(&path)?, "bytes": path.metadata()?.len()}});
    let manifest = directory.join("weights_manifest.json");
    if manifest.is_file() {
        let expected: Value = serde_json::from_slice(&fs::read(manifest)?)?;
        ensure!(expected["files"] == files, "Weight integrity failed");
    }
    Ok(json!({"files": files, "config_sha256": sha256_file(directory.join("config.json"))?}))
}

pub fn ensure_empty_directory(directory: &Path) -> Result<()> {
    if directory.exists() {
        ensure!(
            fs::read_dir(directory)?.next().is_none(),
            "Nonempty output {}; use a new output directory",
            directory.display()
        );
    }
    fs::create_dir_all(directory)?;
    Ok(())
}

/// Candle has no signed i32 dtype; emit NumPy v1 C-order little-endian int32.
pub fn write_token_npy(path: impl AsRef<Path>, ids: &[u32]) -> Result<()> {
    ensure!(
        ids.iter().all(|&v| v <= i32::MAX as u32),
        "Token exceeds int32"
    );
    let mut file = BufWriter::new(File::create(path)?);
    let mut header = format!(
        "{{'descr': '<i4', 'fortran_order': False, 'shape': ({},), }}",
        ids.len()
    );
    // Match NumPy's alignment; the ten bytes precede the header.
    let padding = 64 - (10 + header.len() + 1) % 64;
    header.extend(std::iter::repeat_n(' ', padding));
    header.push('\n');
    file.write_all(b"\x93NUMPY\x01\x00")?;
    file.write_all(&u16::try_from(header.len())?.to_le_bytes())?;
    file.write_all(header.as_bytes())?;
    for &id in ids {
        file.write_all(&(id as i32).to_le_bytes())?;
    }
    file.flush()?;
    Ok(())
}

/// Interleaved FP32 [samples,2], already clipped by pipeline::decode.
/// FLAC stores PCM_24; WAV stores lossless IEEE FLOAT as Python does.
pub fn save_audio(path: impl AsRef<Path>, audio: &Tensor, sample_rate: usize) -> Result<()> {
    let path = path.as_ref();
    let (frames, channels) = audio.dims2()?;
    ensure!(
        frames > 0 && channels == 2 && sample_rate > 0,
        "Expected nonempty stereo audio"
    );
    let samples = audio
        .to_device(&Device::Cpu)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    ensure!(
        samples
            .iter()
            .all(|v| v.is_finite() && (-1.0..=1.0).contains(v)),
        "Audio must be finite and clipped"
    );
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    match path
        .extension()
        .and_then(|s| s.to_str())
        .map(str::to_ascii_lowercase)
        .as_deref()
    {
        Some("wav") => {
            let mut writer = hound::WavWriter::create(
                path,
                hound::WavSpec {
                    channels: 2,
                    sample_rate: u32::try_from(sample_rate)?,
                    bits_per_sample: 32,
                    sample_format: hound::SampleFormat::Float,
                },
            )?;
            for sample in samples {
                writer.write_sample(sample)?;
            }
            writer.finalize()?;
        }
        Some("flac") => {
            let pcm: Vec<i32> = samples
                .into_iter()
                .map(|v| {
                    // Round in f64 so conversion itself cannot lose a half-step.
                    ((v as f64 * 8_388_608.).round_ties_even() as i32).clamp(-8_388_608, 8_388_607)
                })
                .collect();
            let config = flacenc::config::Encoder::default()
                .into_verified()
                .map_err(|(_, e)| anyhow::anyhow!("FLAC config: {e:?}"))?;
            let source = flacenc::source::MemSource::from_samples(&pcm, channels, 24, sample_rate);
            let stream = flacenc::encode_with_fixed_block_size(&config, source, config.block_size)
                .map_err(|e| anyhow::anyhow!("FLAC encode: {e}"))?;
            let mut sink = flacenc::bitsink::ByteSink::new();
            stream
                .write(&mut sink)
                .map_err(|e| anyhow::anyhow!("FLAC stream: {e}"))?;
            fs::write(path, sink.as_slice())?;
        }
        _ => anyhow::bail!("Use .flac or .wav"),
    }
    Ok(())
}

impl SymbolicPlan {
    /// Writes the same five plan artifacts as Python (four when cot=off).
    pub fn save(&self, directory: impl AsRef<Path>) -> Result<()> {
        let directory = directory.as_ref();
        ensure_empty_directory(directory)?;
        if let Some(abc) = &self.abc {
            fs::write(directory.join("score.abc"), abc.as_bytes())?;
        }
        write_token_npy(directory.join("abc_tokens.npy"), &self.abc_ids)?;
        write_token_npy(directory.join("prefix.npy"), &self.prefix)?;
        write_json(directory.join("plan.json"), self)?;
        let mut hashes = serde_json::Map::new();
        for name in ["plan.json", "abc_tokens.npy", "prefix.npy", "score.abc"] {
            if name == "score.abc" && self.abc.is_none() {
                continue;
            }
            hashes.insert(name.into(), json!(sha256_file(directory.join(name))?));
        }
        write_json(directory.join("plan_manifest.json"), &hashes)
    }
}

fn collect_hashes(directory: &Path) -> Result<Value> {
    let mut hashes = serde_json::Map::new();
    for entry in fs::read_dir(directory)? {
        let path = entry?.path();
        if path.file_name().is_some_and(|s| s == "result.json") {
            continue;
        }
        ensure!(path.is_file(), "Unexpected artifact directory");
        let name = path
            .file_name()
            .and_then(|s| s.to_str())
            .context("Non-UTF8 artifact")?;
        hashes.insert(
            name.into(),
            json!({"sha256": sha256_file(&path)?, "bytes": path.metadata()?.len()}),
        );
    }
    Ok(Value::Object(hashes))
}

impl SongResult {
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        save_audio(path, &self.audio, self.sample_rate)
    }

    /// Default file set is identical to Python's save_artifacts, including hashes.
    /// result.json is the last, atomic completion marker.
    pub fn save_artifacts(&self, directory: impl AsRef<Path>) -> Result<Value> {
        self.save_artifacts_with_format(directory, "flac")
    }

    pub fn save_artifacts_with_format(
        &self,
        directory: impl AsRef<Path>,
        format: &str,
    ) -> Result<Value> {
        ensure!(["flac", "wav"].contains(&format), "Use flac or wav");
        let directory = directory.as_ref();
        self.semantic.plan.save(directory)?;
        self.save(directory.join(format!("audio.{format}")))?;
        write_token_npy(directory.join("semantic.npy"), &self.semantic.tokens)?;
        self.latents
            .to_device(&Device::Cpu)?
            .to_dtype(candle_core::DType::F32)?
            .write_npy(directory.join("latent.npy"))?;
        write_json(directory.join("request.json"), &self.semantic.plan.request)?;
        write_json(directory.join("config.json"), &self.config)?;
        let result = json!({"status": "complete", "identity": self.request_identity,
            "truncated": {"abc": self.semantic.plan.truncated, "semantic": self.semantic.truncated},
            "sample_rate": self.sample_rate, "audio_seconds": self.audio.dim(0)? as f64 / self.sample_rate as f64,
            "weights": self.weights, "timing": self.timing, "artifacts": collect_hashes(directory)?});
        write_json(directory.join("result.json"), &result)?;
        Ok(result)
    }
}

/// Read request JSON with Python CLI's aliases, metadata and relative abc_path.
pub fn read_request(
    path: &Path,
    abc_file: Option<&Path>,
) -> Result<(crate::protocol::SongRequest, Option<Value>, Option<Value>)> {
    let mut value: Value = serde_json::from_slice(&fs::read(path)?)?;
    let data = value.as_object_mut().context("Expected a request object")?;
    let abc_sampling = data.remove("abc_sampling");
    let semantic_sampling = data.remove("semantic_sampling");
    for key in ["lang", "eval_index", "clip_id"] {
        data.remove(key);
    }
    let prompt = data.remove("prompt");
    if let Some(tags) = data.remove("tags") {
        match data.get("style").filter(|s| !s.is_null()) {
            Some(style) => ensure!(
                *style == tags || tags.is_null(),
                "style and tags cannot disagree"
            ),
            None => {
                data.insert("style".into(), tags);
            }
        }
    }
    let supplied = if let Some(path) = abc_file {
        data.remove("abc_path");
        Some(PathBuf::from(path))
    } else if let Some(abc_path) = data.remove("abc_path") {
        ensure!(
            data.get("abc").is_none_or(Value::is_null),
            "Pass abc or abc_path, not both"
        );
        Some(
            path.parent()
                .unwrap_or(Path::new("."))
                .join(abc_path.as_str().context("abc_path must be text")?),
        )
    } else {
        None
    };
    if let Some(path) = supplied {
        data.insert(
            "abc".into(),
            json!(fs::read_to_string(&path)
                .with_context(|| format!("Read ABC {}", path.display()))?),
        );
    }
    let request: crate::protocol::SongRequest = serde_json::from_value(value)?;
    if let Some(prompt) = prompt.filter(|p| !p.is_null()) {
        ensure!(
            prompt.as_str() == Some(request.text()?.as_str()),
            "Historical literal prompt does not match request"
        );
    }
    Ok((request, abc_sampling, semantic_sampling))
}
