//! Offline oracle gates. Run explicitly with --ignored and YUE2_FIXTURES set.
use anyhow::{ensure, Context, Result};
use candle_core::{DType, Device, Tensor};
use serde_json::Value;
use std::{collections::HashMap, path::PathBuf};
use yue2::{
    model::{StaticKVCache, YuE2ForCausalLM},
    protocol::*,
    tokenizer::{special_tokens, YuE2TextTokenizer},
};

struct Fixtures {
    root: PathBuf,
    manifest: Value,
}
impl Fixtures {
    fn open() -> Result<Option<Self>> {
        let Some(root) = std::env::var_os("YUE2_FIXTURES") else {
            eprintln!("SKIP: YUE2_FIXTURES is unset");
            return Ok(None);
        };
        let mut root = PathBuf::from(root);
        if !root.join("manifest.json").is_file() {
            root = root.join("first-song");
        }
        let manifest = serde_json::from_slice(&std::fs::read(root.join("manifest.json"))?)?;
        Ok(Some(Self { root, manifest }))
    }
    fn model_dir(&self) -> Result<PathBuf> {
        if let Some(path) = std::env::var_os("YUE2_MODEL_DIR") {
            return Ok(path.into());
        }
        let path = self.manifest["weights"]["model"]["path"]
            .as_str()
            .context("Missing model path")?;
        Ok(PathBuf::from(path)
            .parent()
            .context("Missing model directory")?
            .to_owned())
    }
    fn json(&self, name: &str) -> Result<Value> {
        Ok(serde_json::from_slice(&std::fs::read(
            self.root.join(name),
        )?)?)
    }
    fn tensors(&self, name: &str) -> Result<HashMap<String, Tensor>> {
        Ok(candle_core::safetensors::load(
            self.root.join(name),
            &Device::Cpu,
        )?)
    }
    fn tokenizer(&self) -> Result<YuE2TextTokenizer> {
        YuE2TextTokenizer::new(self.model_dir()?.join("qwen.tiktoken"))
    }
    fn model(&self) -> Result<YuE2ForCausalLM> {
        let device = match std::env::var("YUE2_TEST_DEVICE")
            .as_deref()
            .unwrap_or("cpu")
        {
            "cpu" => Device::Cpu,
            "cuda" => {
                ensure!(
                    std::env::var("CUDA_VISIBLE_DEVICES").as_deref() == Ok("0"),
                    "Only GPU 0 is authorized"
                );
                Device::new_cuda(0)?
            }
            other => anyhow::bail!("Unknown test device {other}"),
        };
        // The oracle and model are BF16 on either device; CPU is slower.
        // SAFETY: the reference checkpoint directory is read-only for this task.
        unsafe { YuE2ForCausalLM::from_pretrained(self.model_dir()?, DType::BF16, &device) }
    }
}

fn ids(t: &Tensor) -> Result<Vec<u32>> {
    Ok(t.to_dtype(DType::U32)?.flatten_all()?.to_vec1()?)
}
fn floats(t: &Tensor) -> Result<Vec<f32>> {
    Ok(t.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?)
}
fn decoded(tokenizer: &YuE2TextTokenizer, tokens: &[u32]) -> Result<String> {
    tokenizer.decode(&tokens.iter().map(|&v| i64::from(v)).collect::<Vec<_>>())
}
fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut dot, mut aa, mut bb) = (0., 0., 0.);
    for (&a, &b) in a.iter().zip(b) {
        let (a, b) = (f64::from(a), f64::from(b));
        dot += a * b;
        aa += a * a;
        bb += b * b;
    }
    dot / (aa * bb).sqrt()
}
fn argmax(row: &[f32]) -> usize {
    row.iter()
        .enumerate()
        .fold((0, f32::NEG_INFINITY), |best, (i, &v)| {
            if v > best.1 {
                (i, v)
            } else {
                best
            }
        })
        .0
}

#[test]
#[ignore = "requires YUE2_FIXTURES and the local tokenizer"]
fn p1a_tokenizer() -> Result<()> {
    let Some(f) = Fixtures::open()? else {
        return Ok(());
    };
    let tokenizer = f.tokenizer()?;
    let metadata = f.json("tokens.json")?;
    let tensors = f.tensors("tokens.safetensors")?;
    let probes = metadata["probes"].as_array().context("Missing probes")?;
    ensure!(probes.len() == 20, "Expected exactly 20 probes");
    for (i, probe) in probes.iter().enumerate() {
        let text = probe.as_str().context("Non-string probe")?;
        let tokens = tokenizer.encode(text);
        assert_eq!(
            tokens,
            ids(&tensors[&format!("probe.{i:02}")])?,
            "probe {i}: {text}"
        );
        assert_eq!(decoded(&tokenizer, &tokens)?, text);
    }
    let special: std::collections::BTreeMap<String, u32> =
        serde_json::from_value(metadata["special_tokens"].clone())?;
    assert_eq!(special_tokens(), special);
    for (text, token) in &special {
        assert_eq!(tokenizer.decode(&[i64::from(*token)])?, *text);
    }
    let norm = &metadata["normalization_probe"];
    let tokens = tokenizer.encode(norm["text"].as_str().unwrap());
    assert_eq!(
        tokens,
        serde_json::from_value::<Vec<u32>>(norm["ids"].clone())?
    );
    assert_eq!(
        decoded(&tokenizer, &tokens)?,
        norm["normalized"].as_str().unwrap()
    );
    assert_eq!(tokenizer.decode(&[-1, 151851, 184623])?, "");
    let case = &metadata["requests"][0];
    let request: SongRequest = serde_json::from_value(case["request"].clone())?;
    let text = request.text()?;
    assert_eq!(tokenizer.encode(&text), ids(&tensors["request.0.text"])?);
    assert_eq!(decoded(&tokenizer, &tokenizer.encode(&text))?, text);
    let abc = ids(&tensors["abc_ids"])?;
    let prefix = token_prefixes(&request, &tokenizer, Some(&abc))?;
    assert_eq!(prefix, ids(&tensors["prefix"])?);
    println!("P1a PASS: 20/20 exact encodings and round-trips; 208/208 specials; NFC; full first-song prefix {}/{} tokens", prefix.len(), prefix.len());
    Ok(())
}

#[test]
#[ignore = "requires YUE2_FIXTURES and the local tokenizer"]
fn p1b_protocol_prefixes() -> Result<()> {
    let Some(f) = Fixtures::open()? else {
        return Ok(());
    };
    let tokenizer = f.tokenizer()?;
    let metadata = f.json("tokens.json")?;
    let tensors = f.tensors("tokens.safetensors")?;
    for case in metadata["requests"].as_array().unwrap() {
        let request: SongRequest = serde_json::from_value(case["request"].clone())?;
        let stem = case["stem"].as_str().unwrap();
        let abc = ids(&tensors[&format!("{stem}.abc")])?;
        assert_eq!(request.text()?, case["text"].as_str().unwrap());
        let prompt = token_prefixes(&request, &tokenizer, None)?;
        assert_eq!(prompt, ids(&tensors[&format!("{stem}.prompt")])?);
        let prefix = token_prefixes(&request, &tokenizer, Some(&abc))?;
        assert_eq!(prefix, ids(&tensors[&format!("{stem}.prefix")])?);
        assert_eq!(
            negative_prefix(&request, &tokenizer, Some(&abc))?,
            ids(&tensors[&format!("{stem}.negative")])?
        );
        println!(
            "P1b {} PASS: prompt={}, saved prefix={}, ABC={}",
            case["name"].as_str().unwrap(),
            prompt.len(),
            prefix.len(),
            abc.len()
        );
    }
    let mut request = SongRequest::new("piano", "[Verse]\nHello");
    request.cot = "off".into();
    let positive = token_prefixes(&request, &tokenizer, Some(&[EOD]))?;
    assert!(positive.ends_with(&[ABC_START, ABC_END, MUSIC_START]));
    let negative = negative_prefix(&request, &tokenizer, None)?;
    assert_eq!(negative.last(), Some(&MUSIC_START));
    assert!(!negative.contains(&ABC_START));
    request.cot = "melody".into();
    request.abc = Some("K:C\nC4 |".into());
    let external = tokenizer.encode(request.abc.as_ref().unwrap());
    assert_eq!(
        token_prefixes(&request, &tokenizer, None)?,
        token_prefixes(&request, &tokenizer, Some(&external))?
    );
    assert!(negative_prefix(&request, &tokenizer, None).is_err());
    assert!(token_prefixes(&request, &tokenizer, Some(&[EOD])).is_err());
    assert!(negative_prefix(&request, &tokenizer, Some(&[ABC_START])).is_err());
    assert!(
        token_prefixes(&request, &tokenizer, Some(&[]))?.ends_with(&[
            ABC_START,
            ABC_END,
            MUSIC_START
        ])
    );
    Ok(())
}

#[test]
#[ignore = "requires YUE2_FIXTURES and the 3B checkpoint; YUE2_TEST_DEVICE=cuda for BF16 gate"]
fn p1c_teacher_forced() -> Result<()> {
    let Some(f) = Fixtures::open()? else {
        return Ok(());
    };
    let model = f.model()?;
    let oracle = f.tensors("ar_logits.safetensors")?;
    let input = oracle["input_ids"]
        .to_dtype(DType::U32)?
        .to_device(model.device())?;
    let output = model.forward(&input, None, 0, true)?;
    let expected = floats(&oracle["logits"])?;
    let actual = floats(&output.logits)?;
    assert_eq!(actual.len(), expected.len());
    ensure!(actual.iter().all(|v| v.is_finite()), "Non-finite logits");
    let max_diff = actual
        .iter()
        .zip(&expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let matches = actual
        .chunks(model.config.vocab_size)
        .zip(expected.chunks(model.config.vocab_size))
        .filter(|(a, b)| argmax(a) == argmax(b))
        .count();
    println!(
        "P1c {:?} {:?}: argmax={matches}/64; max_abs_logit_diff={max_diff:.9}",
        model.device(),
        model.dtype()
    );
    let mut similarities = Vec::new();
    for name in ["embedding", "layer.0", "layer.13", "final_norm"] {
        let a = floats(&output.hidden_states[name])?;
        let b = floats(&oracle[name])?;
        assert_eq!(a.len(), b.len());
        let similarity = cosine(&a, &b);
        let minimum = a
            .chunks(model.config.hidden_size)
            .zip(b.chunks(model.config.hidden_size))
            .map(|(a, b)| cosine(a, b))
            .fold(1f64, f64::min);
        let max = a
            .iter()
            .zip(&b)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        println!("P1c {name}: cosine={similarity:.12}; min_position_cosine={minimum:.12}; max_abs={max:.9}");
        similarities.push((name, similarity, minimum));
    }
    assert_eq!(
        matches, 64,
        "all teacher-forced argmax positions must match"
    );
    assert!(max_diff <= 0.5, "logit gate exceeded: {max_diff}");
    for (name, similarity, minimum) in similarities {
        assert!(
            similarity >= 0.999 && minimum >= 0.999,
            "hidden-state cosine gate failed at {name}: {similarity}, min={minimum}"
        );
    }
    Ok(())
}

fn greedy_64(f: &Fixtures) -> Result<Vec<u32>> {
    let model = f.model()?;
    let tensors = f.tensors("tokens.safetensors")?;
    let prompt = ids(&tensors["request.0.prompt"])?;
    let metadata = f.json("greedy.json")?;
    let sampling: Sampling = serde_json::from_value(metadata["sampling"].clone())?;
    ensure!(
        sampling.temperature == 0. && sampling.max_tokens == 64,
        "Wrong greedy oracle config"
    );
    let mut cache = StaticKVCache::new(
        &model.config,
        1,
        prompt.len() + 64,
        model.dtype(),
        model.device(),
    )?;
    let mut input = Tensor::new(prompt.as_slice(), model.device())?.unsqueeze(0)?;
    let mut history = Vec::<u32>::new();
    for step in 0..64 {
        let output = model.forward(&input, Some(&mut cache), 1, false)?;
        let mut scores = floats(&output.logits)?;
        // Greedy-only gate harness, matching sampling.py distribution. General
        // sampling/CFG is Phase 2 and is not implemented by this test helper.
        for (i, score) in scores.iter_mut().enumerate().skip(EOD as usize) {
            if i != ABC_END as usize {
                *score = f32::NEG_INFINITY;
            }
        }
        if step < sampling.min_tokens {
            scores[ABC_END as usize] = f32::NEG_INFINITY;
        }
        let mut counts = HashMap::<u32, i32>::new();
        for &token in history.iter().rev().take(sampling.penalty_window) {
            *counts.entry(token).or_default() += 1;
        }
        for (token, count) in counts {
            let alpha = (sampling.repetition_penalty as f32).powi(count);
            let score = &mut scores[token as usize];
            *score = if *score < 0. {
                *score * alpha
            } else {
                *score / alpha
            };
        }
        let next = argmax(&scores) as u32;
        ensure!(next != ABC_END, "Greedy ended before 64 steps at {step}");
        history.push(next);
        input = Tensor::new(&[[next]], model.device())?;
    }
    Ok(history)
}

#[test]
#[ignore = "requires --greedy fixtures and the 3B checkpoint"]
fn p1d_greedy_oracle() -> Result<()> {
    let Some(f) = Fixtures::open()? else {
        return Ok(());
    };
    let actual = greedy_64(&f)?;
    let expected = ids(&f.tensors("greedy.safetensors")?["ids"])?;
    let matches = actual.iter().zip(&expected).filter(|(a, b)| a == b).count();
    let sampled = ids(&f.tensors("tokens.safetensors")?["abc_ids"])?;
    let sampled_matches = actual.iter().zip(&sampled).filter(|(a, b)| a == b).count();
    println!("P1d eager greedy oracle: {matches}/64; original sampled abc_tokens.npy: {sampled_matches}/64");
    assert_eq!(
        actual, expected,
        "Rust must match Python's eager greedy 64-step plan"
    );
    Ok(())
}

#[test]
#[ignore = "advisory sampled-artifact comparison; requires fixtures and the 3B checkpoint"]
fn p1d_literal_saved_abc() -> Result<()> {
    let Some(f) = Fixtures::open()? else {
        return Ok(());
    };
    let actual = greedy_64(&f)?;
    let sampled = ids(&f.tensors("tokens.safetensors")?["abc_ids"])?;
    let matches = actual.iter().zip(&sampled).filter(|(a, b)| a == b).count();
    println!("P1d advisory: Rust greedy vs original sampled abc_tokens.npy = {matches}/64");
    Ok(())
}
