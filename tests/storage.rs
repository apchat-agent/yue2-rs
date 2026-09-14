use anyhow::Result;
use serde_json::json;
use std::path::PathBuf;
use yue2::{
    pipeline::{SemanticResult, SongResult, SymbolicPlan},
    protocol::SongRequest,
    storage, Device, Tensor,
};

fn directory(name: &str) -> Result<PathBuf> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("target/storage-tests")
        .join(format!(
            "{name}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_nanos()
        ));
    std::fs::create_dir_all(&path)?;
    Ok(path)
}

#[test]
fn identity_matches_python_float_and_unicode_canonicalization() -> Result<()> {
    // hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
    // separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    let value = json!({"a": [1e-7, 1e-5, 0.0001, 1e15, 1e16, 1.2e20, -0.0, 0.7],
        "z": "é\n你好", "seed": 831001});
    assert_eq!(
        storage::identity(&value)?,
        "eea7b699c832f896e09ea271a732f407d2e6f8f2d163bd2e88529e768cb5c4c5"
    );
    Ok(())
}

#[test]
fn numpy_audio_and_hashed_receipt() -> Result<()> {
    let root = directory("artifacts")?;
    let plan = SymbolicPlan {
        request: SongRequest::new("piano", "sing"),
        abc: Some("X:1\nK:C\nC4 |\n".into()),
        abc_ids: vec![1, 2, 151847],
        prefix: vec![151851, 184620],
        timing: json!({}),
        truncated: true,
    };
    let audio = Tensor::from_vec(vec![-1f32, 1., 0.25, -0.5, 0., 0.75], (3, 2), &Device::Cpu)?;
    let result = SongResult {
        audio: audio.clone(),
        sample_rate: 48000,
        semantic: SemanticResult {
            plan,
            tokens: vec![0, 32767],
            timing: json!({}),
            truncated: false,
        },
        latents: Tensor::zeros((2, 64), yue2::DType::F32, &Device::Cpu)?,
        config: json!({}),
        weights: json!({}),
        timing: json!({}),
        request_identity: "test".into(),
    };
    let receipt = result.save_artifacts(root.join("song"))?;
    assert_eq!(
        receipt["truncated"],
        json!({"abc": true, "semantic": false})
    );
    let expected = [
        "abc_tokens.npy",
        "audio.flac",
        "config.json",
        "latent.npy",
        "plan.json",
        "plan_manifest.json",
        "prefix.npy",
        "request.json",
        "score.abc",
        "semantic.npy",
    ];
    assert_eq!(
        receipt["artifacts"]
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>(),
        expected
    );
    for name in expected {
        assert_eq!(
            receipt["artifacts"][name]["sha256"],
            storage::sha256_file(root.join("song").join(name))?
        );
    }
    let bytes = std::fs::read(root.join("song/semantic.npy"))?;
    assert_eq!(&bytes[..8], b"\x93NUMPY\x01\x00");
    let data_start = 10 + u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
    assert_eq!(data_start % 64, 0);
    assert!(std::str::from_utf8(&bytes[10..data_start])?.contains("'<i4'"));
    assert_eq!(&bytes[data_start..], &[0, 0, 0, 0, 255, 127, 0, 0]);
    assert_eq!(&std::fs::read(root.join("song/audio.flac"))?[..4], b"fLaC");
    result.save(root.join("audio.wav"))?;
    let mut wav = hound::WavReader::open(root.join("audio.wav"))?;
    assert_eq!(wav.spec().channels, 2);
    assert_eq!(wav.spec().sample_rate, 48000);
    assert_eq!(
        wav.samples::<f32>()
            .collect::<std::result::Result<Vec<_>, _>>()?,
        audio.flatten_all()?.to_vec1::<f32>()?
    );
    assert!(
        result.save_artifacts(root.join("song")).is_err(),
        "must not overwrite a saved run"
    );
    let bad_audio = Tensor::new(&[[f32::NAN, 0.]], &Device::Cpu)?;
    assert!(storage::save_audio(root.join("bad.flac"), &bad_audio, 48000).is_err());
    std::fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn request_alias_paths_overrides_and_errors() -> Result<()> {
    let root = directory("request")?;
    let path = root.join("request.json");
    std::fs::write(root.join("score.abc"), "X:1\r\nK:C\r\nC4 |\r\n")?;
    let value = json!({"tags": "piano", "lyrics": "sing", "abc_path": "score.abc", "lang": "en",
        "abc_sampling": {"max_tokens": 32}, "semantic_sampling": {"max_tokens": 200}});
    storage::write_json(&path, &value)?;
    let (request, abc, semantic) = storage::read_request(&path, None)?;
    assert_eq!(request.style, "piano");
    assert!(
        request.abc.unwrap().contains("\r\n"),
        "external ABC bytes must survive unchanged"
    );
    assert_eq!(abc.unwrap()["max_tokens"], 32);
    assert_eq!(semantic.unwrap()["max_tokens"], 200);
    for update in [
        json!({"style": "different"}),
        json!({"abc": "conflicting inline ABC"}),
        json!({"cot": "off"}),
        json!({"surprise": true}),
        json!({"prompt": "wrong"}),
    ] {
        let mut changed = value.clone();
        changed
            .as_object_mut()
            .unwrap()
            .extend(update.as_object().unwrap().clone());
        storage::write_json(&path, &changed)?;
        assert!(storage::read_request(&path, None).is_err());
    }
    let mut changed = value;
    changed["abc"] = json!("replaced inline ABC");
    changed["abc_path"] = json!("nonexistent.abc");
    storage::write_json(&path, &changed)?;
    assert!(storage::read_request(&path, Some(&root.join("score.abc")))?
        .0
        .abc
        .unwrap()
        .starts_with("X:1"));
    std::fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn cli_help_and_render_requires_score_before_loading() -> Result<()> {
    let help = std::process::Command::new(env!("CARGO_BIN_EXE_yue2"))
        .arg("--help")
        .output()?;
    assert!(help.status.success());
    let text = String::from_utf8(help.stdout)?;
    for command in ["generate", "plan", "render"] {
        assert!(text.contains(command));
    }
    let result = std::process::Command::new(env!("CARGO_BIN_EXE_yue2"))
        .args([
            "render",
            "--request",
            "missing-request.json",
            "--output",
            "target/must-not-exist",
        ])
        .output()?;
    assert!(!result.status.success());
    assert!(String::from_utf8(result.stderr)?.contains("render requires --abc-file"));
    Ok(())
}
