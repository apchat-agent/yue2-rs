use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use serde_json::json;
use std::collections::HashMap;
use yue2::{
    model::{StaticKVCache, YuE2Config, YuE2ForCausalLM},
    protocol::*,
};

#[test]
fn generation_overrides_preserve_phase_defaults() -> Result<()> {
    let config = GenerationConfig::from_dict(
        &json!({"abc": {"top_k": 7}, "semantic": {"temperature": 0.3}}),
    )?;
    assert_eq!(config.abc.top_k, 7);
    assert_eq!(config.abc.temperature, 0.7);
    assert_eq!(config.abc.min_tokens, 32);
    assert_eq!(config.semantic.top_k, 100);
    assert_eq!(config.semantic.temperature, 0.3);
    assert_eq!(GenerationConfig::from_dict(&config.to_dict()?)?, config);
    assert_eq!(resolve_sampling(None, &config.abc)?, config.abc);
    assert_eq!(
        resolve_sampling(Some(&json!(null)), &config.abc)?,
        config.abc
    );
    let updated = resolve_sampling(
        Some(&json!({"max_tokens": 64, "temperature": 0})),
        &config.abc,
    )?;
    assert_eq!(updated.min_tokens, 32);
    assert_eq!(updated.temperature, 0.);
    Ok(())
}

#[test]
fn reject_invalid_sampling_and_generation_json() {
    for value in [
        json!({"top_k": true}),
        json!({"top_k": 1.5}),
        json!({"top_k": -1}),
        json!({"top_k": 0}),
        json!({"temperature": 5.1}),
        json!({"top_p": 0}),
        json!({"penalty_window": 101}),
        json!({"repetition_penalty": 0}),
        json!({"min_tokens": 10, "max_tokens": 9}),
        json!({"unknown": 1}),
    ] {
        assert!(
            serde_json::from_value::<Sampling>(value.clone()).is_err(),
            "{value}"
        );
    }
    for value in [
        json!({"context": 2048}),
        json!({"ode_method": "euler"}),
        json!({"ode_steps": false}),
        json!({"ode_steps": 0}),
        json!({"abc": null}),
        json!({"unknown": 0}),
    ] {
        assert!(GenerationConfig::from_dict(&value).is_err(), "{value}");
    }
    let config = Sampling {
        temperature: f64::NAN,
        ..Sampling::default()
    };
    assert!(config.to_dict().is_err());
}

#[test]
fn request_defaults_text_guidance_and_validation() -> Result<()> {
    let mut request: SongRequest =
        serde_json::from_value(json!({"style": "piano", "lyrics": "[Verse]\nCafé"}))?;
    assert_eq!(request.seed, 831001);
    assert_eq!(request.id, "song");
    assert_eq!(request.guidance(), 1.);
    assert_eq!(request.text()?, "Generate a chord-annotated ABC transcription, then generate music with codec tokens from the given conditions.\n[Tags]\npiano\n[Lyrics]\n[Verse]\nCafé\n");
    request.cot = "off".into();
    assert_eq!(request.guidance(), 1.01);
    request.cfg_scale = Some(0.);
    assert_eq!(request.guidance(), 0.);
    assert_eq!(
        serde_json::from_value::<SongRequest>(request.to_dict()?)?,
        request
    );
    for overrides in [
        json!({"seed": true}),
        json!({"seed": -1}),
        json!({"seed": 9223372036854775808u64}),
        json!({"id": "../song"}),
        json!({"id": "."}),
        json!({"id": "é"}),
        json!({"style": null}),
        json!({"cot": "unknown"}),
        json!({"abc": " \n\t"}),
        json!({"abc": "\u{1c}\u{1f}"}),
        json!({"abc": "X:1", "cot": "off"}),
        json!({"cfg_scale": 21}),
        json!({"unknown": 1}),
    ] {
        let mut value = json!({"style": "", "lyrics": ""});
        value
            .as_object_mut()
            .unwrap()
            .extend(overrides.as_object().unwrap().clone());
        assert!(
            serde_json::from_value::<SongRequest>(value.clone()).is_err(),
            "{value}"
        );
    }
    request.cfg_scale = Some(f64::INFINITY);
    assert!(request.validate().is_err());
    Ok(())
}

#[test]
fn chunk_boundaries_and_insufficient_context() -> Result<()> {
    assert_eq!(chunk_ranges(1484, 611, CONTEXT)?, vec![(0, 1484)]);
    assert_eq!(chunk_ranges(1484, 611, 2098)?, vec![(0, 742), (742, 1484)]);
    assert_eq!(chunk_ranges(5, 10, 17)?, vec![(0, 2), (2, 4), (4, 5)]);
    assert!(chunk_ranges(0, 0, CONTEXT).is_err());
    assert!(chunk_ranges(1, CONTEXT, CONTEXT).is_err());
    assert!(chunk_ranges(1, usize::MAX, CONTEXT).is_err());
    Ok(())
}

fn tiny_model() -> Result<YuE2ForCausalLM> {
    tiny_model_dtype(DType::F32)
}

fn tiny_model_dtype(dtype: DType) -> Result<YuE2ForCausalLM> {
    let c = YuE2Config {
        hidden_size: 16,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        head_dim: 4,
        intermediate_size: 24,
        vocab_size: 32,
        rms_norm_eps: 1e-6,
        rope_theta: 1_000_000.,
        max_position_embeddings: 32,
        tie_word_embeddings: false,
    };
    let mut tensors = HashMap::new();
    let mut put = |name: String, shape: Vec<usize>, norm: bool| -> Result<()> {
        let phase = name.bytes().map(usize::from).sum::<usize>() as f32;
        let values = (0..shape.iter().product())
            .map(|i| {
                let v = (i as f32 * 0.19 + phase).sin();
                if norm {
                    1. + 0.05 * v
                } else {
                    0.1 * v
                }
            })
            .collect::<Vec<_>>();
        tensors.insert(name, Tensor::from_vec(values, shape, &Device::Cpu)?);
        Ok(())
    };
    put("model.embed_tokens.weight".into(), vec![32, 16], false)?;
    put("lm_head.weight".into(), vec![32, 16], false)?;
    put("model.norm.weight".into(), vec![16], true)?;
    for i in 0..2 {
        for name in ["input_layernorm", "post_attention_layernorm"] {
            put(format!("model.layers.{i}.{name}.weight"), vec![16], true)?;
        }
        for name in ["q_norm", "k_norm"] {
            put(
                format!("model.layers.{i}.self_attn.{name}.weight"),
                vec![4],
                true,
            )?;
        }
        for (name, shape) in [
            ("q_proj", vec![16, 16]),
            ("k_proj", vec![8, 16]),
            ("v_proj", vec![8, 16]),
            ("o_proj", vec![16, 16]),
        ] {
            put(
                format!("model.layers.{i}.self_attn.{name}.weight"),
                shape,
                false,
            )?;
        }
        for (name, shape) in [
            ("gate_proj", vec![24, 16]),
            ("up_proj", vec![24, 16]),
            ("down_proj", vec![16, 24]),
        ] {
            put(format!("model.layers.{i}.mlp.{name}.weight"), shape, false)?;
        }
    }
    YuE2ForCausalLM::load(c, VarBuilder::from_tensors(tensors, dtype, &Device::Cpu))
}

fn close(a: &Tensor, b: &Tensor) -> Result<()> {
    assert_eq!(a.dims(), b.dims());
    let error = (a - b)?.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
    assert!(error <= 2e-5, "max difference {error}");
    Ok(())
}

#[test]
fn cpu_bf16_forward_and_cache_are_supported() -> Result<()> {
    let model = tiny_model_dtype(DType::BF16)?;
    let input = Tensor::new(&[[1u32, 2, 3, 4]], &Device::Cpu)?;
    let full = model.forward(&input, None, 0, false)?.logits;
    assert_eq!(full.dtype(), DType::BF16);
    let values = full.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    assert!(values.iter().all(|v| v.is_finite()));
    assert!(values.iter().any(|v| v.abs() > 0.01));
    let mut cache = StaticKVCache::new(&model.config, 1, 4, DType::BF16, &Device::Cpu)?;
    let first = model
        .forward(&input.narrow(1, 0, 3)?, Some(&mut cache), 0, false)?
        .logits;
    let last = model
        .forward(&input.narrow(1, 3, 1)?, Some(&mut cache), 0, false)?
        .logits;
    close(
        &Tensor::cat(&[first, last], 1)?.to_dtype(DType::F32)?,
        &full.to_dtype(DType::F32)?,
    )?;
    Ok(())
}

#[test]
fn cached_chunked_and_single_token_forward_match_causal_prefill() -> Result<()> {
    let model = tiny_model()?;
    let input = Tensor::new(&[[1u32, 2, 3, 4, 5, 6]], &Device::Cpu)?;
    let all = model.forward(&input, None, 0, true)?;
    let mut cache = StaticKVCache::new(&model.config, 1, 6, DType::F32, &Device::Cpu)?;
    let first = model.forward(&input.narrow(1, 0, 2)?, Some(&mut cache), 0, false)?;
    assert_eq!(cache.get_seq_length(), 2);
    let second = model.forward(&input.narrow(1, 2, 3)?, Some(&mut cache), 0, false)?;
    let last = model.forward(&input.narrow(1, 5, 1)?, Some(&mut cache), 1, false)?;
    close(
        &Tensor::cat(&[first.logits, second.logits, last.logits], 1)?,
        &all.logits,
    )?;
    assert_eq!(cache.get_seq_length(), 6);
    assert_eq!(cache.capacity(), 6);
    assert!(model
        .forward(&input.narrow(1, 0, 1)?, Some(&mut cache), 1, false)
        .is_err());
    assert_eq!(cache.get_seq_length(), 6);
    cache.reset();
    let reuse = model.forward(&input, Some(&mut cache), 0, false)?;
    close(&reuse.logits, &all.logits)?;
    Ok(())
}

#[test]
fn future_tokens_do_not_change_past_logits_and_batches_are_independent() -> Result<()> {
    let model = tiny_model()?;
    let batch = Tensor::new(&[[1u32, 2, 3, 4], [1, 2, 20, 21]], &Device::Cpu)?;
    let output = model.forward(&batch, None, 0, false)?;
    close(
        &output.logits.narrow(0, 0, 1)?.narrow(1, 0, 2)?,
        &output.logits.narrow(0, 1, 1)?.narrow(1, 0, 2)?,
    )?;
    for i in 0..2 {
        let single = model.forward(&batch.narrow(0, i, 1)?, None, 0, false)?;
        close(&single.logits, &output.logits.narrow(0, i, 1)?)?;
    }
    assert!(model
        .forward(
            &Tensor::zeros((1, 0), DType::U32, &Device::Cpu)?,
            None,
            0,
            false
        )
        .is_err());
    Ok(())
}
