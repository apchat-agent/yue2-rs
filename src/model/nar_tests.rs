use super::*;
use std::cell::Cell;

#[test]
fn acoustic_bias_rounds_once_after_accumulation() -> Result<()> {
    let weight = Tensor::new(&[[1f32, 1. / 256.]], &Device::Cpu)?.to_dtype(DType::BF16)?;
    let bias = Tensor::new(&[1f32 / 256.], &Device::Cpu)?.to_dtype(DType::BF16)?;
    let x = Tensor::new(&[[1f32, 1.]], &Device::Cpu)?.to_dtype(DType::BF16)?;
    let actual = linear_forward(&Linear::new(weight, Some(bias)), &x)?;
    assert_eq!(
        actual.to_dtype(DType::F32)?.to_vec2::<f32>()?,
        [[1.0078125]]
    );
    Ok(())
}

#[test]
#[ignore = "requires YUE2_FIXTURES, CUDA and tools/dump_nar_debug.py"]
fn nar_operations_oracle() -> Result<()> {
    let Some(root) = std::env::var_os("YUE2_FIXTURES") else {
        eprintln!("SKIP: YUE2_FIXTURES is unset");
        return Ok(());
    };
    ensure!(
        std::env::var("CUDA_VISIBLE_DEVICES").as_deref() == Ok("0"),
        "Only GPU 0 authorized"
    );
    let mut root = PathBuf::from(root);
    if !root.join("manifest.json").is_file() {
        root = root.join("first-song");
    }
    let d = Device::new_cuda(0)?;
    let fixture = candle_core::safetensors::load(root.join("nar.safetensors"), &d)?;
    let oracle = candle_core::safetensors::load(root.join("nar-debug.safetensors"), &d)?;
    let compare = |name: &str, a: &Tensor| -> Result<()> {
        let b = &oracle[name];
        ensure!(
            a.shape() == b.shape(),
            "{name}: shape mismatch {:?} {:?}",
            a.shape(),
            b.shape()
        );
        let a = a.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let b = b.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let (mut max, mut dot, mut aa, mut bb, mut neq) = (0f64, 0., 0., 0., 0);
        for (a, b) in a.iter().zip(&b) {
            let (a, b) = (*a as f64, *b as f64);
            max = max.max((a - b).abs());
            dot += a * b;
            aa += a * a;
            bb += b * b;
            neq += usize::from(a != b);
        }
        println!(
            "NAR op {name}: max={max:.12} cos={:.12} different={neq}/{}",
            dot / (aa * bb).sqrt(),
            a.len()
        );
        Ok(())
    };
    // SAFETY: checkpoint files are immutable.
    let model = unsafe {
        YuE2ForCausalLM::from_pretrained_with_nar(snapshot_dir("YuE2-3B")?, DType::BF16, &d)?
    };
    let chunk = Chunk {
        ar_tokens: fixture["two_chunk.0.ar_tokens"]
            .to_dtype(DType::U32)?
            .to_vec1()?,
        noise: fixture["two_chunk.0.noise"].clone(),
        nar_cond_end: 0,
    };
    let engine = CachedNAR::new(&model, &chunk, 128)?;
    for i in [0, 13, 27] {
        compare(&format!("cache.{i}.k"), &engine.cache[i].0.squeeze(0)?)?;
        compare(&format!("cache.{i}.v"), &engine.cache[i].1.squeeze(0)?)?;
    }
    compare("cos", &engine.cos)?;
    compare("sin", &engine.sin)?;
    compare("pos", &engine.pos_emb)?;
    let w = engine.weights;
    let state = &fixture["two_chunk.0.step.00.state"];
    let zero = Tensor::zeros((1, 64), DType::BF16, &d)?;
    let x_nar = Tensor::cat(&[&zero, state, &zero], 0)?.unsqueeze(0)?;
    let mut x = linear_forward(&w.vae2llm, &x_nar)?;
    compare("vae2llm", &x)?;
    let first = linear_forward(&w.time_embedder.first, &oracle["time.0.input"])?;
    compare("time.0.output", &first)?;
    let second = linear_forward(&w.time_embedder.second, &oracle["time.2.input"])?;
    compare("time.2.output", &second)?;
    let time = w
        .time_embedder
        .forward(
            &oracle["shifted"]
                .broadcast_as(engine.nar_length)?
                .contiguous()?,
        )?
        .unsqueeze(0)?;
    compare("time", &time)?;
    x = (x + time)?;
    compare("with_time", &x)?;
    x = (x + &engine.pos_emb)?;
    compare("injected", &x)?;
    for (i, (layer, (ar_k, ar_v))) in w.layers.iter().zip(&engine.cache).enumerate() {
        let norm = layer.nar_input_layernorm.forward(&x)?;
        let (q, k, v) = layer.nar_self_attn.project_qkv(
            &norm,
            &model.model.rotary_emb,
            &engine.cos,
            &engine.sin,
        )?;
        let h = engine.attention(
            &layer.nar_self_attn,
            &q,
            &Tensor::cat(&[ar_k, &k], 1)?,
            &Tensor::cat(&[ar_v, &v], 1)?,
            false,
        )?;
        let o = linear_forward(&layer.nar_self_attn.o_proj, &h)?;
        if i == 0 {
            compare("layer.0.norm", &norm)?;
            compare("layer.0.q", &q)?;
            compare("layer.0.k", &k)?;
            compare("layer.0.v", &v)?;
            compare(
                "layer.0.attn",
                &h.squeeze(0)?.reshape((
                    engine.nar_length,
                    model.config.num_attention_heads,
                    model.config.head_dim,
                ))?,
            )?;
            compare("layer.0.o", &o)?;
        }
        x = (&x + o)?;
        x = (&x
            + layer
                .nar_mlp
                .forward(&layer.nar_pre_mlp_layernorm.forward(&x)?)?)?;
        compare(&format!("layer.{i}.output"), &x)?;
    }
    let norm = model.model.norm.forward(&x)?;
    compare("final_norm", &norm)?;
    compare(
        "velocity",
        &linear_forward(&w.llm2vae, &norm)?
            .squeeze(0)?
            .narrow(0, 1, engine.nar_length - 2)?,
    )?;
    Ok(())
}

fn constant_velocity_model(dtype: DType) -> Result<YuE2ForCausalLM> {
    let config = YuE2Config {
        hidden_size: 4,
        num_hidden_layers: 1,
        num_attention_heads: 2,
        num_key_value_heads: 1,
        head_dim: 2,
        intermediate_size: 4,
        vocab_size: 184704,
        rms_norm_eps: 1e-6,
        rope_theta: 10000.,
        max_position_embeddings: 32,
        tie_word_embeddings: false,
    };
    let mut model = YuE2ForCausalLM::load_with_nar(
        config,
        NarConfig {
            max_latent_frames: 8,
            ..Default::default()
        },
        VarBuilder::zeros(dtype, &Device::Cpu),
    )?;
    let weights = model.nar.as_mut().unwrap();
    weights.llm2vae = Linear::new(
        weights.llm2vae.weight().clone(),
        Some(Tensor::full(1.25f32, 64, &Device::Cpu)?.to_dtype(dtype)?),
    );
    Ok(model)
}

#[test]
fn midpoint_cpu_f32_bf16_and_cancellation() -> Result<()> {
    for dtype in [DType::F32, DType::BF16] {
        let model = constant_velocity_model(dtype)?;
        let chunk = Chunk {
            ar_tokens: vec![1, 2],
            noise: Tensor::full(2f32, (3, 64), &Device::Cpu)?,
            nar_cond_end: 1,
        };
        let engine = CachedNAR::new(&model, &chunk, 2)?;
        assert_eq!(engine.cache[0].0.dim(1)?, 1);
        let mut progress = Vec::new();
        let result = engine.solve(
            4,
            None,
            Some(&mut |done, total| progress.push((done, total))),
        )?;
        assert_eq!(result.dims(), &[3, 64]);
        assert_eq!(result.dtype(), DType::F32);
        assert!(result.device().is_cpu());
        assert_eq!(result.flatten_all()?.to_vec1::<f32>()?, vec![0.75f32; 192]);
        assert_eq!(progress, [(1, 4), (2, 4), (3, 4), (4, 4)]);
        assert!(engine.solve(0, None, None).is_err());
        let calls = Cell::new(0);
        let cancel = || {
            calls.set(calls.get() + 1);
            calls.get() == 2
        };
        let mut callbacks = 0;
        assert!(engine
            .solve(4, Some(&cancel), Some(&mut |_, _| callbacks += 1))
            .is_err());
        assert_eq!(calls.get(), 2);
        assert_eq!(callbacks, 0, "Cancellation between midpoint evaluations");
        let invalid = Chunk {
            noise: Tensor::full(f32::NAN, (3, 64), &Device::Cpu)?,
            ..chunk.clone()
        };
        assert!(CachedNAR::new(&model, &invalid, 2).is_err());
        let invalid = Chunk {
            ar_tokens: vec![184704],
            ..chunk.clone()
        };
        assert!(CachedNAR::new(&model, &invalid, 2).is_err());
        let invalid = Chunk {
            ar_tokens: vec![1; 30],
            ..chunk.clone()
        };
        assert!(CachedNAR::new(&model, &invalid, 2).is_err());
        assert!(CachedNAR::new(&model, &chunk, 0).is_err());
    }
    Ok(())
}

#[test]
fn synthesis_serial_chunks_progress_and_seed_reset() -> Result<()> {
    let model = constant_velocity_model(DType::F32)?;
    let mut progress = Vec::new();
    let actual = crate::nar::synthesize(
        &model,
        &[1, 2],
        &[1; 9],
        7,
        crate::nar::SynthesisOptions {
            steps: 4,
            context: 13,
            query_chunk_size: 2,
            on_progress: Some(&mut |done, total| progress.push((done, total))),
            cancelled: None,
        },
    )?;
    let chunks = crate::nar::song_chunks(&[1, 2], &[1; 9], 7, 13, &Device::Cpu)?;
    let noise = Tensor::cat(&chunks.iter().map(|c| &c.noise).collect::<Vec<_>>(), 0)?;
    let error = (actual - (noise - 1.25)?)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(
        error < 1e-6,
        "constant velocity integrates over original cuts"
    );
    assert_eq!(progress, (1..=12).map(|i| (i, 12)).collect::<Vec<_>>());
    Ok(())
}
