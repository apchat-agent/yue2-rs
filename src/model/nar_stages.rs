//! P3b diagnostics only. The staged forward is asserted equal to production;
//! solve_observed records the production loop, without reference substitution.
use super::*;
use std::collections::HashMap;

type Dump = HashMap<String, Tensor>;
fn put(out: &mut Dump, name: &str, x: &Tensor) -> Result<()> {
    out.insert(name.into(), x.to_device(&Device::Cpu)?.contiguous()?);
    Ok(())
}
fn linear(out: &mut Dump, name: &str, layer: &Linear, x: &Tensor) -> Result<Tensor> {
    put(out, &format!("{name}.input"), x)?;
    let y = linear_forward(layer, x)?;
    put(out, &format!("{name}.output"), &y)?;
    Ok(y)
}
fn norm(out: &mut Dump, name: &str, layer: &RMSNorm, x: &Tensor) -> Result<Tensor> {
    put(out, &format!("{name}.input"), x)?;
    let y = layer.forward(x)?;
    put(out, &format!("{name}.output"), &y)?;
    Ok(y)
}
fn qkv(
    out: &mut Dump,
    name: &str,
    layer: &Attention,
    x: &Tensor,
    rope: &RotaryEmbedding,
    cos: &Tensor,
    sin: &Tensor,
) -> Result<(Tensor, Tensor, Tensor)> {
    let (b, t, _) = x.dims3()?;
    let q = linear(out, &format!("{name}.q_proj"), &layer.q_proj, x)?.reshape((
        b,
        t,
        layer.num_heads,
        layer.head_dim,
    ))?;
    let k = linear(out, &format!("{name}.k_proj"), &layer.k_proj, x)?.reshape((
        b,
        t,
        layer.num_kv_heads,
        layer.head_dim,
    ))?;
    let v = linear(out, &format!("{name}.v_proj"), &layer.v_proj, x)?.reshape((
        b,
        t,
        layer.num_kv_heads,
        layer.head_dim,
    ))?;
    let q = rope.apply(
        &norm(out, &format!("{name}.q_norm"), &layer.q_norm, &q)?,
        cos,
        sin,
    )?;
    let k = rope.apply(
        &norm(out, &format!("{name}.k_norm"), &layer.k_norm, &k)?,
        cos,
        sin,
    )?;
    for (suffix, x) in [("q", &q), ("k", &k), ("v", &v)] {
        put(out, &format!("{name}.{suffix}"), x)?;
    }
    Ok((q, k, v))
}

#[test]
#[ignore = "requires YUE2_FIXTURES, CUDA and dump_reference.py --nar-stages"]
fn nar_stage_dump() -> Result<()> {
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
    let reference =
        candle_core::safetensors::load(root.join("p3b-python.safetensors"), &Device::Cpu)?;
    let fixture = candle_core::safetensors::load(root.join("nar.safetensors"), &Device::Cpu)?;
    // SAFETY: the local checkpoint is immutable throughout the test.
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
    let mut out = Dump::new();
    put(&mut out, "noise", &chunk.noise)?;
    let state = chunk.noise.to_device(&d)?.to_dtype(DType::BF16)?;
    put(&mut out, "state", &state)?;
    let ar_positions = Tensor::arange(0u32, chunk.ar_tokens.len() as u32, &d)?.unsqueeze(0)?;
    let nar_positions = Tensor::arange(
        chunk.ar_tokens.len() as u32,
        (chunk.ar_tokens.len() + engine.nar_length) as u32,
        &d,
    )?
    .unsqueeze(0)?;
    let audio_positions = Tensor::arange(0u32, engine.nar_length as u32, &d)?;
    for (name, x) in [
        ("ar.positions", &ar_positions),
        ("nar.positions", &nar_positions),
        ("audio.positions", &audio_positions),
        ("audio.input", &audio_positions),
    ] {
        put(&mut out, name, x)?;
    }
    put(&mut out, "audio.output", &engine.pos_emb.squeeze(0)?)?;
    let backbone = &model.model;
    let rope = &backbone.rotary_emb;
    put(&mut out, "rope.inv_freq", &rope.inv_freq)?;
    put(&mut out, "time.freqs", &engine.weights.time_embedder.freqs)?;
    let (cos, sin) = rope.forward(&ar_positions)?;
    put(&mut out, "ar.cos", &cos)?;
    put(&mut out, "ar.sin", &sin)?;
    put(&mut out, "nar.cos", &engine.cos)?;
    put(&mut out, "nar.sin", &engine.sin)?;
    let ids = Tensor::new(chunk.ar_tokens.as_slice(), &d)?.unsqueeze(0)?;
    let mut x = backbone.embed_tokens.forward(&ids)?;
    put(&mut out, "ar.embedding.output", &x)?;
    for (i, layer) in backbone.layers.iter().enumerate() {
        let name = format!("ar.layer.{i:02}");
        let normalized = if i == 0 {
            norm(
                &mut out,
                &format!("{name}.norm"),
                &layer.input_layernorm,
                &x,
            )?
        } else {
            layer.input_layernorm.forward(&x)?
        };
        let (q, k, v) = if i == 0 {
            qkv(
                &mut out,
                &name,
                &layer.self_attn,
                &normalized,
                rope,
                &cos,
                &sin,
            )?
        } else {
            layer.self_attn.project_qkv(&normalized, rope, &cos, &sin)?
        };
        if [0, 13, 27].contains(&i) {
            for (suffix, actual, cached) in
                [("k", &k, &engine.cache[i].0), ("v", &v, &engine.cache[i].1)]
            {
                ensure!(
                    actual
                        .to_dtype(DType::F32)?
                        .flatten_all()?
                        .to_vec1::<f32>()?
                        == cached
                            .to_dtype(DType::F32)?
                            .flatten_all()?
                            .to_vec1::<f32>()?,
                    "Staged prefill differs"
                );
                put(&mut out, &format!("cache.{i:02}.{suffix}"), actual)?;
            }
        }
        let h = engine.attention(&layer.self_attn, &q, &k, &v, true)?;
        let o = if i == 0 {
            linear(
                &mut out,
                &format!("{name}.o_proj"),
                &layer.self_attn.o_proj,
                &h,
            )?
        } else {
            linear_forward(&layer.self_attn.o_proj, &h)?
        };
        x = (&x + o)?;
        x = (&x
            + layer
                .mlp
                .forward(&layer.post_attention_layernorm.forward(&x)?)?)?;
        put(&mut out, &format!("{name}.output"), &x)?;
    }
    let w = engine.weights;
    let shifted = shifted_time(20., w.config.timestep_shift, model.dtype(), &d)?;
    put(&mut out, "shifted", &shifted)?;
    let times = shifted.broadcast_as(engine.nar_length)?.contiguous()?;
    put(&mut out, "time.input", &times)?;
    let args = times
        .to_dtype(DType::F32)?
        .unsqueeze(D::Minus1)?
        .broadcast_mul(&w.time_embedder.freqs)?;
    let emb = Tensor::cat(&[args.cos()?, args.sin()?], D::Minus1)?.to_dtype(model.dtype())?;
    let first = linear(&mut out, "time.0", &w.time_embedder.first, &emb)?;
    put(&mut out, "time.1.input", &first)?;
    let activation = first
        .to_dtype(DType::F32)?
        .silu()?
        .to_dtype(model.dtype())?;
    put(&mut out, "time.1.output", &activation)?;
    let time = linear(&mut out, "time.2", &w.time_embedder.second, &activation)?;
    put(&mut out, "time.output", &time)?;
    let zero = Tensor::zeros((1, 64), model.dtype(), &d)?;
    let padded = Tensor::cat(&[&zero, &state, &zero], 0)?.unsqueeze(0)?;
    x = linear(&mut out, "vae2llm", &w.vae2llm, &padded)?;
    x = (x + time.unsqueeze(0)?)?;
    put(&mut out, "with_time", &x)?;
    x = (x + &engine.pos_emb)?;
    put(&mut out, "injected", &x)?;
    for (i, (layer, (ar_k, ar_v))) in w.layers.iter().zip(&engine.cache).enumerate() {
        let name = format!("nar.layer.{i:02}");
        let normalized = if i == 0 {
            norm(
                &mut out,
                &format!("{name}.norm"),
                &layer.nar_input_layernorm,
                &x,
            )?
        } else {
            layer.nar_input_layernorm.forward(&x)?
        };
        let (q, k, v) = if i == 0 {
            qkv(
                &mut out,
                &name,
                &layer.nar_self_attn,
                &normalized,
                rope,
                &engine.cos,
                &engine.sin,
            )?
        } else {
            layer
                .nar_self_attn
                .project_qkv(&normalized, rope, &engine.cos, &engine.sin)?
        };
        let h = engine.attention(
            &layer.nar_self_attn,
            &q,
            &Tensor::cat(&[ar_k, &k], 1)?,
            &Tensor::cat(&[ar_v, &v], 1)?,
            false,
        )?;
        let o = if i == 0 {
            linear(
                &mut out,
                &format!("{name}.o_proj"),
                &layer.nar_self_attn.o_proj,
                &h,
            )?
        } else {
            linear_forward(&layer.nar_self_attn.o_proj, &h)?
        };
        x = (&x + o)?;
        x = (&x
            + layer
                .nar_mlp
                .forward(&layer.nar_pre_mlp_layernorm.forward(&x)?)?)?;
        put(&mut out, &format!("{name}.output"), &x)?;
    }
    let normalized = norm(&mut out, "final_norm", &backbone.norm, &x)?;
    let projection = linear(&mut out, "projection", &w.llm2vae, &normalized)?;
    let velocity = projection.squeeze(0)?.narrow(0, 1, engine.nar_length - 2)?;
    put(&mut out, "velocity", &velocity)?;
    ensure!(
        velocity
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?
            == engine
                .velocity(&state, 20.)?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?,
        "Staged velocity differs from production"
    );
    let latents =
        engine.solve_observed(32, None, None, Some(&mut |name, x| put(&mut out, name, x)))?;
    put(&mut out, "latents", &latents)?;
    // Isolate every first-layer operation using the exact Python input.
    let get = |name: &str| -> Result<Tensor> { Ok(reference[name].to_device(&d)?) };
    for (branch, n, a) in [
        (
            "ar",
            &backbone.layers[0].input_layernorm,
            &backbone.layers[0].self_attn,
        ),
        (
            "nar",
            &w.layers[0].nar_input_layernorm,
            &w.layers[0].nar_self_attn,
        ),
    ] {
        let prefix = format!("{branch}.layer.00");
        put(
            &mut out,
            &format!("isolated.{prefix}.norm.output"),
            &n.forward(&get(&format!("{prefix}.norm.input"))?)?,
        )?;
        for (name, module) in [
            ("q_proj", &a.q_proj),
            ("k_proj", &a.k_proj),
            ("v_proj", &a.v_proj),
            ("o_proj", &a.o_proj),
        ] {
            put(
                &mut out,
                &format!("isolated.{prefix}.{name}.output"),
                &linear_forward(module, &get(&format!("{prefix}.{name}.input"))?)?,
            )?;
        }
        for (name, module) in [("q_norm", &a.q_norm), ("k_norm", &a.k_norm)] {
            put(
                &mut out,
                &format!("isolated.{prefix}.{name}.output"),
                &module.forward(&get(&format!("{prefix}.{name}.input"))?)?,
            )?;
        }
        for (name, n) in [("q", "q_norm"), ("k", "k_norm")] {
            for (control, cs, ss) in [
                (
                    "own_rope",
                    out[&format!("{branch}.cos")].to_device(&d)?,
                    out[&format!("{branch}.sin")].to_device(&d)?,
                ),
                (
                    "reference_rope",
                    get(&format!("{branch}.cos"))?,
                    get(&format!("{branch}.sin"))?,
                ),
            ] {
                put(
                    &mut out,
                    &format!("isolated.{control}.{prefix}.{name}"),
                    &rope.apply(&get(&format!("{prefix}.{n}.output"))?, &cs, &ss)?,
                )?;
            }
        }
        let q = get(&format!("{prefix}.q"))?;
        let (k, v) = if branch == "ar" {
            (get(&format!("{prefix}.k"))?, get(&format!("{prefix}.v"))?)
        } else {
            (
                Tensor::cat(&[get("cache.00.k")?, get(&format!("{prefix}.k"))?], 1)?,
                Tensor::cat(&[get("cache.00.v")?, get(&format!("{prefix}.v"))?], 1)?,
            )
        };
        put(
            &mut out,
            &format!("isolated.{prefix}.o_proj.input"),
            &engine.attention(a, &q, &k, &v, branch == "ar")?,
        )?;
    }
    for (name, module) in [
        ("vae2llm", &w.vae2llm),
        ("time.0", &w.time_embedder.first),
        ("time.2", &w.time_embedder.second),
        ("projection", &w.llm2vae),
    ] {
        put(
            &mut out,
            &format!("isolated.{name}.output"),
            &linear_forward(module, &get(&format!("{name}.input"))?)?,
        )?;
    }
    for shift in [0.5, 1., 3.] {
        let values = reference["shift.raw"]
            .to_vec1::<f64>()?
            .into_iter()
            .map(|t| shifted_time(t, shift, model.dtype(), &d))
            .collect::<Result<Vec<_>>>()?;
        put(
            &mut out,
            &format!("shift.{shift}"),
            &Tensor::stack(&values, 0)?,
        )?;
    }
    // The solver's arithmetic, separately on the exact Python inputs.
    for i in 0..32 {
        let name = format!("step.{i:02}");
        for (suffix, raw_suffix) in [("shifted", "raw_t"), ("shifted_mid", "raw_mid")] {
            let raw = out[&format!("{name}.{raw_suffix}")].to_scalar::<f64>()?;
            put(
                &mut out,
                &format!("{name}.{suffix}"),
                &shifted_time(raw, w.config.timestep_shift, model.dtype(), &d)?,
            )?;
        }
        let state = get(&format!("{name}.state"))?;
        put(
            &mut out,
            &format!("isolated.{name}.mid"),
            &(&state - scalar_mul(&get(&format!("{name}.first"))?, 1. / 64.)?)?,
        )?;
        put(
            &mut out,
            &format!("isolated.{name}.next"),
            &(&state - scalar_mul(&get(&format!("{name}.second"))?, 1. / 32.)?)?,
        )?;
    }
    candle_core::safetensors::save(&out, root.join("p3b-rust.safetensors"))?;
    println!("P3b Rust: {} tensors; staged prefill/velocity exact against production; actual solver observed",out.len());
    drop(engine);
    drop(model);
    // Precision control only: the integration gate still loads BF16.
    // SAFETY: the same local checkpoint remains immutable.
    let model = unsafe {
        YuE2ForCausalLM::from_pretrained_with_nar(snapshot_dir("YuE2-3B")?, DType::F32, &d)?
    };
    let mut controls = Dump::new();
    for i in 0..2 {
        let chunk = Chunk {
            ar_tokens: fixture[&format!("two_chunk.{i}.ar_tokens")]
                .to_dtype(DType::U32)?
                .to_vec1()?,
            noise: fixture[&format!("two_chunk.{i}.noise")].clone(),
            nar_cond_end: 0,
        };
        let engine = CachedNAR::new(&model, &chunk, 128)?;
        put(
            &mut controls,
            &format!("control.fp32.chunk.{i}"),
            &engine.solve(32, None, None)?,
        )?;
        println!("P3b FP32 control chunk {i} complete");
    }
    candle_core::safetensors::save(&controls, root.join("p3b-rust-fp32.safetensors"))?;
    Ok(())
}
