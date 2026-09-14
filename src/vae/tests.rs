use super::*;
use std::collections::HashMap;

#[test]
fn decoder_default_tanh_and_config_json_override() -> Result<()> {
    assert!(DecoderConfig::default().final_tanh);
    let omitted: YuE2VAEConfig = serde_json::from_str(r#"{"decoder_config": {}}"#)?;
    assert!(omitted.decoder_config.final_tanh);
    // Same nested config.json field used by the released checkpoint loader.
    let explicit: YuE2VAEConfig =
        serde_json::from_str(r#"{"decoder_config": {"final_tanh": false}}"#)?;
    assert!(!explicit.decoder_config.final_tanh);
    let enabled: YuE2VAEConfig =
        serde_json::from_str(r#"{"decoder_config": {"final_tanh": true}}"#)?;
    assert!(enabled.decoder_config.final_tanh);
    Ok(())
}

#[test]
fn transposed_weight_norm_axis_and_padding_match_scalar_scatter() -> Result<()> {
    // Unequal channel counts expose an accidental dim=1 norm or transposition.
    for stride in [2, 3, 5, 6] {
        let (batch, input, output, frames, kernel) = (2, 2, 3, 4, 2 * stride);
        let values = (0..input * output * kernel)
            .map(|i| (i as f32 * 0.7).sin())
            .collect::<Vec<_>>();
        let gains = [0.3f32, 0.9];
        let bias = [0.1f32, -0.2, 0.3];
        let tensors = HashMap::from([
            (
                "weight_v".into(),
                Tensor::from_vec(values.clone(), (input, output, kernel), &Device::Cpu)?,
            ),
            (
                "weight_g".into(),
                Tensor::new(&gains, &Device::Cpu)?.reshape((input, 1, 1))?,
            ),
            ("bias".into(), Tensor::new(&bias, &Device::Cpu)?),
        ]);
        let layer = WNConvTranspose1d::load(
            input,
            output,
            stride,
            VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu),
        )?;
        let x = (0..batch * input * frames)
            .map(|i| (i as f32 * 0.3).cos())
            .collect::<Vec<_>>();
        let actual = layer.forward(&Tensor::from_vec(
            x.clone(),
            (batch, input, frames),
            &Device::Cpu,
        )?)?;
        let length = frames * stride - stride % 2;
        assert_eq!(actual.dims(), [batch, output, length]);
        let mut expected = vec![0f64; batch * output * length];
        for b in 0..batch {
            for (o, &bias) in bias.iter().enumerate() {
                expected[(b * output + o) * length..(b * output + o + 1) * length]
                    .fill(bias as f64);
            }
            for (i, &gain) in gains.iter().enumerate() {
                let norm = values[i * output * kernel..(i + 1) * output * kernel]
                    .iter()
                    .map(|&v| (v as f64).powi(2))
                    .sum::<f64>()
                    .sqrt();
                for t in 0..frames {
                    for o in 0..output {
                        for k in 0..kernel {
                            let pos = (t * stride + k) as isize - stride.div_ceil(2) as isize;
                            if (0..length as isize).contains(&pos) {
                                expected[(b * output + o) * length + pos as usize] +=
                                    x[(b * input + i) * frames + t] as f64
                                        * values[(i * output + o) * kernel + k] as f64
                                        * gain as f64
                                        / norm;
                            }
                        }
                    }
                }
            }
        }
        for (index, (a, b)) in actual
            .flatten_all()?
            .to_vec1::<f32>()?
            .iter()
            .zip(expected)
            .enumerate()
        {
            assert!(
                (*a as f64 - b).abs() < 3e-7,
                "stride={stride} index={index}: {a} vs {b}"
            );
        }
    }
    Ok(())
}

fn conv(
    tensors: &mut HashMap<String, Tensor>,
    name: &str,
    shape: (usize, usize, usize),
    bias: Option<usize>,
) -> Result<()> {
    let (a, b, k) = shape;
    let v = (0..a * b * k)
        .map(|i| (0.4 * i as f32 + 0.1).sin())
        .collect::<Vec<_>>();
    tensors.insert(
        format!("{name}.weight_v"),
        Tensor::from_vec(v, shape, &Device::Cpu)?,
    );
    tensors.insert(
        format!("{name}.weight_g"),
        Tensor::full(0.4f32, (a, 1, 1), &Device::Cpu)?,
    );
    if let Some(channels) = bias {
        tensors.insert(
            format!("{name}.bias"),
            Tensor::full(0.01f32, channels, &Device::Cpu)?,
        );
    }
    Ok(())
}

fn activation(tensors: &mut HashMap<String, Tensor>, name: &str, channels: usize) -> Result<()> {
    for (suffix, value) in [("alpha", 0.2f32), ("beta", -0.1)] {
        tensors.insert(
            format!("{name}.{suffix}"),
            Tensor::full(value, channels, &Device::Cpu)?,
        );
    }
    Ok(())
}

fn tiny() -> Result<YuE2VAE> {
    let config = YuE2VAEConfig {
        decoder_config: DecoderConfig {
            channels: 2,
            latent_dim: 2,
            c_mults: vec![1, 2],
            strides: vec![2, 3],
            ..Default::default()
        },
        latent_dim: 2,
        downsampling_ratio: 6,
        decode_core_frames: 17,
        decode_halo_frames: 40,
        ..Default::default()
    };
    let mut tensors = HashMap::new();
    conv(&mut tensors, "decoder.layers.0", (4, 2, 7), Some(4))?;
    for (block, input, stride) in [(1, 4, 3), (2, 2, 2)] {
        let stem = format!("decoder.layers.{block}.layers");
        activation(&mut tensors, &format!("{stem}.0"), input)?;
        conv(
            &mut tensors,
            &format!("{stem}.1"),
            (input, 2, 2 * stride),
            Some(2),
        )?;
        for unit in 2..5 {
            let stem = format!("{stem}.{unit}.layers");
            activation(&mut tensors, &format!("{stem}.0"), 2)?;
            conv(&mut tensors, &format!("{stem}.1"), (2, 2, 7), Some(2))?;
            activation(&mut tensors, &format!("{stem}.2"), 2)?;
            conv(&mut tensors, &format!("{stem}.3"), (2, 2, 1), Some(2))?;
        }
    }
    activation(&mut tensors, "decoder.layers.3", 2)?;
    conv(&mut tensors, "decoder.layers.4", (2, 2, 7), None)?;
    YuE2VAE::load(
        config,
        VarBuilder::from_tensors(tensors, DType::F32, &Device::Cpu),
    )
}

#[test]
fn tiled_decode_preserves_batch_boundaries_and_natural_tail() -> Result<()> {
    let model = tiny()?;
    // Nontrivial two-batch input covers multiple cores and a short final tile.
    let input = Tensor::from_vec(
        (0..2 * 2 * 89).map(|i| (i as f32 * 1.7).sin()).collect(),
        (2, 2, 89),
        &Device::Cpu,
    )?;
    let full = model.decode(&input)?;
    assert_eq!(full.dims(), [2, 2, 89 * 6 - 2]);
    for batch in 0..2 {
        let separate = model.decode(&input.narrow(0, batch, 1)?)?;
        let max = (full.narrow(0, batch, 1)? - separate)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()?;
        assert!(max < 2e-6, "batched vs separate max_abs={max}");
    }
    let halo = model.required_halo(17)?;
    let mut calls = Vec::new();
    let mut progress = |done, total| calls.push((done, total));
    let tiled = model.decode_tiled(&input, Some(17), Some(halo), Some(&mut progress))?;
    assert_eq!(calls, (1..=6).map(|i| (i, 6)).collect::<Vec<_>>());
    assert_eq!(tiled.shape(), full.shape());
    let max = (full - tiled)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(max < 2e-6, "tiled max_abs={max}");
    let short = input.narrow(2, 0, 1)?;
    assert_eq!(model.decode(&short)?.dims(), [2, 2, 4]);
    assert_eq!(
        model.decode_tiled(&short, None, None, None)?.dims(),
        [2, 2, 4]
    );
    assert!(model.decode_tiled(&input, Some(0), None, None).is_err());
    assert!(model
        .decode_tiled(&input, Some(17), Some(halo - 1), None)
        .is_err());
    assert!(model
        .decode(&Tensor::zeros((1, 3, 4), DType::F32, &Device::Cpu)?)
        .is_err());
    assert!(model
        .decode(&Tensor::full(f32::NAN, (1, 2, 4), &Device::Cpu)?)
        .is_err());
    assert!(model.natural_output_length(0).is_err());
    assert!(model.natural_output_length(usize::MAX).is_err());
    Ok(())
}

#[test]
fn decoder_config_rejects_unsupported_architectures_and_precision() -> Result<()> {
    let mut config = YuE2VAEConfig::default();
    config.validate()?;
    config.decoder_config.use_filter = true;
    assert!(config.validate().is_err());
    config.decoder_config.use_filter = false;
    config.downsampling_ratio = 100;
    assert!(config.validate().is_err());
    assert!(YuE2VAE::load(
        YuE2VAEConfig::default(),
        VarBuilder::zeros(DType::BF16, &Device::Cpu)
    )
    .is_err());
    Ok(())
}
