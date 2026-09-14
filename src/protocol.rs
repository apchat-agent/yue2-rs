//! Checkpoint-native prompting and defaults; port of yue2-infer 0.1.6 protocol.py.
use crate::tokenizer::YuE2TextTokenizer;
use anyhow::{bail, ensure, Result};
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value;

pub const EOD: u32 = 151643;
pub const ABC_START: u32 = 151847;
pub const ABC_END: u32 = 151848;
pub const MUSIC_START: u32 = 151851;
pub const MUSIC_END: u32 = 151852;
pub const CODEC_OFFSET: u32 = 151853;
pub const CODEC_SIZE: u32 = 32768;
pub const LATENT_START: u32 = 184621;
pub const LATENT_END: u32 = 184622;
pub const LATENT_PAD: u32 = 184623;
pub const VOCAB_SIZE: usize = 184704;
pub const CONTEXT: usize = 24576;
pub const PROTOCOL_VERSION: &str = "yue2-native-v1";

pub fn instruction(cot: &str) -> Result<&'static str> {
    match cot {
        "off" => Ok("Generate music with codec tokens from the given conditions."),
        "melody" => Ok("Generate a melody-only ABC transcription without chord symbols, then generate music with codec tokens from the given conditions."),
        "full" => Ok("Generate a chord-annotated ABC transcription, then generate music with codec tokens from the given conditions."),
        _ => bail!("cot must be off, melody or full"),
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Sampling {
    pub temperature: f64,
    pub top_p: f64,
    pub top_k: usize,
    pub repetition_penalty: f64,
    pub penalty_window: usize,
    pub min_tokens: usize,
    pub max_tokens: usize,
}

impl Default for Sampling {
    fn default() -> Self {
        Self {
            temperature: 1.,
            top_p: 0.95,
            top_k: 100,
            repetition_penalty: 1.2,
            penalty_window: 50,
            min_tokens: 200,
            max_tokens: 9000,
        }
    }
}

impl Sampling {
    pub fn abc_default() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.9,
            top_k: 30,
            repetition_penalty: 1.005,
            penalty_window: 100,
            min_tokens: 32,
            max_tokens: 4096,
        }
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            [self.temperature, self.top_p, self.repetition_penalty]
                .into_iter()
                .all(f64::is_finite),
            "Sampling numbers must be finite"
        );
        ensure!(
            (0.0..=5.0).contains(&self.temperature)
                && self.top_p > 0.
                && self.top_p <= 1.
                && self.top_k >= 1,
            "Invalid sampling temperature/top_p/top_k"
        );
        ensure!(
            self.repetition_penalty > 0. && (1..=100).contains(&self.penalty_window),
            "Invalid repetition penalty/window"
        );
        ensure!(
            self.min_tokens <= self.max_tokens && self.max_tokens >= 1,
            "Require 0 <= min_tokens <= max_tokens"
        );
        Ok(())
    }

    pub fn to_dict(&self) -> Result<Value> {
        self.validate()?;
        Ok(serde_json::to_value(self)?)
    }
}

impl<'de> Deserialize<'de> for Sampling {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Fields {
            temperature: f64,
            top_p: f64,
            top_k: usize,
            repetition_penalty: f64,
            penalty_window: usize,
            min_tokens: usize,
            max_tokens: usize,
        }
        let value = Value::deserialize(deserializer)?;
        let mut merged = serde_json::to_value(Self::default()).map_err(serde::de::Error::custom)?;
        merge_object(&mut merged, &value).map_err(serde::de::Error::custom)?;
        let f: Fields = serde_json::from_value(merged).map_err(serde::de::Error::custom)?;
        let result = Self {
            temperature: f.temperature,
            top_p: f.top_p,
            top_k: f.top_k,
            repetition_penalty: f.repetition_penalty,
            penalty_window: f.penalty_window,
            min_tokens: f.min_tokens,
            max_tokens: f.max_tokens,
        };
        result.validate().map_err(serde::de::Error::custom)?;
        Ok(result)
    }
}

fn merge_object(base: &mut Value, overrides: &Value) -> Result<()> {
    let Some(values) = overrides.as_object() else {
        bail!("Expected a dictionary of overrides")
    };
    let base = base.as_object_mut().expect("default config is an object");
    for (key, value) in values {
        base.insert(key.clone(), value.clone());
    }
    Ok(())
}

pub fn resolve_sampling(value: Option<&Value>, default: &Sampling) -> Result<Sampling> {
    let Some(value) = value.filter(|v| !v.is_null()) else {
        default.validate()?;
        return Ok(default.clone());
    };
    let mut merged = default.to_dict()?;
    merge_object(&mut merged, value)?;
    Ok(serde_json::from_value(merged)?)
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct GenerationConfig {
    pub abc: Sampling,
    pub semantic: Sampling,
    pub ode_steps: usize,
    pub ode_method: String,
    pub context: usize,
    pub version: String,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            abc: Sampling::abc_default(),
            semantic: Sampling::default(),
            ode_steps: 32,
            ode_method: "midpoint".into(),
            context: CONTEXT,
            version: PROTOCOL_VERSION.into(),
        }
    }
}

impl GenerationConfig {
    pub fn validate(&self) -> Result<()> {
        self.abc.validate()?;
        self.semantic.validate()?;
        ensure!(
            self.context == CONTEXT && self.ode_method == "midpoint" && self.ode_steps >= 1,
            "Require context=24576 and midpoint with positive integer steps"
        );
        Ok(())
    }
    pub fn to_dict(&self) -> Result<Value> {
        self.validate()?;
        Ok(serde_json::to_value(self)?)
    }
    pub fn from_dict(value: &Value) -> Result<Self> {
        Ok(serde_json::from_value(value.clone())?)
    }
}

impl<'de> Deserialize<'de> for GenerationConfig {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Fields {
            abc: Sampling,
            semantic: Sampling,
            ode_steps: usize,
            ode_method: String,
            context: usize,
            version: String,
        }
        let value = Value::deserialize(deserializer)?;
        let mut merged = serde_json::to_value(Self::default()).map_err(serde::de::Error::custom)?;
        for key in ["abc", "semantic"] {
            if let Some(overrides) = value.get(key) {
                merge_object(&mut merged[key], overrides).map_err(serde::de::Error::custom)?;
            }
        }
        let Some(values) = value.as_object() else {
            return Err(serde::de::Error::custom("Expected generation dictionary"));
        };
        for (key, value) in values {
            if key != "abc" && key != "semantic" {
                merged[key] = value.clone();
            }
        }
        let f: Fields = serde_json::from_value(merged).map_err(serde::de::Error::custom)?;
        let result = Self {
            abc: f.abc,
            semantic: f.semantic,
            ode_steps: f.ode_steps,
            ode_method: f.ode_method,
            context: f.context,
            version: f.version,
        };
        result.validate().map_err(serde::de::Error::custom)?;
        Ok(result)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "SongRequestFields")]
pub struct SongRequest {
    pub style: String,
    pub lyrics: String,
    pub cot: String,
    pub seed: u64,
    pub abc: Option<String>,
    pub cfg_scale: Option<f64>,
    pub id: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SongRequestFields {
    style: String,
    lyrics: String,
    #[serde(default = "default_cot")]
    cot: String,
    #[serde(default = "default_seed")]
    seed: u64,
    abc: Option<String>,
    cfg_scale: Option<f64>,
    #[serde(default = "default_id")]
    id: String,
}
fn default_cot() -> String {
    "full".into()
}
fn default_seed() -> u64 {
    831001
}
fn default_id() -> String {
    "song".into()
}

impl TryFrom<SongRequestFields> for SongRequest {
    type Error = anyhow::Error;
    fn try_from(f: SongRequestFields) -> Result<Self> {
        let result = Self {
            style: f.style,
            lyrics: f.lyrics,
            cot: f.cot,
            seed: f.seed,
            abc: f.abc,
            cfg_scale: f.cfg_scale,
            id: f.id,
        };
        result.validate()?;
        Ok(result)
    }
}

impl SongRequest {
    pub fn new(style: impl Into<String>, lyrics: impl Into<String>) -> Self {
        Self {
            style: style.into(),
            lyrics: lyrics.into(),
            cot: default_cot(),
            seed: default_seed(),
            abc: None,
            cfg_scale: None,
            id: default_id(),
        }
    }
    pub fn validate(&self) -> Result<()> {
        instruction(&self.cot)?;
        ensure!(
            self.seed < 1u64 << 63,
            "seed must be an integer in [0, 2**63)"
        );
        ensure!(
            !self.id.is_empty()
                && self.id.len() <= 180
                && self.id.as_bytes()[0].is_ascii_alphanumeric()
                && self
                    .id
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b"_.-".contains(&b)),
            "id must be a filename-safe identifier"
        );
        if let Some(abc) = &self.abc {
            ensure!(
                self.cot != "off"
                    && !abc
                        .trim_matches(
                            |c: char| c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
                        )
                        .is_empty(),
                "External ABC requires nonempty text and cot=melody/full"
            );
        }
        if let Some(cfg) = self.cfg_scale {
            ensure!(
                cfg.is_finite() && (0.0..=20.0).contains(&cfg),
                "cfg_scale must be finite and in [0,20]"
            );
        }
        Ok(())
    }
    pub fn guidance(&self) -> f64 {
        self.cfg_scale
            .unwrap_or(if self.cot == "off" { 1.01 } else { 1.0 })
    }
    pub fn text(&self) -> Result<String> {
        self.validate()?;
        Ok(format!(
            "{}\n[Tags]\n{}\n[Lyrics]\n{}\n",
            instruction(&self.cot)?,
            self.style,
            self.lyrics
        ))
    }
    pub fn to_dict(&self) -> Result<Value> {
        self.validate()?;
        Ok(serde_json::to_value(self)?)
    }
}

fn validate_abc(abc_ids: &[u32]) -> Result<()> {
    ensure!(
        abc_ids.iter().all(|&t| t < EOD),
        "ABC IDs must remain inside the ordinary text vocabulary"
    );
    Ok(())
}

pub fn token_prefixes(
    request: &SongRequest,
    tokenizer: &YuE2TextTokenizer,
    abc_ids: Option<&[u32]>,
) -> Result<Vec<u32>> {
    let mut base = vec![EOD];
    base.extend(tokenizer.encode(&request.text()?));
    if request.cot == "off" {
        base.extend([ABC_START, ABC_END, MUSIC_START]);
        return Ok(base);
    }
    let external = request.abc.as_ref().map(|abc| tokenizer.encode(abc));
    let abc_ids = abc_ids.or(external.as_deref());
    base.push(ABC_START);
    if let Some(abc_ids) = abc_ids {
        validate_abc(abc_ids)?;
        base.extend_from_slice(abc_ids);
        base.extend([ABC_END, MUSIC_START]);
    }
    Ok(base)
}

pub fn negative_prefix(
    request: &SongRequest,
    tokenizer: &YuE2TextTokenizer,
    abc_ids: Option<&[u32]>,
) -> Result<Vec<u32>> {
    request.validate()?;
    let mut base = vec![EOD];
    base.extend(tokenizer.encode(instruction(&request.cot)?));
    if request.cot == "off" {
        base.push(MUSIC_START);
        return Ok(base);
    }
    let Some(abc_ids) = abc_ids else {
        bail!("Symbolic CFG must retain the exact positive-branch ABC IDs")
    };
    validate_abc(abc_ids)?;
    base.push(ABC_START);
    base.extend_from_slice(abc_ids);
    base.extend([ABC_END, MUSIC_START]);
    Ok(base)
}

pub fn chunk_ranges(
    frames: usize,
    prefix_tokens: usize,
    context: usize,
) -> Result<Vec<(usize, usize)>> {
    let size = context
        .checked_sub(prefix_tokens)
        .and_then(|n| n.checked_sub(3))
        .map(|n| (n / 2).min(CONTEXT))
        .unwrap_or(0);
    ensure!(
        frames >= 1 && size >= 1,
        "Empty codec or prefix leaves no acoustic context"
    );
    Ok((0..frames)
        .step_by(size)
        .map(|a| (a, a.saturating_add(size).min(frames)))
        .collect())
}
