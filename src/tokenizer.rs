//! Frozen qwen.tiktoken BPE, ported from yue2-infer 0.1.6 tokenization_yue2.py.
use anyhow::{ensure, Context, Result};
use base64::{engine::general_purpose::STANDARD, Engine};
use std::{collections::BTreeMap, path::Path};
use tiktoken_rs::CoreBPE;
use unicode_normalization::UnicodeNormalization;

pub const ORDINARY_TOKENS: u32 = 151643;
pub const TEXT_VOCAB_SIZE: u32 = ORDINARY_TOKENS + 208;
const PATTERN: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

pub struct YuE2TextTokenizer {
    enc: CoreBPE,
}

pub fn special_tokens() -> BTreeMap<String, u32> {
    let mut specials: Vec<String> = [
        "<|endoftext|>",
        "<|im_start|>",
        "<|im_end|>",
        "<R>",
        "<S>",
        "<X>",
        "<mask>",
        "<sep>",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect();
    specials.extend((0..200).map(|i| format!("<extra_{i}>")));
    specials[204] = "<abc>".into();
    specials[205] = "</abc>".into();
    specials
        .into_iter()
        .enumerate()
        .map(|(i, s)| (s, ORDINARY_TOKENS + i as u32))
        .collect()
}

impl YuE2TextTokenizer {
    pub fn new(merge_file: impl AsRef<Path>) -> Result<Self> {
        let file = std::fs::read_to_string(merge_file.as_ref())
            .with_context(|| format!("Read {}", merge_file.as_ref().display()))?;
        let mut entries = Vec::new();
        let mut seen_ranks = std::collections::HashSet::new();
        let mut seen_bytes = std::collections::HashSet::new();
        for line in file.lines().filter(|line| !line.is_empty()) {
            let mut parts = line.split_whitespace();
            let bytes = STANDARD.decode(parts.next().context("Missing BPE bytes")?)?;
            let rank: u32 = parts.next().context("Missing BPE rank")?.parse()?;
            ensure!(parts.next().is_none(), "Unexpected BPE fields");
            ensure!(
                rank < ORDINARY_TOKENS && seen_ranks.insert(rank),
                "Invalid/duplicate BPE rank"
            );
            ensure!(seen_bytes.insert(bytes.clone()), "Duplicate BPE bytes");
            entries.push((bytes, rank));
        }
        ensure!(
            entries.len() == ORDINARY_TOKENS as usize,
            "Expected checkpoint-native qwen.tiktoken (151643 ordinary tokens)"
        );
        let enc = CoreBPE::new(
            entries.into_iter().collect(),
            special_tokens().into_iter().collect(),
            PATTERN,
        )?;
        Ok(Self { enc })
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        self.enc.encode_ordinary(&text.nfc().collect::<String>())
    }

    /// Python filters IDs outside the TEXT vocabulary and replaces invalid UTF-8.
    /// NFC normalization means decomposed input round-trips to its NFC form.
    pub fn decode(&self, ids: &[i64]) -> Result<String> {
        let ids: Vec<u32> = ids
            .iter()
            .copied()
            .filter(|&i| (0..i64::from(TEXT_VOCAB_SIZE)).contains(&i))
            .map(|i| i as u32)
            .collect();
        let bytes = self.enc.decode_bytes(&ids)?;
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }
}
