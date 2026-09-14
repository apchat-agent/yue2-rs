# Phase 0 report

Completed fixtures and the CPU/CUDA candle 0.11.0 skeleton before starting Phase 1.
No Phase 2–5 implementation, weight conversion, network model access, or push.

## Work

- One edition-2021 crate, library and thin placeholder CLI; CPU default, optional
  `cuda`, `flash-attn`, and `metal` features. Dependencies pinned by Cargo.lock.
- Read the installed tokenizer, protocol, sampling, NAR, backbone, VAE, pipeline
  and storage sources. Installed backbone/VAE files also match the cached model
  source files byte for byte (`cmp`, exit 0).
- `tools/dump_reference.py` imports the installed Python 0.1.6 implementation,
  records eager BF16 AR logits and FP32 hidden states, and instruments the actual
  CachedNAR.solve velocity calls. It asserts every captured midpoint/update against
  CUDA arithmetic. All 32 steps include input state, both velocities, midpoint,
  next state and both raw times. No per-layer NAR hidden activations are claimed.
- Tokens include the original three .npy arrays, 20 probes, all 208 tokenizer
  specials, and all three requests' original ABC/saved prefixes, generation
  prompts, request text encodings, and CFG negative prefixes.
- VAE fixtures contain all 1,484 reference latent frames, the first 192,000
  unclipped stereo samples from full-song halo/crop decoding, and six DecoderBlock
  outputs plus final audio for a 64-frame slice.
- JSON sidecars describe layouts, requests, seeds, configuration, package versions,
  source hashes, weight SHA256s and every tensor shape/dtype.
- Fixtures are ignored and reside in `~/work/yue2-rs-fixtures/first-song/`.
  Run logs reside in `~/work/yue2-rs-logs/`.

## Gates and self-audit

Initial runs: `p0-dump.log`, `p0-build-cpu.log`, `p0-build-cuda.log`,
`p0-test.log`: all exit 0. Then ran **cargo clean** (all build artifacts,
including dependencies) and repeated every P0 gate. Numbers below are from this
second run. Compiler progress is omitted from pasted build/test output; complete
output remains in the named logs.

Exact build audit commands (from repository root):

```bash
export PATH=/home/ayourtch/.cargo/bin:/usr/local/cuda/bin:$PATH CUDA_VISIBLE_DEVICES=0 CUDA_COMPUTE_CAP=120
cargo clean > /home/ayourtch/work/yue2-rs-logs/p0-audit-clean.log 2>&1 && cargo test > /home/ayourtch/work/yue2-rs-logs/p0-audit-test.log 2>&1 && cargo build > /home/ayourtch/work/yue2-rs-logs/p0-audit-build-cpu.log 2>&1 && cargo build --features cuda > /home/ayourtch/work/yue2-rs-logs/p0-audit-build-cuda.log 2>&1 && cargo fmt --check > /home/ayourtch/work/yue2-rs-logs/p0-audit-fmt.log 2>&1 && cargo clippy --all-targets -- -D warnings > /home/ayourtch/work/yue2-rs-logs/p0-audit-clippy.log 2>&1
```

Clean output:

```text
     Removed 2692 files, 2.0GiB total
```

CPU test gate:

```text
   Compiling tokenizers v0.22.2
   Compiling candle-core v0.11.0
   Compiling candle-nn v0.11.0
   Compiling yue2 v0.1.0 (/home/ayourtch/work/yue2-rs)
    Finished `test` profile [optimized + debuginfo] target(s) in 25.92s
     Running unittests src/lib.rs (target/debug/deps/yue2-cf389c1e71801c47)

running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

     Running unittests src/bin/yue2.rs (target/debug/deps/yue2-5a8bed59e4c7f151)

running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

   Doc-tests yue2

running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
```

CPU build, CUDA build, clippy respectively:

```text
   Compiling yue2 v0.1.0 (/home/ayourtch/work/yue2-rs)
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 7.46s
   Compiling yue2 v0.1.0 (/home/ayourtch/work/yue2-rs)
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 45.50s
    Checking yue2 v0.1.0 (/home/ayourtch/work/yue2-rs)
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 4.15s
```

`cargo fmt --check`: exit 0, no output. All other audit commands exit 0.

Exact fixture audit command (GPU availability checked immediately before it):

```bash
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv && PYTHONDONTWRITEBYTECODE=1 HF_HOME=/home/ayourtch/yue2/hf HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 /home/ayourtch/yue2/.venv/bin/python -u tools/dump_reference.py --greedy --two-chunks > /home/ayourtch/work/yue2-rs-logs/p0-audit-dump.log 2>&1
```

GPU 0 had 34,858 MiB free (62,432 MiB used by the existing server). The dumper
caps the PyTorch allocator at 28 GiB and never touches another GPU.

```text
first-song: generation prompt=128, saved semantic prefix=611, ABC=481
song2: generation prompt=200, saved semantic prefix=1670, ABC=1468
song3: generation prompt=202, saved semantic prefix=1456, ABC=1252
tokens.safetensors: 116,448 bytes; 39 tensors
ar_logits.safetensors: 49,382,360 bytes; 6 tensors
greedy.safetensors: 47,284,880 bytes; 2 tensors
greedy oracle: 64 steps; original sampled ABC matches=61/64; first difference=16
NAR native.0: context=24576, frames=1484, AR=2096
NAR two_chunk.0: context=2098, frames=742, AR=1354
NAR two_chunk.1: context=2098, frames=742, AR=1354
nar.safetensors: 62,407,080 bytes; 681 tensors
/home/ayourtch/yue2/.venv/lib/python3.12/site-packages/torch/nn/utils/weight_norm.py:144: FutureWarning: `torch.nn.utils.weight_norm` is deprecated in favor of `torch.nn.utils.parametrizations.weight_norm`.
  WeightNorm.apply(module, name, dim)
vae.safetensors: 79,160,624 bytes; 10 tensors
Total fixture bytes: 238,468,586 <= 1,500,000,000
model model.safetensors sha256: 1d55c42c1a9875c34f5d736e15078449992b044e807ce2a138e6cf289a1e59e9
vae model.safetensors sha256: 807ce9d5149fa27c5ad3e6582058469852e908f6c5acc8c8aa338e7ab7751346
Peak CUDA allocated/reserved: 7,630,939,136/7,772,045,312 bytes
P0 dump PASS (16.84s)
```

## Unsure list and design objections

1. **The reference has one native NAR chunk**, not two: 1,484 frames at context
   24,576, prefix length 611. The fixture preserves that native chunk and adds a
   separately labeled `two_chunk` case with context **2098**, yielding two
   742-frame chunks. Both use the original full-song seeded noise draw. This is
   an explicit supplementary test case, not an assertion that the saved run used
   two chunks. The native second chunk does not exist.
2. **prefix.npy is the semantic-generation prefix**, including the sampled ABC
   and ABC_END/MUSIC_START: 611 tokens. The ABC-generation prompt is 128 tokens.
   All three requests have both forms recorded. P1b must compare the saved
   prefix using `token_prefixes(request, tokenizer, saved_abc_ids)`; calling
   without ABC IDs correctly returns the shorter generation prompt.
3. **The original ABC was sampled**, temperature 0.7, so literal P1d equality
   against that immutable .npy conflicts with “greedy in both.” Eager greedy
   (temperature 0, otherwise release ABC settings) matches 61/64 tokens, first
   mismatch at zero-based position 16. `--greedy` writes a separate
   `greedy.safetensors` and records this discrepancy without altering the source
   or calling the sampled-reference gate passed.
4. P0's literal first 64 positions of prefix + ABC all fall inside the request
   text; they do not exercise ABC positions. P1d additionally exercises the full
   generation prompt and 64 new ABC tokens.
5. Only CUDA and CPU feature builds are gates here. Metal and flash-attn are
   declared optional dependencies; their builds/behavior have not been verified.
6. No gate thresholds were relaxed. No objections to the fixed architecture.
