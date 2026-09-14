# Phase 1 report

Phase 1 was executed after the completed Phase 0 self-audit and REPORT-P0.md.
Tokenizer/protocol and AR inference are implemented. **This is not an
unconditional all-gates-pass result:** P1c needs the reference cuBLAS version,
and the literal P1d comparison with the original sampled ABC fails. All failures
are retained as executable tests and reported below. No Phase 2–5 implementation
or push was performed.

## What changed

- `tokenizer`: native qwen.tiktoken ranks via tiktoken-rs CoreBPE, exact regex
  and 208 specials, NFC normalization, ordinary encoding of special-token-looking
  text, filtered/lossy UTF-8 decode.
- `protocol`: constants/instructions, validated SongRequest/Sampling/
  GenerationConfig, phase-specific partial overrides, guidance, positive and
  negative prefixes, and original context chunk ranges. Rust types reject
  noninteger counts/seeds; Python's extra strip control characters are handled.
- `model`: checkpoint-shaped AR embedding, RMSNorm, RoPE, GQA causal attention,
  SwiGLU, all 28 AR layers, final norm/lm_head, teacher-forcing captures, and a
  fixed-capacity in-place StaticKVCache. Chunked prefill uses absolute causal
  offsets; capacity exhaustion errors rather than truncating generation.
- Model weights are mmap-loaded through candle VarBuilder from an explicit
  directory. Local HF_HOME snapshot resolution is available; nothing converts,
  copies, or downloads weights. CPU BF16 linear operations accumulate in FP32
  because candle's CPU matmul does not implement BF16.
- Attention remains plain candle tensor operations, with bounded query tiles.
  RMSNorm preserves Python's intermediate BF16 rounding; SiLU computes in FP32
  and casts once. The attention softmax numerator rounds to BF16 before the
  value product, matching the observed eager Torch CUDA SDPA arithmetic.
- Added seven small CPU regression tests and ignored fixture gates. General
  sampling is not implemented; a test-only greedy harness covers P1d.
- Added optional operation diagnostics, a reference-cuBLAS linker helper, and a
  clean-build audit script. The CLI remains the Phase 0 placeholder. NAR/VAE
  routing, TimestepEmbedder and AudioPositionEmbedding are deferred to Phase 3.

## Audited gate results

| Gate | Result from the clean audit |
| --- | --- |
| P1a | PASS: 20/20 probes and round-trips; 208/208 specials; full 611-token saved prefix exact |
| P1b | PASS with exact saved ABC IDs: first-song 611/611, song2 1670/1670, song3 1456/1456; generation prompts and CFG negatives also exact |
| P1c, reference cuBLAS 12.8.4.1 | PASS: argmax 64/64; maximum absolute logit difference 0.375 |
| P1c, default system cuBLAS 13.6 | **FAIL**: argmax 63/64; maximum absolute logit difference 1.0 |
| P1d, Python eager greedy oracle | PASS: 64/64, both CUDA and CPU |
| P1d, literal original abc_tokens.npy | **FAIL**: 61/64; the original artifact is sampled |

P1c hidden-state metrics using reference cuBLAS:

| Capture | Whole-tensor cosine | Minimum per-position cosine | Maximum absolute difference |
| --- | ---: | ---: | ---: |
| Embedding | 1.000000000000 | 1.000000000000 | 0 |
| After layer 0 | 0.999999994917 | 0.999999583650 | 0.5 |
| After layer 13 | 0.999999144118 | 0.999965031256 | 16 |
| After final norm | 0.999973469979 | 0.999865967444 | 0.25 |

All required layer cosines exceed 0.999; the test additionally checks every
position. Intermediate residual magnitudes can be large in BF16, so their maximum
absolute differences are shown for context, not substituted for the cosine gate.

## Exact commands, pasted output, and self-audit

All implementation edits preceded the final audit. Every requested P1 gate was
run, then rerun after **cargo clean** and a fresh **cargo test**. CPU/CUDA builds,
both clippy configurations, formatting and unset-fixture behavior were checked
again. Every gate number above comes from this second run. Full logs live outside
the repository in `~/work/yue2-rs-logs/`; compiler progress is omitted from the
pasted excerpts below. The audit exits **1**, preserving the two documented
failing comparisons instead of filtering them out.

Exact top-level audit command:

```bash
export PATH=/home/ayourtch/.cargo/bin:/usr/local/cuda/bin:$PATH CUDA_VISIBLE_DEVICES=0 CUDA_COMPUTE_CAP=120 YUE2_FIXTURES=/home/ayourtch/work/yue2-rs-fixtures
./tools/audit_phase1.sh > /home/ayourtch/work/yue2-rs-logs/p1-audit-commands.log 2>&1
```

The script sets `YUE2_TEST_DEVICE=cuda`. Its exact command/exit transcript:

```text
clean command: cargo clean
clean exit=0 log=/home/ayourtch/work/yue2-rs-logs/p1-audit-clean.log
test command: cargo test
test exit=0 log=/home/ayourtch/work/yue2-rs-logs/p1-audit-test.log
build-cpu command: cargo build
build-cpu exit=0 log=/home/ayourtch/work/yue2-rs-logs/p1-audit-build-cpu.log
build-cuda command: cargo build --features cuda
build-cuda exit=0 log=/home/ayourtch/work/yue2-rs-logs/p1-audit-build-cuda.log
fmt command: cargo fmt --check
fmt exit=0 log=/home/ayourtch/work/yue2-rs-logs/p1-audit-fmt.log
clippy command: cargo clippy --all-targets -- -D warnings
clippy exit=0 log=/home/ayourtch/work/yue2-rs-logs/p1-audit-clippy.log
clippy-cuda command: cargo clippy --features cuda --all-targets -- -D warnings
clippy-cuda exit=0 log=/home/ayourtch/work/yue2-rs-logs/p1-audit-clippy-cuda.log
unset-fixtures command: env -u YUE2_FIXTURES cargo test -- --ignored --nocapture --test-threads=1
unset-fixtures exit=0 log=/home/ayourtch/work/yue2-rs-logs/p1-audit-unset-fixtures.log
gpu command: nvidia-smi --query-gpu=index\,memory.used\,memory.free --format=csv
gpu exit=0 log=/home/ayourtch/work/yue2-rs-logs/p1-audit-gpu.log
parity command: tools/with_reference_blas.sh cargo test --features cuda --test parity -- --ignored --nocapture --test-threads=1
parity exit=101 log=/home/ayourtch/work/yue2-rs-logs/p1-audit-parity.log
system-cuda command: cargo test --features cuda --test parity p1c -- --ignored --nocapture
system-cuda exit=101 log=/home/ayourtch/work/yue2-rs-logs/p1-audit-system-cuda.log
cpu-greedy command: env YUE2_TEST_DEVICE=cpu cargo test --test parity p1d_greedy_oracle -- --ignored --nocapture
cpu-greedy exit=0 log=/home/ayourtch/work/yue2-rs-logs/p1-audit-cpu-greedy.log
```

Clean and build/clippy completion output, in command order:

```text
     Removed 8904 files, 7.0GiB total
   Compiling yue2 v0.1.0 (/home/ayourtch/work/yue2-rs)
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 7.53s
   Compiling yue2 v0.1.0 (/home/ayourtch/work/yue2-rs)
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 42.68s
    Checking yue2 v0.1.0 (/home/ayourtch/work/yue2-rs)
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 4.45s
    Checking yue2 v0.1.0 (/home/ayourtch/work/yue2-rs)
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 40.49s
```

`cargo fmt --check`: exit 0, no output. CPU test output:

```text
     Running unittests src/lib.rs (target/debug/deps/yue2-cf389c1e71801c47)

running 1 test
test model::diagnostics::ar_operations_oracle ... ignored, requires YUE2_FIXTURES, CUDA, and tools/dump_ar_debug.py

test result: ok. 0 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out; finished in 0.00s

     Running unittests src/bin/yue2.rs (target/debug/deps/yue2-5a8bed59e4c7f151)

running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

     Running tests/core.rs (target/debug/deps/core-358a88d276d5d768)

running 7 tests
test chunk_boundaries_and_insufficient_context ... ok
test generation_overrides_preserve_phase_defaults ... ok
test request_defaults_text_guidance_and_validation ... ok
test reject_invalid_sampling_and_generation_json ... ok
test future_tokens_do_not_change_past_logits_and_batches_are_independent ... ok
test cpu_bf16_forward_and_cache_are_supported ... ok
test cached_chunked_and_single_token_forward_match_causal_prefill ... ok

test result: ok. 7 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.03s

     Running tests/parity.rs (target/debug/deps/parity-00628a4316c09dc3)

running 5 tests
test p1a_tokenizer ... ignored, requires YUE2_FIXTURES and the local tokenizer
test p1b_protocol_prefixes ... ignored, requires YUE2_FIXTURES and the local tokenizer
test p1c_teacher_forced ... ignored, requires YUE2_FIXTURES and the 3B checkpoint; YUE2_TEST_DEVICE=cuda for BF16 gate
test p1d_greedy_oracle ... ignored, requires --greedy fixtures and the 3B checkpoint
test p1d_literal_saved_abc ... ignored, literal TASK gate; original abc_tokens.npy was sampled, so this can fail independently of greedy parity

test result: ok. 0 passed; 0 failed; 5 ignored; 0 measured; 0 filtered out; finished in 0.00s

   Doc-tests yue2

running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
```

Unset-fixture invocation (all ignored tests return cleanly without loading files):

```text
     Running unittests src/lib.rs (target/debug/deps/yue2-cf389c1e71801c47)

running 1 test
test model::diagnostics::ar_operations_oracle ... SKIP: YUE2_FIXTURES is unset
ok

test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

     Running unittests src/bin/yue2.rs (target/debug/deps/yue2-5a8bed59e4c7f151)

running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

     Running tests/core.rs (target/debug/deps/core-358a88d276d5d768)

running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 7 filtered out; finished in 0.00s

     Running tests/parity.rs (target/debug/deps/parity-00628a4316c09dc3)

running 5 tests
test p1a_tokenizer ... SKIP: YUE2_FIXTURES is unset
ok
test p1b_protocol_prefixes ... SKIP: YUE2_FIXTURES is unset
ok
test p1c_teacher_forced ... SKIP: YUE2_FIXTURES is unset
ok
test p1d_greedy_oracle ... SKIP: YUE2_FIXTURES is unset
ok
test p1d_literal_saved_abc ... SKIP: YUE2_FIXTURES is unset
ok

test result: ok. 5 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

   Doc-tests yue2

running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
```

GPU availability immediately before parity:

```text
index, memory.used [MiB], memory.free [MiB]
0, 62432 MiB, 34858 MiB
1, 74354 MiB, 22936 MiB
2, 89069 MiB, 8220 MiB
```

All P1 fixture gates, including the literal failing P1d gate:

```text
     Running tests/parity.rs (target/debug/deps/parity-aa72986d0be60796)

running 5 tests
test p1a_tokenizer ... P1a PASS: 20/20 exact encodings and round-trips; 208/208 specials; NFC; full first-song prefix 611/611 tokens
ok
test p1b_protocol_prefixes ... P1b first-song PASS: prompt=128, saved prefix=611, ABC=481
P1b song2 PASS: prompt=200, saved prefix=1670, ABC=1468
P1b song3 PASS: prompt=202, saved prefix=1456, ABC=1252
ok
test p1c_teacher_forced ... P1c Cuda(CudaDevice(DeviceId(1))) BF16: argmax=64/64; max_abs_logit_diff=0.375000000
P1c embedding: cosine=1.000000000000; min_position_cosine=1.000000000000; max_abs=0.000000000
P1c layer.0: cosine=0.999999994917; min_position_cosine=0.999999583650; max_abs=0.500000000
P1c layer.13: cosine=0.999999144118; min_position_cosine=0.999965031256; max_abs=16.000000000
P1c final_norm: cosine=0.999973469979; min_position_cosine=0.999865967444; max_abs=0.250000000
ok
test p1d_greedy_oracle ... P1d eager greedy oracle: 64/64; original sampled abc_tokens.npy: 61/64
ok
test p1d_literal_saved_abc ... Literal P1d: Rust greedy vs original sampled abc_tokens.npy = 61/64

thread 'p1d_literal_saved_abc' (2812100) panicked at tests/parity.rs:361:5:
assertion `left == right` failed: Original artifact is sampled, see REPORT-P0.md and greedy.json
  left: 61
 right: 64
note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace
FAILED

failures:

failures:
    p1d_literal_saved_abc

test result: FAILED. 4 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out; finished in 4.01s

error: test failed, to rerun pass `--test parity`
```

Default-system-cuBLAS diagnostic, with unchanged thresholds and implementation:

```text
     Running tests/parity.rs (target/debug/deps/parity-64226562dd60d8bd)

running 1 test
P1c Cuda(CudaDevice(DeviceId(1))) BF16: argmax=63/64; max_abs_logit_diff=1.000000000
P1c embedding: cosine=1.000000000000; min_position_cosine=1.000000000000; max_abs=0.000000000
P1c layer.0: cosine=0.999996624975; min_position_cosine=0.999989197793; max_abs=32.000000000
P1c layer.13: cosine=0.999970074351; min_position_cosine=0.999929459889; max_abs=128.000000000
P1c final_norm: cosine=0.999909304424; min_position_cosine=0.996942777479; max_abs=0.875000000

thread 'p1c_teacher_forced' (2814170) panicked at tests/parity.rs:265:5:
assertion `left == right` failed: all teacher-forced argmax positions must match
  left: 63
 right: 64
note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace
test p1c_teacher_forced ... FAILED

failures:

failures:
    p1c_teacher_forced

test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 4 filtered out; finished in 1.15s

error: test failed, to rerun pass `--test parity`
```

Full CPU greedy rerun:

```text
     Running tests/parity.rs (target/debug/deps/parity-00628a4316c09dc3)

running 1 test
test p1d_greedy_oracle has been running for over 60 seconds
P1d eager greedy oracle: 64/64; original sampled abc_tokens.npy: 61/64
test p1d_greedy_oracle ... ok

test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 4 filtered out; finished in 125.37s
```

Verified the actual linked libraries with:

```bash
tools/with_reference_blas.sh ldd target/debug/deps/parity-aa72986d0be60796 > /home/ayourtch/work/yue2-rs-logs/p1-audit-linked-libraries.log 2>&1
```

Relevant pasted output:

```text
Reference cuBLAS: /home/ayourtch/yue2/.venv/lib/python3.12/site-packages/nvidia/cublas/lib
	libcublas.so.12 => /home/ayourtch/yue2/.venv/lib/python3.12/site-packages/nvidia/cublas/lib/libcublas.so.12 (0x0000735cb6200000)
	libcudart.so.13 => /usr/local/cuda/targets/x86_64-linux/lib/libcudart.so.13 (0x0000735cb5e00000)
	libcublasLt.so.12 => /home/ayourtch/yue2/.venv/lib/python3.12/site-packages/nvidia/cublas/lib/libcublasLt.so.12 (0x0000735c83200000)
```

The CUDA compiler is 13.3.73. The helper supplies `-L native=...` at link time
and the reference library directory at runtime; changing only LD_LIBRARY_PATH
does not change a binary already linked to the cuBLAS 13 SONAME. It creates only
two symlinks inside target/. No custom CUDA kernels, bespoke GEMM implementation,
algorithm chosen for a fixture shape, library copies, or weight copies remain.

## Optional operation audit

After the clean audit's build, reran the final diagnostic tool and comparison:

```bash
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv > /home/ayourtch/work/yue2-rs-logs/p1-audit-debug-gpu.log
PYTHONDONTWRITEBYTECODE=1 HF_HOME=/home/ayourtch/yue2/hf HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 /home/ayourtch/yue2/.venv/bin/python -u tools/dump_ar_debug.py > /home/ayourtch/work/yue2-rs-logs/p1-audit-debug-dump.log 2>&1
export PATH=/home/ayourtch/.cargo/bin:/usr/local/cuda/bin:$PATH CUDA_VISIBLE_DEVICES=0 CUDA_COMPUTE_CAP=120 YUE2_FIXTURES=/home/ayourtch/work/yue2-rs-fixtures
tools/with_reference_blas.sh cargo test --features cuda --lib ar_operations_oracle -- --ignored --nocapture > /home/ayourtch/work/yue2-rs-logs/p1-audit-operations.log 2>&1
```

```text
allow_bf16_reduced_precision_reduction: True
q_proj FP32 vs reference: exact 130956 / 131072 max 0.001953125
k_proj FP32 vs reference: exact 44317 / 65536 max 0.03125
v_proj FP32 vs reference: exact 42310 / 65536 max 0.0625
o_proj FP32 vs reference: exact 130919 / 131072 max 0.001953125
Saved 35 first-layer operation tensors
     Running unittests src/lib.rs (target/debug/deps/yue2-db14caa9d1033bcb)

running 1 test
input_layernorm.output: exact 131072/131072; max_abs=0
self_attn.q_proj.output: exact 131072/131072; max_abs=0
self_attn.k_proj.output: exact 65536/65536; max_abs=0
self_attn.v_proj.output: exact 65536/65536; max_abs=0
self_attn.o_proj.output: exact 131072/131072; max_abs=0
self_attn.q_norm.output: exact 131072/131072; max_abs=0
self_attn.k_norm.output: exact 65536/65536; max_abs=0
cos: exact 4060/4096; max_abs=0.00000011920929
sin: exact 3935/4096; max_abs=0.00000011920929
q_rot: exact 131072/131072; max_abs=0
k_rot: exact 65536/65536; max_abs=0
self_attn.o_proj.input: exact 130993/131072; max_abs=0.00012207031
post_attention_layernorm.output: exact 131072/131072; max_abs=0
mlp.gate_proj.output: exact 393216/393216; max_abs=0
mlp.up_proj.output: exact 393216/393216; max_abs=0
mlp.down_proj.output: exact 131072/131072; max_abs=0
mlp.down_proj.input: exact 393216/393216; max_abs=0
test model::diagnostics::ar_operations_oracle ... ok

test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 1.02s
```

This is diagnostic output, not an additional relaxed gate. Its file is separate
under the ignored `first-song/debug/` directory.

## Unsure list and design objections

1. **Literal P1d is unsatisfied.** The immutable saved plan used temperature 0.7.
   Python greedy itself differs from it at zero-based positions 16, 17 and 26.
   Rust matches the separate eager greedy dump exactly. Both comparisons remain
   executable; neither the source artifact nor the gate threshold was changed.
2. **P1c is sensitive to cuBLAS version.** The standard CUDA 13.3 build works, but
   its system cuBLAS 13.6 does not meet the fixture gate. Linking against the
   installed Python cuBLAS 12.8.4.1 does. The helper makes this reproducible, but
   numerical parity on other GPU/library combinations has not been established.
   This is an additional environment requirement for the strict parity gate.
3. CPU builds, small F32/BF16 regression tests, and the full greedy test pass.
   CPU's FP32-accumulated BF16 fallback is not claimed to pass the strict
   CUDA-reference teacher-forced logit gate; it does not reproduce cuBLAS's
   reduction choices. A preliminary CPU P1c diagnostic failed, so no CPU P1c pass
   is implied by the CPU greedy result.
4. As detailed in REPORT-P0.md, saved prefix.npy includes ABC. P1b uses
   `token_prefixes(request, tokenizer, Some(saved_abc_ids))`; the separately
   checked shorter generation prompts are the result with no ABC IDs. The
   original reference has only one native NAR chunk; the extra two-chunk fixture
   is explicitly a context override.
5. P1c's first 64 input positions are all inside the prompt. The 64-step greedy
   gate provides additional coverage of the full prompt, actual ABC tokens, and
   append-only KV caching. Padded batches, arbitrary position_ids, NAR paths,
   flash-attn, Metal, and full-context numerical parity are outside this phase's
   verified surface.
6. No objections to the fixed model architecture; no thresholds were lowered.
   The findings above are retained for owner/reviewer decisions before later
   phases.
