# Phase 3 report

Phase 3 NAR implementation and the requested P2 updates were executed.
**P3 is not satisfied:** both chunks pass cosine similarity, but both exceed the
maximum absolute latent-error limit of 0.01. The strict failing gate remains
executable. No fixture noise, solver steps, context cuts or thresholds were
changed to obtain a pass. Phases 0–2 remain committed work; no VAE, final CLI,
Phase 5 optimization or push was performed.

## What changed

- `nar`: `Chunk`, `song_chunks`, `CachedNAR`, midpoint `solve`, and serial
  `synthesize`, returning CPU FP32 `[frames,64]` tensors. Full-song noise is drawn
  before the existing historical context cuts. Prefix/codec validation, context
  errors, boundary padding, finite checks, cancellation between velocity calls,
  and progress across chunks follow the Python structure.
- NAR explicitly owns its noise seed. CUDA noise comes from a fresh candle
  device object with a private cuRAND generator on the same physical GPU, then
  transfers to CPU before chunk slicing. It neither reseeds nor advances the
  model device RNG used by Phase 2. CPU uses seeded `StdRng` plus `StandardNormal`
  from candle's existing rand/rand_distr dependency versions, since candle 0.11
  cannot seed its CPU RNG. Noise parity with torch is not claimed; P3 injects the
  immutable dumped torch noise directly.
- `model`: opt-in `load_with_nar` / `from_pretrained_with_nar` load the acoustic
  checkpoint tensors while preserving AR-only loading. The new model submodule
  owns the separate NAR attention, norms and MLPs, `TimestepEmbedder`, biased
  latent projections, and checkpoint `AudioPositionEmbedding` buffer.
- Each chunk prefills AR once and stores its projected K/V at all 28 layers.
  The velocity path attends to visible AR keys and all NAR keys, with absolute
  RoPE positions and local audio positions including START/END. Restricted
  `nar_cond_end` copies only the visible cache. Cache lifetime is scoped to the
  chunk; Rust drops it on success, cancellation or error.
- Extended existing plain tensor SDPA with noncausal attention and configurable
  query tiles; AR calls retain their causal offset and 128-row tile. CUDA graphs,
  external flash kernels and AR offloading were not introduced. The BF16 solver
  preserves multiply/subtract rounding boundaries and returns an FP32 cast.
- Fixed biased projection arithmetic: torch adds bias before BF16 rounding;
  candle's ordinary linear rounds the matmul before bias. The four acoustic
  auxiliary linears now accumulate/add bias in FP32 and cast once. Existing AR
  linears have no bias and keep their established CUDA path. CPU BF16 linears
  now also retain any bias in their FP32 fallback.
- `pipeline::synthesize` validates the semantic result's retained exact plan
  prefix and forwards its request seed to NAR. The VAE and final CLI are deferred.
- Four ordinary CPU tests cover seed/reset/chunk independence, analytic constant
  velocity integration in F32/BF16, progress, cancellation, invalid inputs and
  the bias-rounding boundary. Two ignored P3 tests cover real-checkpoint parity
  and CUDA RNG isolation. An optional first-velocity diagnostic records operation
  errors without substituting them for the final-latent gate.
- P2a now reports section tags, key/meter counts, length ratio and truncation;
  only parsing fails its gate. `REPORT-P2-addendum.md` records the once-measured
  Python eager baseline and both eager/graph comparisons against the committed
  Rust P2 measurements. Original P2 generation artifacts and reports are intact.

## Audited gates

P3 uses seed **831001**, **32 midpoint steps**, BF16 backbone/state, and the
existing `two_chunk` case in `nar.json`: context **2098**, prefix **611** tokens,
two **742-frame** chunks, **1354 AR** positions and **744 NAR** positions each.
The original song has only one chunk at the native 24576 context; this explicit
two-chunk fixture override was documented and committed in Phase 0. The gate
does not represent these as two native chunks.

| Gate / check | Clean-audit result |
| --- | --- |
| P3 chunk 0, frames 0–741 | **FAIL** max absolute error: **0.839233398438**; cosine **0.999378217075** passes |
| P3 chunk 1, frames 742–1483 | **FAIL** max absolute error: **0.269531250000**; cosine **0.999945369729** passes |
| NAR CUDA noise ownership | PASS: **576/576** values repeat after AR reseeding and different context cuts; another seed differs; AR RNG stream unchanged |
| P2a updated advisory check | PASS: saved ABC parses; tags intro/verse/chorus/outro versus intro/verse/chorus; K=1, M=1; 617/481 tokens, ratio **1.282744283** |
| P2c Python eager, single measurement | Plan **109.329852100 tok/s**; semantic **117.951828259 tok/s**; full details in the addendum |

P3 solve-only times were **5.592684 s / 5.625642 s**, excluding model load and AR
prefill, including final CPU transfer and finite validation. Initial and clean
runs have identical per-chunk maximum errors and cosine values.

Selected first-velocity diagnostics from the clean audit:

| Capture, chunk 0 / step 0 | Maximum absolute difference | Cosine |
| --- | ---: | ---: |
| AR cache, layer 0 K | 0.001953125000 | 0.999999999996 |
| AR cache, layer 0 V | 0 | 1.000000000000 |
| AR cache, layer 13 K / V | 0.148437500000 / 0.140625000000 | 0.999961574412 / 0.999898720234 |
| AR cache, layer 27 K / V | 0.191406250000 / 0.154785156250 | 0.999958513120 / 0.999943979146 |
| Loaded audio position embedding | 0 | 1.000000000000 |
| vae2llm projection | 0.003906250000 | 0.999999999938 |
| Timestep embedding | 0.001953125000 | 0.999999915097 |
| Hidden state after latent/time/position injection | 0.015625000000 | 0.999999997501 |
| Final norm | 0.250000000000 | 0.999950325741 |
| First velocity | 0.062500000000 | 0.999955537267 |

These are diagnostic observations, not substitute passing gates.

The P3 test evaluates both chunks before failing, checks finite FP32 values, and
writes the actual Rust output to
`~/work/yue2-rs-fixtures/first-song/p3/latents.safetensors` for review. It never
substitutes reference states inside `solve`; optional velocity diagnostics use
reference states separately from the complete generated trajectory.

## Exact commands, output, and self-audit

The final implementation first ran `cargo test` and the complete P3 CUDA gate
(`p3-final-initial-test.log`, `p3-final-initial-gates.log`). The advisory P2a
checker also ran on the unchanged saved score (`p3-initial-p2a.log`). Then
`tools/audit_phase3.sh` performed **cargo clean**, a fresh **cargo test**, CPU/CUDA
builds, formatting, CPU/CUDA clippy and fixture-skip checks, and reran P3 and the
advisory checker. All P3 numbers quoted here come from that clean self-audit.
The script exits **1** because P3 exits **101**; no failure is filtered out.

Exact top-level audit command from the repository root:

```bash
env PATH=/home/ayourtch/.cargo/bin:/usr/local/cuda/bin:$PATH CUDA_VISIBLE_DEVICES=0 CUDA_COMPUTE_CAP=120 YUE2_FIXTURES=/home/ayourtch/work/yue2-rs-fixtures tools/audit_phase3.sh > /home/ayourtch/work/yue2-rs-logs/p3-audit-commands.log 2>&1
```

The script exports `YUE2_TEST_DEVICE=cuda` and `YUE2_NAR_DIAGNOSTIC=1`.
Exact command/exit transcript, followed by gate output (compiler progress omitted):

```text
clean command: cargo clean
clean exit=0 log=/home/ayourtch/work/yue2-rs-logs/p3-audit-clean.log
test command: cargo test
test exit=0 log=/home/ayourtch/work/yue2-rs-logs/p3-audit-test.log
build-cpu command: cargo build
build-cpu exit=0 log=/home/ayourtch/work/yue2-rs-logs/p3-audit-build-cpu.log
build-cuda command: cargo build --features cuda
build-cuda exit=0 log=/home/ayourtch/work/yue2-rs-logs/p3-audit-build-cuda.log
fmt command: cargo fmt --check
fmt exit=0 log=/home/ayourtch/work/yue2-rs-logs/p3-audit-fmt.log
clippy command: cargo clippy --all-targets -- -D warnings
clippy exit=0 log=/home/ayourtch/work/yue2-rs-logs/p3-audit-clippy.log
clippy-cuda command: cargo clippy --features cuda --all-targets -- -D warnings
clippy-cuda exit=0 log=/home/ayourtch/work/yue2-rs-logs/p3-audit-clippy-cuda.log
unset-fixtures command: env -u YUE2_FIXTURES cargo test -- --ignored --nocapture --test-threads=1
unset-fixtures exit=0 log=/home/ayourtch/work/yue2-rs-logs/p3-audit-unset-fixtures.log
p2a-advisory command: env PYTHONDONTWRITEBYTECODE=1 /home/ayourtch/yue2/.venv/bin/python tools/check_phase2.py /home/ayourtch/work/yue2-rs-fixtures/first-song/p2
p2a-advisory exit=0 log=/home/ayourtch/work/yue2-rs-logs/p3-audit-p2a-advisory.log
gpu command: nvidia-smi --query-gpu=index\,memory.used\,memory.free --format=csv
gpu exit=0 log=/home/ayourtch/work/yue2-rs-logs/p3-audit-gpu.log
gates command: tools/with_reference_blas.sh cargo test --features cuda --test phase3 -- --ignored --nocapture --test-threads=1
gates exit=101 log=/home/ayourtch/work/yue2-rs-logs/p3-audit-gates.log
operations command: tools/with_reference_blas.sh cargo test --features cuda --lib nar_operations_oracle -- --ignored --nocapture
operations exit=0 log=/home/ayourtch/work/yue2-rs-logs/p3-audit-operations.log
```

Clean, CPU/CUDA build and CPU/CUDA clippy completion output, respectively:

```text
     Removed 11652 files, 7.5GiB total
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 7.46s
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 39.30s
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 4.47s
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 37.97s
```

`cargo fmt --check`: exit 0, no output. Clean CPU `cargo test`: **15 passed**,
**12 ignored**, no failures. This includes the seven existing core and four
existing sampling tests. New unit/integration output:

```text
     Running unittests src/lib.rs (target/debug/deps/yue2-3717c6ee475f2f29)

running 5 tests
test model::diagnostics::ar_operations_oracle ... ignored, requires YUE2_FIXTURES, CUDA, and tools/dump_ar_debug.py
test model::nar::tests::nar_operations_oracle ... ignored, requires YUE2_FIXTURES, CUDA and tools/dump_nar_debug.py
test model::nar::tests::acoustic_bias_rounds_once_after_accumulation ... ok
test model::nar::tests::midpoint_cpu_f32_bf16_and_cancellation ... ok
test model::nar::tests::synthesis_serial_chunks_progress_and_seed_reset ... ok

test result: ok. 3 passed; 0 failed; 2 ignored; 0 measured; 0 filtered out; finished in 0.09s

     Running tests/nar.rs (target/debug/deps/nar-1ee4997bfd447c5b)

running 1 test
test seeded_noise_precedes_chunk_cuts_and_validates_inputs ... ok

test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
```

Unset-fixture invocation: all 12 ignored tests return cleanly without loading
weights. The new NAR diagnostic also prints `SKIP: YUE2_FIXTURES is unset`. P3
integration output:

```text
     Running tests/phase3.rs (target/debug/deps/phase3-e45a832cb5bd6b7d)

running 2 tests
test nar_cuda_noise_owns_seed ... SKIP: YUE2_FIXTURES is unset
ok
test p3_dumped_noise_latents ... SKIP: YUE2_FIXTURES is unset
ok

test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
```

Advisory P2a output from the unchanged Phase 2 artifacts:

```text
P2a reference ABC parse PASS: Vocal=22 measures/56 notes, Ins=22 measures/24 notes
P2a Rust ABC parse PASS: Vocal=24 measures/61 notes, Ins=24 measures/34 notes
P2a sections: Rust=['% intro', '% verse', '% chorus', '% outro'], reference=['% intro', '% verse', '% chorus']
P2a key/meter line counts: Rust={'K': 1, 'M': 1}, reference={'K': 1, 'M': 1}
P2a advisory length: 617/481, ratio=1.282744283, target within 30%=True, truncated=False
P2a PASS: ABC parses; section tags and key/meter counts are advisory
```

GPU availability immediately before the P3 gates:

```text
index, memory.used [MiB], memory.free [MiB]
0, 62432 MiB, 34858 MiB
1, 74354 MiB, 22936 MiB
2, 89069 MiB, 8220 MiB
```

The script samples process memory every 500 ms using:

```bash
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_gpu_memory --format=csv -lms 500
```

The P3 gate process (PID 2906826) peaked at **7918 MiB observed** on GPU 0;
its operation-diagnostic process (PID 2907118) peaked at **8142 MiB observed**.
Existing server PIDs 2564265 and 2689298 were not modified. Neither task process
approached the approximately 30 GB limit. Full samples: `p3-audit-memory.log`.

All P3 CUDA gate output:

```text
     Running tests/phase3.rs (target/debug/deps/phase3-3201476339a2abbf)

running 2 tests
test nar_cuda_noise_owns_seed ... NAR CUDA RNG PASS: 576/576 values repeat after AR reseed; context-independent cuts; different seed differs; AR stream unchanged
ok
test p3_dumped_noise_latents ... P3 velocity two_chunk.0.step.00: max_abs=0.062500000000 cosine=0.999955537267
P3 velocity two_chunk.0.step.01: max_abs=0.093750000000 cosine=0.999952744767
P3 velocity two_chunk.0.step.16: max_abs=0.078125000000 cosine=0.999920463960
P3 velocity two_chunk.0.step.31: max_abs=0.156250000000 cosine=0.999944826663
P3 two_chunk.0 step 8/32
P3 two_chunk.0 step 16/32
P3 two_chunk.0 step 24/32
P3 two_chunk.0 step 32/32
P3 two_chunk.0: frames=742 max_abs=0.839233398438 cosine=0.999378217075 seconds=5.592684
P3 velocity two_chunk.1.step.00: max_abs=0.136718750000 cosine=0.999957140503
P3 velocity two_chunk.1.step.01: max_abs=0.070312500000 cosine=0.999956381175
P3 velocity two_chunk.1.step.16: max_abs=0.062500000000 cosine=0.999941573005
P3 velocity two_chunk.1.step.31: max_abs=0.093750000000 cosine=0.999955341993
P3 two_chunk.1 step 8/32
P3 two_chunk.1 step 16/32
P3 two_chunk.1 step 24/32
P3 two_chunk.1 step 32/32
P3 two_chunk.1: frames=742 max_abs=0.269531250000 cosine=0.999945369729 seconds=5.625642
Error: P3 requires max_abs <= 0.01 and cosine >= 0.999 for BOTH chunks
FAILED

failures:

failures:
    p3_dumped_noise_latents

test result: FAILED. 1 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out; finished in 13.77s

error: test failed, to rerun pass `--test phase3`
```

The Python eager measurement is intentionally excluded from repeat audits,
honoring the request to measure it once. Its exact command/output and comparison
table are in `REPORT-P2-addendum.md`. Phase 0/1 gates and full Phase 2 generation
were not repeated.

Optional first-velocity diagnostic fixture preparation (before the clean audit):

```bash
env PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 /home/ayourtch/yue2/.venv/bin/python -u tools/dump_nar_debug.py > /home/ayourtch/work/yue2-rs-logs/p3-dump-debug.log 2>&1
```

```text
nar-debug.safetensors: 145,491,338 bytes; velocity exact against P0
```

This adds a separate diagnostic file without replacing the original NAR oracle.
Total first-song fixtures/artifacts including this supplement and generated
outputs are **415,094,985 bytes**, below 1.5 GB. The clean audit's operation
comparison uses `tools/with_reference_blas.sh`; its full output is retained in
`~/work/yue2-rs-logs/p3-audit-operations.log`.

## Unsure list and design objections

1. **P3 fails maximum absolute error on both chunks.** The final cosine scores
   exceed 0.999, but that is only one of the two requirements. The implementation
   is not certified for latent parity, and no audio-quality claim follows from
   these measurements. The failing gate and Rust latents remain available for
   review; no tolerance or reference artifact was relaxed.
2. Remaining numerical differences appear before the ODE integration, in AR
   caches and the first acoustic velocity. The trace confirms exact audio
   position embeddings and identifies bias rounding as one corrected cause.
   It does not isolate every remaining difference among BF16 GEMM reduction,
   RMSNorm/RoPE arithmetic and fused torch SDPA versus plain candle attention.
   A diagnostic trial of unbatched cuBLAS GEMM did not change the first NAR value
   projection discrepancy; that experimental operation was not retained.
3. The output being FP32 does not make the reference ODE arithmetic FP32:
   Python runs its state and velocity updates in BF16 and casts at return. At
   magnitudes above 2, one BF16 step is already larger than 0.01. Thus this gate
   requires much closer trajectory agreement than the inherited AR cosine
   requirement alone provides. This is an objection to interpreting the earlier
   AR pass as evidence that P3 should pass, not a waiver of P3's threshold.
4. CPU builds and analytic F32/BF16 tests pass; full 3B CPU latent parity was not
   run. CUDA uses candle's private seeded RNG; CPU/Metal use the documented host
   fallback. Cross-backend or torch/candle noise identity is not promised.
   Metal, flash-attn, offloaded AR weights, native full-song latent parity, and
   non-default timestep-shift checkpoint parity were not certified in this phase.
5. All CUDA gates use the required reference cuBLAS 12.8.4.1 helper. System
   cuBLAS parity was not retried. GPU memory values are sampled process usage,
   not allocator-instrumented maxima. Timings are observations on shared GPU 0,
   not a speed claim or an end-to-end throughput gate.
6. The fixed module architecture has no additional objections. Optional NAR
   loading avoids increasing the established AR-only memory footprint. The
   original P1 caveats remain in REPORT-P1.md; the P2a/P2c policy and baseline
   updates are addressed only by the requested addendum.
