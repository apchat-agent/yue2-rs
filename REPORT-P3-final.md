# Phase 3 final gate

**PASS under the owner's two-tier wording in TASK.md (2026-09-14).**
`tests/phase3.rs` now requires both FP32 chunks to have max absolute error
<= 1e-3 and cosine >= 0.999999, and both production CUDA BF16 chunks to have
cosine >= 0.999. BF16 max absolute error, native dtypes and the first divergent
operation are printed without numerical assertions.

Tier (a) reuses the existing P3b Python FP32 control and reruns Rust's same
FP32 checkpoint path. Both tiers retain the original dumped noise, two 742-frame
chunks (the P0 context=2098 case), and 32 midpoint steps. No existing backbone/NAR
code or oracle was changed. Rust FP32 results are saved separately as
`p3/latents-fp32.safetensors`; BF16 keeps `p3/latents.safetensors`.

## Commands, output and self-audit

The gate passed before Phase 4 began. After completing Phase 4, reran both
P3 tiers following `cargo clean` and a fresh `cargo test` (18 passed, 16 ignored).
All trajectory numbers below are from that final clean audit; they match the
initial run. Exact audit command (including P4), and the P3 command it executes:

```bash
export PATH=/home/ayourtch/.cargo/bin:/usr/local/cuda/bin:$PATH CUDA_VISIBLE_DEVICES=0 CUDA_COMPUTE_CAP=120 YUE2_FIXTURES=/home/ayourtch/work/yue2-rs-fixtures
tools/audit_phase4.sh > /home/ayourtch/work/yue2-rs-logs/p4-audit-commands.log 2>&1
# Inside the audit, with YUE2_TEST_DEVICE=cuda:
tools/with_reference_blas.sh cargo test --features cuda --test phase3 -- --ignored --nocapture --test-threads=1
```

```text
P3(a) FP32 two_chunk.0: frames=742 max_abs=0.000020578504 cosine=1.000000000000 seconds=9.897435
P3(a) FP32 two_chunk.1: frames=742 max_abs=0.000025272369 cosine=1.000000000000 seconds=9.938940
P3(b) BF16 two_chunk.0: frames=742 max_abs=0.839233398438 cosine=0.999378217075 seconds=7.058173
P3(b) BF16 two_chunk.1: frames=742 max_abs=0.269531250000 cosine=0.999945369729 seconds=7.090679
NAR CUDA RNG PASS: 576/576 values repeat after AR reseed; context-independent cuts; different seed differs; AR stream unchanged
test result: ok. 3 passed; 0 failed; 0 ignored
```

Full gate log: `~/work/yue2-rs-logs/p4-audit-p3-final.log`. Cosines shown as
1.000000000000 are rounded to twelve decimals, not bit identity. Observed task
GPU peak was 15,948 MiB on GPU 0 (34,858 MiB free beforehand).

## Reported BF16 diagnostics

The test explicitly labels its retained `p3b-comparison.json` stage evidence;
the completed P3b investigation was not rerun. Python/Rust native embeddings,
projections, norm outputs, transformer outputs, velocities and midpoint/update
states are BF16. Noise, RoPE frequencies/angles, timestep frequencies and final
API latents are FP32; raw solver times are FP64. Internal RMS reductions and
attention accumulations use FP32 as documented in REPORT-P3b.md.

The first operation above the P3b localization threshold of 1e-3 is AR prefill
layer 0 Q after RoPE: max absolute error **0.00390625**. The earlier nonzero
FP32 precursor is `rope.inv_freq`: **1.862645149230957e-9**. These are reported
observations, not extra gate criteria.

## Unsure list and design objections

- Certification remains limited to the original two-chunk fixture and reference
  cuBLAS environment. FP32 uses the existing Torch math-SDPA oracle; BF16 uses
  the original production oracle. Full checkpoint CPU parity was not rerun.
- The large BF16 maximum errors remain visible: **0.839233398438** and
  **0.269531250000**. The owner-authorized gate treats them as advisory.
- No additional design objection; earlier reports remain historical records of
  the earlier gate, and Phase 5 remains outside this task.
