# Phase 4 report

**PASS: P4a and P4b.** All six decoder blocks satisfy max absolute error <= 1e-3
on CUDA and CPU. The CUDA decoder's first-four-second audio SNR is
**120.086907 dB** against the original FP32 oracle (required >= 40 dB).
The full **2,849,216-frame, 48 kHz stereo** waveform is written as WAV and FLAC.
The finalized P3 gate also passes both tiers; see REPORT-P3-final.md.

Read TASK.md fully and the released `modeling_vae.py`, fixture dumper, and audio
handling in `pipeline.py` before porting. Phases 0–3 implementation and original
fixtures were retained. The requested P3 test update preceded Phase 4 work.
Phase 5 pipeline/CLI integration was not implemented. No push.

## What changed

- Added `vae`: `YuE2VAEConfig`, `DecoderConfig`, `YuE2VAE`, `OobleckDecoder`,
  `DecoderBlock`, `ResidualUnit` and `SnakeBeta`, following the Python module
  hierarchy. The residual dilations are 1/3/9. Snake parameters use the stored
  logarithmic alpha/beta, exponentiation, and the reference 1e-9 denominator.
  The released checkpoint has no final tanh; ELU and configured tanh are also
  represented, while the release's unsupported options are rejected.
- `from_pretrained(directory, device)` mmap-loads the original safetensors
  through candle VarBuilder, selecting decoder tensors and checking their
  native FP32 dtype. No encoder allocation, copied/converted checkpoint file,
  new dependency or download was introduced. Tests use `YUE2_VAE_DIR` or the
  existing local `HF_HOME` snapshot resolver.
- Weight normalization is folded once into plain FP32 weights. The normalization
  axis is **dimension 0 for both convolution kinds**: `[out,in,k]` for Conv1d,
  `[in,out,k]` for ConvTranspose1d. The stored g/v are not retained in the model.
- Transposed convolutions use candle's zero-padding GEMM/col2im path and then
  crop `ceil(stride/2)` from both ends. This preserves the odd stride=5 stage's
  one-sample loss, propagated to the final `1920*T - 64` length. CPU uses a
  `[batch,time,channel]` allocation with public `[B,C,T]` axes to work around
  candle 0.11's batch-merging MatMul assumption for a shared kernel. A scalar
  scatter test with unequal input/output channels exposed and verifies this fix.
- `decode` returns unclipped FP32 `[B,2,samples]` on the decoder device.
  `decode_tiled` ports the dependency-interval halo calculation, exact core
  cropping, natural final length, CPU output and tile progress callback. It
  rejects empty/nonfinite/malformed inputs and insufficient halos. There is no
  crossfade, boundary smoothing or extra audio padding.
- P4a observes each block during one continuous 64-latent-frame forward pass;
  it never replaces an intermediate with its oracle. P4b decodes all 1,484
  reference frames with core=1024 / halo=16, then compares the first 192,000
  stereo samples against `vae.safetensors`. The required halo computes to 12;
  the gate keeps the reference's configured 16.
- `tools/check_phase4_audio.py` consumes **Rust-produced audio only**, clamps
  as Python's pipeline does, writes FLOAT WAV / PCM_24 FLAC via soundfile,
  and verifies format, length, roundtrip error and SNR. Native Rust storage and
  pipeline/CLI routing remain Phase 5 work. No Python VAE decode is used to
  create the output artifacts.
- Three ordinary CPU tests cover scalar transposed-convolution/weight-norm
  parity (strides 2/3/5/6, batch 2), batch independence, full/tiled boundaries,
  short final tiles, progress, invalid inputs/configuration and FP32 enforcement.
  Added ignored real-checkpoint P4 gates, an audit script, and the upstream MIT
  notices for the ported Oobleck/SnakeBeta source.

## Audited gates

P4a maximum absolute errors, all outputs natively FP32:

| Block | Shape | CUDA max absolute error | CPU max absolute error | Gate <= 1e-3 |
| --- | --- | ---: | ---: | --- |
| 0 | `[1, 1024, 384]` | 0.000028729439 | 0.000041842461 | PASS |
| 1 | `[1, 512, 1919]` | 0.000240266323 | 0.000342428684 | PASS |
| 2 | `[1, 256, 7676]` | 0.000133156776 | 0.000189840794 | PASS |
| 3 | `[1, 128, 30704]` | 0.000032126904 | 0.000046029687 | PASS |
| 4 | `[1, 64, 61408]` | 0.000013113022 | 0.000028252602 | PASS |
| 5 | `[1, 64, 122816]` | 0.000002920628 | 0.000007927418 | PASS |

| P4b check | Clean-audit result |
| --- | --- |
| Raw decoded audio, first 4 seconds | **PASS: SNR 120.086907 dB**, max absolute error **0.000000260770** |
| Full waveform | `[1,2,2849216]`, **59.358666667 seconds**, all finite |
| WAV | FLOAT, 48 kHz stereo, **2,849,216 frames**; first-4s SNR **120.086907 dB**; roundtrip max **0** |
| FLAC | PCM_24, 48 kHz stereo, **2,849,216 frames**; first-4s SNR **115.228304 dB**; roundtrip max **0.000000059605** |
| Peak / clipping | Peak **0.503509939**, **0** samples clipped |

CUDA's 64-frame slice audio (additional observation): max absolute error
**0.000000871718**, SNR **118.876906 dB**. CPU slice audio: max absolute error
**0.000001902692**, SNR **111.226924 dB**. The CUDA full-song decode took
**0.575814 seconds**, excluding loading, including CPU crop transfers. These are
shared-GPU observations, not a Phase 5 speed gate.

Output artifacts under `~/work/yue2-rs-fixtures/first-song/p4/`:
`audio.safetensors` (unclipped Rust output), `audio.wav` (**22,793,816 bytes**),
and `audio.flac` (**8,341,876 bytes**). Total recursive fixtures/artifacts are
**1,478,805,166 bytes**, below the 1.5 GB budget. No artifacts are committed.

## Exact commands, pasted output and self-audit

Initial P4 CUDA gates and WAV/FLAC checks passed. The added CPU scalar test first
exposed incorrect results after batch 0 in candle's transposed-convolution
MatMul path; the local layout fix made it and the full/tiled/batch tests pass.
The real-checkpoint CPU P4a gate then passed too.

After all implementation changes, `tools/audit_phase4.sh` ran **cargo clean**,
a fresh **cargo test**, CPU/CUDA builds, formatting and CPU/CUDA clippy, unset
fixture checks, CPU P4a, both finalized P3 tiers, CUDA P4a/P4b and audio export
verification. **Every number quoted above is from this clean audit**, not the
initial run. Initial and audited P3/P4 errors and SNR values are identical.
The audit exits **0**, preserving any failure rather than filtering it out.

Exact top-level command:

```bash
export PATH=/home/ayourtch/.cargo/bin:/usr/local/cuda/bin:$PATH CUDA_VISIBLE_DEVICES=0 CUDA_COMPUTE_CAP=120 YUE2_FIXTURES=/home/ayourtch/work/yue2-rs-fixtures
tools/audit_phase4.sh > /home/ayourtch/work/yue2-rs-logs/p4-audit-commands.log 2>&1
```

The script exports `YUE2_TEST_DEVICE=cuda` and `PYTHONDONTWRITEBYTECODE=1`.
Exact command/exit transcript:

```text
clean command: cargo clean
clean exit=0 log=/home/ayourtch/work/yue2-rs-logs/p4-audit-clean.log
test command: cargo test
test exit=0 log=/home/ayourtch/work/yue2-rs-logs/p4-audit-test.log
build-cpu command: cargo build
build-cpu exit=0 log=/home/ayourtch/work/yue2-rs-logs/p4-audit-build-cpu.log
build-cuda command: cargo build --features cuda
build-cuda exit=0 log=/home/ayourtch/work/yue2-rs-logs/p4-audit-build-cuda.log
fmt command: cargo fmt --check
fmt exit=0 log=/home/ayourtch/work/yue2-rs-logs/p4-audit-fmt.log
clippy command: cargo clippy --all-targets -- -D warnings
clippy exit=0 log=/home/ayourtch/work/yue2-rs-logs/p4-audit-clippy.log
clippy-cuda command: cargo clippy --features cuda --all-targets -- -D warnings
clippy-cuda exit=0 log=/home/ayourtch/work/yue2-rs-logs/p4-audit-clippy-cuda.log
unset-fixtures command: env -u YUE2_FIXTURES cargo test --test phase3 --test phase4 -- --ignored --nocapture --test-threads=1
unset-fixtures exit=0 log=/home/ayourtch/work/yue2-rs-logs/p4-audit-unset-fixtures.log
cpu-blocks command: env YUE2_TEST_DEVICE=cpu cargo test --test phase4 p4a_decoder_blocks -- --ignored --nocapture
cpu-blocks exit=0 log=/home/ayourtch/work/yue2-rs-logs/p4-audit-cpu-blocks.log
gpu command: nvidia-smi --query-gpu=index\,memory.used\,memory.free --format=csv
gpu exit=0 log=/home/ayourtch/work/yue2-rs-logs/p4-audit-gpu.log
p3-final command: tools/with_reference_blas.sh cargo test --features cuda --test phase3 -- --ignored --nocapture --test-threads=1
p3-final exit=0 log=/home/ayourtch/work/yue2-rs-logs/p4-audit-p3-final.log
gates command: tools/with_reference_blas.sh cargo test --features cuda --test phase4 -- --ignored --nocapture --test-threads=1
gates exit=0 log=/home/ayourtch/work/yue2-rs-logs/p4-audit-gates.log
audio command: /home/ayourtch/yue2/.venv/bin/python tools/check_phase4_audio.py --root /home/ayourtch/work/yue2-rs-fixtures/first-song
audio exit=0 log=/home/ayourtch/work/yue2-rs-logs/p4-audit-audio.log
```

Clean/build/clippy completion excerpts:

```text
clean: Removed 9443 files, 6.4GiB total
test: Finished `test` profile [optimized + debuginfo] target(s) in 27.41s
build-cpu: Finished `dev` profile [unoptimized + debuginfo] target(s) in 7.52s
build-cuda: Finished `dev` profile [unoptimized + debuginfo] target(s) in 41.95s
clippy: Finished `dev` profile [unoptimized + debuginfo] target(s) in 4.78s
clippy-cuda: Finished `dev` profile [unoptimized + debuginfo] target(s) in 42.86s
```

`cargo fmt --check`: exit 0, no output. Clean CPU `cargo test`: **18 passed**,
**16 ignored**, no failures. All five P3/P4 ignored tests return cleanly when
`YUE2_FIXTURES` is unset; no checkpoint is loaded in that check.

CUDA P4 gate output (compiler progress omitted):

```text
test p4a_decoder_blocks ... P4a block.0: shape=[1, 1024, 384] dtype=F32 max_abs=0.000028729439
P4a block.1: shape=[1, 512, 1919] dtype=F32 max_abs=0.000240266323
P4a block.2: shape=[1, 256, 7676] dtype=F32 max_abs=0.000133156776
P4a block.3: shape=[1, 128, 30704] dtype=F32 max_abs=0.000032126904
P4a block.4: shape=[1, 64, 61408] dtype=F32 max_abs=0.000013113022
P4a block.5: shape=[1, 64, 122816] dtype=F32 max_abs=0.000002920628
P4a slice audio (reported): shape=[1, 2, 122816] max_abs=0.000000871718 snr_db=118.876906 seconds=0.188737
ok
test p4b_reference_audio ... P4b config: frames=1484 core=1024 halo=16 required_halo=12 FP32, unclipped
P4b tile 1/2
P4b tile 2/2
P4b first 4 seconds: samples=192000 channels=2 max_abs=0.000000260770 snr_db=120.086907
P4b full audio: shape=[1, 2, 2849216] seconds=0.575814
ok

test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 1.20s
```

WAV/FLAC artifact gate output:

```text
P4b Rust raw first-4s SNR=120.086907 dB; peak=0.503509939; clipped_values=0
P4b audio.wav: 2849216 frames, 48000 Hz, 2 channels, FLOAT; first-4s SNR=120.086907 dB; roundtrip_max=0.000000000000; bytes=22793816
P4b audio.flac: 2849216 frames, 48000 Hz, 2 channels, PCM_24; first-4s SNR=115.228304 dB; roundtrip_max=0.000000059605; bytes=8341876
Total recursive fixture/artifact bytes: 1,478,805,166 <= 1,500,000,000
```

CPU P4a output:

```text
P4a block.0: shape=[1, 1024, 384] dtype=F32 max_abs=0.000041842461
P4a block.1: shape=[1, 512, 1919] dtype=F32 max_abs=0.000342428684
P4a block.2: shape=[1, 256, 7676] dtype=F32 max_abs=0.000189840794
P4a block.3: shape=[1, 128, 30704] dtype=F32 max_abs=0.000046029687
P4a block.4: shape=[1, 64, 61408] dtype=F32 max_abs=0.000028252602
P4a block.5: shape=[1, 64, 122816] dtype=F32 max_abs=0.000007927418
P4a slice audio (reported): shape=[1, 2, 122816] max_abs=0.000001902692 snr_db=111.226924 seconds=2.952944
test p4a_decoder_blocks ... ok

test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 1 filtered out; finished in 3.23s
```

GPU availability before the audited CUDA gates:

```text
index, memory.used [MiB], memory.free [MiB]
0, 62432 MiB, 34858 MiB
1, 74354 MiB, 22936 MiB
2, 89069 MiB, 8220 MiB
```

Memory was sampled every 500 ms with:

```bash
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_gpu_memory --format=csv -lms 500
```

Observed task-process GPU peaks: finalized P3, PID 2959692, **15,948 MiB**;
P4, PID 2960133, **9,490 MiB**. Both ran serially on GPU 0, below the approximately
30 GB limit. Existing server processes were not modified. Full samples are in
`~/work/yue2-rs-logs/p4-audit-memory.log`.

Read-only SHA-256 checks matched the original P0 manifest for both checkpoint
files and the immutable NAR/VAE oracles. The VAE checkpoint remains
`807ce9d5149fa27c5ad3e6582058469852e908f6c5acc8c8aa338e7ab7751346`;
`vae.safetensors` remains
`97d1f24a57d686e6fea2ca2b52e370544979fc715878cd14908f2501e37ca57d`.
All four hashes are recorded in `~/work/yue2-rs-logs/p4-audit-identities.log`.
No original fixture was regenerated or overwritten.

## Unsure list and design objections

1. P4 parity is certified for the released standard checkpoint, all six blocks
   of the recorded 64-frame slice, and the first four seconds of reference audio.
   The remaining full-song output is finite and has the correct length, but
   there is no full-song Python waveform oracle in the existing fixture. The
   Rust NAR-to-VAE end-to-end waveform is not covered by P4's reference-latent gate.
2. CPU real-checkpoint block parity and small full/tiled batch tests pass.
   A full-song CPU audio gate, Metal, alternate checkpoints, ELU/final-tanh
   checkpoint parity and alternate CUDA/cuBLAS versions were not certified.
3. The batch-layout workaround is required by the inspected candle 0.11 CPU
   MatMul optimization. No dependency source was modified. The scalar regression
   should remain when upgrading candle; the workaround can be reconsidered then.
4. Folded FP32 weight normalization and different convolution/reduction kernels
   produce small floating-point differences. All original P4 tolerances pass
   unchanged. The P3 BF16 absolute differences remain advisory under the owner's
   revised wording and are recorded separately, not hidden by P4's audio result.
5. The fixed library-first structure has no further design objection. WAV/FLAC
   serialization is currently an audit tool using the existing Python environment;
   native Rust storage is deliberately left for Phase 5. GPU peaks are sampled
   observations rather than allocator-instrumented maxima. No Phase 5 work or push.
