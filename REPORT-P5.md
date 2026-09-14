# Phase 5 report

**P5 PASS; the inherited literal P1d failure remains 61/64.** All other requested
P1a–P4b comparisons pass with their original tolerances under reference cuBLAS.
The clean audit returns 1 solely because it deliberately retains that P1d test.

Phase 5 implements the complete pipeline and `yue2 generate`, `yue2 plan`, and
`yue2 render --abc-file`. Both reviewer housekeeping requests are included.
TASK.md was read fully; the Python CLI, storage and pipeline source were read
before implementation. Phases 0–4 were retained, and no oracle was regenerated.
No inference arithmetic, attention, sampling or solver optimization was made.

## What changed

- Added `YuE2Pipeline` with lazy checkpoint loading and public `plan`,
  `generate_semantic`, `synthesize`, `decode` and `generate(request)` methods.
  The existing free stage functions remain public; the new `decode` accepts
  `[T,64]` or `[1,64,T]` latents and returns finite, clipped CPU FP32
  `[samples,2]` audio. The shared external-plan branch handles supplied ABC and
  cot=off without running the planner. Existing token/CFG/truncation logic is
  unchanged. CUDA backbone is BF16; the VAE is FP32 with core=1024 / halo=16.
- Backbone weights are released before VAE loading. Checkpoints are memory-mapped
  directly from explicit directories or the offline HF_HOME snapshot resolver.
  Weight files/configs are SHA-256 hashed and an existing weights manifest is
  checked. No checkpoint is converted, copied, downloaded or modified.
- Added native Rust storage: signed int32 NumPy token arrays, FP32 NumPy latents,
  PCM_24 FLAC via flacenc, FLOAT WAV via hound, UTF-8 ABC, atomic JSON completion
  markers, plan manifests and complete artifact hashes. The default file set is
  exactly Python's eleven files, including `request.json` and
  `plan_manifest.json`. Plan-only output has five files; cot=off omits score.abc.
  Request/config/weight identity uses Python's sorted compact UTF-8 JSON,
  including float exponent formatting. Runtime/backend identity identifies Rust.
- The clap CLI accepts `--request`, `--output`, `--model-dir`, `--vae-dir`,
  `--config`, `--abc-file`, `--device`, `--audio-format` and `--quiet`.
  `--output` names the exact destination directory. It rejects nonempty output,
  preserves external ABC bytes, accepts Python request aliases/metadata and
  relative abc_path, applies per-stage sampling overrides, reports progress on
  stderr and a JSON summary on stdout, and writes failure.json on execution
  failure. The CUDA path permits only GPU 0.
- P5 artifact verification uses Python's actual `SymbolicPlan.load`,
  `verify_result`, tokenizer, native two-voice ABC parser, NumPy and soundfile.
  It checks the complete file set, identities/hashes, array dtypes/shapes, codec
  range, retained prefixes, audio length/finiteness, edited bytes and zero
  generated planner tokens for render. Both generation and rendering run the
  full default 32 midpoint steps; no shortened smoke-test preset is used.
- Extended `tools/measure_python_eager.py` with a cold end-to-end mode and an
  explicit output directory. It still supports the P2 fixed-Python-plan
  measurement. Both modes were measured again in this session, on GPU 0, with
  backend=torch-eager and native SDPA. These are separate from the saved fast run.
- Added storage/CLI CPU regressions and a real-waveform native WAV/FLAC check;
  updated README and added a reproducible clean audit script. Generated runs are
  under ignored `runs/`; logs remain in `~/work/yue2-rs-logs/`.

## Reviewer housekeeping

`DecoderConfig::default().final_tanh` is now **true**, matching the Python decoder
constructor. A regression checks the omitted field plus explicit false/true in
the nested decoder_config JSON read by the checkpoint loader. The standard
checkpoint explicitly sets false; its P4 numerics are checked again below.

The retained FP32 control was verified against its original p3b-python.json
sidecar, then added to the fixture manifest's files mapping with its original
shape/dtype metadata, byte count and SHA-256:

```text
p3b-python.safetensors: bytes=471650068
sha256=e6c122b87f78be8312ba6ed5eed63514789b1aed9a3cb9823666edda5a018994
```

`tools/pin_p3_control.py` performs this idempotent manifest update or a read-only
`--check`; it refuses a changed control. The P3 gate now verifies the manifest
hashes of both nar.safetensors and p3b-python.safetensors before loading, and
also checks the FP32 digest against the committed constant. No tensor file was
rewritten. The manifest remains an uncommitted fixture; the pinning tool, digest,
and enforcing test are committed so the immutable identity is reviewable.

## Audited gates and timings

**P5 PASS.** First-song request, seed 831001, cot=full, default sampling,
32 midpoint steps, GPU 0, original checkpoint files. Generate and edited-score
render produce complete nontruncated audio and the exact Python file set.

| P5 check | Clean-audit result |
| --- | --- |
| Generate | 617 ABC tokens, 747-token semantic prefix, 1,624 codec frames, FP32 latents `[1624,64]` |
| Generated audio | **3,118,016 stereo samples / 64.958666667 s**, 48 kHz PCM_24 FLAC, peak 0.911643863, finite |
| Artifact compatibility | **11/11 files**, every hash, request/config/weight identity, NumPy dtype/layout, and Python plan/result loaders pass |
| Plan command | **5/5 files**, 617 ABC tokens, native two-voice ABC parse passes; 6.532271088 s AR / 11.569631781 s CLI wall |
| Render edited ABC | Tempo 86→90 and LF→CRLF preserved exactly; 617 external ABC tokens, **0 generated ABC tokens / 0 s planner** |
| Rendered audio | **1,565 codec frames / 3,004,736 stereo samples / 62.598666667 s**, peak 0.712540030; all 11 artifacts verified |
| Render timings | Semantic 19.434644326 s; NAR 16.696415295 s; VAE 0.636814148 s; E2E **37.976581774 s**; CLI wall **41.890370985 s** |
| Truncation | abc=false and semantic=false for both complete audio runs; standalone plan also false |

End-to-end wall time by stage (seconds):

| Stage | Rust eager, release, clean audit | Python eager, measured here, clean audit | Python saved CUDA graph + flash |
| --- | ---: | ---: | ---: |
| ABC | **6.513994413** | **4.650217381** | 2.971908743 |
| Semantic | **20.101909204** | **13.334667873** | 7.963954709 |
| NAR | **17.169783640** | **3.703770217** | 3.451511450 |
| VAE, including stage transition/loading | **0.644224213** | **2.782660426** | 2.851955019 |
| E2E, including backbone loading | **45.640991732** | **25.907251205** | **19.607553694** |
| Produced audio | **64.958666667** | **59.918666667** | **59.358666667** |
| Wall including integrity setup and artifact storage | **49.569832984** | **29.829235606** | Not stored |

The last Python column is the immutable supplied first-song result, **not eager**.
All other numbers are from this session's final audit. Python eager uses
torch 2.10.0+cu128, backend=torch-eager, execution=eager, attention=sdpa, and its
native NAR SDPA path. It generates 481 ABC tokens and 1,498 codec frames from its
own generated plan. Rust generates its own sampled plan too; differing durations
are disclosed rather than padding/truncating either song to match.

Stage AR timings synchronize CUDA and include prefill and the emitted end token,
excluding weight loading. E2E includes loading and all four stages, but excludes
initial file integrity verification and final serialization, matching Python's
pipeline timer boundary. The final row includes that setup/storage and excludes
process startup/imports. Rust/Python resolve-and-integrity times are
3.703139305 / 3.832558192 s. Rust's E2E/audio ratio is about 0.703; Python eager's
is about 0.432. There is no minimum-speed P5 gate. The NAR cost is the largest
speed gap; the inherited tiled attention path was preserved.

Rust P5 throughput is **94.872663502 ABC / 80.838092716 semantic tok/s**;
Python's end-to-end eager run is **103.651068434 / 112.413748455 tok/s**.
The separate P2c comparison below uses the exact immutable Python plan in both
languages, as required by that gate.

### P1a–P4b regression gates, all rerun at the end

| Gate | Clean-audit result |
| --- | --- |
| P1a | **PASS:** 20/20 encodings and roundtrips, 208/208 specials, NFC, full first-song prefix 611/611 |
| P1b | **PASS:** first-song 611/611, song2 1670/1670, song3 1456/1456; generation prompts 128/200/202, supplied ABC IDs 481/1468/1252 |
| P1c | **PASS:** teacher-forced argmax **64/64**, maximum logit error **0.375000000** (bound 0.5) |
| P1d eager greedy oracle | **PASS: 64/64** |
| P1d original sampled abc_tokens.npy | **Inherited FAIL: 61/64**; no change to the test or artifact |
| P2a | **PASS:** native ABC parses; 617/481 tokens, ratio **1.282744283**; truncation=false; structure advisory |
| P2b | **PASS:** 1495/1484 codec frames, ratio **1.007412399**, indices **67–32623**, truncation=false, EOS excluded; exact Python plan |
| P2 controls | **PASS:** supplied ABC, cot=off CFG, exact prefix validation, independent ABC/semantic truncation, codec subtraction |
| P2 sampling oracle | **PASS:** 11/11 exact supports, max score error **4.7683716e-7** in ordinary semantic cases, 0 elsewhere; BF16 CFG **1024/1024 exact** |
| P3(a), FP32 chunk 0 | **PASS:** max error **0.000020578504**, cosine **1.000000000000**, 742 frames, 8.225855 s |
| P3(a), FP32 chunk 1 | **PASS:** max error **0.000025272369**, cosine **1.000000000000**, 742 frames, 8.330834 s |
| P3(b), BF16 chunk 0 | **PASS:** cosine **0.999378217075**, advisory max error **0.839233398438**, 742 frames, 5.825732 s |
| P3(b), BF16 chunk 1 | **PASS:** cosine **0.999945369729**, advisory max error **0.269531250000**, 742 frames, 5.866686 s |
| NAR RNG control | **PASS:** 576/576 values repeat after AR reseed, context-independent cuts, changed seed differs, AR RNG stream unchanged |
| P4a | **PASS:** all six decoder blocks, largest maximum error **0.000240266323** (bound 1e-3) |
| P4b | **PASS:** first-four-second SNR **120.086907 dB** (bound 40), max audio error **0.000000260770**; full `[1,2,2849216]` waveform |

P1c whole-tensor and minimum-position cosines:

| Layer | Cosine | Minimum position cosine | Max absolute error |
| --- | ---: | ---: | ---: |
| Embedding | 1.000000000000 | 1.000000000000 | 0 |
| Layer 0 | 0.999999994917 | 0.999999583650 | 0.5 |
| Layer 13 | 0.999999144118 | 0.999965031256 | 16 |
| Final norm | 0.999973469979 | 0.999865967444 | 0.25 |

P2a Rust sections: intro / verse / chorus / outro; reference: intro / verse /
chorus. Both have one K line and one M line. Rust Vocal/Ins parts have
24 measures with 61/34 notes; reference parts have 22 measures with 56/24 notes.

P2c throughput (including prefill and EOS, excluding loading):

| Stage | Rust gate seconds / outputs / tok/s | New Python eager seconds / outputs / tok/s | Saved Python graph tok/s | Rust / eager | Rust / graph |
| --- | --- | --- | ---: | ---: | ---: |
| Plan | **6.841150443 / 618 / 90.335683325** | **4.485921791 / 482 / 107.447258883** | 162.185329927 | 84.07% | 55.70% |
| Semantic, exact Python plan | **18.015605734 / 1496 / 83.039117423** | **12.861249269 / 1499 / 116.551663734** | 186.465148819 | 71.25% | 44.53% |

P3 uses the original noise, original two 742-frame chunks and 32 midpoint steps.
FP32 requires max error <=1e-3 and cosine >=0.999999; BF16 requires cosine
>=0.999. Cosines rounded to 1.000000000000 do not mean bit identity.
The diagnostic output explicitly reuses the retained P3b trace, rather than
claiming a new localization experiment. Native embeddings, projections, norms,
transformer outputs, velocities and midpoint/update states are BF16 in both
languages; noise, RoPE frequencies/angles, timestep frequencies and API latents
are FP32; raw solver times are FP64. First operation above 1e-3 is AR layer 0 Q
after RoPE, max error **0.00390625**; earlier nonzero FP32 rope.inv_freq error is
**1.862645149230957e-9**. Both complete trajectories were rerun.

P4a block maximum absolute errors (all FP32):

| Block | Shape | Max absolute error |
| --- | --- | ---: |
| 0 | `[1,1024,384]` | 0.000028729439 |
| 1 | `[1,512,1919]` | 0.000240266323 |
| 2 | `[1,256,7676]` | 0.000133156776 |
| 3 | `[1,128,30704]` | 0.000032126904 |
| 4 | `[1,64,61408]` | 0.000013113022 |
| 5 | `[1,64,122816]` | 0.000002920628 |

P4b full decode took **0.473520 s**, excluding checkpoint loading. The Python
artifact checker exported the Rust waveform to FLOAT WAV / PCM_24 FLAC: their
first-four-second SNRs are **120.086907 / 115.228304 dB**, roundtrip errors
**0 / 0.000000059605**. The additional P5 storage test independently writes both
formats through the native Rust encoders and soundfile reads them back: same
2,849,216 frames, 48 kHz stereo, exact FLOAT WAV, and FLAC max roundtrip error
**0.000000059605**. The reference waveform peak is 0.503509939 with zero clipping.

## Exact commands, pasted output and self-audit

Initial CPU checks, generation, planning, edited-score rendering, Python eager
measurements and native audio exports passed before the audit. The first P5
verification harness attempted to import unavailable music21; it was corrected
to use the already-installed native ABC parser used by P2. That parser then
rejected a dangling comment added by the initial edit helper. The helper now
makes a valid tempo edit and uses CRLF line endings. Those verification failures
remain in the initial logs; no generated score or numerical gate was repaired.

After the final implementation changes, ran **cargo clean**, then a fresh
**cargo test**, CPU/CUDA builds, formatting, both clippy configurations and every
requested gate. All gate/timing numbers in this report come from this final
audit, except the explicitly labeled immutable fast-path baseline and retained
P3b localization evidence. No implementation or gate code changed afterward.

The script exits **1 solely for the inherited P1d literal test** (its cargo test
exits 101). Every other recorded command exits 0. The 23 ordinary CPU tests pass;
17 fixture/diagnostic tests are ignored normally, and all 17 return cleanly with
YUE2_FIXTURES unset. Formatting and CPU/CUDA clippy with -D warnings pass.

Clean removed 14,514 files / 10.0 GiB. Fresh cargo test completed in 29.24 s,
CPU build in 8.13 s, CUDA release build in 59.17 s, CPU clippy in 4.87 s and CUDA
clippy in 41.10 s. The audit kept the test profile's original opt-level=2.

Exact top-level command from the repository root:

```bash
export PATH=/home/ayourtch/.cargo/bin:/usr/local/cuda/bin:$PATH CUDA_VISIBLE_DEVICES=0 CUDA_COMPUTE_CAP=120 YUE2_FIXTURES=/home/ayourtch/work/yue2-rs-fixtures
./tools/audit_phase5.sh > /home/ayourtch/work/yue2-rs-logs/p5-audit-commands.log 2>&1
```

The script additionally exports YUE2_TEST_DEVICE=cuda, HF_HUB_OFFLINE=1,
HF_HOME=/home/ayourtch/yue2/hf and PYTHONDONTWRITEBYTECODE=1. Exact command/exit
transcript (no gate is filtered out):

```text
clean command: cargo clean
clean exit=0 log=/home/ayourtch/work/yue2-rs-logs/p5-audit-clean.log
test command: cargo test
test exit=0 log=/home/ayourtch/work/yue2-rs-logs/p5-audit-test.log
build-cpu command: cargo build
build-cpu exit=0 log=/home/ayourtch/work/yue2-rs-logs/p5-audit-build-cpu.log
build-cuda command: tools/with_reference_blas.sh cargo build --release --features cuda
build-cuda exit=0 log=/home/ayourtch/work/yue2-rs-logs/p5-audit-build-cuda.log
fmt command: cargo fmt --check
fmt exit=0 log=/home/ayourtch/work/yue2-rs-logs/p5-audit-fmt.log
clippy command: cargo clippy --all-targets -- -D warnings
clippy exit=0 log=/home/ayourtch/work/yue2-rs-logs/p5-audit-clippy.log
clippy-cuda command: tools/with_reference_blas.sh cargo clippy --features cuda --all-targets -- -D warnings
clippy-cuda exit=0 log=/home/ayourtch/work/yue2-rs-logs/p5-audit-clippy-cuda.log
unset-fixtures command: env -u YUE2_FIXTURES cargo test -- --ignored --nocapture --test-threads=1
unset-fixtures exit=0 log=/home/ayourtch/work/yue2-rs-logs/p5-audit-unset-fixtures.log
pin-control command: /home/ayourtch/yue2/.venv/bin/python tools/pin_p3_control.py --root /home/ayourtch/work/yue2-rs-fixtures --check
pin-control exit=0 log=/home/ayourtch/work/yue2-rs-logs/p5-audit-pin-control.log
gpu command: nvidia-smi -i 0 --query-gpu=index\,memory.used\,memory.free --format=csv
gpu exit=0 log=/home/ayourtch/work/yue2-rs-logs/p5-audit-gpu.log
generate command: tools/with_reference_blas.sh target/release/yue2 generate --request /home/ayourtch/yue2/YuE/examples/song.json --output /home/ayourtch/work/yue2-rs/runs/p5-audit/generate --device cuda
generate exit=0 log=/home/ayourtch/work/yue2-rs-logs/p5-audit-generate.log
check-generate command: /home/ayourtch/yue2/.venv/bin/python tools/check_phase5.py /home/ayourtch/work/yue2-rs/runs/p5-audit/generate
check-generate exit=0 log=/home/ayourtch/work/yue2-rs-logs/p5-audit-check-generate.log
plan command: tools/with_reference_blas.sh target/release/yue2 plan --request /home/ayourtch/yue2/YuE/examples/song.json --output /home/ayourtch/work/yue2-rs/runs/p5-audit/plan --device cuda
plan exit=0 log=/home/ayourtch/work/yue2-rs-logs/p5-audit-plan.log
check-plan command: /home/ayourtch/yue2/.venv/bin/python tools/check_phase5.py /home/ayourtch/work/yue2-rs/runs/p5-audit/plan --plan-only
check-plan exit=0 log=/home/ayourtch/work/yue2-rs-logs/p5-audit-check-plan.log
edit command: /home/ayourtch/yue2/.venv/bin/python tools/check_phase5.py /home/ayourtch/work/yue2-rs/runs/p5-audit --make-edit
edit exit=0 log=/home/ayourtch/work/yue2-rs-logs/p5-audit-edit.log
render command: tools/with_reference_blas.sh target/release/yue2 render --request /home/ayourtch/yue2/YuE/examples/song.json --abc-file /home/ayourtch/work/yue2-rs/runs/p5-audit/edited.abc --output /home/ayourtch/work/yue2-rs/runs/p5-audit/render --device cuda
render exit=0 log=/home/ayourtch/work/yue2-rs-logs/p5-audit-render.log
check-render command: /home/ayourtch/yue2/.venv/bin/python tools/check_phase5.py /home/ayourtch/work/yue2-rs/runs/p5-audit/render --abc-file /home/ayourtch/work/yue2-rs/runs/p5-audit/edited.abc
check-render exit=0 log=/home/ayourtch/work/yue2-rs-logs/p5-audit-check-render.log
python-eager command: /home/ayourtch/yue2/.venv/bin/python -u tools/measure_python_eager.py --end-to-end --output /home/ayourtch/work/yue2-rs/runs/p5-audit/python-eager
python-eager exit=0 log=/home/ayourtch/work/yue2-rs-logs/p5-audit-python-eager.log
python-p2 command: /home/ayourtch/yue2/.venv/bin/python -u tools/measure_python_eager.py --output /home/ayourtch/work/yue2-rs/runs/p5-audit/python-p2
python-p2 exit=0 log=/home/ayourtch/work/yue2-rs-logs/p5-audit-python-p2.log
p1 command: tools/with_reference_blas.sh cargo test --features cuda --test parity -- --ignored --nocapture --test-threads=1
p1 exit=101 log=/home/ayourtch/work/yue2-rs-logs/p5-audit-p1.log
p2 command: tools/with_reference_blas.sh cargo test --features cuda --test phase2 -- --ignored --nocapture --test-threads=1
p2 exit=0 log=/home/ayourtch/work/yue2-rs-logs/p5-audit-p2.log
p3 command: tools/with_reference_blas.sh cargo test --features cuda --test phase3 -- --ignored --nocapture --test-threads=1
p3 exit=0 log=/home/ayourtch/work/yue2-rs-logs/p5-audit-p3.log
p4 command: tools/with_reference_blas.sh cargo test --features cuda --test phase4 -- --ignored --nocapture --test-threads=1
p4 exit=0 log=/home/ayourtch/work/yue2-rs-logs/p5-audit-p4.log
p4-audio command: /home/ayourtch/yue2/.venv/bin/python tools/check_phase4_audio.py --root /home/ayourtch/work/yue2-rs-fixtures/first-song
p4-audio exit=0 log=/home/ayourtch/work/yue2-rs-logs/p5-audit-p4-audio.log
native-audio command: env YUE2_P5_AUDIO_DIR=/home/ayourtch/work/yue2-rs/runs/p5-audit/native-audio tools/with_reference_blas.sh cargo test --features cuda --test phase5 -- --ignored --nocapture --test-threads=1
native-audio exit=0 log=/home/ayourtch/work/yue2-rs-logs/p5-audit-native-audio.log
check-native-audio command: /home/ayourtch/yue2/.venv/bin/python tools/check_phase5.py /home/ayourtch/work/yue2-rs/runs/p5-audit/native-audio --audio-export
check-native-audio exit=0 log=/home/ayourtch/work/yue2-rs-logs/p5-audit-check-native-audio.log
```

P5 stdout and Python verifier output (progress and verbose timing JSON omitted;
the detailed stage numbers are in the tables above and result.json):

```text
{"audio_seconds":64.95866666666667,"output":"/home/ayourtch/work/yue2-rs/runs/p5-audit/generate","seconds":45.640991732,"status":"complete","truncated":{"abc":false,"semantic":false},"wall_seconds":49.569832984}
{"output":"/home/ayourtch/work/yue2-rs/runs/p5-audit/plan","stage":"plan","timing":{"attention":"sdpa","cfg_branches":1,"content_tokens":617,"execution":"eager","output_tokens":618,"output_tps":94.60721878724333,"prefill_seconds":0.169454086,"prefix_tokens":128,"seconds":6.532271088,"ttft_seconds":0.17150472},"truncated":false,"wall_seconds":11.569631781}
{"audio_seconds":62.59866666666667,"output":"/home/ayourtch/work/yue2-rs/runs/p5-audit/render","seconds":37.976581774,"status":"complete","truncated":{"abc":false,"semantic":false},"wall_seconds":41.890370985}
P5 ABC parse PASS: 2 voices; 617 ABC tokens; prefix=747
P5 artifacts PASS: exact Python file set (11 files), hashes and request/config/weight identity verified
P5 audio PASS: 1624 codec frames (12..32690), latents=(1624, 64) float32; 3118016 stereo samples, 64.958666667 s, peak=0.911643863
P5 ABC parse PASS: 2 voices; 617 ABC tokens; prefix=747
P5 plan PASS: Python SymbolicPlan.load verified all 5 files
P5 ABC parse PASS: 2 voices; 617 ABC tokens; prefix=747
P5 render PASS: edited ABC bytes/prefix preserved; zero generated ABC tokens
P5 artifacts PASS: exact Python file set (11 files), hashes and request/config/weight identity verified
P5 audio PASS: 1565 codec frames (36..32761), latents=(1565, 64) float32; 3004736 stereo samples, 62.598666667 s, peak=0.712540030
```

P1 output, including the retained failure (compiler progress omitted):

```text
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

thread 'p1d_literal_saved_abc' (3006835) panicked at tests/parity.rs:361:5:
assertion `left == right` failed: Original artifact is sampled, see REPORT-P0.md and greedy.json
  left: 61
 right: 64
note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace
FAILED

failures:

failures:
    p1d_literal_saved_abc

test result: FAILED. 4 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out; finished in 4.04s

error: test failed, to rerun pass `--test parity`
```

P2 output:

```text
running 3 tests
test p2_full_generation_gates ... Abc: 250 output tokens
Abc: 500 output tokens
P2a ABC tokens=617/481 ratio=1.282744283; truncated=false
P2c plan timing: {"attention":"sdpa","cfg_branches":1,"content_tokens":617,"execution":"eager","output_tokens":618,"output_tps":90.33568332536083,"prefill_seconds":0.173808713,"prefix_tokens":128,"seconds":6.841150443,"ttft_seconds":0.176386912}
P2a reference ABC parse PASS: Vocal=22 measures/56 notes, Ins=22 measures/24 notes
P2a Rust ABC parse PASS: Vocal=24 measures/61 notes, Ins=24 measures/34 notes
P2a sections: Rust=['% intro', '% verse', '% chorus', '% outro'], reference=['% intro', '% verse', '% chorus']
P2a key/meter line counts: Rust={'K': 1, 'M': 1}, reference={'K': 1, 'M': 1}
P2a advisory length: 617/481, ratio=1.282744283, target within 30%=True, truncated=False
P2a PASS: ABC parses; section tags and key/meter counts are advisory
Semantic: 250 output tokens
Semantic: 500 output tokens
Semantic: 750 output tokens
Semantic: 1000 output tokens
Semantic: 1250 output tokens
P2b semantic tokens=1495/1484 ratio=1.007412399; codec min=67 max=32623; truncated=false
P2c semantic timing: {"attention":"sdpa","cfg_branches":1,"content_tokens":1495,"execution":"eager","output_tokens":1496,"output_tps":83.03911742343861,"prefill_seconds":0.048247533,"prefix_tokens":611,"seconds":18.015605734,"ttft_seconds":0.049687365}
P2b PASS; codec range, exact Python plan, length and EOS accounting valid
P2a advisory length target within 30%: true
P2a/P2b PASS; P2c measured on Cuda(CudaDevice(DeviceId(1))); seed=831001
ok
test pipeline_modes_and_truncation ... Pipeline controls PASS: provided ABC, cot=off CFG, exact prefix validation, independent plan/semantic truncation and codec subtraction
ok
test sampling_python_oracle ... Sampling case.0 Abc legacy=false: exact support; max_abs=0
Sampling case.1 Abc legacy=false: exact support; max_abs=0
Sampling case.2 Abc legacy=false: exact support; max_abs=0
Sampling case.3 Abc legacy=false: exact support; max_abs=0
Sampling case.4 Semantic legacy=false: exact support; max_abs=0.00000047683716
Sampling case.5 Semantic legacy=false: exact support; max_abs=0.00000047683716
Sampling case.6 Semantic legacy=true: exact support; max_abs=0
Sampling case.7 Semantic legacy=true: exact support; max_abs=0
Sampling case.8 Semantic legacy=true: exact support; max_abs=0
Sampling case.9 Abc legacy=false: exact support; max_abs=0
Sampling case.10 Semantic legacy=true: exact support; max_abs=0
BF16 CFG subtraction/multiply/add: 1024/1024 exact
ok

test result: ok. 3 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 25.98s
```

P3 output excerpt (per-step progress and repeated hash/dtype rows omitted):

```text
test nar_cuda_noise_owns_seed ... NAR CUDA RNG PASS: 576/576 values repeat after AR reseed; context-independent cuts; different seed differs; AR stream unchanged
P3(a) FP32 two_chunk.0: frames=742 max_abs=0.000020578504 cosine=1.000000000000 seconds=8.225855
P3(a) FP32 two_chunk.1: frames=742 max_abs=0.000025272369 cosine=1.000000000000 seconds=8.330834
P3(b) first divergent operation above 1e-3: ar.layer.00.q (AR layer 0 Q after RoPE), max_abs=0.00390625; earlier nonzero FP32 precursor rope.inv_freq max_abs=1.862645149230957e-9
P3(b) BF16 two_chunk.0: frames=742 max_abs=0.839233398438 cosine=0.999378217075 seconds=5.825732
P3(b) BF16 two_chunk.1: frames=742 max_abs=0.269531250000 cosine=0.999945369729 seconds=5.866686
test result: ok. 3 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 32.53s
```

P4 output and both Python/native storage checks:

```text
running 2 tests
test p4a_decoder_blocks ... P4a block.0: shape=[1, 1024, 384] dtype=F32 max_abs=0.000028729439
P4a block.1: shape=[1, 512, 1919] dtype=F32 max_abs=0.000240266323
P4a block.2: shape=[1, 256, 7676] dtype=F32 max_abs=0.000133156776
P4a block.3: shape=[1, 128, 30704] dtype=F32 max_abs=0.000032126904
P4a block.4: shape=[1, 64, 61408] dtype=F32 max_abs=0.000013113022
P4a block.5: shape=[1, 64, 122816] dtype=F32 max_abs=0.000002920628
P4a slice audio (reported): shape=[1, 2, 122816] max_abs=0.000000871718 snr_db=118.876906 seconds=0.170832
ok
test p4b_reference_audio ... P4b config: frames=1484 core=1024 halo=16 required_halo=12 FP32, unclipped
P4b tile 1/2
P4b tile 2/2
P4b first 4 seconds: samples=192000 channels=2 max_abs=0.000000260770 snr_db=120.086907
P4b full audio: shape=[1, 2, 2849216] seconds=0.473520
ok

test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 1.03s
P4b Rust raw first-4s SNR=120.086907 dB; peak=0.503509939; clipped_values=0
P4b audio.wav: 2849216 frames, 48000 Hz, 2 channels, FLOAT; first-4s SNR=120.086907 dB; roundtrip_max=0.000000000000; bytes=22793816
P4b audio.flac: 2849216 frames, 48000 Hz, 2 channels, PCM_24; first-4s SNR=115.228304 dB; roundtrip_max=0.000000059605; bytes=8341876
Total recursive fixture/artifact bytes: 1,478,864,347 <= 1,500,000,000
P5 native audio.wav: 2849216 frames, 48000 Hz, 2 channels, FLOAT; roundtrip_max=0.000000000000
P5 native audio.flac: 2849216 frames, 48000 Hz, 2 channels, PCM_24; roundtrip_max=0.000000059605
```

The full Python eager measurement is
`runs/p5-audit/python-eager/measurement.json`; its generated song retains the
full artifact set under `python-eager/song/`. The independently measured P2c
baseline is `runs/p5-audit/python-p2/measurement.json`, including its request,
generation settings, attention/execution labels and GPU information. Rust outputs
are `runs/p5-audit/{generate,plan,render,native-audio}/`.

Both generate and render reproduced **7/7 content files byte-for-byte** across
their initial successful runs and the clean audit: score.abc, abc_tokens.npy,
prefix.npy, semantic.npy, latent.npy, audio.flac and request.json. Timings and
executable runtime identity are naturally different. The comparisons are logged
in `p5-audit-repeatability.log`.

Before CUDA runs, GPU 0 had **34,858 MiB free**. Only GPU 0 was visible to every
CUDA process. The script sampled memory every 500 ms with:

```bash
nvidia-smi -i 0 --query-compute-apps=gpu_uuid,pid,used_gpu_memory --format=csv -lms 500
```

| Audit process | PID | Observed GPU 0 peak, MiB |
| --- | ---: | ---: |
| Rust generate | 3001377 | 9,550 |
| Rust plan | 3001787 | 8,108 |
| Rust render | 3001929 | 9,550 |
| Python eager E2E | 3002341 | 8,888 |
| Python eager P2c | 3002605 | 8,810 |
| P1 | 3006787 | 4,876 |
| P2 | 3007045 | 5,932 |
| P3, both tiers | 3007424 | **15,916** |
| P4 | 3007825 | 9,490 |

All stayed below the approximately 30 GB allowance. Python's own E2E peak
allocated/reserved numbers are **8,406,090,752 / 8,600,420,352 bytes**; the fixed-plan
measurement reports **8,406,090,752 / 8,518,631,424 bytes**. Existing server
processes were untouched. Full memory samples are in p5-audit-memory.log.

All six manifest-pinned fixture SHA-256 values were read-only verified, including
the new FP32 control; identities are logged in p5-audit-fixture-identities.log.
Both checkpoint identities also match the original Python result. The final
recursive fixture size is **1,478,864,347 bytes**, below 1.5 GB. Generated P5 songs
live in ignored repository runs/ rather than inflating the fixture directory.

## Unsure list and design objections

1. The original literal P1d test compares greedy output to a sampled file. This
   inherited failure is retained and reported, while the separate Python eager
   greedy oracle remains the algorithmic comparison. No test was skipped or
   tolerance lowered to manufacture an all-green audit.
2. End-to-end sampled songs differ in content and length because the fixed
   design deliberately uses candle's RNG instead of Torch's stream. P5 verifies
   the complete pipeline/artifact protocol; it does not establish perceptual
   equivalence or waveform identity for newly sampled songs. P3/P4 retain their
   fixed-noise/fixed-latent numerical gates. BF16 max errors remain advisory under
   the owner's P3 wording.
3. Timings are individual observations on a shared GPU, not distributions from
   repeated benchmarks on an isolated device. Rust P5 uses the release profile;
   existing parity/throughput gates retain opt-level=2 in the test profile.
   Native Torch SDPA and candle's tiled matmul/softmax attention differ in
   implementation. No CUDA graphs or new fused kernels were introduced.
4. The decoder method drops the backbone instead of retaining a CPU copy; a
   subsequent request reloads it. The pipeline hashes both checkpoint files even
   for plan-only work, and loads both AR/NAR weights when it needs the backbone.
   These are avoidable initialization costs, but no latency requirement mandates
   a new cache/offload design in P5. The high-level wrapper runs serially; low-level
   stage callbacks/cancellation remain available.
5. Default FLAC output is consumable by Python's result verifier. Explicit WAV
   output intentionally substitutes audio.wav; the upstream verify_result helper
   hardcodes audio.flac and therefore does not accept that optional file set.
   Container bytes/compression metadata are not promised to match libsndfile;
   decoded sample precision is checked. The CLI's exact --output directory is
   documented, whereas Python's generate CLI appends the request id.
6. Certification covers the supplied standard checkpoints and GPU 0 with the
   reference cuBLAS. CPU builds/regressions pass; full-song CPU execution, Metal,
   flash-attn, other checkpoints, batch/resume and audio-to-ABC transcription are
   not certified here. The CLI restricts CUDA to GPU 0; the reusable low-level
   library still accepts a caller-provided Device. Process memory peaks are
   sampled observations, not allocator-instrumented Rust maxima.
7. No further objection to the fixed library/CLI architecture. The principal
   performance gap is reported without altering numerical gates. All requested
   work is local; no push was performed.
