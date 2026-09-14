# yue2-rs

Checkpoint-native YuE2 inference on candle 0.11: request → ABC plan → semantic
tokens → NAR latents → 48 kHz stereo audio. One library and a thin CLI, implemented
through Phase 5 of [TASK.md](TASK.md). Covers from supplied ABC are supported;
SheetSage2/MERT2 audio transcription is outside scope.

```bash
export PATH="$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH"
cargo build
cargo test
export CUDA_VISIBLE_DEVICES=0 CUDA_COMPUTE_CAP=120
export HF_HOME="$HOME/yue2/hf"
tools/with_reference_blas.sh cargo build --release --features cuda

tools/with_reference_blas.sh target/release/yue2 generate \
  --request "$HOME/yue2/YuE/examples/song.json" --output runs/song --device cuda

tools/with_reference_blas.sh target/release/yue2 plan \
  --request "$HOME/yue2/YuE/examples/song.json" --output runs/plan --device cuda

# Copy/edit runs/plan/score.abc, then supply the edited file:
tools/with_reference_blas.sh target/release/yue2 render \
  --request "$HOME/yue2/YuE/examples/song.json" --abc-file edited.abc \
  --output runs/render --device cuda
```

`--output` is the exact destination directory; it must be empty. `generate`
(default FLAC) and `render` write the same eleven files as Python:
`score.abc`, `plan.json`, `plan_manifest.json`, `prefix.npy`, `abc_tokens.npy`,
`semantic.npy`, `latent.npy`, `audio.flac`, `request.json`, `config.json`, and
`result.json`. `plan` writes the first five. `cot=off` omits `score.abc`.
Token arrays are NumPy int32, latents are FP32 `[T,64]`, and FLAC is PCM_24 stereo.
`--audio-format wav` selects FLOAT WAV. Audio encoding runs entirely in Rust.
The final result contains timings, truncation flags, weight identities and hashes
of every artifact. Python's `SymbolicPlan.load` and `verify_result` can read the
default output. Runtime/backend identities describe Rust's execution.

Requests follow the existing Python protocol: `style` (or `tags`), `lyrics`,
`cot`, `seed`, `abc`, `cfg_scale`, `id`, and optional `abc_sampling` /
`semantic_sampling` dictionaries. `abc_path` resolves relative to the request;
`--abc-file` overrides both ABC fields. Historical metadata and a validated
literal `prompt` are accepted. `--config` accepts GenerationConfig JSON or a
saved effective `config.json`. Defaults preserve the 32-step midpoint solver.
`--quiet` hides progress; stdout is a JSON summary including total CLI wall time.
A failed run writes `failure.json` and returns nonzero.

`--model-dir` and `--vae-dir` take explicit local checkpoint directories; otherwise
the offline snapshot resolver uses `HF_HOME` (default `~/yue2/hf`). Checkpoints are
hashed and memory-mapped without conversion or copies. Keep those files immutable
while the pipeline runs. The backbone is released before loading the FP32 VAE.
The CLI's CUDA device is restricted to physical GPU 0. Default features build for
CPU; a CUDA build defaults to CUDA. CPU backbone uses FP32. Metal can be selected
with a `metal` build but has not been certified. `flash-attn` is declared; the
certified inference path uses eager attention and no CUDA graphs.

Library entry points are `pipeline::{plan, generate_semantic, synthesize, decode}`
and `YuE2Pipeline::{from_pretrained, plan, generate_semantic, synthesize, decode,
generate}`. `SymbolicPlan::save` and `SongResult::{save, save_artifacts}` implement
storage. Low-level stages retain callbacks/cancellation. `from_pretrained` is
unsafe because memory-mapped checkpoints must remain immutable for the pipeline's
lifetime. The high-level pipeline processes one request at a time.

The reference Python environment uses cuBLAS 12.8; system cuBLAS 13.6 produces
BF16 reduction differences large enough to fail P1c. `with_reference_blas.sh`
creates build-directory symlinks and supplies link/runtime search paths; CUDA
kernels still compile with the installed CUDA 13.3 toolchain. Override the library
directory with `YUE2_REFERENCE_CUBLAS`. No reference files are edited.

Existing parity fixtures stay under `~/work/yue2-rs-fixtures/first-song/` and are
not committed, including `manifest.json`. P3 requires its `files` SHA-256 entries
for `nar.safetensors` and `p3b-python.safetensors`. To regenerate the base manifest
and its P3 fixture entries using the reference venv and GPU 0:

```bash
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
"$HOME/yue2/.venv/bin/python" tools/dump_reference.py --greedy --two-chunks
"$HOME/yue2/.venv/bin/python" tools/dump_reference.py --nar-stages
"$HOME/yue2/.venv/bin/python" tools/pin_p3_control.py --root "$HOME/work/yue2-rs-fixtures"
```

For existing fixtures, only the last command is needed to add the retained FP32
control entry; append `--check` for read-only verification. It verifies the
committed control digest before updating the manifest.

Running `python tools/measure_python_eager.py` without arguments uses a timestamped
subdirectory under `~/work/yue2-rs-fixtures/first-song/p2-eager/` when that default
directory is nonempty, and prints the destination to stderr. Explicit `--output`
must be empty.

Ignored tests skip when `YUE2_FIXTURES` is unset. The Phase 5 audit
rebuilds cleanly, runs all three CLI commands, measures Python eager, verifies
artifacts with Python, then reruns every P1–P4 gate:

```bash
export YUE2_FIXTURES="$HOME/work/yue2-rs-fixtures"
export YUE2_P5_OUTPUT="$PWD/runs/p5-audit-$(date -u +%Y%m%dT%H%M%S)"
tools/audit_phase5.sh > "$HOME/work/yue2-rs-logs/p5-audit-commands.log" 2>&1
```

The 2026-09-14 cleanup audit exited **0**, with all 26 steps passing; logs are in
`~/work/yue2-rs-logs/p5-cleanup-20260914/`. P1d still asserts the Python eager
greedy oracle match (64/64). The original sampled `abc_tokens.npy` comparison
reports 61/64 as advisory under the clarified TASK.md gate. P2a structural
comparisons and P3 BF16 maximum absolute errors also remain advisory.
See [REPORT-P5.md](REPORT-P5.md) for Phase 5 timings and limitations; phase reports
retain their historical results, including the former literal P1d failure.
