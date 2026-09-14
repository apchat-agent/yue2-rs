# yue2-rs

YuE2 inference on candle 0.11, currently implemented through Phase 1 of
[TASK.md](TASK.md): tokenizer, request protocol, and the AR backbone with a bounded
KV cache. Sampling, NAR, VAE, end-to-end generation and the working CLI are future
phases. The current CLI is a placeholder.

```bash
export PATH="$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH"
cargo build
cargo test
CUDA_VISIBLE_DEVICES=0 CUDA_COMPUTE_CAP=120 cargo build --features cuda
```

Default features use CPU. CUDA is opt-in; `metal` and `flash-attn` feature
dependencies are declared but have not been validated. AR weights are read from
an explicit model directory using safetensors/VarBuilder, without conversion.
`model::snapshot_dir("YuE2-3B")` resolves a local snapshot under `HF_HOME` (default
`~/yue2/hf`). Nothing downloads model files.

Generate the offline Python oracle using the installed reference venv:

```bash
nvidia-smi
PYTHONDONTWRITEBYTECODE=1 HF_HOME="$HOME/yue2/hf" HF_HUB_OFFLINE=1 \
CUDA_VISIBLE_DEVICES=0 "$HOME/yue2/.venv/bin/python" \
  tools/dump_reference.py --greedy --two-chunks
```

Fixtures stay under `~/work/yue2-rs-fixtures/first-song/`. They include original
saved tokens, three request prefixes, eager AR outputs, all midpoint solver loop
intermediates, and VAE decoder traces. The short reference has only one native
NAR chunk; `--two-chunks` adds an explicitly labeled smaller-context case.

Run all Phase 1 gates (GPU 0 only, serial model tests):

```bash
export CUDA_VISIBLE_DEVICES=0 CUDA_COMPUTE_CAP=120
export YUE2_FIXTURES="$HOME/work/yue2-rs-fixtures"
export YUE2_TEST_DEVICE=cuda
tools/with_reference_blas.sh cargo test --features cuda --test parity \
  -- --ignored --nocapture --test-threads=1
```

**Reference-library requirement:** Python uses cuBLAS 12.8 while the CUDA 13.3
toolchain links system cuBLAS 13.6. Their BF16 GEMM reductions differ enough to
fail the strict P1c gate. `with_reference_blas.sh` creates two build-directory
symlinks and supplies link/runtime search paths for the reference venv's cuBLAS;
CUDA kernels still compile with the installed CUDA 13.3 toolchain. Override its
library directory with `YUE2_REFERENCE_CUBLAS`. This helper changes only its child
process environment. CPU BF16 matmul uses FP32 accumulation with BF16 operation
boundaries because candle CPU has no native BF16 matmul.

**Known literal gate failure:** `abc_tokens.npy` contains a sampled plan. The
separate Python eager greedy oracle matches Rust for all 64 steps, but greedy
matches only 61/64 of the original sampled tokens. The command above deliberately
runs and reports the failing `p1d_literal_saved_abc` test. To run just the
greedy-to-greedy comparison, select `p1d_greedy_oracle`.

`tools/audit_phase1.sh` starts with `cargo clean`, repeats builds, lint checks and
every gate, and puts all output in `~/work/yue2-rs-logs/`. It returns nonzero for
the known literal and system-cuBLAS gate failures. Ignored tests skip cleanly if
`YUE2_FIXTURES` is unset; `cargo test` always runs the small CPU regression suite.
Optional operation diagnostics: run `tools/dump_ar_debug.py` in the same Python
environment, then select `ar_operations_oracle` with `--lib --features cuda`
through the reference-BLAS helper. `--investigate-gemm` also profiles Python's
GEMM reduction.

See [REPORT-P0.md](REPORT-P0.md) and [REPORT-P1.md](REPORT-P1.md) for gate evidence,
numerical results, and limitations. No later phase is implemented.
