#!/usr/bin/env bash
# CUDA 13.x compiles the kernels; use the reference venv's cuBLAS for parity.
# Only build-directory symlinks are created. No libraries or weights are copied.
set -euo pipefail
task_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
reference_blas=${YUE2_REFERENCE_CUBLAS:-$HOME/yue2/.venv/lib/python3.12/site-packages/nvidia/cublas/lib}
task_links=$task_root/target/reference-cublas
for lib in libcublas libcublasLt; do
    if [[ ! -f $reference_blas/$lib.so.12 ]]; then
        printf 'Missing reference library: %s\n' "$reference_blas/$lib.so.12" >&2
        exit 1
    fi
done
mkdir -p -- "$task_links"
for lib in libcublas libcublasLt; do
    ln -sfn -- "$reference_blas/$lib.so.12" "$task_links/$lib.so"
done
export RUSTFLAGS="${RUSTFLAGS:+$RUSTFLAGS }-L native=$task_links"
export LD_LIBRARY_PATH="$reference_blas${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
printf 'Reference cuBLAS: %s\n' "$reference_blas" >&2
exec "$@"
