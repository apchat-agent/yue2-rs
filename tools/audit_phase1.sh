#!/usr/bin/env bash
# Rebuild from scratch, run every Phase 1 gate, and retain even failed gate logs.
# Run from the repository root. The sampled-artifact P1d gate is intentionally
# included; its failure must be reported rather than hidden by --skip.
set -uo pipefail
: "${YUE2_FIXTURES:?Set YUE2_FIXTURES to the dumped fixture directory}"
[[ ${CUDA_VISIBLE_DEVICES-} == 0 ]] || { printf 'Set CUDA_VISIBLE_DEVICES=0\n' >&2; exit 1; }
export YUE2_TEST_DEVICE=cuda
task_logs=${YUE2_LOG_DIR:-$HOME/work/yue2-rs-logs}
mkdir -p -- "$task_logs"
task_status=0
record() {
    local label=$1
    shift
    printf '%s command:' "$label"
    printf ' %q' "$@"
    printf '\n'
    "$@" > "$task_logs/p1-audit-$label.log" 2>&1
    local status=$?
    printf '%s exit=%s log=%s\n' "$label" "$status" "$task_logs/p1-audit-$label.log"
    if (( status != 0 )); then task_status=1; fi
    return "$status"
}
record clean cargo clean || exit 1
record test cargo test
record build-cpu cargo build
record build-cuda cargo build --features cuda
record fmt cargo fmt --check
record clippy cargo clippy --all-targets -- -D warnings
record clippy-cuda cargo clippy --features cuda --all-targets -- -D warnings
record unset-fixtures env -u YUE2_FIXTURES cargo test -- --ignored --nocapture --test-threads=1
record gpu nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv
record parity tools/with_reference_blas.sh cargo test --features cuda --test parity -- --ignored --nocapture --test-threads=1
record system-cuda cargo test --features cuda --test parity p1c -- --ignored --nocapture
record cpu-greedy env YUE2_TEST_DEVICE=cpu cargo test --test parity p1d_greedy_oracle -- --ignored --nocapture
exit "$task_status"
