#!/usr/bin/env bash
# Run from the repository root; preserve every gate's failure in the audit exit.
set -uo pipefail
: "${YUE2_FIXTURES:?Set YUE2_FIXTURES to the dumped fixture directory}"
[[ ${CUDA_VISIBLE_DEVICES-} == 0 ]] || { printf 'Set CUDA_VISIBLE_DEVICES=0\n' >&2; exit 1; }
export YUE2_TEST_DEVICE=cuda
task_logs=${YUE2_LOG_DIR:-$HOME/work/yue2-rs-logs}
task_python=${YUE2_PYTHON:-$HOME/yue2/.venv/bin/python}
mkdir -p -- "$task_logs"
task_status=0
record() {
    local label=$1
    shift
    printf '%s command:' "$label"
    printf ' %q' "$@"
    printf '\n'
    "$@" > "$task_logs/p2-audit-$label.log" 2>&1
    local status=$?
    printf '%s exit=%s log=%s\n' "$label" "$status" "$task_logs/p2-audit-$label.log"
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
record dump env PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 "$task_python" -u tools/dump_reference.py --phase2-sampling
record sampling-cpu env YUE2_TEST_DEVICE=cpu cargo test --test phase2 sampling_python_oracle -- --ignored --nocapture
record pipeline-cpu env YUE2_TEST_DEVICE=cpu cargo test --test phase2 pipeline_modes_and_truncation -- --ignored --nocapture
record gpu-before-generation nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_gpu_memory --format=csv -lms 500 > "$task_logs/p2-audit-memory.log" 2>&1 &
task_monitor=$!
trap 'kill "$task_monitor" 2>/dev/null || true' EXIT
record gates tools/with_reference_blas.sh cargo test --features cuda --test phase2 -- --ignored --nocapture --test-threads=1
kill "$task_monitor" 2>/dev/null || true
wait "$task_monitor" 2>/dev/null || true
trap - EXIT
exit "$task_status"
