#!/usr/bin/env bash
# P3 clean self-audit. P0/P1/P2 generation and fixtures remain completed work.
set -uo pipefail
: "${YUE2_FIXTURES:?Set YUE2_FIXTURES to the existing fixture directory}"
[[ ${CUDA_VISIBLE_DEVICES-} == 0 ]] || { printf 'Set CUDA_VISIBLE_DEVICES=0\n' >&2; exit 1; }
export YUE2_TEST_DEVICE=cuda
export YUE2_NAR_DIAGNOSTIC=1
task_logs=${YUE2_LOG_DIR:-$HOME/work/yue2-rs-logs}
task_python=${YUE2_PYTHON:-$HOME/yue2/.venv/bin/python}
task_fixtures=$YUE2_FIXTURES
[[ -f $task_fixtures/manifest.json ]] || task_fixtures=$task_fixtures/first-song
mkdir -p -- "$task_logs"
task_status=0
record() {
    local label=$1
    shift
    printf '%s command:' "$label"
    printf ' %q' "$@"
    printf '\n'
    "$@" > "$task_logs/p3-audit-$label.log" 2>&1
    local status=$?
    printf '%s exit=%s log=%s\n' "$label" "$status" "$task_logs/p3-audit-$label.log"
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
record p2a-advisory env PYTHONDONTWRITEBYTECODE=1 "$task_python" tools/check_phase2.py "$task_fixtures/p2"
record gpu nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_gpu_memory --format=csv -lms 500 > "$task_logs/p3-audit-memory.log" 2>&1 &
task_monitor=$!
trap 'kill "$task_monitor" 2>/dev/null || true' EXIT
record gates tools/with_reference_blas.sh cargo test --features cuda --test phase3 -- --ignored --nocapture --test-threads=1
record operations tools/with_reference_blas.sh cargo test --features cuda --lib nar_operations_oracle -- --ignored --nocapture
kill "$task_monitor" 2>/dev/null || true
wait "$task_monitor" 2>/dev/null || true
trap - EXIT
exit "$task_status"
