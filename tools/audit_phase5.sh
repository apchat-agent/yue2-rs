#!/usr/bin/env bash
# Phase 5 clean self-audit plus every inherited P1a-P4b gate. Never dump fixtures.
# The known literal sampled-artifact P1d failure is retained in the exit status.
set -uo pipefail
: "${YUE2_FIXTURES:?Set YUE2_FIXTURES to the existing fixture directory}"
[[ ${CUDA_VISIBLE_DEVICES-} == 0 ]] || { printf 'Set CUDA_VISIBLE_DEVICES=0\n' >&2; exit 1; }
export YUE2_TEST_DEVICE=cuda PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1
export HF_HOME=${HF_HOME:-$HOME/yue2/hf}
task_logs=${YUE2_LOG_DIR:-$HOME/work/yue2-rs-logs}
task_python=${YUE2_PYTHON:-$HOME/yue2/.venv/bin/python}
task_output=${YUE2_P5_OUTPUT:-$PWD/runs/p5-audit}
task_request=$HOME/yue2/YuE/examples/song.json
task_fixtures=$YUE2_FIXTURES
[[ -f $task_fixtures/manifest.json ]] || task_fixtures=$task_fixtures/first-song
[[ ! -e $task_output ]] || { printf 'Use a new YUE2_P5_OUTPUT directory\n' >&2; exit 1; }
mkdir -p -- "$task_logs"
task_status=0
record() {
    local label=$1
    shift
    printf '%s command:' "$label"
    printf ' %q' "$@"
    printf '\n'
    "$@" > "$task_logs/p5-audit-$label.log" 2>&1
    local status=$?
    printf '%s exit=%s log=%s\n' "$label" "$status" "$task_logs/p5-audit-$label.log"
    if (( status != 0 )); then task_status=1; fi
    return "$status"
}
record clean cargo clean || exit 1
record test cargo test
record build-cpu cargo build
record build-cuda tools/with_reference_blas.sh cargo build --release --features cuda
record fmt cargo fmt --check
record clippy cargo clippy --all-targets -- -D warnings
record clippy-cuda tools/with_reference_blas.sh cargo clippy --features cuda --all-targets -- -D warnings
record unset-fixtures env -u YUE2_FIXTURES cargo test -- --ignored --nocapture --test-threads=1
record pin-control "$task_python" tools/pin_p3_control.py --root "$YUE2_FIXTURES" --check
record gpu nvidia-smi -i 0 --query-gpu=index,memory.used,memory.free --format=csv
nvidia-smi -i 0 --query-compute-apps=gpu_uuid,pid,used_gpu_memory --format=csv -lms 500 > "$task_logs/p5-audit-memory.log" 2>&1 &
task_monitor=$!
trap 'kill "$task_monitor" 2>/dev/null || true' EXIT
record generate tools/with_reference_blas.sh target/release/yue2 generate --request "$task_request" --output "$task_output/generate" --device cuda
record check-generate "$task_python" tools/check_phase5.py "$task_output/generate"
record plan tools/with_reference_blas.sh target/release/yue2 plan --request "$task_request" --output "$task_output/plan" --device cuda
record check-plan "$task_python" tools/check_phase5.py "$task_output/plan" --plan-only
record edit "$task_python" tools/check_phase5.py "$task_output" --make-edit
record render tools/with_reference_blas.sh target/release/yue2 render --request "$task_request" --abc-file "$task_output/edited.abc" --output "$task_output/render" --device cuda
record check-render "$task_python" tools/check_phase5.py "$task_output/render" --abc-file "$task_output/edited.abc"
record python-eager "$task_python" -u tools/measure_python_eager.py --end-to-end --output "$task_output/python-eager"
record python-p2 "$task_python" -u tools/measure_python_eager.py --output "$task_output/python-p2"
# All inherited gates are rerun at the end, using the original tolerances/oracles.
record p1 tools/with_reference_blas.sh cargo test --features cuda --test parity -- --ignored --nocapture --test-threads=1
record p2 tools/with_reference_blas.sh cargo test --features cuda --test phase2 -- --ignored --nocapture --test-threads=1
record p3 tools/with_reference_blas.sh cargo test --features cuda --test phase3 -- --ignored --nocapture --test-threads=1
record p4 tools/with_reference_blas.sh cargo test --features cuda --test phase4 -- --ignored --nocapture --test-threads=1
record p4-audio "$task_python" tools/check_phase4_audio.py --root "$task_fixtures"
record native-audio env YUE2_P5_AUDIO_DIR="$task_output/native-audio" tools/with_reference_blas.sh cargo test --features cuda --test phase5 -- --ignored --nocapture --test-threads=1
record check-native-audio "$task_python" tools/check_phase5.py "$task_output/native-audio" --audio-export
kill "$task_monitor" 2>/dev/null || true
wait "$task_monitor" 2>/dev/null || true
trap - EXIT
exit "$task_status"
