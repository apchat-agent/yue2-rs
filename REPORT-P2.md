# Phase 2 report

Phase 2 implementation and all requested gates were executed. **P2a is not
satisfied:** the generated ABC parses and meets the length and key/meter checks,
but adds an `% outro` section absent from the reference. P2b passes; P2c was
measured on GPU 0. The failing assertion remains executable. No thresholds,
reference artifacts, request seed, or sampling settings were changed to obtain
a pass. No Phase 3–5 implementation or push was performed.

## What changed

- `sampling`: eager `generate_tokens`, vocabulary and minimum-length masks,
  frequency-powered sliding-window repetition penalty, temperature, top-k with
  boundary ties, top-p with the Python strict cumulative-mass comparison, and
  categorical/greedy draws. Prefixes do not enter repetition history.
- CFG retains separate positive and negative StaticKVCache instances and feeds
  each the same output tokens. Historical BF16 subtraction/multiply/add boundaries
  are preserved before any distribution upcast. Two oracle-detected rounding
  details were fixed: BF16 scalar-base exponentiation in the repetition penalty,
  and FP32 scalar multiplication followed by BF16 rounding in CFG.
- The CUDA RNG uses seeded candle tensor draws, captured per request before
  generation. Each stage resets its seed. CPU uses a documented seeded fallback
  because candle 0.11's CPU `set_seed` returns an error. `half` and `rand` were
  already transitive dependencies; no new dependency versions were introduced.
- `pipeline`: public `plan` and `generate_semantic`, `SymbolicPlan` and
  `SemanticResult`. Supports provided ABC, cot=off, exact saved-plan IDs without
  decode/re-encode, request guidance and negative prefixes, codec-offset removal,
  callbacks including EOS, cancellation, context errors, and independent ABC/
  semantic truncation flags. The AR backbone and Phase 1 test implementations
  are unchanged.
- Timing follows Python: synchronized prefill and total generation time, time
  to first token, content/output counts, prefix length and CFG branch count.
  Output throughput includes EOS and prefill, excludes model loading and the
  request RNG setup. Execution remains eager with plain tensor SDPA.
- Four ordinary CPU regression tests and three ignored fixture tests cover
  sampling, pipeline behavior, the Python distribution/CFG oracles, and full
  P2a/P2b/P2c generation. Fixture tests return cleanly when `YUE2_FIXTURES` is unset.
- `tools/dump_reference.py --phase2-sampling` adds a separate safetensors/JSON
  supplement using the installed Python implementation; existing P0/P1 fixtures
  are untouched. `tools/check_phase2.py` invokes the existing read-only native
  ABC parser at `~/yue2/YuE/skills/yue2-music/scripts/abc_tools.py` and checks
  section tags and key/meter field counts without repairing the generated score.
- `tools/audit_phase2.sh` performs a clean rebuild and retains all gate failures.
  Generated score/plan/semantic review artifacts live in
  `~/work/yue2-rs-fixtures/first-song/p2/`; logs live in `~/work/yue2-rs-logs/`.
  NAR, VAE, storage, and the final CLI remain for their assigned phases.

## Audited gates

All results below are from the clean self-audit, using seed **831001** and the
unchanged generation settings in the saved first-song config (ABC:
temperature 0.7, top-p 0.9, top-k 30, penalty 1.005/window 100, min 32/max 4096;
semantic: temperature 1, top-p 0.95, top-k 100, penalty 1.2/window 50,
min 200/max 9000).

| Gate | Clean-audit result |
| --- | --- |
| P2a ABC parsing | PASS: both voices parse; Rust 24 measures per voice, reference 22 |
| P2a section tags | **FAIL:** intro/verse/chorus/**outro** versus intro/verse/chorus |
| P2a key/meter line counts | PASS: K=1, M=1 in both scores |
| P2a length and completion | PASS: 617/481 tokens, ratio 1.282744283 (+28.274%); not truncated |
| P2b semantic length | PASS: 1495/1484 tokens, ratio 1.007412399 (+0.741%) |
| P2b range and flags | PASS: codec indices 67–32623, all within 0–32767; not truncated; EOS excluded from content |
| P2c plan | **74.693410563 tok/s**, 618 outputs including EOS in 8.273822220 s |
| P2c semantic | **66.647384880 tok/s**, 1496 outputs including EOS in 22.446492127 s |

Semantic generation uses the exact Python plan and its 611-token prefix, not
Rust's newly sampled plan. Its raw codec-token range is 151920–184476 before
subtracting CODEC_OFFSET. ABC prefill/TTFT were 0.184513986/0.187044582 seconds;
semantic prefill/TTFT were 0.069401118/0.071675491 seconds.

P2c uses BF16 on GPU 0 with the existing optimized test profile (opt-level=2).
Against TASK's supplied 162/186 tok/s figures, measured throughput is about
46.1%/35.8%; see the baseline-label objection below. This phase has no minimum
throughput threshold.

The 11 Python sampling cases match the surviving token set exactly on both CPU
and CUDA. Maximum finite score difference is 0 for all legacy BF16 cases and
0.00000047683716 for ordinary semantic cases. The 1024-element BF16 CFG oracle
matches exactly on both devices.

## Exact commands, output, and self-audit

After the implementation's ordinary tests and full GPU gate invocation
(`p2-final-initial-test.log`, `p2-final-initial-gates.log`), ran **cargo clean**
and **cargo test**, then reran every Phase 2 gate. All quoted gate numbers come
from this second run. No implementation or gate code changed after this audit.
The script exits **1**, preserving P2a's failure; its gate test exits **101**.

Exact top-level command from the repository root:

```bash
export PATH=/home/ayourtch/.cargo/bin:/usr/local/cuda/bin:$PATH CUDA_VISIBLE_DEVICES=0 CUDA_COMPUTE_CAP=120 YUE2_FIXTURES=/home/ayourtch/work/yue2-rs-fixtures
./tools/audit_phase2.sh > /home/ayourtch/work/yue2-rs-logs/p2-audit-commands.log 2>&1
```

The script exports `YUE2_TEST_DEVICE=cuda`. Exact command/exit transcript:

```text
clean command: cargo clean
clean exit=0 log=/home/ayourtch/work/yue2-rs-logs/p2-audit-clean.log
test command: cargo test
test exit=0 log=/home/ayourtch/work/yue2-rs-logs/p2-audit-test.log
build-cpu command: cargo build
build-cpu exit=0 log=/home/ayourtch/work/yue2-rs-logs/p2-audit-build-cpu.log
build-cuda command: cargo build --features cuda
build-cuda exit=0 log=/home/ayourtch/work/yue2-rs-logs/p2-audit-build-cuda.log
fmt command: cargo fmt --check
fmt exit=0 log=/home/ayourtch/work/yue2-rs-logs/p2-audit-fmt.log
clippy command: cargo clippy --all-targets -- -D warnings
clippy exit=0 log=/home/ayourtch/work/yue2-rs-logs/p2-audit-clippy.log
clippy-cuda command: cargo clippy --features cuda --all-targets -- -D warnings
clippy-cuda exit=0 log=/home/ayourtch/work/yue2-rs-logs/p2-audit-clippy-cuda.log
unset-fixtures command: env -u YUE2_FIXTURES cargo test -- --ignored --nocapture --test-threads=1
unset-fixtures exit=0 log=/home/ayourtch/work/yue2-rs-logs/p2-audit-unset-fixtures.log
gpu command: nvidia-smi --query-gpu=index\,memory.used\,memory.free --format=csv
gpu exit=0 log=/home/ayourtch/work/yue2-rs-logs/p2-audit-gpu.log
dump command: env PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 /home/ayourtch/yue2/.venv/bin/python -u tools/dump_reference.py --phase2-sampling
dump exit=0 log=/home/ayourtch/work/yue2-rs-logs/p2-audit-dump.log
sampling-cpu command: env YUE2_TEST_DEVICE=cpu cargo test --test phase2 sampling_python_oracle -- --ignored --nocapture
sampling-cpu exit=0 log=/home/ayourtch/work/yue2-rs-logs/p2-audit-sampling-cpu.log
pipeline-cpu command: env YUE2_TEST_DEVICE=cpu cargo test --test phase2 pipeline_modes_and_truncation -- --ignored --nocapture
pipeline-cpu exit=0 log=/home/ayourtch/work/yue2-rs-logs/p2-audit-pipeline-cpu.log
gpu-before-generation command: nvidia-smi --query-gpu=index\,memory.used\,memory.free --format=csv
gpu-before-generation exit=0 log=/home/ayourtch/work/yue2-rs-logs/p2-audit-gpu-before-generation.log
gates command: tools/with_reference_blas.sh cargo test --features cuda --test phase2 -- --ignored --nocapture --test-threads=1
gates exit=101 log=/home/ayourtch/work/yue2-rs-logs/p2-audit-gates.log
```

Clean output and CPU build, CUDA build, CPU clippy, CUDA clippy completion
output, respectively (compiler progress omitted):

```text
     Removed 9730 files, 7.0GiB total
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 7.66s
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 45.33s
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 4.48s
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 41.67s
```

`cargo fmt --check`: exit 0, no output. Clean `cargo test` passed all seven
existing core tests plus the four new tests, with nine fixture/diagnostic tests
ignored. New-test and doc-test output:

```text
     Running tests/sampling.rs (target/debug/deps/sampling-d740caaad41f3b95)

running 4 tests
test masks_ties_and_top_p_crossing ... ok
test eos_minimum_budget_and_callback_accounting ... ok
test context_cfg_and_cancellation_are_explicit ... ok
test seeded_cpu_sampling_resets_per_request ... ok

test result: ok. 4 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.05s

   Doc-tests yue2

running 0 tests

test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
```

Unset-fixture behavior for the new tests (the six existing ignored tests also
returned cleanly):

```text
     Running tests/phase2.rs (target/debug/deps/phase2-16a9c9ff99da93b0)

running 3 tests
test p2_full_generation_gates ... SKIP: YUE2_FIXTURES is unset
ok
test pipeline_modes_and_truncation ... SKIP: YUE2_FIXTURES is unset
ok
test sampling_python_oracle ... SKIP: YUE2_FIXTURES is unset
ok

test result: ok. 3 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s
```

Fresh sampling fixture dump:

```text
sampling.safetensors: 12,198,736 bytes; 25 tensors
P2 sampling oracle: 11 distributions and BF16 CFG; original seed=831001
Total fixture bytes including supplements: 269,180,406 <= 1,500,000,000
```

GPU availability immediately before full generation:

```text
index, memory.used [MiB], memory.free [MiB]
0, 62432 MiB, 34858 MiB
1, 74354 MiB, 22936 MiB
2, 89069 MiB, 8220 MiB
```

The script samples process memory every 500 ms with:

```bash
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_gpu_memory --format=csv -lms 500
```

The AR test process (PID 2858352) peaked at **5932 MiB observed** on GPU 0,
well below the approximately 30 GB limit. PID 2564265 is the pre-existing server.
This is sampled process usage, not an allocator-instrumented peak. Full samples
are in `p2-audit-memory.log`.

All Phase 2 CUDA gate output, including the retained failure (dependency/compiler
progress omitted):

```text
     Running tests/phase2.rs (target/debug/deps/phase2-f42cfe68f873d558)

running 3 tests
test p2_full_generation_gates ... Abc: 250 output tokens
Abc: 500 output tokens
P2a ABC tokens=617/481 ratio=1.282744283; truncated=false
P2c plan timing: {"attention":"sdpa","cfg_branches":1,"content_tokens":617,"execution":"eager","output_tokens":618,"output_tps":74.69341056254893,"prefill_seconds":0.184513986,"prefix_tokens":128,"seconds":8.27382222,"ttft_seconds":0.187044582}
P2a reference ABC parse PASS: Vocal=22 measures/56 notes, Ins=22 measures/24 notes
P2a Rust ABC parse PASS: Vocal=24 measures/61 notes, Ins=24 measures/34 notes
P2a sections: Rust=['% intro', '% verse', '% chorus', '% outro'], reference=['% intro', '% verse', '% chorus']
P2a key/meter line counts: Rust={'K': 1, 'M': 1}, reference={'K': 1, 'M': 1}
Traceback (most recent call last):
  File "/home/ayourtch/work/yue2-rs/tools/check_phase2.py", line 38, in <module>
    main()
  File "/home/ayourtch/work/yue2-rs/tools/check_phase2.py", line 30, in main
    assert sections(actual) == sections(reference), (sections(actual), sections(reference))
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: (['% intro', '% verse', '% chorus', '% outro'], ['% intro', '% verse', '% chorus'])
Semantic: 250 output tokens
Semantic: 500 output tokens
Semantic: 750 output tokens
Semantic: 1000 output tokens
Semantic: 1250 output tokens
P2b semantic tokens=1495/1484 ratio=1.007412399; codec min=67 max=32623; truncated=false
P2c semantic timing: {"attention":"sdpa","cfg_branches":1,"content_tokens":1495,"execution":"eager","output_tokens":1496,"output_tps":66.64738488026468,"prefill_seconds":0.069401118,"prefix_tokens":611,"seconds":22.446492127,"ttft_seconds":0.071675491}
P2b PASS; codec range, exact Python plan, length and EOS accounting valid

thread 'p2_full_generation_gates' (2858353) panicked at tests/phase2.rs:304:5:
P2a ABC parser/structure gate failed
note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace
FAILED
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

failures:

failures:
    p2_full_generation_gates

test result: FAILED. 2 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out; finished in 31.92s

error: test failed, to rerun pass `--test phase2`
```

The CPU sampling-oracle rerun printed the same 11 support/error values and
1024/1024 CFG result shown above, then:

```text
test sampling_python_oracle ... ok

test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 2 filtered out; finished in 0.04s
```

The CPU pipeline-control rerun:

```text
running 1 test
Pipeline controls PASS: provided ABC, cot=off CFG, exact prefix validation, independent plan/semantic truncation and codec subtraction
test pipeline_modes_and_truncation ... ok

test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 2 filtered out; finished in 0.14s
```

The initial final-implementation artifacts were retained at
`first-song/p2-before-audit/`. Post-audit comparison of the JSON token arrays
confirms the sampled content repeated exactly across the clean build:

```bash
jq -n --slurpfile before /home/ayourtch/work/yue2-rs-fixtures/first-song/p2-before-audit/plan.json --slurpfile after /home/ayourtch/work/yue2-rs-fixtures/first-song/p2/plan.json '{abc_ids_exact: ($before[0].abc_ids == $after[0].abc_ids), prefix_exact: ($before[0].prefix == $after[0].prefix), abc_count: ($after[0].abc_ids | length)}'
jq -n --slurpfile before /home/ayourtch/work/yue2-rs-fixtures/first-song/p2-before-audit/semantic.json --slurpfile after /home/ayourtch/work/yue2-rs-fixtures/first-song/p2/semantic.json '{semantic_tokens_exact: ($before[0].tokens == $after[0].tokens), semantic_count: ($after[0].tokens | length)}'
```

```text
{
  "abc_ids_exact": true,
  "prefix_exact": true,
  "abc_count": 617
}
{
  "semantic_tokens_exact": true,
  "semantic_count": 1495
}
```

## Unsure list and design objections

1. **P2a fails the exact section-tag requirement.** Rust produces intro, verse,
   chorus, outro; the reference has intro, verse, chorus. The generated score
   remains unchanged. Different candle/torch random streams and floating-point
   reductions make sampled structural identity uncertain even with the same
   numeric seed. No alternate seed, reranking, section filter, or retry-until-pass
   mechanism was introduced. The design explicitly disclaims bit-exact sampling,
   but this does not waive the stricter structural gate.
2. **CPU RNG exception:** candle-core 0.11 cannot seed its CPU tensor RNG
   (`cpu_backend/mod.rs`, `set_seed`: "cannot seed the CPU rng with set_seed").
   CPU therefore uses `rand::rngs::StdRng`, the same rand dependency candle uses,
   seeded per request. CUDA uses candle's own seeded device RNG. CPU and CUDA
   random streams are not claimed to match; repeated CPU requests are tested.
   Capturing CUDA draws under a mutex isolates this crate's generators from one
   another, but unrelated callers using the same device RNG are not coordinated.
3. Sampling filtering and cumulative sums run on the host: candle's full-vocabulary
   CUDA sort requires more shared memory than is available, and its `cumsum`
   constructs a quadratic matrix. Host float32 reductions can differ slightly
   from torch's parallel reductions; tied top-p candidates use ascending token
   IDs where torch sort tie order is unspecified. These are sampling-numerics
   limitations, not teacher-forcing changes. Host transfers and the existing
   eager attention operations limit throughput; Phase 5 speed work was not done.
4. **The supplied throughput baseline is mislabeled as eager in TASK.md.** The
   immutable first-song `result.json` records the quoted 162.185/186.465 tok/s
   with `execution="cuda_graph"` and `attention="flash"`. This report compares
   against the supplied figures but does not call that an eager-to-eager test.
   No new Python eager throughput measurement is claimed.
5. CUDA runs use the existing `tools/with_reference_blas.sh` helper to select
   the reference cuBLAS 12.8.4.1, as required by the Phase 1 findings. System
   cuBLAS, Metal, flash-attn, and full real-checkpoint CFG numerical parity are
   not newly certified here. CFG arithmetic is checked against Python on both
   CPU and CUDA, and CFG control flow is exercised with a tiny CPU backbone.
   Full sampled CPU generation was not a GPU throughput gate and was not run.
6. No other objections to the fixed architecture. The inherited P1c/P1d caveats
   remain documented in REPORT-P1.md; Phase 2 does not alter or resolve them.
