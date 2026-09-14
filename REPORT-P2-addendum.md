# Phase 2 addendum

Applies the 2026-09-14 TASK.md revision during Phase 3. REPORT-P2.md remains the
historical report; its exact-section failure is superseded by the advisory gate.
Phase 2 generation and sampling fixtures were not repeated or changed.

## P2a: parsing required, structure advisory

Removed the section/key/meter equality assertions from `tools/check_phase2.py`
and the length/truncation assertion from the full-generation test. The test still
fails on a parser error. It reports tags, key/meter counts, token ratio and
truncation without filtering or repairing the score.

Initial check and clean P3 self-audit both ran:

```bash
env PYTHONDONTWRITEBYTECODE=1 /home/ayourtch/yue2/.venv/bin/python tools/check_phase2.py /home/ayourtch/work/yue2-rs-fixtures/first-song/p2
```

Clean-audit output (exit 0):

```text
P2a reference ABC parse PASS: Vocal=22 measures/56 notes, Ins=22 measures/24 notes
P2a Rust ABC parse PASS: Vocal=24 measures/61 notes, Ins=24 measures/34 notes
P2a sections: Rust=['% intro', '% verse', '% chorus', '% outro'], reference=['% intro', '% verse', '% chorus']
P2a key/meter line counts: Rust={'K': 1, 'M': 1}, reference={'K': 1, 'M': 1}
P2a advisory length: 617/481, ratio=1.282744283, target within 30%=True, truncated=False
P2a PASS: ABC parses; section tags and key/meter counts are advisory
```

## P2c: measured Python eager baseline

Ran **once**, as requested, on GPU 0 using Python yue2-infer 0.1.6,
torch 2.10.0+cu128, BF16, `backend="torch-eager"`, and native `attention="sdpa"`.
The original first-song request, seed **831001**, and saved generation settings
were unchanged. Plan generation starts from the request. Semantic generation uses
the exact saved Python plan and 611-token prefix, matching the Rust P2 gate.
Neither the VAE nor NAR was run for this measurement.

```bash
env PYTHONDONTWRITEBYTECODE=1 HF_HOME=/home/ayourtch/yue2/hf HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 /home/ayourtch/yue2/.venv/bin/python -u tools/measure_python_eager.py > /home/ayourtch/work/yue2-rs-logs/p2-python-eager.log 2>&1
```

Exit 0. Pasted timing output:

```json
  "abc": {
    "seconds": 4.408676960039884,
    "prefill_seconds": 0.19018075801432133,
    "ttft_seconds": 0.3365083810640499,
    "output_tokens": 482,
    "content_tokens": 481,
    "output_tps": 109.32985210049037,
    "prefix_tokens": 128,
    "cfg_branches": 1,
    "execution": "eager",
    "attention": "sdpa"
  },
  "semantic": {
    "seconds": 12.708577918005176,
    "prefill_seconds": 0.014216193929314613,
    "ttft_seconds": 0.014576992020010948,
    "output_tokens": 1499,
    "content_tokens": 1498,
    "output_tps": 117.95182825894757,
    "prefix_tokens": 611,
    "cfg_branches": 1,
    "execution": "eager",
    "attention": "sdpa"
  },
```

| Stage | Rust eager, committed P2 audit | Python eager, measured once | Python CUDA graph + flash, saved result | Rust / Python eager | Rust / Python graph |
| --- | ---: | ---: | ---: | ---: | ---: |
| Plan | 74.693410563 tok/s | 109.329852100 tok/s | 162.185329927 tok/s | 68.32% | 46.05% |
| Semantic | 66.647384880 tok/s | 117.951828259 tok/s | 186.465148819 tok/s | 56.50% | 35.74% |

The package synchronizes each stage's timer; throughput includes prefill and EOS,
excludes model loading, and is not a decode-only rate. Both stages completed
without truncation. Peak torch allocated/reserved memory was
**8,406,090,752 / 8,518,631,424 bytes**, within the 24 GiB pipeline budget.
The GPU had 34,858 MiB free immediately before the run.

The complete measurement and generated artifacts are in
`~/work/yue2-rs-fixtures/first-song/p2-eager/`; log:
`~/work/yue2-rs-logs/p2-python-eager.log`. This is a single observation on shared
hardware, with different sampled output lengths and no variance estimate. It was
intentionally not rerun during the clean P3 audit, honoring the one-run request.
