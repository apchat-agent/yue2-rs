# Phase 3b: NAR divergence root cause

**P3 still FAILS. The evidence identifies numerical backend differences in BF16,
then amplification through the nonlinear midpoint trajectory, rather than a
structural NAR port error.** No tolerance, original fixture, production dtype,
attention backend, context cut, noise, or solver step was changed to obtain a
pass. This report proposes gate wording for the owner; it does not enact it.

The first stage above **1e-3** is **AR prefill layer 0, Q after RoPE**:
**0.00390625**, in **9 / 2,772,992** values. K after RoPE differs by
**0.001953125**, in **3 / 1,386,496** values. The embedding, input RMSNorm,
Q/K/V projections and Q/K norms before that rotation are **exact**. The first
acoustic-input projection, `vae2llm`, independently crosses the threshold at
**0.00390625**; the timestep MLP's second linear crosses at **0.001953125**.
Thus neither the first completed transformer layer nor the final latent is the
first divergent operation.

The decisive control uses the same checkpoint values, noise, boundaries, masks,
positions and 32 midpoint steps in **FP32 in both languages**. Final latent error
falls to **0.000020578503609** for chunk 0 and **0.000025272369385** for chunk 1.
This control is diagnostic; it is not substituted for the required BF16 P3 run.

## Method and reproducible artifacts

Read TASK.md and REPORT-P3.md in full, and read the installed `nar.py` and relevant
`modeling_yue2.py` implementations before tracing. The baseline is bfc95e0.

`tools/dump_reference.py --nar-stages` invokes `tools/dump_nar_stages.py` and adds
separate safetensors and a dtype/configuration/hash JSON sidecar. Hooks observe
the **installed** CachedNAR prefill and first velocity. Since CachedNAR calls
layer components directly, layer outputs are observed at the final residual
addition using the actual pre-MLP residual and actual MLP output; injection is
also checked against the actual layer-0 input. The first velocity is asserted
bit-identical to the original P0 oracle. The existing `trace_chunk` observes
all 64 real velocity calls, raw times, 32 initial states, midpoint states, updated
states and final latents, and asserts each saved original intermediate is exact.
The P0 `nar.safetensors` SHA-256 is checked before and after the diagnostic.

Rust's ignored `nar_stage_dump` test records matching native-dtype stages in
`src/model/nar_stages.rs`. Its staged prefill cache and first velocity are
asserted bit-identical to the production methods. A private observer on the
**production solver** captures its unmodified, freely evolving trajectory.
Reference inputs are used only in separately named `isolated.*` operation
controls, never inside that trajectory. `shifted_time` was extracted without
changing its arithmetic so the production operation itself can be checked.

The layer table covers every one of the **28 AR prefill layers** and **28 NAR
layers at the first velocity call**. All subsequent midpoint iterations have
state, velocity and timestep dumps. Dumping every layer at all 64 calls would
exceed the fixture budget; first-operation localization does not need that.
`tools/compare_nar_stages.py` checks shapes/dtypes/finite values, computes errors
in FP64, and emits full stage, isolated-operation and solver tables plus JSON.

Artifacts are under `~/work/yue2-rs-fixtures/first-song/`:

- `p3b-python.safetensors`, `p3b-python.json`: production Python trace and
  explicitly named numerical controls.
- `p3b-rust.safetensors`: production Rust trace and same-input controls.
- `p3b-rust-fp32.safetensors`: separate full FP32 trajectories for both chunks.
- `p3b-comparison.json`: all measured stage metrics and FP32 comparisons.
- `p3/latents.safetensors`: the unchanged strict gate's actual Rust BF16 outputs,
  cast to FP32 as required by the API.

The case remains `two_chunk`, seed **831001**, context **2098**, prefix **611**,
**742** frames/chunk, **1354** AR positions and **744** NAR positions including
START/END. This is the original Phase-0 explicit context override, not two
native-context chunks. No new fixture is committed.

## Why this is a numerical difference

1. **RoPE's first mismatch is numerical, not an offset.** Both integer position
   arrays match exactly. CPU `f32::powf` generates Rust's inverse frequencies;
   Python generates them using CUDA FP32 tensor power. Their slight FP32
   differences and CUDA transcendental rounding straddle a BF16 cast boundary.
   Supplying Python sin/cos to Rust's rotation makes every first-layer Q/K
   value exact in both AR and NAR. This is a diagnostic substitution only.
2. **Attention differs even on identical Q/K/V and cache values.** Candle's
   global row maximum/exponential/sum and explicit numerator rounding cannot
   reproduce all rounding from Torch fused SDPA. On identical inputs, AR
   attention differs by **0.00390625** and NAR attention by **0.0625**. A Python
   transcription of candle's plain attention reduces the Rust comparison to
   **29 / 2,772,992** differing AR values and **8 / 1,523,712** NAR values (both
   max **0.001953125**). The visibility and equations agree; floating-point
   reduction/exp implementations still differ at rounding boundaries.
3. **GEMM rounding differs independently of attention.** On the exact Python
   layer-0 inputs, NAR K and V projections each differ by **0.125**; Q and O
   projections are exact. The four biased acoustic auxiliary linears use FP32
   accumulation followed by a single BF16 cast in Rust, while Python dispatches
   BF16 addmm.
   Python's own FP32-linear-then-BF16 control reproduces the `vae2llm` error
   **0.00390625 in exactly 53 values**, and the second timestep linear's
   **0.001953125 in 2232 values**. Disabling Torch's reduced-precision BF16
   reduction changes NAR V on the identical input by **0.125** as well.
   This is evidence of reduction/rounding behavior, not a claim that every
   differing GEMM uses the same internal cuBLAS algorithm.
4. **The solver amplifies these perturbations.** All 32 midpoint/update pairs
   are bit-exact when given identical states and velocities. In the actual
   trajectories, the first midpoint and first updated latent already differ
   by **0.015625**. Subsequent velocities see different states, and chunk-0
   maximum latent error grows to **0.839233398438**. Resetting each step to a
   reference state would hide this; the gate and trajectory dump do not reset.
5. **Controls reproduce the effect entirely within Python**, and **FP32 removes
   it across languages**. See the numerical-control table below. Changing only
   Python attention to math SDPA yields **0.835815429688** chunk-0 error against
   its original fused-SDPA result. Disabling BF16 reduced-precision GEMM alone
   also fails the 0.01 bound. The FP32 cross-language results are about 2e-5.
   Together with the operation controls, this establishes BF16 numerical
   sensitivity as the remaining cause; cosine alone would not establish it.

## Explicit suspect checks

| Suspect | Check and conclusion |
| --- | --- |
| BF16 vs FP32 / implicit upcasts | All checkpoint parameters and audio PE are BF16 in the original run. Every embedding/projection/norm/layer/velocity/state output is saved with its actual BF16 dtype; final API latents are FP32. RMS variance/rsqrt, timestep frequencies/trig and RoPE angles are FP32 internally; the reciprocal RMS is cast back before two BF16 multiplies in both implementations. Torch fused SDPA uses backend-specific internal reductions; Rust's scores, row sum and value accumulation are FP32 with a BF16 numerator boundary. This numerical backend difference is **confirmed**, not ruled out. FP32 does not silently enter the Python solver state. |
| RoPE / audio position off by one | Exact AR positions 0..1353, absolute NAR positions 1354..2097, and local audio positions 0..743. START/END occupy local positions 0/743; the output slice drops precisely them. Loaded audio PE and initial state are exact. Same-sin/cos isolated rotations are exact. **Indexing error ruled out**; tiny numerical sin/cos differences remain. |
| `timestep_shift` | Checkpoint shift is 1. All 64 actual raw solver times and shifted BF16 values match. Additional shifts 0.5, 1 and 3 at raw times 20, 4, 1, 0, -1, -4, -20 match exactly. Both evaluate `shift*sigmoid(raw)/(1+(shift-1)*sigmoid(raw))` with the same BF16 boundaries. **Ruled out** for this gate and these non-default scalar probes. |
| CFG in NAR | Installed `CachedNAR.velocity/solve/synthesize` has one conditional path and no unconditional model call, negative prefix or guidance interpolation. Rust follows that path. CFG belongs to the AR sampler, not this acoustic solver. **Ruled out**. |
| Weight norm / layer norm eps | NAR uses ordinary linear weights and RMSNorm, not VAE weight-normalized convolutions or mean-subtracting LayerNorm. Every Python RMSNorm epsilon and Rust's checkpoint-derived epsilon is 1e-6. Same-input AR/NAR input and Q/K norm controls are exact. **Wrong norm kind/epsilon or missing weight-norm folding ruled out**. Small deeper FP32 reduction differences are covered by the precision control. |
| Attention mask | AR prefill is causal `key_index <= absolute_query_index`, including later 128-row query tiles. NAR is noncausal over all 1354 visible AR keys plus all 744 NAR keys. `nar_cond_end=0` exposes all AR keys for both gate chunks. GQA groups repeat each of 8 KV heads for 16 Q heads. The Python plain-attention transcription uses these exact sets and agrees at the operation level above. **Mask/key-set error ruled out** for this fixture; fused versus plain arithmetic remains. |
| Noise draw order | Python redraw with its private CPU generator reproduces both P0 slices exactly: one full-song FP32 `[1484,64]` draw before slicing. Rust also draws once before cuts using its own RNG; the gate injects the immutable Python noise directly. Chunk-0 noise and BF16 initial state are exact; the existing CUDA ownership/seed/context test passes. **Draw order ruled out**. Cross-library RNG identity is not claimed. |

## Stage measurements from the clean audit

Stage and solver tables use chunk 0; the control/gate table reports both chunks.
Errors are absolute maxima over the entire tensor, not selected positions.
BF16 output values are compared after lossless widening; this does not turn the
underlying computation into FP32. Cosines are printed as computed; excursions
above 1 by about 2e-12 are FP64 reduction roundoff.

Inputs, AR first operations and acoustic injection:

| Stage | Python / Rust dtype | Max abs diff | Cosine |
| --- | --- | ---: | ---: |
| `noise` | float32 | 0 | 1.000000000000 |
| `state` | bfloat16 | 0 | 1.000000000000 |
| `ar.positions` | int64 / uint32 | 0 | 1.000000000000 |
| `nar.positions` | int64 / uint32 | 0 | 1.000000000000 |
| `audio.positions` | int64 / uint32 | 0 | 1.000000000000 |
| `audio.output` | bfloat16 | 0 | 1.000000000000 |
| `rope.inv_freq` | float32 | 1.86264514923e-09 | 1.000000000000 |
| `ar.cos` | float32 | 3.81097197533e-06 | 1.000000000000 |
| `ar.sin` | float32 | 3.81469726562e-06 | 1.000000000000 |
| `ar.embedding.output` | bfloat16 | 0 | 1.000000000000 |
| `ar.layer.00.norm.output` | bfloat16 | 0 | 1.000000000000 |
| `ar.layer.00.q_proj.output` | bfloat16 | 0 | 1.000000000000 |
| `ar.layer.00.k_proj.output` | bfloat16 | 0 | 1.000000000000 |
| `ar.layer.00.v_proj.output` | bfloat16 | 0 | 1.000000000000 |
| `ar.layer.00.q_norm.output` | bfloat16 | 0 | 1.000000000000 |
| `ar.layer.00.k_norm.output` | bfloat16 | 0 | 1.000000000000 |
| `ar.layer.00.q` | bfloat16 | 0.00390625 | 0.999999999996 |
| `ar.layer.00.k` | bfloat16 | 0.001953125 | 0.999999999996 |
| `ar.layer.00.v` | bfloat16 | 0 | 1.000000000000 |
| `ar.layer.00.o_proj.input` | bfloat16 | 0.00390625 | 0.999999860257 |
| `ar.layer.00.o_proj.output` | bfloat16 | 0.125 | 0.999999911814 |
| `nar.cos` | float32 | 3.81469726562e-06 | 1.000000000000 |
| `nar.sin` | float32 | 3.81469726562e-06 | 1.000000000000 |
| `shifted` | bfloat16 | 0 | 1.000000000000 |
| `vae2llm.output` | bfloat16 | 0.00390625 | 0.999999999938 |
| `time.freqs` | float32 | 0 | 1.000000000000 |
| `time.0.input` | bfloat16 | 0 | 1.000000000000 |
| `time.0.output` | bfloat16 | 4.76837158203e-07 | 1.000000000000 |
| `time.1.output` | bfloat16 | 2.38418579102e-07 | 1.000000000000 |
| `time.2.output` | bfloat16 | 0.001953125 | 0.999999915097 |
| `time.output` | bfloat16 | 0.001953125 | 0.999999915097 |
| `with_time` | bfloat16 | 0.0078125 | 0.999999995094 |
| `injected` | bfloat16 | 0.015625 | 0.999999997501 |

Every transformer-layer output (all BF16 in both languages):

| Layer (zero based) | AR max abs | AR cosine | NAR max abs | NAR cosine |
| --- | ---: | ---: | ---: | ---: |
| 0 | 0.25 | 0.999999998506 | 1 | 0.999998732567 |
| 1 | 64 | 0.999998313155 | 1.5 | 0.999998348999 |
| 2 | 64 | 0.999998288856 | 1.5 | 0.999998095057 |
| 3 | 64 | 0.999998304391 | 1.5 | 0.999997741460 |
| 4 | 64 | 0.999998285391 | 1.5 | 0.999996858497 |
| 5 | 64 | 0.999998268758 | 1.5 | 0.999996326351 |
| 6 | 64 | 0.999998238503 | 1.5 | 0.999995826617 |
| 7 | 64 | 0.999998202702 | 1.5 | 0.999995419522 |
| 8 | 64 | 0.999998166891 | 1.5 | 0.999994608556 |
| 9 | 64 | 0.999998035471 | 1.5 | 0.999993910824 |
| 10 | 64 | 0.999997969388 | 1.5 | 0.999989984961 |
| 11 | 64 | 0.999997901718 | 1.5 | 0.999987102348 |
| 12 | 64 | 0.999997814732 | 1.5 | 0.999984596994 |
| 13 | 64 | 0.999997714420 | 1.5 | 0.999980399872 |
| 14 | 64 | 0.999997563379 | 1.5 | 0.999977402659 |
| 15 | 64 | 0.999997364279 | 1.5 | 0.999972309647 |
| 16 | 64 | 0.999997121136 | 1.5 | 0.999967396729 |
| 17 | 64 | 0.999996875952 | 1.5 | 0.999962969225 |
| 18 | 64 | 0.999996577325 | 1.5 | 0.999961024163 |
| 19 | 64 | 0.999996239657 | 1.5 | 0.999958438475 |
| 20 | 64 | 0.999995751661 | 1.75 | 0.999958314099 |
| 21 | 64 | 0.999995112484 | 1.5 | 0.999957769168 |
| 22 | 64 | 0.999994361630 | 2 | 0.999954679327 |
| 23 | 64 | 0.999993295462 | 3 | 0.999954600196 |
| 24 | 64 | 0.999991760069 | 3 | 0.999954731153 |
| 25 | 64 | 0.999989833833 | 3 | 0.999964522351 |
| 26 | 64 | 0.999986040867 | 4 | 0.999977427791 |
| 27 | 64 | 0.999979801686 | 6 | 0.999977576837 |

| Stage | Python / Rust dtype | Max abs diff | Cosine |
| --- | --- | ---: | ---: |
| `final_norm.output` | bfloat16 | 0.25 | 0.999950325741 |
| `projection.output` | bfloat16 | 0.0625 | 0.999955503454 |
| `velocity` | bfloat16 | 0.0625 | 0.999955537267 |
| `latents` | float32 | 0.839233398438 | 0.999378217075 |

Isolated operations use the **identical Python input**, so these errors do not
include inherited state or cache divergence. Norms grouped below are each exact:
AR/NAR input RMSNorm and AR/NAR Q/K RMSNorm (six operations).

| Stage | Python / Rust dtype | Max abs diff | Cosine |
| --- | --- | ---: | ---: |
| `isolated.ar.layer.00.q_proj.output` | bfloat16 | 0 | 1.000000000000 |
| `isolated.ar.layer.00.k_proj.output` | bfloat16 | 0 | 1.000000000000 |
| `isolated.ar.layer.00.v_proj.output` | bfloat16 | 0 | 1.000000000000 |
| `isolated.ar.layer.00.o_proj.input` | bfloat16 | 0.00390625 | 0.999999860257 |
| `isolated.ar.layer.00.o_proj.output` | bfloat16 | 0 | 1.000000000002 |
| `isolated.nar.layer.00.q_proj.output` | bfloat16 | 0 | 1.000000000000 |
| `isolated.nar.layer.00.k_proj.output` | bfloat16 | 0.125 | 0.999996410577 |
| `isolated.nar.layer.00.v_proj.output` | bfloat16 | 0.125 | 0.999995994320 |
| `isolated.nar.layer.00.o_proj.input` | bfloat16 | 0.0625 | 0.999999811766 |
| `isolated.nar.layer.00.o_proj.output` | bfloat16 | 0 | 1.000000000000 |
| `isolated.own_rope.ar.layer.00.q` | bfloat16 | 0.00390625 | 0.999999999996 |
| `isolated.own_rope.ar.layer.00.k` | bfloat16 | 0.001953125 | 0.999999999996 |
| `isolated.own_rope.nar.layer.00.q` | bfloat16 | 0.0078125 | 0.999999999814 |
| `isolated.own_rope.nar.layer.00.k` | bfloat16 | 0.015625 | 0.999999999539 |
| `isolated.reference_rope.ar.layer.00.q` | bfloat16 | 0 | 1.000000000000 |
| `isolated.reference_rope.ar.layer.00.k` | bfloat16 | 0 | 1.000000000000 |
| `isolated.reference_rope.nar.layer.00.q` | bfloat16 | 0 | 1.000000000000 |
| `isolated.reference_rope.nar.layer.00.k` | bfloat16 | 0 | 1.000000000000 |
| `isolated.vae2llm.output` | bfloat16 | 0.00390625 | 0.999999999938 |
| `isolated.time.0.output` | bfloat16 | 4.76837158203e-07 | 1.000000000000 |
| `isolated.time.2.output` | bfloat16 | 0.001953125 | 0.999999915097 |
| `isolated.projection.output` | bfloat16 | 0.03125 | 0.999995239976 |

All 32 actual sampling steps (all five intermediate columns are BF16 in both
implementations). Each cell is max absolute diff; the final API cast is FP32:

| Step | State | First velocity | Midpoint | Second velocity | Updated latent | Updated cosine |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | 0.0625 | 0.015625 | 0.0625 | 0.015625 | 0.999999358986 |
| 1 | 0.015625 | 0.0625 | 0.015625 | 0.078125 | 0.015625 | 0.999998646674 |
| 2 | 0.015625 | 0.0625 | 0.015625 | 0.0732421875 | 0.03125 | 0.999997837575 |
| 3 | 0.03125 | 0.078125 | 0.03125 | 0.08203125 | 0.03125 | 0.999996867303 |
| 4 | 0.03125 | 0.109375 | 0.03125 | 0.1171875 | 0.03125 | 0.999995669119 |
| 5 | 0.03125 | 0.125 | 0.03125 | 0.19140625 | 0.03125 | 0.999994115238 |
| 6 | 0.03125 | 0.228515625 | 0.03125 | 0.31640625 | 0.03125 | 0.999992329763 |
| 7 | 0.03125 | 0.3203125 | 0.0390625 | 0.42578125 | 0.0390625 | 0.999990151669 |
| 8 | 0.0390625 | 0.515701293945 | 0.046875 | 0.57568359375 | 0.05859375 | 0.999986986937 |
| 9 | 0.05859375 | 0.66796875 | 0.06640625 | 0.7265625 | 0.08203125 | 0.999982359201 |
| 10 | 0.08203125 | 0.76953125 | 0.09375 | 0.775390625 | 0.107421875 | 0.999976187054 |
| 11 | 0.107421875 | 0.75390625 | 0.119140625 | 0.75390625 | 0.12890625 | 0.999967510599 |
| 12 | 0.12890625 | 0.796875 | 0.140625 | 0.796875 | 0.15625 | 0.999956484007 |
| 13 | 0.15625 | 0.732421875 | 0.16796875 | 0.759765625 | 0.177734375 | 0.999942191120 |
| 14 | 0.177734375 | 0.85888671875 | 0.189453125 | 0.943908691406 | 0.19921875 | 0.999924152462 |
| 15 | 0.19921875 | 1.037109375 | 0.2099609375 | 1.068359375 | 0.2236328125 | 0.999902218287 |
| 16 | 0.2236328125 | 1.134765625 | 0.234375 | 1.171875 | 0.248046875 | 0.999876037153 |
| 17 | 0.248046875 | 1.2109375 | 0.2587890625 | 1.3125 | 0.26953125 | 0.999845429233 |
| 18 | 0.26953125 | 1.3046875 | 0.2802734375 | 1.40234375 | 0.2919921875 | 0.999810211698 |
| 19 | 0.2919921875 | 1.390625 | 0.30322265625 | 1.4296875 | 0.313720703125 | 0.999772031119 |
| 20 | 0.313720703125 | 1.48828125 | 0.324340820312 | 1.59375 | 0.3486328125 | 0.999729915387 |
| 21 | 0.3486328125 | 1.64453125 | 0.373046875 | 1.69921875 | 0.400390625 | 0.999685817151 |
| 22 | 0.400390625 | 1.74609375 | 0.4267578125 | 1.71484375 | 0.4521484375 | 0.999640376736 |
| 23 | 0.4521484375 | 1.70703125 | 0.478515625 | 1.69921875 | 0.50390625 | 0.999595401879 |
| 24 | 0.50390625 | 1.67578125 | 0.52978515625 | 1.6171875 | 0.55615234375 | 0.999551733980 |
| 25 | 0.55615234375 | 1.59375 | 0.58203125 | 1.60546875 | 0.6044921875 | 0.999511200997 |
| 26 | 0.6044921875 | 1.58984375 | 0.6298828125 | 1.53125 | 0.650390625 | 0.999473450677 |
| 27 | 0.650390625 | 1.48046875 | 0.675048828125 | 1.4453125 | 0.692626953125 | 0.999441170240 |
| 28 | 0.692626953125 | 1.4375 | 0.715270996094 | 1.3671875 | 0.733520507812 | 0.999415082773 |
| 29 | 0.733520507812 | 1.359375 | 0.755249023438 | 1.25390625 | 0.773803710938 | 0.999396380484 |
| 30 | 0.773803710938 | 1.16796875 | 0.792724609375 | 1.099609375 | 0.809692382812 | 0.999383874881 |
| 31 | 0.809692382812 | 1.025390625 | 0.826171875 | 0.9404296875 | 0.839233398438 | 0.999378217075 |

All **64 raw FP64 times**, **64 shifted BF16 times**, **64 isolated midpoint/update
results**, and **21 alternate-shift scalar probes** are exact. The comparison
script asserts the noise, initial state, audio embedding and position indices
are exact too. All 64 solver velocity outputs are BF16; no hidden state upcast.

Numerical controls (each row compares complete final trajectories):

| Comparison | Chunk 0 max abs | Chunk 0 cosine | Chunk 1 max abs | Chunk 1 cosine |
| --- | ---: | ---: | ---: | ---: |
| Python BF16, reduced GEMM precision disabled vs original Python | 0.182739257812 | 0.999967810598 | 0.173828125 | 0.999972756509 |
| Python BF16 math SDPA vs original Python | 0.835815429688 | 0.999443163343 | 0.140625 | 0.999939655858 |
| Python BF16 candle attention transcription vs original Python | 0.768310546875 | 0.999640686051 | 0.24853515625 | 0.999949423326 |
| Rust FP32 vs Python FP32 math SDPA, separate diagnostic | 2.05785036087e-05 | 1.000000000000 | 2.52723693848e-05 | 1.000000000000 |
| **Strict P3: Rust BF16 vs original Python BF16** | **0.839233398438 FAIL** | **0.999378217075** | **0.269531250000 FAIL** | **0.999945369729** |

The numerical control rows retain the original parameters, mask, boundaries and
noise. The two attention controls alter attention arithmetic in **both AR prefill
and NAR velocity**, so they include its effect on the invariant caches.

## Proposed gate wording — not applied

> With identical checkpoint values, dumped noise, original chunk boundaries and
> 32 midpoint steps, require a paired FP32 Python/Rust diagnostic to have final
> max absolute latent error <= 1e-3 and cosine >= 0.999999 for each chunk. For
> production CUDA BF16 against the original Python BF16 oracle, require cosine
> >= 0.999 for each chunk and report maximum absolute error, native stage dtypes,
> the first divergent operation and same-Python numerical controls. Treat the
> BF16 absolute-error bound as advisory pending owner-approved calibration on a
> representative corpus.

This is a proposed distinction between algorithmic parity and backend-dependent
BF16 trajectory parity. It is **not** an audio-quality gate or evidence that an
arbitrary absolute error is acceptable. If strict BF16 max <= 0.01 is retained,
P3 remains blocked on matching Torch's numerical kernels much more closely;
the inherited AR cosine pass is insufficient. TASK.md and tests/phase3.rs are
unchanged, and the audit deliberately returns failure for P3.

## Commands, validation and self-audit

The initial diagnostic and strict gate were run, then `tools/audit_phase3b.sh`
ran **cargo clean**, a fresh **cargo test**, CPU/CUDA builds, fmt, CPU/CUDA Clippy,
unset-fixture checks, a fresh Python dump, Rust dump, comparison and strict P3.
All numerical values in this report come from that clean audit. Every shared
stage metric and both FP32 results match the initial run exactly. The audit exits
**1**, because the unchanged strict P3 test exits **101**.

Exact top-level command:

```bash
env PATH=/home/ayourtch/.cargo/bin:/usr/local/cuda/bin:$PATH CUDA_VISIBLE_DEVICES=0 CUDA_COMPUTE_CAP=120 YUE2_FIXTURES=/home/ayourtch/work/yue2-rs-fixtures tools/audit_phase3b.sh > /home/ayourtch/work/yue2-rs-logs/p3b-audit-commands.log 2>&1
```

The script exports `YUE2_TEST_DEVICE=cuda`, `YUE2_NAR_DIAGNOSTIC=1`,
`PYTHONDONTWRITEBYTECODE=1`, and `HF_HUB_OFFLINE=1`. Exact command/exit transcript:

```text
clean command: cargo clean
clean exit=0 log=/home/ayourtch/work/yue2-rs-logs/p3b-audit-clean.log
test command: cargo test
test exit=0 log=/home/ayourtch/work/yue2-rs-logs/p3b-audit-test.log
build-cpu command: cargo build
build-cpu exit=0 log=/home/ayourtch/work/yue2-rs-logs/p3b-audit-build-cpu.log
build-cuda command: cargo build --features cuda
build-cuda exit=0 log=/home/ayourtch/work/yue2-rs-logs/p3b-audit-build-cuda.log
fmt command: cargo fmt --check
fmt exit=0 log=/home/ayourtch/work/yue2-rs-logs/p3b-audit-fmt.log
clippy command: cargo clippy --all-targets -- -D warnings
clippy exit=0 log=/home/ayourtch/work/yue2-rs-logs/p3b-audit-clippy.log
clippy-cuda command: cargo clippy --features cuda --all-targets -- -D warnings
clippy-cuda exit=0 log=/home/ayourtch/work/yue2-rs-logs/p3b-audit-clippy-cuda.log
unset-fixtures command: env -u YUE2_FIXTURES cargo test -- --ignored --nocapture --test-threads=1
unset-fixtures exit=0 log=/home/ayourtch/work/yue2-rs-logs/p3b-audit-unset-fixtures.log
gpu command: nvidia-smi --query-gpu=index\,memory.used\,memory.free --format=csv
gpu exit=0 log=/home/ayourtch/work/yue2-rs-logs/p3b-audit-gpu.log
python command: /home/ayourtch/yue2/.venv/bin/python -u tools/dump_reference.py --nar-stages --output /home/ayourtch/work/yue2-rs-fixtures/first-song
python exit=0 log=/home/ayourtch/work/yue2-rs-logs/p3b-audit-python.log
rust command: tools/with_reference_blas.sh cargo test --features cuda --lib nar_stage_dump -- --ignored --nocapture
rust exit=0 log=/home/ayourtch/work/yue2-rs-logs/p3b-audit-rust.log
comparison command: /home/ayourtch/yue2/.venv/bin/python tools/compare_nar_stages.py --root /home/ayourtch/work/yue2-rs-fixtures/first-song
comparison exit=0 log=/home/ayourtch/work/yue2-rs-logs/p3b-audit-comparison.log
gates command: tools/with_reference_blas.sh cargo test --features cuda --test phase3 -- --ignored --nocapture --test-threads=1
gates exit=101 log=/home/ayourtch/work/yue2-rs-logs/p3b-audit-gates.log
```

Clean CPU tests: **15 passed, 13 ignored, 0 failures**. Unset-fixture tests return
cleanly, including the new NAR stage dump. `cargo fmt --check` emits no output;
both Clippy configurations pass with `-D warnings`. Relevant clean-test output:

```text
running 6 tests
test model::diagnostics::ar_operations_oracle ... ignored, requires YUE2_FIXTURES, CUDA, and tools/dump_ar_debug.py
test model::nar::stages::nar_stage_dump ... ignored, requires YUE2_FIXTURES, CUDA and dump_reference.py --nar-stages
test model::nar::tests::nar_operations_oracle ... ignored, requires YUE2_FIXTURES, CUDA and tools/dump_nar_debug.py
test model::nar::tests::acoustic_bias_rounds_once_after_accumulation ... ok
test model::nar::tests::midpoint_cpu_f32_bf16_and_cancellation ... ok
test model::nar::tests::synthesis_serial_chunks_progress_and_seed_reset ... ok

test result: ok. 3 passed; 0 failed; 3 ignored; 0 measured; 0 filtered out; finished in 0.10s
```

Clean-audit gpu output (compiler progress omitted):

```text
index, memory.used [MiB], memory.free [MiB]
0, 62432 MiB, 34858 MiB
1, 74354 MiB, 22936 MiB
2, 89069 MiB, 8220 MiB
```

Clean-audit python output (compiler progress omitted):

```text
P3b original velocity, all 32 solver steps, two full-song noise slices: EXACT
P3b numeric control no_reduced chunk 0: {'max_abs': 0.1827392578125, 'cosine': 0.9999678105975508, 'different': 34247, 'count': 47488}
P3b numeric control no_reduced chunk 1: {'max_abs': 0.173828125, 'cosine': 0.9999727565088387, 'different': 33104, 'count': 47488}
P3b numeric control math_sdpa chunk 0: {'max_abs': 0.8358154296875, 'cosine': 0.9994431633430217, 'different': 39906, 'count': 47488}
P3b numeric control math_sdpa chunk 1: {'max_abs': 0.140625, 'cosine': 0.9999396558575843, 'different': 36433, 'count': 47488}
P3b numeric control plain_sdpa chunk 0: {'max_abs': 0.768310546875, 'cosine': 0.9996406860509885, 'different': 38960, 'count': 47488}
P3b numeric control plain_sdpa chunk 1: {'max_abs': 0.24853515625, 'cosine': 0.9999494233255068, 'different': 35734, 'count': 47488}
P3b FP32 control chunk 0 complete
P3b FP32 control chunk 1 complete
p3b-python.safetensors: 471,650,068 bytes; 439 tensors
Total recursive fixture bytes: 1,424,476,394; peak CUDA allocated 15,195,621,888
```

Clean-audit rust output (compiler progress omitted):

```text
running 1 test
P3b Rust: 512 tensors; staged prefill/velocity exact against production; actual solver observed
P3b FP32 control chunk 0 complete
P3b FP32 control chunk 1 complete
test model::nar::stages::nar_stage_dump ... ok

test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 5 filtered out; finished in 26.52s
```

Clean-audit gates output (compiler progress omitted):

```text
running 2 tests
test nar_cuda_noise_owns_seed ... NAR CUDA RNG PASS: 576/576 values repeat after AR reseed; context-independent cuts; different seed differs; AR stream unchanged
ok
test p3_dumped_noise_latents ... P3 velocity two_chunk.0.step.00: max_abs=0.062500000000 cosine=0.999955537267
P3 velocity two_chunk.0.step.01: max_abs=0.093750000000 cosine=0.999952744767
P3 velocity two_chunk.0.step.16: max_abs=0.078125000000 cosine=0.999920463960
P3 velocity two_chunk.0.step.31: max_abs=0.156250000000 cosine=0.999944826663
P3 two_chunk.0 step 8/32
P3 two_chunk.0 step 16/32
P3 two_chunk.0 step 24/32
P3 two_chunk.0 step 32/32
P3 two_chunk.0: frames=742 max_abs=0.839233398438 cosine=0.999378217075 seconds=5.742257
P3 velocity two_chunk.1.step.00: max_abs=0.136718750000 cosine=0.999957140503
P3 velocity two_chunk.1.step.01: max_abs=0.070312500000 cosine=0.999956381175
P3 velocity two_chunk.1.step.16: max_abs=0.062500000000 cosine=0.999941573005
P3 velocity two_chunk.1.step.31: max_abs=0.093750000000 cosine=0.999955341993
P3 two_chunk.1 step 8/32
P3 two_chunk.1 step 16/32
P3 two_chunk.1 step 24/32
P3 two_chunk.1 step 32/32
P3 two_chunk.1: frames=742 max_abs=0.269531250000 cosine=0.999945369729 seconds=5.748339
Error: P3 requires max_abs <= 0.01 and cosine >= 0.999 for BOTH chunks
FAILED

failures:

failures:
    p3_dumped_noise_latents

test result: FAILED. 1 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out; finished in 14.00s

error: test failed, to rerun pass `--test phase3`
```

Process GPU memory was sampled every 500 ms with
`nvidia-smi --query-compute-apps=gpu_uuid,pid,used_gpu_memory --format=csv -lms 500`.
Observed task-process peaks: PID 2928106: **18332 MiB**, PID 2933308: **15566 MiB**, PID 2933947: **7982 MiB**.
Python allocator peak: **15,195,621,888 bytes**; reserved peak: **18,503,172,096 bytes**.
All runs were serial on GPU 0, under the approximately 30 GB limit. Existing
server processes were not changed. These are measured peaks, not a guarantee
about unsampled allocations.

Total recursive first-song fixtures/artifacts: **1,424,495,594 bytes**, below 1.5 GB.
Original `nar.safetensors` SHA-256 (unchanged):
`e1d4dc5fdf82a57a60df39365e5d75f86023c182514960b0a7b0a9d674e41eaa`.
Existing P0/P1 fixture files were not overwritten; P2 generation was not rerun.

## Unsure list and design objections

- No structural port bug was found in the tested path. Evidence covers the
  requested two chunks and exact first-song checkpoint/environment, not a proof
  for arbitrary requests or all hardware. Non-default shift checks cover scalar
  arithmetic, not a complete alternate-checkpoint trajectory. Restricted
  `nar_cond_end` is not exercised by the P3 fixture.
- The remaining numerical errors include FP32 transcendental/reduction rounding
  before BF16 casts, BF16-input GEMM reduction choices, fused/plain attention
  arithmetic and their propagation through BF16 state updates. Describing all
  of them merely as "BF16 accumulation" would be imprecise. No internal Torch
  kernel algorithm is asserted solely from an endpoint error.
- Shared GPU 0 and the existing reference cuBLAS 12.8.4.1 helper were used.
  Neither alternate cuBLAS/GPU versions nor Metal, flash-attn or full checkpoint
  CPU trajectories are certified. The FP32 control uses Torch math SDPA and
  candle plain SDPA; it does not compare FP32 results to the BF16 oracle as a gate.
- The requested 0.01 BF16 absolute bound rejects even the measured Python-only
  numerical-backend controls. This is the design objection motivating the
  proposed wording, not permission to change the gate.
- No VAE, audio-quality, pipeline/CLI, P4 or P5 work was performed. No push.
