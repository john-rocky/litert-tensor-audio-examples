# diffusion sdpa + cache_update multi-step (#8796) — bench + contract findings (2026-07-22 pm)

Continuation of the runtime-bmm campaign (notes/runtime-bmm-bench.md), queue
items "sdpa for diffusion's full-seq MHA" and "multi-step cache_rbmm harness".
Tools: `diffusion_step/diffusion_main --attention=raw|sdpa|sdpa_nofold|
sdpa_folddirect` and `talker_step/attn_bench --variant=cache_rbmm
--multi_steps=N [--multi_host_cache=zeros|real|once] [--feedback_loops]`.
Machine: M4 Max, macOS, Metal accelerator prebuilt (cwd = litert/prebuilt/
macos_arm64), diffusion GPU runs are fp32 (no fp16 flag in diffusion_main).
Host was under interactive load — small-graph medians swing hard; flagged
below wherever it matters.

## A. diffusion sdpa (`--attention=sdpa`) — adopted

Graph forms:
- raw: proj -> BNSD transposes -> GqaAttention fold (unchanged talker-style).
- sdpa (MHA, G==H): everything stays BSND [1,S,H,D]; rope tables
  pre-transposed once to [1,S,1,D]; composite inputs q/k/v BSND, no mask
  (bidirectional); the two graph-edge transposes MOVE INSIDE the
  decomposition. Attrs: {scale}.
- sdpa (GQA, G<H): pre-fold OUTSIDE the composite to batched MHA —
  q [G, g*S, 1, D], k/v [G, S, 1, D] (BNSD memory order makes the reshape
  exactly the raw fold). Decomposition body = raw fold ops (reshape-only
  axis swaps; no Transpose->BMM fusion ambiguity).
- sdpa_nofold: GQA WITHOUT the fold — kept only as the Metal wrong-result
  repro (FINDINGS #13).
- sdpa_folddirect: the folded body as plain ops, no composite wrapper —
  CPU-parity workaround for FINDINGS #14.

### A1. CPU parity (multi-step ids traces, synthetic weights)

| config | run | result |
|---|---|---|
| GQA 16/8, 2L, seq64, vocab512, k=4 (14 steps) | raw vs sdpa(nofold BSND) | ids bit-identical |
| LLaDA 32/32, 2L, seq64, vocab2048, k=4 | raw vs sdpa (MHA BSND) | ids bit-identical |
| GQA folded v1 (Transpose in decomposition) | raw vs sdpa | DIFFER from step 0 (8/64 = top-k cap) |
| GQA folded v2 (reshape-only decomposition) | raw vs sdpa | still DIFFER -> see #14, not a decomposition-math issue |
| GQA folded body as plain ops (folddirect) | raw vs folddirect | ids bit-identical (k=56 argmax 0/64) |

### A2. Decisive correctness probe (single step, k=56 full unmask -> argmax
exposed at all positions; 2L, seq64; count = positions differing /64, the 8
prompt positions always match)

| arms | GQA 16/8 synthetic (vocab512) | LLaDA MHA synthetic (vocab2048) | LLaDA MHA REAL weights (2L sliced, vocab2048) |
|---|---|---|---|
| CPU raw vs CPU sdpa | 0 | 0 | 0 |
| GPU raw vs CPU ref | 0 | 19 | 0 |
| GPU sdpa vs CPU ref | 56 (= ALL generated) | 19 | 0 |
| GPU raw vs GPU sdpa | 56 | 13 (flip sets overlap 13/19, same regions) | — |

Reading: synthetic near-uniform logits make 4096-dim fp32 drift flip ~19
argmaxes on BOTH GPU arms (same regions, 13 shared -> same numerical
neighborhood); REAL weights peak the logits and both GPU arms go exact ->
**MHA sdpa on Metal = correct**. GQA un-folded: raw-GPU drift baseline is 0
yet sdpa-GPU flips ALL 56 -> **claimed but computing something else**
(FINDINGS #13). GQA folded GPU: 0/64 (both decomposition versions) ->
**fold restores correctness on Metal**.

Earlier k=4 probes saturate at 2k=8 differing positions (top-k selection
cap) — they cannot distinguish drift from breakage; k=56 can. Lesson kept.

### A3. FINDINGS #14 isolation chain (CPU path of the composite)

Identical folded math run three ways, GQA 2L seq64 vocab512 k=56 vs raw-CPU:
1. inside StableHLOComposite, decomposition with Transposes: 56/64 wrong
2. inside StableHLOComposite, reshape-only decomposition (op-for-op = raw): 56/64 wrong
3. same body as plain ops (no composite): 0/64
4. same composite on Metal GPU: 0/64
=> the CPU runtime substitutes its own sdpa lowering for the authored
decomposition and mishandles the batched presentation ([G, g*S, 1, D],
batch>1, heads=1). Plain-BSND (batch=1) presentations happen to match our
fold bit-for-bit (A1 rows 1-2), consistent with a per-head lowering of the
same op order. CPU-parity verification must use --attention=raw or
sdpa_folddirect (mirrors the --norms=raw rule).

### A4. GPU perf, LLaDA shape (MHA 32/32, 4L, vocab 8192 synthetic, fp32;
medians of 26-28 timed steps; two interleaved reps)

seq_len=256, unmask_k=8, prompt 32 (28 steps): raw 34.841 / sdpa 34.630
ms/step -> neutral (attention ~3% of step FLOPs; FC-dominated).

seq_len=1024, unmask_k=32, prompt 64 (30 steps):

| arm | rep1 | rep2 | read |
|---|---|---|---|
| raw / raw norms | 284.172 | 284.409 | stable baseline |
| sdpa / raw | 258.243 | 261.136 | **-8.6%** |
| sdpa / rms composite | 267.532 | 254.688 | overlaps sdpa-alone -> rms does NOT stack on sdpa |
| raw / rms composite | 269.049 | 276.379 | rms alone -4~5% |

(The rep1-only ordering "stack worse than sdpa" was variance; rep2 inverted
it. Claimed only: sdpa -8.6%, rms-alone -4~5%, stack ~= sdpa-alone.)

### A5. GPU perf, 1024-dim GQA shape (16/8, vocab 8192, seq256, k=8) — and
the --norms flag bug correction

Flag bug found: `config.use_rms_norm_composite = (--norms)` sat INSIDE the
`if (--llada)` block in diffusion_main.cc, so --norms=composite was silently
ignored for the default 1024-dim shape. The 7/21 "1024-dim neutral" rms
entry was raw-vs-raw. Fixed 7/22 (assignment moved out + flag validation
added); COMPOSITES.md corrected.

4L (steps ~6-15 ms) — noise-dominated under host load, sign flips between
reps; kept only to document the variance, no claims from it:
raw 11.795 / 7.385; rms 14.728 / 10.844; sdpa(nofold, later shown wrong)
14.228 / 5.801.

28L (real 0.6B depth), interleaved reps:

| arm | rep1 | rep2 |
|---|---|---|
| raw | 75.358, 76.863 | 51.519, 85.920 |
| rms composite | 78.500 | 58.068 |
| sdpa un-folded (INVALID — wrong results, #13) | 33.170 | 36.459 |
| sdpa folded (correct) | 66.412 | 43.925 |

Reads: (1) folded sdpa beat raw in both interleaved pairings (76.9->66.4,
85.9->43.9) but the raw band is 51-86 under contention -> "consistently
faster, magnitude 15-50%, clean-machine re-measure pending". (2) rms
composite at this shape is a consistent small REGRESSION (~+3 ms/step at
both 4L pairs and 28L pairs, every pairing ordered the same way even as
absolutes swung). (3) the un-folded 33-36 ms "3x win" was a broken kernel
computing garbage — perf numbers from wrong-result configs are void.

## B. cache_update multi-step + PR #8796 FeedbackLoopConfig

Setup: attn_bench --variant=cache_rbmm, talker decode geometry (16/8/128,
cache 1024), fp32. Chained decode pos=0..N-1 on ONE runner; per-step CPU
reference is stateless (host caches carry the true prefix rows each step).

### B1. Without feedback loops: the delegate cache resets every Run()

--multi_steps=128 --multi_host_cache=zeros: pos=0 exact; every pos>=1
MISMATCH; worst max_rel 46.3 at pos=5; per-step median 0.390 ms (min 0.294,
max 4.060 first-step). Late-step sample (pos 117-127): max_rel 5.0-10.4 with
got ~ v/(pos+1) scale — the profile of "previous rows are zero".

Host-cache mode discrimination (--multi_steps=8): zeros, real, and once
produce IDENTICAL per-step numbers (pos1..7 max_rel = 12.6 / 7.86 / 11.3 /
5.08 / 46.3 / 24.2 / 12.3; pos0 = 0 in all) =>
- the GPU path NEVER reads the host cache inputs (real vs zeros: no change), and
- state does not persist regardless of binding behavior (once: no change)
=> the delegate cache is re-initialized per Run(). Yesterday's fill=1
exactness was the write+read INSIDE one Run; "delegate-managed cache"
holds only within a Run without a feedback contract.

### B2. With FeedbackLoopConfig (#8796, landed 7/22 03:10 JST; cherry-pick
2e89a93e onto pin a19d8fa = local LiteRT commit 8b2d63b4; full main was
5449 commits ahead, not pulled)

API: LitertDynamicRunner::Create(env, path, options, {{"cache_k_in",
"cache_k_new"}, {"cache_v_in","cache_v_new"}}) — on GPU it enables
external-tensors mode + AddExternalTensorPattern per loop name; Run() swaps
input<->output buffers per loop after the first run (ping-pong; Reset()
restores). Caches must be signature OUTPUTS (harness adds cache_k_new /
cache_v_new to outputs under --feedback_loops).

CPU (--accelerator=cpu --multi_steps=64 --feedback_loops; measured runner =
a second, loop-registered CPU runner; reference = stateless CPU runner):
**max_rel = 0 at every one of 64 chained steps** (per-step median 0.506 ms,
min 0.396, max 1.106 first-step). The DUS decomposition + swap chain carry
state exactly. Their own lambda_model_runner_test is CPU-only and a toy
(x=y+1); this is the same contract at 10-input signature + two DUS caches.

Metal (--accelerator=gpu, default AND --gpu_buffer_storage=buffer): first
Run() exact (pos=0 max_rel 0; 4.0-12.9 ms warm-up step), second Run() fails:
"ERROR: [litert_compiled_model.cc:164] Failed to register input tensor
buffer: The given buffer type is not supported." (delegate logs "External
tensors mode is enabled."; "Failed to get buffer requirements" warnings on
cache/pos inputs precede it, as in the single-step probe). Binding zeros at
pos=0 vs never binding the caches: same failure => the swapped-in OUTPUT
buffer itself is rejected on re-registration as an input. FINDINGS #15;
talker cache_update adoption blocked on it.

Harness detail worth keeping: in feedback mode the harness never re-binds
cache inputs (the swap chain owns them); unwritten rows are never read
because step t's mask admits only rows 0..t, all written by steps <= t.

## Cross-refs

- FINDINGS.md #13/#14/#15 (public repo) — one-command repros for all three.
- notes/repro-logs-0722/ — issue-ready RAW LOGS for #13/#14/#15, regenerated
  from the one-liners after the session (numbers reproduce bit-for-bit incl.
  worst-element values); attach these (or a fresh capture) if Ping green-lights
  the GitHub issues.
- COMPOSITES.md — sdpa row (adopted for diffusion), rms row (corrected
  numbers), cache_update row (multi-step + #8796 status), adoption matrix.
- Repro one-liners: FINDINGS #13 `diffusion_main --attention=sdpa_nofold
  --layers=2 --seq_len=64 --vocab=512 --unmask_k=56 --prompt_len=8
  --warmup=0 --trace_ids --accelerator=gpu` (compare step-0 ids vs the same
  command with --accelerator=cpu; raw GPU matches, sdpa_nofold does not);
  #14 swap --attention between sdpa / sdpa_folddirect on CPU; #15
  `attn_bench --variant=cache_rbmm --accelerator=gpu --gpu_precision=fp32
  --multi_steps=8 --feedback_loops`.


## 7/23 addendum — clean-machine re-baseline + Ping's 4 follow-ups

Host idle (early morning); every arm re-run. Two 7/22 labeling/condition
errors corrected below; talker numbers REPRODUCED, diffusion numbers were
contaminated.

### Corrections to 7/22

1. PRECISION LABEL: diffusion_main had no precision flag, and the Metal
   delegate DEFAULT is fp16 (F16 when round-to-nearest is supported,
   delegate_metal.mm:396) - every 7/22 diffusion "fp32" GPU number was
   actually fp16-compute. Flag added 7/23 (--gpu_precision=default|fp16|fp32).
2. HOST CONTENTION: the 7/22 diffusion absolutes were ~2x inflated and the
   deltas exaggerated. Clean 3-rep medians (spread <0.3%), llada 4L seq1024:
   fp32 raw 154.1 / sdpa 148.2 (-3.8%); fp16 raw 134.8 / sdpa 131.5 (-2.5%);
   rms fp32 153.9-154.6 (neutral); stack fp32 148.0-150.5 (~= sdpa alone).
   seq256 fp32: raw 38.46 / sdpa 38.23 (neutral). 28L GQA seq256: fp32 raw
   27.45 vs folded-sdpa 27.25-27.48 (NEUTRAL - the 7/22 "15-50% faster" was
   contention amplifying dispatch overhead); rms fp32 26.4-27.2 (neutral),
   fp16 raw 25.40 vs rms 24.46-24.50 (-3.6%).
   Talker control (same morning): raw fp16 22.34 / stack 20.40 (-8.7%);
   fp32 30.58 / 28.11 (-8.1%) - matches the published 22.23/20.31 and
   30.41/28.1, so the talker campaign was clean; only the 7/22-evening
   diffusion window was contaminated.
3. Real-weight k=56 probe at EXPLICIT precisions: GPU raw and sdpa both
   0/64 vs CPU at fp32 AND fp16 - the exactness claim survives with labels.

### fp16 sweep (Ping 3:31 "GPU seems fp32-only") — CORRECTED same day

attn_bench fill=64, default fp16, Metal: raw max_rel 0.056 (rounding-scale,
fine); rbmm 0.069, 0.945 ms (fine); GQA sdpa = CPU-fallback bit-exact
(known #7); cache_rbmm = max_rel 18.0. The morning read blamed fp16; the
window-2 control showed fp32 at fill=64 mismatches IDENTICALLY (18.05) and
fill=1 is fine at both precisions (fp32 exact, fp16 4.6e-4) -> the fill>1
single-step mismatch is the per-Run cache-reset contract (GPU never saw
the CPU reference's prefix rows), precision-irrelevant. FINDINGS #16
rewritten as a harness-contract note. Lesson: the fp32 control was
initially assumed from the 7/22 fill=1 result instead of being run -
control arms must be executed, not extrapolated.

### int4 Metal status change (Ping 1:10)

int4_fc_repro (2-sig blockwise, the #1 minimal repro): PASSES 5/5 today.
Production GPTQ int4 talker (fp16 scales, 2 signatures, 28L, cache 1024):
both signatures compile; decode 19.18 ms/frame raw graph, 17.29 ms/frame
with the composite stack (-9.9%, RTF 0.216) - int4 e2e is now the fastest
talker config on Metal, and the composite share grew exactly as predicted
once FCs shrank. Environment: same pin a19d8fa + only the #8796 runner
cherry-pick, same prebuilt dylib; no explanation on our side for the
change. Mali/CL re-check pending a supervised device session.

### Op counts (Ping 1:05), serialized main-graph nodes

talker 28L (per signature, prefill == decode): raw 7809; rbmm 7781; rbmm+
rms 5551 (-29%, 502 composites per signature).
diffusion LLaDA 32L (single denoise signature): raw 1816; sdpa 1402; rms
1491; sdpa+rms 1077 (-41%, 97 composites). Decomposition subgraphs ride
along (6 ops per rms, 14 per sdpa) and never run when the delegate claims.

### Window-2 reproducibility check (7/23 07:2x, user active + a CPU-bound
python job; load avg 3.4 - deliberately a DIFFERENT condition)

Every claim reproduced within 1%: diff 1k fp32 154.7/148.7 (-3.9%), fp16
135.3/131.9 (-2.5%); talker fp16 22.52/20.58 (-8.6%), fp32 30.79/28.33
(-8.0%), int4 19.30/17.32 (-10.2%); 28L GQA 27.60/27.43 (neutral); int4
repro 3/3 PASS; cache fill=64 fp16 18.04. The one non-reproducing item was
the assumed cache-fp32 control -> #16 corrected (above). Note the 7/22-eve
contamination was GPU contention specifically: window 2's CPU-bound load
did not move Metal numbers.

Logs for all of the above: notes/repro-logs-0722/ (7/23 files) plus this
session's command lines in the KICKOFF entry.
