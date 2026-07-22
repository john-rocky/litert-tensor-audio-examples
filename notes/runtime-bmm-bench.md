# odml.runtime_bmm vs sdpa vs raw attention — bench + contract findings (2026-07-22)

Prompted by a suggestion from the LiteRT team (2026-07-22): use odml.runtime_bmm for the qwen3_tts talker and compare
against scaled_dot_product_attention. Tool: `attn_bench.cc` (talker_step/, synced
both copies; BUILD target `attn_bench`). 28 chained decode-shape attention blocks
(heads 16 / kv 8 / head_dim 128 / cache 1024, q_seq 1), one graph per variant,
identical logical inputs, CPU decompositions double as numeric reference. Active
length is runtime-only (mask / control tensor) so one flatbuffer serves all fills.

## Contract (read from ml_drift_delegate/delegate/composite/, confirmed by run)

- `odml.runtime_bmm` = STABLEHLO_COMPOSITE, 3 runtime inputs: LHS [B,H,M,K],
  RHS [B,H,N,K] (fp32; int8 via `scale` attr), control tensor **int32 [1,1,1,7],
  element 2 = active token count**; 1 output [B,H,M,N]. Flexbuffer attrs:
  `is_global` (required, value unused by parser), `is_src` (false = bound output
  channels -> the QK case; true = bound reduction/source channels -> the AV case),
  `rhs_cache_update` (FC-kernel path; auto when RHS comes from odml.cache_update).
  fp32 BMM path sets transpose_right, so RHS rows = positions; for AV we feed V
  transposed [1,G,D,S] (positions on channels).
- Parser registry is linked into the APP-side LiteRT (attn_bench binary contains
  the composite names), NOT the accelerator dylib — the earlier "Metal prebuilt
  lacks runtime_bmm" strings-check conclusion was about the wrong layer. Built
  from this checkout, **runtime_bmm works on Metal today**.
- Emission from the Tensor API: `StableHLOComposite({name, flexbuffer bytes,
  version}, decomposition lambda, inputs...)`; unused decomposition inputs (the
  control tensor) serialize fine.
- No 4-alignment requirement observed for element 2 (fill 511 exact on Metal).

## Finding 1 (the big one): bounded QK + full-width softmax is unsound

- A dst-bounded QK composite **leaves output columns >= active UNWRITTEN**
  (stale memory, zero only on fresh allocations), and the scale-Mul/mask-Add
  after it get **fused into the bounded kernel's epilogue**, so they never touch
  the tail either. A full-width Softmax then reads unmasked tail garbage.
  Demo: run fills descending (1024 then 64) in `--rbmm_mode=corrected` ->
  max_rel 14.3 at fill 64 (tail = stale scores of the 1024 run); ascending order
  looks fine (allocation zeros) — a trap.
- Softmax tail probe (`--debug_softmax`): tail probabilities uniform
  ~e^0/Z -> output scaled by ~active/S exactly; measured active_sum 0.509 at
  fill 512 (= the 0.5x output shrink observed).
- **Sound composition ("avbound", now the tool default building block):
  QK runs UNBOUNDED** (second control tensor input with element2 = S; zero cache
  rows give deterministic zero tail scores and the fused mask epilogue covers
  full width) **+ AV side src-bounded**. Correct at every fill/order (max_rel
  ~2e-5 fp32).
- A "corrected" mode (renormalize by 1/(1 - tail_count*p0), exact algebra) is
  kept in the tool but is unsound on GPU for the same unwritten-tail reason.

## Numbers — M4 Max Metal fp32 (buffer storage), ms per 28-block frame, runs=40

| config | fill 64 | 512 | 1024 | numerics |
|---|---|---|---|---|
| raw (GQA fold, today's talker) | 2.554 | 2.564 | 2.571 | ok ~2e-5 |
| sdpa GQA-shaped | 2.851* | 2.811* | 2.834* | *NOT on GPU: rejected -> CPU fallback (bit-exact vs XNNPACK ref; time is the fallback's, multithread caveat) |
| sdpa MHA-shaped (on GPU) | 4.549 | 4.419 | 4.441 | ok |
| raw MHA-shaped | 3.397 | 3.383 | 3.378 | ok |
| **rbmm avbound (GQA)** | **0.758** | **0.796** | **0.846** | **ok ~2e-5 all fills incl. 511** |
| rbmm both-bounded (masked) | 0.690 | 0.775 | 0.849 | BROKEN < full fill (Finding 1) |
| rbmm avbound MHA-shaped | 0.845 | 1.052 | 1.173 | ok |

- **runtime_bmm avbound = 3.0-3.4x faster than raw, >5x faster than the sdpa
  Metal kernel**, at exact fp32 correctness. The sound pattern costs only ~10%
  vs the unsound both-bounded ceiling at fill 64.
- sdpa's fused Metal kernel is SLOWER than the plain BMM chain at decode shapes
  (M=1-2 rows) — 4.44 vs 3.38 ms MHA-shaped.
- Blocks=1 numbers: avbound 0.36-0.40 ms vs raw 0.60-0.68 — dispatch floor
  dominates tiny graphs; 28-block figures are the meaningful ratio.

## Numbers — CPU (decompositions; deployability of the same flatbuffer)

M4 Max XNNPACK (runs=30): raw 14.0-14.8 / sdpa-decomp 28.8-30.1 (BSND
transposes) / rbmm-decomp 13.8-14.3 ms.
Pixel 8a XNNPACK (runs=15): raw 32.2-35.7 / sdpa-decomp 46.6-49.9 /
**rbmm-decomp 25.2-26.9 ms — the composite graph is also the fastest CPU form**
(pre-folded, no transposes), so shipping the composite file loses nothing where
kernels are unmatched.

## Pixel 8a CL delegate findings

- raw: 308/308 nodes delegated, init in seconds, fp32 correct, 18.4 ms median
  (spread 15.8-52.5, thermal).
- **runtime_bmm (all modes, GQA and MHA): partitions 141/141 -> kernel init
  FAILS gracefully** at `ml_drift/cl/tensor.cc:396` -> `cl_arguments.cc:487`
  ("Failed to initialize kernel", plan restored, no crash). sdpa (no int32
  runtime input) initializes fine on the same device -> suspect the int32
  [1,1,1,7] control tensor storage on CL. param_align4 changes nothing.
- sdpa GQA-shaped on CL: 1 block runs; **4+ blocks die with clEnqueueMapBuffer
  -36 (command queue killed by a kernel fault) -> CHECK abort** in
  litert_buffer.h:55. sdpa MHA-shaped works: 14.8/3.9/6.2 ms at fills
  64/512/1024 (very spiky, max 224 ms).
- Metal vs CL GQA handling differs: Metal rejects grouped sdpa cleanly (CPU
  fallback), CL accepts then faults at >=4 nodes.

## Memory timeline of the Mali delegate-init OOM panic

Instrumented re-run of the Round-4 repro (4L talker, 845 MB fp32 flatbuffer,
serialization on, 8 compile threads; 1 Hz MemAvailable + VmRSS, fsync per line;
raw log notes/memlog-panic-0722.txt). Launch at t=0:

| t | MemAvailable | talker VmRSS |
|---|---|---|
| 0.1 s | 4.65 GB | 153 MB |
| 1.3 s | 3.09 GB | 894 MB |
| 2.6 s | 1.60 GB | 3.23 GB |
| 5.5 s | **0** | **4.87 GB peak = 5.8x the flatbuffer** |
| 8.9 s | 0 | 3.58 GB |
| 29.6 s | 0 | 2.04 GB (reclaim/thrash) |
| ~35-40 s | kernel panic -> reboot (bootreason kernel_panic) | |

- Last talker log line: "Replacing 6105/6105 node(s) ... 1 partition" then the
  OpenCL init line — death is inside delegate init, before any program cache
  write (clcache empty again). No lmkd kill record — kernel died first.
  Un-fsynced stdout log was lost in the panic; only the per-line-synced memlog
  survived (log-discipline lesson).
- This run panicked ~35-40 s after launch (Round 4: ~65 s; Round 1 28L: 62 min
  compile then reboot) — fuse length varies with boot state, the allocation
  shape is the constant: **init wants ~5-6x the model bytes within seconds**.

## Repro commands

- Metal matrix: `attn_bench --variant=raw|sdpa|rbmm [--rbmm_mode=avbound|masked]
  [--mha] --accelerator=gpu --gpu_precision=fp32 --gpu_buffer_storage=buffer
  --fills=64,512,1024 --blocks=28` (cwd = runfiles dir with the Metal dylib).
- Staleness demo: `--variant=rbmm --rbmm_mode=corrected --fills=1024,64 --blocks=1`.
- Pixel: same binary cross-built android_arm64 (NDK env var), run from
  /data/local/tmp/qwen3tts with LD_LIBRARY_PATH=. ; OOM repro via launcher.sh
  (memlog.sh + talker_main 4L GPU serialization) — REBOOTS THE DEVICE.

## Open questions (raised with the LiteRT team)

1. Intended composition for bounding the softmax between two runtime_bmms
   (runtime-aware softmax? -inf tail fill? our avbound workaround keeps 3x).
2. Expected storage/handling of the int32 [1,1,1,7] control tensor on CL
   (tensor.cc:396 init failure repro).
3. GQA guidance: pre-fold to MHA shapes for both composites, or native grouped
   support planned? (Metal rejects grouped sdpa; CL faults at >=4 nodes.)
4. Metal roadmap for cache_update (runtime_bmm already works app-side).

## End-to-end: runtime_bmm wired into the talker decode step

`talker_main --attention=rbmm` builds the backbone attention as the
odml.runtime_bmm pair (avbound composition; value caches stored transposed
[1, kv, head_dim, max_seq]; control tensors runtime_param /
runtime_param_full / cache_position_vt added to both signatures).

Correctness: CPU fp32 ids are bit-identical to --attention=raw (synthetic 4L
and real-weight 28L, chained frames), and Metal GPU fp32 ids match raw
exactly. Full-frame medians (28L real weights, max_seq 1024, M4 Max, buffer
storage, 24 steps):

| config | raw | rbmm |
|---|---|---|
| Metal fp16 | 22.23 ms/frame | 21.29 (-4.2%) |
| Metal fp32 | 30.41 ms/frame | 29.02 (-4.6%) |
| CPU fp32 | 57.27 ms/frame | 57.47 (decompositions, parity) |

The attention block itself is 3x faster (see above), but the full frame is
dominated by the FC projections and the 16-step code-predictor unroll, so
the end-to-end gain today is ~4-5%. The share grows as the rest shrinks
(e.g. once int4 weights run on Metal) and with longer caches.

## odml.cache_update pairing (first probe, 2026-07-22)

`attn_bench --variant=cache_rbmm` builds one decode step as
odml.cache_update (7-input form; CPU decomposition = DYNAMIC_UPDATE_SLICE)
feeding the odml.runtime_bmm pair. On Metal the delegate matches the
composite and the step runs at ~0.35 ms, exact vs CPU at fill=1. Learned
contract: the GPU caches are delegate-managed (K written in the QK
FC-weights layout kOSpatialIOGroupO4I4, V in kOSpatialIOGroupI4O4; the
host-provided cache inputs are consumed only by the CPU decomposition -
the runner warns "Failed to get buffer requirements" for them, which is
the expected signature of the composite being claimed). Control tensor
slots: [0] write offset, [1] active tokens (cache_update), [2] active
bound (runtime_bmm). Multi-fill single-shot numerics intentionally diverge
on GPU (state accumulates across calls, not from host inputs); a
multi-step harness is the next step before talker adoption.

## odml.rms_norm adoption (2026-07-22)

`--norms=composite` in talker_main and diffusion_main emits every RMS norm
as the odml.rms_norm composite ({epsilon} attr, constant 1-D scale;
decomposition = the raw MEAN/RSQRT/MUL math, CPU bit-identical). Talker 28L
real weights on M4 Metal: fp32 29.0 -> 28.1 ms/frame, fp16 21.3 -> 20.3 on
top of --attention=rbmm; the combined composite stack is -7.5% (fp32) /
-8.6% (fp16) vs the original raw graph. Diffusion (both the 1024-dim and
4096-dim shapes): neutral within run-to-run noise. Caveat: the fused
kernel's reduction order differs from the decomposition, so fp32 GPU greedy
id trajectories shift; CPU-parity verification should run --norms=raw.
