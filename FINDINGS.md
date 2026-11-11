# Findings ledger — LiteRT / ML Drift, from the audio-LM examples

One line per finding; details and repro commands in the linked notes. Status:
`open` = not resolved; `workaround` = we ship around it;
`superseded` = replaced by a
later finding. Repros stay ready.

| # | finding | backend | status | repro / notes |
|---|---|---|---|---|
| 1 | Blockwise-int4 FC weights shared across signatures fail at delegate init (SIGSEGV/SIGTRAP/spin; per-channel and fp32 fine; single-signature blockwise also failed for one 8B graph) | Metal | NOT REPRODUCING as of 7/23 on our M4 (same checkout a19d8fa + only the PR #8796 runner commit cherry-picked, same prebuilt dylib): minimal 2-sig repro passes 5/5, and the production GPTQ int4 talker (fp16 scales, 2 signatures, 28L) compiles both signatures and decodes at 17.3-19.2 ms/frame. Cause of the change unknown on our side; repro kept; CL/Mali re-check pending a supervised device session | `talker_step/int4_fc_repro.cc`; also reproduced 7/20-22 with the converter-produced model, and LiteRT#8831 reports a same-family blockwise serializer gap on Google-Tensor AOT |
| 2 | Delegate init allocates ~5-6x the model bytes within seconds; on an 8 GB phone this outruns reclaim and kernel-panics (reboot; no lmkd record; program cache never written) | CL (Pixel 8a) | open | `talker_step/oom_repro/`, timeline `notes/memlog-panic-0722.txt` |
| 3 | max_seq >= 1024 + texture storage silently runs the whole graph on CPU (no runtime indication) | Metal | workaround (buffer storage) | |
| 4 | >62 min on-device kernel compile for a 7388-node graph, then reboot | CL (Pixel 8a) | superseded by #2 (same OOM class, longer fuse) | |
| 5 | dst-bounded odml.runtime_bmm leaves its output tail unwritten and elementwise ops fuse into the bounded epilogue, so a full-width Softmax between two bounded matmuls reads unmasked memory | Metal | workaround ("avbound": QK unbounded + AV bounded; exact) | `notes/runtime-bmm-bench.md`; intended pattern unknown |
| 6 | odml.runtime_bmm kernel init fails at ml_drift/cl/tensor.cc:396 -> cl_arguments.cc:487 (graceful; sdpa without the int32 control input inits fine) | CL (Pixel 8a) | open - repro is one command | suspect int32 [1,1,1,7] control-tensor storage |
| 7 | Grouped-K/V sdpa: clean CPU fallback on Metal; on CL one node runs but 4+ nodes fault (clEnqueueMapBuffer -36 -> CHECK abort) | Metal + CL | workaround (pre-fold GQA to MHA/batch shapes) | pre-fold vs native grouped support: open question (`notes/runtime-bmm-bench.md`) |
| 8 | GATHER needs 1-D indices; DYNAMIC_UPDATE_SLICE needs runtime start_indices (constant rejected); with either violated, partitioned compile fails without an op-level diagnostic | Metal | workaround (in-graph) | |
| 9 | Gather axis-1 on 4D tensors is a hard compile fail | Metal | workaround (GQA fold, no repeat) | |
| 10 | fp16 default precision flips greedy codec ids over long rollouts (precision, not a bug) | Metal | guidance note (fp32 option for verification) | |
| 11 | XNNPACK blockwise-int4 fast kernels are GEMV-shape only; M>=64 GEMM (diffusion-style steps) hits a slow path on arm64 macOS and Android | CPU | open observation | int4 4.6x slower than fp32 at M=64; int4 wins only for AR decode |
| 12 | No output->input buffer aliasing contract in the runner (naive copy pattern costs +24-35%/frame) | runtime | landed 7/22 (PR #8796 FeedbackLoopConfig) and verified the same day: 64 chained cache_update decode steps bit-exact on CPU via the swap chain | `attn_bench --variant=cache_rbmm --multi_steps=64 --feedback_loops`; Metal side = #15 |
| 13 | Un-folded grouped-K/V sdpa at q_seq>1: Metal CLAIMS the composite and computes silently wrong results (single-step full-unmask probe: every generated id differs from the CPU reference while raw GPU matches it exactly); at q_seq=1 the same shape rejects cleanly to CPU | Metal | workaround (pre-fold GQA to batched MHA [G, g*S, 1, D] - exact; extends #7) | `diffusion_main --attention=sdpa_nofold` is the repro; `--attention=sdpa` auto-folds |
| 14 | sdpa composite on CPU: the runtime substitutes its own lowering for the authored decomposition; correct (bit-exact) for plain-BSND batch-1 presentations, wrong for the folded batched presentation ([G, g*S, 1, D], batch>1 heads=1) - the same graph is exact on Metal and exact as plain ops | CPU | workaround (CPU-parity verification via `--attention=raw` or `sdpa_folddirect`) | isolated by running the identical fold body with and without the composite wrapper |
| 15 | FeedbackLoopConfig on Metal: with external-tensors mode + swapped cache buffers, the second Run() fails at input-buffer re-registration ("The given buffer type is not supported", litert_compiled_model.cc:164; default and buffer storage both) - first Run() is exact, CPU path is exact over 64 steps | Metal | open; talker cache_update adoption blocked on this | `attn_bench --variant=cache_rbmm --accelerator=gpu --multi_steps=8 --feedback_loops` |
| 16 | Harness-contract note (NOT a kernel bug; we briefly misread it as an fp16 issue and corrected it the same day after a cross-window control run): single-step cache_rbmm GPU-vs-CPU comparisons are only meaningful at fill=1 - at fill>1 BOTH precisions mismatch identically (max_rel 18.04 fp16 / 18.05 fp32 at fill 64) because the delegate cache starts empty every Run() and the host cache inputs are never read (the multi-step contract), so the CPU reference holds prefix rows the GPU never saw. At fill=1: fp32 exact, fp16 max_rel 4.6e-4 (normal rounding). Chained state needs --multi_steps + feedback loops (#15) | Metal | documented (compare at fill=1, or chain steps) | `attn_bench --variant=cache_rbmm --fills=1` vs `--fills=64`, either precision |

Wins adopted from the LiteRT side: odml composites confirmed in converted
graphs (rms_norm etc.); odml.runtime_bmm adopted as the talker's attention
(3x attention, ~4-5% e2e; `--attention=rbmm`); odml.scaled_dot_product_
attention adopted as the diffusion step's attention (`--attention=sdpa`:
exact on Metal at the MHA shape at fp32 and fp16; -3.8% fp32 / -2.5% fp16
at S=1024 on a quiet machine; GQA via pre-fold); full composite stacks cut
serialized node counts -29% (talker per signature) / -41% (diffusion 32L).
