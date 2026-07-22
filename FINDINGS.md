# Findings ledger — LiteRT / ML Drift, from the audio-LM examples

One line per finding; details and repro commands in the linked notes. Status:
`open` = reported, awaiting guidance; `workaround` = we ship around it;
`fixed-in-flight` = a LiteRT change is underway; `superseded` = replaced by a
later finding. Routing (in-thread vs GitHub issue) is always the LiteRT
team's call - we keep repros ready either way.

| # | finding | backend | status | repro / notes |
|---|---|---|---|---|
| 1 | Blockwise-int4 FC weights shared across signatures fail at delegate init (SIGSEGV/SIGTRAP/spin; per-channel and fp32 fine; single-signature blockwise also failed for one 8B graph) | Metal | open - issue text ready | `talker_step/int4_fc_repro.cc`; also reproduced with the converter-produced model, and LiteRT#8831 reports a same-family blockwise serializer gap on Google-Tensor AOT |
| 2 | Delegate init allocates ~5-6x the model bytes within seconds; on an 8 GB phone this outruns reclaim and kernel-panics (reboot; no lmkd record; program cache never written) | CL (Pixel 8a) | open - issue text ready | `talker_step/oom_repro/`, timeline `notes/memlog-panic-0722.txt` |
| 3 | max_seq >= 1024 + texture storage silently runs the whole graph on CPU (no runtime indication) | Metal | workaround (buffer storage) | reported 7/19 |
| 4 | >62 min on-device kernel compile for a 7388-node graph, then reboot | CL (Pixel 8a) | superseded by #2 (same OOM class, longer fuse) | |
| 5 | dst-bounded odml.runtime_bmm leaves its output tail unwritten and elementwise ops fuse into the bounded epilogue, so a full-width Softmax between two bounded matmuls reads unmasked memory | Metal | workaround ("avbound": QK unbounded + AV bounded; exact) | `notes/runtime-bmm-bench.md`; intended pattern asked |
| 6 | odml.runtime_bmm kernel init fails at ml_drift/cl/tensor.cc:396 -> cl_arguments.cc:487 (graceful; sdpa without the int32 control input inits fine) | CL (Pixel 8a) | open - repro is one command | suspect int32 [1,1,1,7] control-tensor storage; asked |
| 7 | Grouped-K/V sdpa: clean CPU fallback on Metal; on CL one node runs but 4+ nodes fault (clEnqueueMapBuffer -36 -> CHECK abort) | Metal + CL | workaround (pre-fold GQA to MHA/batch shapes) | guidance asked |
| 8 | GATHER needs 1-D indices; DYNAMIC_UPDATE_SLICE needs runtime start_indices (constant rejected); with either violated, partitioned compile fails without an op-level diagnostic | Metal | workaround (in-graph) | reported 7/18 |
| 9 | Gather axis-1 on 4D tensors is a hard compile fail | Metal | workaround (GQA fold, no repeat) | |
| 10 | fp16 default precision flips greedy codec ids over long rollouts (precision, not a bug) | Metal | guidance note (fp32 option for verification) | |
| 11 | XNNPACK blockwise-int4 fast kernels are GEMV-shape only; M>=64 GEMM (diffusion-style steps) hits a slow path on arm64 macOS and Android | CPU | open observation | int4 4.6x slower than fp32 at M=64; int4 wins only for AR decode |
| 12 | No output->input buffer aliasing contract in the runner (naive copy pattern costs +24-35%/frame) | runtime | fixed-in-flight (LiteRT PR #8796, prompted by this finding) | we verify and post numbers when it lands |

Wins adopted from the LiteRT side: odml composites confirmed in converted
graphs (rms_norm etc.); odml.runtime_bmm adopted as the talker's attention
(3x attention, ~4-5% e2e; `--attention=rbmm`).
