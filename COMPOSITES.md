# odml composite coverage program

Goal: adopt and benchmark each odml composite / custom op across these
examples, one at a time, and report what we find - a running showcase of the
composite path from an external C++ Tensor API user. Suggested by the LiteRT
team's pointers (sdpa, runtime_bmm); we continue down the list.

| composite | status here | measured / notes |
|---|---|---|
| odml.runtime_bmm | ADOPTED (talker `--attention=rbmm`) | attention 3.0-3.4x vs raw ops on Metal fp32, exact at every fill; e2e frame -4-5%; decompositions also the fastest CPU form. CL init issue open (ledger #6) |
| odml.scaled_dot_product_attention | ADOPTED for the diffusion step (`--attention=sdpa` in diffusion_main); decode shapes stay rbmm | Decode (q_seq=1, talker): Metal kernel slower than the raw BMM chain (4.4 vs 2.6 ms/28 blocks) - rbmm stays the decode choice. Full-sequence (diffusion): at the LLaDA shape (MHA 32/32, 4096-dim) Metal sdpa is exact (real-weight probe: GPU ids == CPU ids at every position) and -8.6% e2e at S=1024 (284 -> 259 ms/step, 4L; S=256 neutral); at the 1024-dim GQA shape the group-fold presented as batched MHA ([G, g*S, 1, D]) is exact on Metal and consistently faster at 28L (raw 51-86 -> folded 44-66 ms/step under host contention; clean re-measure pending). Two correctness boundaries found on the way, ledger #13/#14: Metal CLAIMS un-folded grouped K/V at q_seq>1 but computes wrong results (q_seq=1 rejects cleanly), and the CPU runtime substitutes its own sdpa lowering for the authored decomposition, which breaks for batched presentations - CPU-parity runs must use `--attention=raw` (or `sdpa_folddirect`) |
| odml.cache_update | BENCHED single-step + multi-step harness (attn_bench `--variant=cache_rbmm [--multi_steps=N --feedback_loops]`) | WORKS on Metal within one Run(): delegate matches the composite and pairs with runtime_bmm; single decode step ~0.35 ms, exact vs CPU at fill=1. Contract learned: caches are delegate-managed on GPU (K stored in the QK FC-weights layout, V in the AV layout; the host-side cache inputs feed only the CPU decomposition), and the control tensor packs write offset (elem 0), active tokens (elem 1) and the runtime_bmm bound (elem 2) in one [1,1,1,7] int32. Multi-step (7/22): WITHOUT feedback loops the Metal delegate cache resets every Run() (identical mismatches whether host caches carry zeros or real prefixes - they are never read); WITH PR #8796 FeedbackLoopConfig (cache_k/v_in <-> cache_k/v_new), 64 chained decode steps are bit-exact on CPU, and Metal fails the second Run() at buffer re-registration (ledger #15). Talker adoption waits on that gap |
| odml.rms_norm | ADOPTED as an option (`--norms=composite` in talker_main and diffusion_main) | talker 28L Metal: fp32 29.0 -> 28.1 ms/frame, fp16 21.3 -> 20.3 (about -1 ms on top of runtime_bmm; combined stack -7.5 to -8.6% vs the original raw graph). diffusion CORRECTED 7/22: the earlier "neutral at 1024-dim" was a flag-wiring bug on our side (--norms was silently ignored without --llada; fixed in diffusion_main) - the real 1024-dim GQA result is a consistent small REGRESSION (~+3 ms/step at both 4L and 28L, seq256); at the LLaDA 4096-dim shape it is neutral at seq256 and -4-5% at seq1024, but the gain does NOT stack on top of sdpa there (sdpa alone ~= sdpa+rms at S=1024). CPU decompositions bit-identical. Note: the fused kernel's reduction order differs from the decomposition, so fp32 GPU greedy trajectories shift - use --norms=raw for CPU-parity verification runs |
| odml.group_norm / layer_norm | N/A here | no group-norm in these models |
| moe (custom) | FUTURE | delegate has a MoE parser; connects to the generative-API MoE lane |
| custom_call.rotary_positional_embedding | EVALUATE | our rope rows are host-precomputed inputs by design (one serialized graph serves any position); worth an A/B against the custom op |

## Adoption by model

| model | runtime_bmm | rms_norm | sdpa | cache_update |
|---|---|---|---|---|
| talker (Qwen3-TTS 0.6B decode) | adopted (`--attention=rbmm`) | adopted (`--norms=composite`) | benched (slower at decode shapes) | multi-step harness done; adoption waits on the Metal feedback-loop gap (ledger #15) |
| codec (speech_tokenizer decoder) | n/a (conv/vocoder stack) | n/a (LayerScale, no RMS) | n/a | n/a |
| diffusion (LLaDA / bidirectional denoise step) | n/a (no KV cache) | option kept; honest read: small regression at the 0.6B shape, seq1024-only win at the 8B shape | ADOPTED (`--attention=sdpa`): MHA exact + -8.6% at S=1024; GQA via pre-fold, exact on Metal (ledger #13/#14 caveats) | n/a |
