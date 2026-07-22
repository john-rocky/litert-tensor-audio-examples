# odml composite coverage program

Goal: adopt and benchmark each odml composite / custom op across these
examples, one at a time, and report what we find - a running showcase of the
composite path from an external C++ Tensor API user. Suggested by the LiteRT
team's pointers (sdpa, runtime_bmm); we continue down the list.

| composite | status here | measured / notes |
|---|---|---|
| odml.runtime_bmm | ADOPTED (talker `--attention=rbmm`) | attention 3.0-3.4x vs raw ops on Metal fp32, exact at every fill; e2e frame -4-5%; decompositions also the fastest CPU form. CL init issue open (ledger #6) |
| odml.scaled_dot_product_attention | BENCHED | Metal kernel slower than the raw BMM chain at decode shapes (4.4 vs 2.6 ms/28 blocks); MHA-shaped only on Metal; CL grouped faults at 4+ nodes (ledger #7) |
| odml.cache_update | BENCHED (attn_bench `--variant=cache_rbmm`) | WORKS on Metal: delegate matches the composite (UCL shader compiles) and pairs with runtime_bmm; single decode step ~0.35 ms, exact vs CPU at fill=1. Contract learned: caches are delegate-managed on GPU (K stored in the QK FC-weights layout, V in the AV layout; the host-side cache inputs feed only the CPU decomposition), and the control tensor packs write offset (elem 0), active tokens (elem 1) and the runtime_bmm bound (elem 2) in one [1,1,1,7] int32. Next: multi-step loop harness so GPU cache state accumulates across calls, then talker integration |
| odml.rms_norm | ADOPTED as an option (`--norms=composite` in talker_main and diffusion_main) | talker 28L Metal: fp32 29.0 -> 28.1 ms/frame, fp16 21.3 -> 20.3 (about -1 ms on top of runtime_bmm; combined stack -7.5 to -8.6% vs the original raw graph). diffusion: neutral within noise at both the 1024-dim and 4096-dim shapes. CPU decompositions bit-identical. Note: the fused kernel's reduction order differs from the decomposition, so fp32 GPU greedy trajectories shift - use --norms=raw for CPU-parity verification runs |
| odml.group_norm / layer_norm | N/A here | no group-norm in these models |
| moe (custom) | FUTURE | delegate has a MoE parser; connects to the generative-API MoE lane |
| custom_call.rotary_positional_embedding | EVALUATE | our rope rows are host-precomputed inputs by design (one serialized graph serves any position); worth an A/B against the custom op |

## Adoption by model

| model | runtime_bmm | rms_norm | sdpa | cache_update |
|---|---|---|---|---|
| talker (Qwen3-TTS 0.6B decode) | adopted (`--attention=rbmm`) | adopted (`--norms=composite`) | benched (slower at decode shapes) | probe done; multi-step harness next |
| codec (speech_tokenizer decoder) | n/a (conv/vocoder stack) | n/a (LayerScale, no RMS) | n/a | n/a |
| diffusion (LLaDA / bidirectional denoise step) | n/a (no KV cache) | adopted (neutral, no loss) | NEXT: full-seq MHA is sdpa's home shape - A/B planned | n/a |
