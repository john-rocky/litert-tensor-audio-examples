# LiteRT Tensor API audio-LM examples (Qwen3-TTS 0.6B)

A working audio-LM decode loop authored directly with the [LiteRT Tensor
API](https://github.com/google-ai-edge/LiteRT/tree/main/tensor) — no
converter. Companion code to the "Audio-LM Loop Design Note (Tensor API)".

- `talker_step/` — the talker decode-step graph: prefill/decode signatures,
  KV caches as signature I/O (DynamicUpdateSlice, in-place-aliasing
  verified), cb0 + 15-group code predictor + feedback aggregation in-graph
  (both code-predictor forms: re-forward and KV-cached incremental), int8
  per-channel / int4 blockwise-32 weight quantization, pre-quantized (GPTQ)
  companion-file loading, Gumbel noise-as-input temperature sampling.
  Also `int4_fc_repro.cc`: standalone minimal repro for the Metal-delegate
  hang with two signatures sharing blockwise-int4 FC weights.
- `codec_decode/` — the full speech_tokenizer decoder (RVQ + transformer +
  vocoder) as one Tensor API graph.
- `verify/` — numpy references for both graphs, GPTQ quantizer/eval
  (`gptq_quantize.py` emits the exact blockwise-32 layout the C++ loader
  consumes), speaker x-vector enrollment, and `audio_smoke.py`, the
  end-to-end text -> wav driver.

## Building

The directories are overlays for a LiteRT checkout:

```
cp -r talker_step codec_decode <litert>/tensor/examples/
cd <litert>
bazel build //tensor/examples/talker_step:talker_main
bazel build //tensor/examples/codec_decode:codec_main
# Android: ANDROID_NDK_HOME=<ndk> bazel build -c opt --config=android_arm64 ...
```

macOS GPU runs need `libLiteRtMetalAccelerator.dylib` (shipped in the
target's runfiles) in the working directory.

## Weights

Not included. Get the Qwen3-TTS-12Hz-0.6B-Base checkpoint (safetensors)
from HuggingFace; `verify/gptq_quantize.py` produces the optional int4
companion file. See `verify/audio_smoke.py --help` for the pipeline flags.

## Measured highlights

- Pixel 8a CPU, GPTQ int4 blockwise, KV-cached CP: 99.6 ms/frame = RTF 1.24
  (80 ms frame budget), thermal-controlled.
- M4 Max: CPU int4 34.2 ms/frame; Metal fp16 25.2 ms/frame.
- Codec on GPU: 17.6 ms (Mali) / 47.4 ms (Metal) per 89-frame window.
- KV-cache rebind by buffer handle is zero-copy and in-place-DUS-eligible;
  naive copy-rebind costs +24%/frame (M4) / +35%/frame (Pixel 8a).
