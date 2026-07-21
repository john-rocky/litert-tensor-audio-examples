// Codec decoder checkpoint loading + host-side preprocessing.
//
// Reads speech_tokenizer/model.safetensors (via the gemma3 example's
// SafetensorLoader) and returns graph-ready constants:
//  - conv weights permuted to TFLite layouts: Conv [out,1,k,in], depthwise
//    [1,1,k,C], transpose-conv [out,1,k,in] (from torch [in,out,k]);
//  - EMA codebooks divided out: codebook_table_q = embedding_sum /
//    max(cluster_usage, eps), q = 0..15;
//  - RVQ 1x1 output projections squeezed to 2-D matmul weights;
//  - SnakeBeta parameters folded: <base>.alpha_exp = exp(alpha),
//    <base>.beta_inv = 1/(exp(beta)+1e-9);
//  - the gemma3 loader's +1.0 Gemma-norm offset undone on *layernorm* /
//    *norm.weight tensors (Qwen norms are raw; ConvNeXt LayerNorm bias is
//    unaffected).

#ifndef SPEECHLM_STAGE1_CODEC_DECODE_CODEC_WEIGHTS_H_
#define SPEECHLM_STAGE1_CODEC_DECODE_CODEC_WEIGHTS_H_

#include <string>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "tensor/examples/codec_decode/codec_config.h"
#include "tensor/examples/codec_decode/codec_graph.h"

namespace litert::tensor::examples::codec {

absl::StatusOr<WeightMap> LoadCodecWeights(const CodecConfig& config,
                                           const std::string& path);

}  // namespace litert::tensor::examples::codec

#endif  // SPEECHLM_STAGE1_CODEC_DECODE_CODEC_WEIGHTS_H_
