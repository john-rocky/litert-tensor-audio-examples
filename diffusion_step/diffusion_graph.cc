// Diffusion-LM denoise-step scaffold — Stage 0. See diffusion_graph.h.

#include "tensor/examples/diffusion_step/diffusion_graph.h"

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "tensor/arithmetic.h"
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/tensor.h"

namespace litert::tensor::examples::diffusion {

namespace {

TfTensor ConstF(float value) {
  return TfTensor({.type = Type::kFP32,
                   .shape = {1},
                   .buffer = OwningCpuBuffer::Copy<Type::kFP32>({value})});
}

TfTensor ConstI(int32_t value) {
  return TfTensor({.type = Type::kI32,
                   .shape = {1},
                   .buffer = OwningCpuBuffer::Copy<Type::kI32>({value})});
}

std::vector<float> SyntheticData(size_t count, unsigned& state, float scale) {
  std::vector<float> data(count);
  for (size_t i = 0; i < count; ++i) {
    state = state * 1664525u + 1013904223u;
    data[i] = scale * (static_cast<float>(state >> 8) /
                           static_cast<float>(1u << 24) * 2.0f -
                       1.0f);
  }
  return data;
}

TfTensor SyntheticWeight(const std::string& name, const std::vector<int>& shape,
                         unsigned& state, float scale = 0.02f) {
  size_t count = 1;
  for (int d : shape) count *= static_cast<size_t>(d);
  return TfTensor({.name = name,
                   .type = Type::kFP32,
                   .shape = shape,
                   .buffer = OwningCpuBuffer::Copy<Type::kFP32>(
                       SyntheticData(count, state, scale))});
}

const TfTensor& GetWeight(const WeightMap& weights, const std::string& name) {
  auto it = weights.find(name);
  ABSL_CHECK(it != weights.end()) << "missing weight: " << name;
  return it->second;
}

bool g_rms_norm_composite = false;

TfTensor RmsNormRaw(const TfTensor& input, const TfTensor& scale, float eps) {

  TfTensor x_squared = Mul(input, input);
  int last_axis = static_cast<int>(input.GetShape().size()) - 1;
  TfTensor mean_squared = Mean(x_squared, {last_axis}, /*keep_dims=*/true);
  TfTensor inv_rms = Rsqrt(Add(mean_squared, ConstF(eps)));
  return Mul(Mul(input, inv_rms), scale);
}

std::vector<uint8_t> RmsNormAttributes(float eps) {
  flexbuffers::Builder fbb;
  fbb.Map([&]() { fbb.Float("epsilon", eps); });
  fbb.Finish();
  return fbb.GetBuffer();
}

// RMS norm, optionally as the odml.rms_norm composite (fused ML Drift
// kernels; the decomposition is the raw math).
TfTensor RmsNorm(const TfTensor& input, const TfTensor& scale, float eps) {
  if (!g_rms_norm_composite) return RmsNormRaw(input, scale, eps);
  StableHLOCompositeOptions opts{.name = "odml.rms_norm",
                                 .composite_attributes =
                                     RmsNormAttributes(eps)};
  return StableHLOComposite(
      opts,
      [eps](TfTensor x, TfTensor s) { return RmsNormRaw(x, s, eps); }, input,
      scale);
}

TfTensor Silu(const TfTensor& x) { return Mul(x, Logistic(x)); }

TfTensor ApplyRope(const TfTensor& x, const TfTensor& cos,
                   const TfTensor& sin) {
  const auto& s = x.GetShape();
  int half = s[3] / 2;
  TfTensor x1 = Slice(x, {0, 0, 0, 0}, {s[0], s[1], s[2], half});
  TfTensor x2 = Slice(x, {0, 0, 0, half}, {s[0], s[1], s[2], half});
  TfTensor rotated = Concatenation({Mul(x2, ConstF(-1.0f)), x1}, /*axis=*/3);
  return Add(Mul(x, cos), Mul(rotated, sin));
}

// HF-pairing GQA attention core without materializing repeated K/V: fold the
// kv groups into the batch axis and each group's q heads into the sequence
// axis, so q head h reads kv group h / (heads/kv) — repeat_interleave
// semantics. (A plain Tile pairs head h with group h % kv instead — wrong
// whenever 1 < kv < heads; Gather-axis-1 repeat is rejected by the Metal
// delegate.) Bidirectional: no mask. q [1, heads, S, hd], k/v [1, kv, S, hd].
TfTensor GqaAttention(const TfTensor& q, const TfTensor& k, const TfTensor& v,
                      int n_heads, int n_kv_groups, int head_dim) {
  int g = n_heads / n_kv_groups;
  int S = q.GetShape()[2];
  float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

  TfTensor qg = Reshape(q, {n_kv_groups, 1, g * S, head_dim});
  TfTensor kg = Reshape(k, {n_kv_groups, 1, S, head_dim});
  TfTensor vg = Reshape(v, {n_kv_groups, 1, S, head_dim});

  TfTensor scores = BatchMatMul(qg, kg, /*adj_x=*/false, /*adj_y=*/true);
  scores = Reshape(scores, {1, n_heads, S, S});
  scores = Mul(scores, ConstF(scale));
  TfTensor attn = Softmax(scores);
  attn = Reshape(attn, {n_kv_groups, 1, g * S, S});
  TfTensor out = BatchMatMul(attn, vg);
  return Reshape(out, {1, n_heads, S, head_dim});
}

// Bidirectional GQA attention over the full sequence — no causal mask, no
// cache. This is the structural difference from the talker step graph.
TfTensor BidirectionalAttention(const DiffusionConfig& config,
                                const TfTensor& input, const std::string& prefix,
                                const DiffusionInputs& inputs,
                                const WeightMap& weights) {
  int S = config.seq_len;

  TfTensor q = FullyConnected(input, GetWeight(weights, prefix + ".q_proj.weight"));
  TfTensor k = FullyConnected(input, GetWeight(weights, prefix + ".k_proj.weight"));
  TfTensor v = FullyConnected(input, GetWeight(weights, prefix + ".v_proj.weight"));

  q = Reshape(q, {1, S, config.n_heads, config.head_dim});
  q = Transpose(q, {0, 2, 1, 3});
  k = Reshape(k, {1, S, config.n_kv_groups, config.head_dim});
  k = Transpose(k, {0, 2, 1, 3});
  v = Reshape(v, {1, S, config.n_kv_groups, config.head_dim});
  v = Transpose(v, {0, 2, 1, 3});

  if (config.use_qk_norm) {
    q = RmsNorm(q, GetWeight(weights, prefix + ".q_norm.weight"),
                config.rms_norm_eps);
    k = RmsNorm(k, GetWeight(weights, prefix + ".k_norm.weight"),
                config.rms_norm_eps);
  }

  q = ApplyRope(q, inputs.rope_cos, inputs.rope_sin);
  k = ApplyRope(k, inputs.rope_cos, inputs.rope_sin);

  TfTensor out = GqaAttention(q, k, v, config.n_heads, config.n_kv_groups,
                              config.head_dim);

  out = Transpose(out, {0, 2, 1, 3});
  out = Reshape(out, {1, S, config.qkv_out_dim()});
  return FullyConnected(out, GetWeight(weights, prefix + ".o_proj.weight"));
}

TfTensor FeedForward(const TfTensor& input, const std::string& prefix,
                     const WeightMap& weights) {
  TfTensor gate = FullyConnected(input, GetWeight(weights, prefix + ".gate_proj.weight"));
  TfTensor up = FullyConnected(input, GetWeight(weights, prefix + ".up_proj.weight"));
  return FullyConnected(Mul(Silu(gate), up),
                        GetWeight(weights, prefix + ".down_proj.weight"));
}

}  // namespace

std::vector<TfTensor> DiffusionInputs::AsList() const {
  return {token_ids, rope_cos, rope_sin};
}

std::vector<TfTensor> DiffusionOutputs::AsList() const {
  return {new_token_ids, masked_count};
}

DiffusionInputs MakeDiffusionInputs(const DiffusionConfig& config) {
  DiffusionInputs inputs;
  inputs.token_ids = TfTensor({.name = "token_ids",
                               .type = Type::kI32,
                               .shape = {1, config.seq_len}});
  inputs.rope_cos = TfTensor({.name = "rope_cos",
                              .type = Type::kFP32,
                              .shape = {1, 1, config.seq_len, config.head_dim}});
  inputs.rope_sin = TfTensor({.name = "rope_sin",
                              .type = Type::kFP32,
                              .shape = {1, 1, config.seq_len, config.head_dim}});
  return inputs;
}

WeightMap MakeSyntheticWeights(const DiffusionConfig& config, unsigned seed) {
  WeightMap weights;
  unsigned state = seed;
  auto add = [&](const std::string& name, const std::vector<int>& shape,
                 float scale = 0.02f) {
    weights[name] = SyntheticWeight(name, shape, state, scale);
  };

  add("model.embed_tokens.weight", {config.vocab, config.emb_dim});
  for (int i = 0; i < config.n_layers; ++i) {
    std::string p = absl::StrCat("model.layers.", i);
    add(p + ".input_layernorm.weight", {config.emb_dim}, 1.0f);
    add(p + ".self_attn.q_proj.weight", {config.qkv_out_dim(), config.emb_dim});
    add(p + ".self_attn.k_proj.weight", {config.kv_out_dim(), config.emb_dim});
    add(p + ".self_attn.v_proj.weight", {config.kv_out_dim(), config.emb_dim});
    add(p + ".self_attn.o_proj.weight", {config.emb_dim, config.qkv_out_dim()});
    if (config.use_qk_norm) {
      add(p + ".self_attn.q_norm.weight", {config.head_dim}, 1.0f);
      add(p + ".self_attn.k_norm.weight", {config.head_dim}, 1.0f);
    }
    add(p + ".post_attention_layernorm.weight", {config.emb_dim}, 1.0f);
    add(p + ".mlp.gate_proj.weight", {config.hidden_dim, config.emb_dim});
    add(p + ".mlp.up_proj.weight", {config.hidden_dim, config.emb_dim});
    add(p + ".mlp.down_proj.weight", {config.emb_dim, config.hidden_dim});
  }
  add("model.norm.weight", {config.emb_dim}, 1.0f);
  add("lm_head.weight", {config.vocab, config.emb_dim});
  return weights;
}

DiffusionOutputs BuildDenoiseStep(const DiffusionConfig& config,
                                  const DiffusionInputs& inputs,
                                  const WeightMap& weights) {
  g_rms_norm_composite = config.use_rms_norm_composite;
  int S = config.seq_len;

  // --- Embedding (1-D Gather indices for the Metal delegate) ---
  TfTensor ids_flat = Reshape(inputs.token_ids, {S});  // [S]
  TfTensor hidden = Gather(GetWeight(weights, "model.embed_tokens.weight"),
                           ids_flat, /*axis=*/0);      // [S, emb]
  hidden = Reshape(hidden, {1, S, config.emb_dim});

  // --- Bidirectional backbone ---
  for (int i = 0; i < config.n_layers; ++i) {
    std::string p = absl::StrCat("model.layers.", i);
    TfTensor attn_in = RmsNorm(
        hidden, GetWeight(weights, p + ".input_layernorm.weight"),
        config.rms_norm_eps);
    hidden = Add(hidden, BidirectionalAttention(config, attn_in,
                                                p + ".self_attn", inputs,
                                                weights));
    TfTensor ffn_in = RmsNorm(
        hidden, GetWeight(weights, p + ".post_attention_layernorm.weight"),
        config.rms_norm_eps);
    hidden = Add(hidden, FeedForward(ffn_in, p + ".mlp", weights));
  }
  hidden = RmsNorm(hidden, GetWeight(weights, "model.norm.weight"),
                   config.rms_norm_eps);

  // --- In-graph remasking schedule ---
  TfTensor logits = FullyConnected(hidden, GetWeight(weights, "lm_head.weight"));
  // [1, S, vocab]
  TfTensor probs = Softmax(logits);
  TfTensor pred_ids = ArgMax(logits, -1, Type::kI32);        // [1, S]
  TfTensor conf = ReduceMax(probs, {2}, /*keep_dims=*/false);  // [1, S]

  // Only still-masked positions compete for unmasking.
  TfTensor is_masked = Equal(inputs.token_ids, ConstI(config.mask_id));  // bool
  TfTensor conf_masked = Mul(conf, Cast(is_masked, Type::kFP32));

  // Top-k most confident masked positions -> a 0/1 update mask via OneHot+Sum
  // (ScatterNd-free — Phase 0 noted ScatterNd is absent from the op set).
  std::vector<TfTensor> topk = TopK(conf_masked, config.unmask_k);
  TfTensor topk_indices = topk[1];                            // [1, k] kI32
  TfTensor onehot = OneHot(topk_indices, ConstI(S), ConstF(1.0f), ConstF(0.0f),
                           /*axis=*/-1);                      // [1, k, S]
  TfTensor update_mask = Sum(onehot, {1}, /*keep_dims=*/false);  // [1, S]
  TfTensor do_update = Greater(update_mask, ConstF(0.5f));       // bool [1, S]

  TfTensor new_ids = SelectV2(do_update, pred_ids, inputs.token_ids);
  new_ids.SetName("new_token_ids");

  // Remaining masked count (after this update) for the host stop condition.
  TfTensor still_masked = Equal(new_ids, ConstI(config.mask_id));
  TfTensor masked_count =
      Sum(Cast(still_masked, Type::kI32), {1}, /*keep_dims=*/false);  // [1]
  masked_count.SetName("masked_count");

  DiffusionOutputs outputs;
  outputs.new_token_ids = new_ids;
  outputs.masked_count = masked_count;
  return outputs;
}

}  // namespace litert::tensor::examples::diffusion
