// Minimal repro: FULLY_CONNECTED with INT4 BLOCKWISE weights hangs the
// LiteRT GPU (ML Drift) delegate at initialization on Metal (macOS arm64).
//
//   bazel build -c opt //tensor/examples/talker_step:int4_fc_repro
//   int4_fc_repro --accelerator=gpu                # HANGS at Create (Metal)
//   int4_fc_repro --accelerator=cpu                # OK on XNNPACK
//   int4_fc_repro --accelerator=gpu --per_channel  # OK on Metal
//
// One op, no checkpoint: y = FullyConnected(x [1,8,512], W [512,512] int4).
// Blockwise = block 32 along the input dim, fp16-rounded scales, codes in
// [-7,7], TFLite kI4 bit-packing. Observed hang stack (sampled, CPU-bound,
// flat RSS): ml_drift::GpuModelBuilder::FullyConnected(Int4Attributes) ->
// FullyConnectedInt4ExternalWeights -> AddTensor ->
// absl::flat_hash_map::find_or_prepare_insert_large.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "absl/flags/flag.h"  // from @com_google_absl
#include "absl/flags/parse.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/options/litert_gpu_options.h"
#include "tensor/arithmetic.h"
#include "tensor/backends/tflite/tflite_flatbuffer_conversion.h"
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/runners/litert/litert_dynamic_runner.h"
#include "tensor/tensor.h"

ABSL_FLAG(std::string, accelerator, "gpu", "cpu|gpu");
ABSL_FLAG(bool, per_channel, false,
          "use per-channel int4 metadata instead of blockwise (compiles and "
          "runs fine on Metal — isolates blockwise as the trigger)");
ABSL_FLAG(bool, buffer_storage, false,
          "set GpuOptions BufferStorageType=kBuffer (the talker config that "
          "hangs; routes weights through the ExternalWeights path)");
ABSL_FLAG(std::string, tflite_path, "/tmp/int4_fc_repro.tflite",
          "where to write the one-op flatbuffer");
ABSL_FLAG(int, fc_out, 512, "FC output dim");
ABSL_FLAG(int, fc_in, 512, "FC input dim (must be divisible by 32)");
ABSL_FLAG(int, chain, 1,
          "number of chained square FCs after the first (distinct weights)");
ABSL_FLAG(int, seq, 8, "input sequence length");
ABSL_FLAG(bool, two_sigs, false,
          "add a second signature that SHARES the same int4 weight tensors "
          "(the talker prefill/decode pattern)");
ABSL_FLAG(bool, unshared, false,
          "with --two_sigs: give the second signature its own COPY of the "
          "weights (isolates sharing as the trigger)");

namespace {

using ::litert::tensor::BlockwiseQuantization;
using ::litert::tensor::Create;
using ::litert::tensor::FullyConnected;
using ::litert::tensor::LitertDynamicRunner;
using ::litert::tensor::ModelFactory;
using ::litert::tensor::OwningCpuBuffer;
using ::litert::tensor::PerChannelAffineQuantization;
using ::litert::tensor::TensorHandle;
using ::litert::tensor::Type;
using TfTensor = ::litert::tensor::Tensor<::litert::tensor::TfLiteMixinTag>;

constexpr int kBlock = 32;

TfTensor MakeInt4Weight(const std::string& name, int out_dim, int in_dim,
                        bool per_channel, int seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> dist(0.0f, 0.02f);
  std::vector<float> w(static_cast<size_t>(out_dim) * in_dim);
  for (auto& v : w) v = dist(rng);

  const int block = per_channel ? in_dim : kBlock;
  const int blocks_per_row = in_dim / block;
  std::vector<float> scales(static_cast<size_t>(out_dim) * blocks_per_row);
  std::vector<int8_t> q(w.size());
  for (int c = 0; c < out_dim; ++c) {
    for (int b = 0; b < blocks_per_row; ++b) {
      float amax = 0.0f;
      for (int i = b * block; i < (b + 1) * block; ++i)
        amax = std::max(amax, std::fabs(w[c * in_dim + i]));
      float scale = amax > 0.0f ? amax / 7.0f : 1.0f;
      scales[c * blocks_per_row + b] = scale;
      for (int i = b * block; i < (b + 1) * block; ++i) {
        float v = std::round(w[c * in_dim + i] / scale);
        q[c * in_dim + i] =
            static_cast<int8_t>(std::min(7.0f, std::max(-7.0f, v)));
      }
    }
  }
  std::vector<char> packed((q.size() + 1) / 2, 0);
  for (size_t i = 0; i < q.size(); ++i) {
    char nib = static_cast<char>(q[i] & 0xF);
    packed[i / 2] |= (i % 2 == 0) ? nib : static_cast<char>(nib << 4);
  }
  TfTensor weight({.name = name,
                   .type = Type::kI4,
                   .shape = {out_dim, in_dim},
                   .buffer = OwningCpuBuffer::Copy(packed.data(),
                                                   packed.size())});
  if (per_channel) {
    weight.SetQuantization(std::make_shared<PerChannelAffineQuantization>(
        std::move(scales), std::vector<int64_t>(out_dim, 0),
        /*quantized_dimension=*/0));
  } else {
    weight.SetQuantization(std::make_shared<BlockwiseQuantization>(
        std::move(scales), std::vector<int64_t>(), kBlock,
        /*quantized_dimension=*/0));
  }
  return weight;
}

absl::Status RunRepro() {
  const int out_dim = absl::GetFlag(FLAGS_fc_out);
  const int in_dim = absl::GetFlag(FLAGS_fc_in);
  const int chain = absl::GetFlag(FLAGS_chain);
  const int seq = absl::GetFlag(FLAGS_seq);
  const bool per_channel = absl::GetFlag(FLAGS_per_channel);

  std::vector<TfTensor> weights;
  weights.push_back(
      MakeInt4Weight("weight_0", out_dim, in_dim, per_channel, 42));
  for (int i = 1; i < chain; ++i) {
    weights.push_back(MakeInt4Weight(absl::StrCat("weight_", i), out_dim,
                                     out_dim, per_channel, 42 + i));
  }

  TfTensor x({.name = "x", .type = Type::kFP32, .shape = {1, seq, in_dim}});
  TfTensor y = FullyConnected(x, weights[0]);
  for (int i = 1; i < chain; ++i) y = FullyConnected(y, weights[i]);

  ModelFactory factory;
  std::vector<TensorHandle> ins = {x};
  std::vector<TensorHandle> outs = {y};
  auto st = factory.AddSignature(ins, outs, "main");
  if (!st.ok()) return st;

  if (absl::GetFlag(FLAGS_two_sigs)) {
    // Second signature sharing the SAME weight tensor handles — the talker
    // prefill/decode pattern (decode seq 1).
    TfTensor x2({.name = "x2", .type = Type::kFP32, .shape = {1, 1, in_dim}});
    TfTensor y2;
    if (absl::GetFlag(FLAGS_unshared)) {
      y2 = FullyConnected(x2, MakeInt4Weight("weight2_0", out_dim, in_dim,
                                             per_channel, 42));
      for (int i = 1; i < chain; ++i) {
        y2 = FullyConnected(y2, MakeInt4Weight(absl::StrCat("weight2_", i),
                                               out_dim, out_dim, per_channel,
                                               42 + i));
      }
    } else {
      y2 = FullyConnected(x2, weights[0]);
      for (int i = 1; i < chain; ++i) y2 = FullyConnected(y2, weights[i]);
    }
    std::vector<TensorHandle> ins2 = {x2};
    std::vector<TensorHandle> outs2 = {y2};
    st = factory.AddSignature(ins2, outs2, "second");
    if (!st.ok()) return st;
    std::cout << "two signatures sharing int4 weights" << std::endl;
  }
  const std::string path = absl::GetFlag(FLAGS_tflite_path);
  st = factory.Save(path);
  if (!st.ok()) return st;
  std::cout << "Serialized: " << path << std::endl;

  auto env = ::litert::Environment::Create({});
  if (!env) return absl::InternalError("Environment::Create failed");
  auto options = ::litert::Options::Create();
  if (!options) return absl::InternalError("Options::Create failed");
  const bool use_gpu = absl::GetFlag(FLAGS_accelerator) == "gpu";
  options->SetHardwareAccelerators(use_gpu ? ::litert::HwAccelerators::kGpu
                                           : ::litert::HwAccelerators::kCpu);
  if (use_gpu && absl::GetFlag(FLAGS_buffer_storage)) {
    auto gpu_options = options->GetGpuOptions();
    if (gpu_options.HasValue()) {
      gpu_options->SetBufferStorageType(
          ::litert::GpuOptions::BufferStorageType::kBuffer);
      std::cout << "gpu buffer storage: buffer" << std::endl;
    }
  }
  std::cout << "Creating compiled model (" << (use_gpu ? "gpu" : "cpu")
            << ") — blockwise int4 on Metal HANGS HERE..." << std::endl;
  auto runner_or = LitertDynamicRunner::Create(*env, path, *options);
  if (!runner_or.ok()) return runner_or.status();
  auto runner = std::move(*runner_or);

  std::vector<float> in(static_cast<size_t>(seq) * in_dim, 0.5f);
  st = runner.SetInput(
      "main", "x",
      Create("x", Type::kFP32, {1, seq, in_dim}, std::move(in)));
  if (!st.ok()) return st;
  st = runner.Run("main");
  if (!st.ok()) return st;
  auto out = runner.GetOutput("main", 0);
  if (!out.ok()) return out.status();
  std::cout << "OK: model created and ran (no hang)" << std::endl;
  return absl::OkStatus();
}

}  // namespace

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  auto st = RunRepro();
  if (!st.ok()) {
    std::cerr << "ERROR: " << st.message() << std::endl;
    return 1;
  }
  return 0;
}
