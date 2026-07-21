// Load-probe: create a LitertDynamicRunner (= compile all signatures on the
// chosen accelerator) for an ARBITRARY existing .tflite, without running it.
// Used to A/B the converter-produced Qwen3-TTS talker against the
// Tensor-API-authored one on the Metal delegate (blockwise-int4 hang check).
//
//   bazel build -c opt //tensor/examples/talker_step:load_probe
//   load_probe --model=/path/model.tflite --accelerator=gpu

#include <iostream>
#include <string>

#include "absl/flags/flag.h"  // from @com_google_absl
#include "absl/flags/parse.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/options/litert_gpu_options.h"
#include "tensor/runners/litert/litert_dynamic_runner.h"

ABSL_FLAG(std::string, model, "", "path to an existing .tflite flatbuffer");
ABSL_FLAG(std::string, accelerator, "gpu", "cpu|gpu");
ABSL_FLAG(bool, buffer_storage, false,
          "set GpuOptions BufferStorageType=kBuffer");

namespace {

using ::litert::tensor::LitertDynamicRunner;

absl::Status RunProbe() {
  const std::string model_path = absl::GetFlag(FLAGS_model);
  if (model_path.empty()) {
    return absl::InvalidArgumentError("--model is required");
  }

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

  std::cout << "load_probe: compiling " << model_path << " on "
            << (use_gpu ? "gpu" : "cpu") << " ..." << std::endl;
  auto runner_or = LitertDynamicRunner::Create(*env, model_path, *options);
  if (!runner_or.ok()) return runner_or.status();
  std::cout << "OK: compiled model created (all signatures initialized)"
            << std::endl;
  return absl::OkStatus();
}

}  // namespace

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  auto st = RunProbe();
  if (!st.ok()) {
    std::cerr << "ERROR: " << st.message() << std::endl;
    return 1;
  }
  return 0;
}
