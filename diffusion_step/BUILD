# Diffusion-LM denoise-step scaffold — Stage 0. Drop this directory into
# LiteRT/tensor/examples/diffusion_step/ and build from the LiteRT repo root.

load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("//litert/build_common:litert_build_defs.bzl", "commandline_flag_copts")
load(
    "//litert/build_common:special_rule.bzl",
    "litert_android_linkopts",
    "litert_gpu_accelerator_prebuilts",
)

package(default_visibility = ["//visibility:public"])

cc_library(
    name = "diffusion_graph",
    srcs = ["diffusion_graph.cc"],
    hdrs = ["diffusion_graph.h"],
    deps = [
        "//tensor",
        "//tensor:arithmetic",
        "//tensor:buffer",
        "//tensor:datatypes",
        "//tensor/backends/tflite:arithmetic_tflite",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/log:absl_check",
        "@com_google_absl//absl/strings",
        "@flatbuffers",
    ],
)

cc_library(
    name = "diffusion_weights",
    srcs = ["diffusion_weights.cc"],
    hdrs = ["diffusion_weights.h"],
    deps = [
        ":diffusion_graph",
        "//tensor",
        "//tensor:buffer",
        "//tensor:datatypes",
        "//tensor/examples/gemma3:safetensor_loader",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
        "@com_google_absl//absl/strings",
    ],
)

cc_binary(
    name = "diffusion_main",
    srcs = ["diffusion_main.cc"],
    copts = select({
        "//litert:android": [],
        "//conditions:default": ["-DGOOGLE_COMMANDLINEFLAGS_FULL_API=1"],
    }) + commandline_flag_copts(),
    data = litert_gpu_accelerator_prebuilts(),
    linkopts = litert_android_linkopts(),
    deps = [
        ":diffusion_graph",
        ":diffusion_weights",
        "//litert/cc:litert_api_with_dynamic_runtime",
        "//litert/cc:litert_common",
        "//litert/cc:litert_environment",
        "//litert/cc:litert_options",
        "//litert/cc/options:litert_gpu_options",
        "//tensor",
        "//tensor:datatypes",
        "//tensor/backends/tflite:tflite_flatbuffer_conversion",
        "//tensor/runners/litert:litert_dynamic_runner",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
    ],
)
