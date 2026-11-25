#include "sherpa-onnx/c-api/c-api.h"

#if defined(__ANDROID__) && defined(__aarch64__)
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <exception>
#include <memory>
#include <string>

#include "sherpa-onnx/csrc/nemotron-neon/runtime.h"
#include "sherpa-onnx/csrc/neon-cpu-affinity.h"
#include "sherpa-onnx/csrc/neon-threads.h"

struct SherpaOnnxNemotronDiarization {
  transai::nemotron::Runtime impl;
  sherpa_onnx::neon::CpuAffinity affinity;
};

namespace {

bool IsNeonModel(const char *model, const char *provider) {
  return model && *model && provider && std::strcmp(provider, "neon") == 0;
}

template <typename F>
auto Call(const char *method, F fn) -> decltype(fn()) {
  try {
    return fn();
  } catch (const std::exception &e) {
    std::fprintf(stderr, "%s: %s\n", method, e.what());
    return nullptr;
  }
}

template <typename Handle, typename Config>
const Handle *Create(const Config *config,
                     const SherpaOnnxNeonCpuConfig *cpu, const char *method) {
  if (!config || !IsNeonModel(config->model, config->provider)) return nullptr;
  using namespace sherpa_onnx::neon;
  if (!ValidNumThreads(config->num_threads)) {
    std::fprintf(stderr, "%s: num_threads must be 0–8\n", method);
    return nullptr;
  }
  return Call(method, [&] {
    auto affinity = cpu
        ? CpuAffinity::Resolve(cpu->cpu_cores, cpu->num_cpu_cores, config->num_threads)
        : CpuAffinity{};
    ScopedCpuAffinityRequest placement(affinity);
    auto *handle = new Handle{{config->model, ResolveNumThreads(config->num_threads)}, {}};
    placement.ApplyWorkers();
    handle->affinity = affinity;
    return handle;
  });
}

}  // namespace

static_assert(SHERPA_ONNX_NEMOTRON_MAX_SAMPLES == transai::nemotron::kMaxSamples);

const SherpaOnnxNemotronDiarization *SherpaOnnxCreateNemotronDiarizationWithCpuConfig(
    const SherpaOnnxNemotronDiarizationConfig *config,
    const SherpaOnnxNeonCpuConfig *cpu) {
  return Create<SherpaOnnxNemotronDiarization>(config, cpu, __func__);
}

void SherpaOnnxDestroyNemotronDiarization(
    const SherpaOnnxNemotronDiarization *diarizer) {
  delete diarizer;
}

const char *SherpaOnnxNemotronDiarizationProcess(
    const SherpaOnnxNemotronDiarization *diarizer, const int16_t *samples,
    int32_t n, const char *recording_id) {
  if (!diarizer || !samples || n <= 0 ||
      n > SHERPA_ONNX_NEMOTRON_MAX_SAMPLES || !recording_id) {
    return nullptr;
  }
  return Call(__func__, [&] {
    const sherpa_onnx::neon::ScopedInferenceCore affinity(diarizer->affinity);
    const auto text = diarizer->impl.Process(samples, n, recording_id);
    auto result = std::make_unique<char[]>(text.size() + 1);
    std::memcpy(result.get(), text.c_str(), text.size() + 1);
    return result.release();
  });
}

void SherpaOnnxDestroyNemotronDiarizationResult(const char *rttm) {
  delete[] rttm;
}

#else

const SherpaOnnxNemotronDiarization *SherpaOnnxCreateNemotronDiarizationWithCpuConfig(
    const SherpaOnnxNemotronDiarizationConfig *, const SherpaOnnxNeonCpuConfig *) {
  return nullptr;
}

void SherpaOnnxDestroyNemotronDiarization(
    const SherpaOnnxNemotronDiarization *) {}

const char *SherpaOnnxNemotronDiarizationProcess(
    const SherpaOnnxNemotronDiarization *, const int16_t *, int32_t,
    const char *) {
  return nullptr;
}

void SherpaOnnxDestroyNemotronDiarizationResult(const char *) {}

#endif

