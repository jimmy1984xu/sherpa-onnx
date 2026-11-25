#include "sherpa-onnx/c-api/c-api.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <exception>
#include <memory>
#include <string>

#include "sherpa-onnx/csrc/voxlingua107-cpu/runtime.h"

#if defined(__ANDROID__) && defined(__aarch64__)
#include "sherpa-onnx/csrc/neon-cpu-affinity.h"
#include "sherpa-onnx/csrc/neon-threads.h"
#include "sherpa-onnx/csrc/voxlingua107-neon/runtime.h"
#endif

enum class VoxLinguaBackend { kCpu, kNeon };

struct SherpaOnnxVoxLingua107 {
  VoxLinguaBackend backend = VoxLinguaBackend::kCpu;
  std::unique_ptr<transai::voxlingua107::cpu::Runtime> cpu;
#if defined(__ANDROID__) && defined(__aarch64__)
  std::unique_ptr<transai::voxlingua107::Runtime> neon;
  sherpa_onnx::neon::CpuAffinity affinity;
#endif
};

namespace {

struct VoxResult : SherpaOnnxVoxLingua107Result {
  std::string language_storage;
};

bool IsNeonProvider(const char *provider) {
  return provider && std::strcmp(provider, "neon") == 0;
}

template <typename F>
auto Call(const char *method, F fn) -> decltype(fn()) {
  try {
    return fn();
  } catch (const std::exception &e) {
    std::fprintf(stderr, "%s: %s\n", method, e.what());
    return decltype(fn()){};
  }
}

void FillResult(const transai::voxlingua107::cpu::Result &source, VoxResult *result) {
  result->language_id = static_cast<int32_t>(source.language_id);
  result->language_storage = source.language;
  result->language = result->language_storage.c_str();
  result->probability = source.probability;
  std::copy(source.log_probabilities.begin(), source.log_probabilities.end(),
            result->log_probabilities);
  std::copy(source.embedding.begin(), source.embedding.end(), result->embedding);
}

#if defined(__ANDROID__) && defined(__aarch64__)
void FillResult(transai::voxlingua107::Result source, VoxResult *result) {
  result->language_id = static_cast<int32_t>(source.language_id);
  result->language_storage = std::move(source.language);
  result->language = result->language_storage.c_str();
  result->probability = source.probability;
  std::copy(source.log_probabilities.begin(), source.log_probabilities.end(),
            result->log_probabilities);
  std::copy(source.embedding.begin(), source.embedding.end(), result->embedding);
}
#endif

}  // namespace

const SherpaOnnxVoxLingua107 *SherpaOnnxCreateVoxLingua107WithCpuConfig(
    const SherpaOnnxVoxLingua107Config *config, const SherpaOnnxNeonCpuConfig *cpu) {
  if (!config || !config->model || !*config->model) return nullptr;
  const int32_t threads = config->num_threads > 0 ? config->num_threads : 1;
  return Call(__func__, [&] {
    auto *handle = new SherpaOnnxVoxLingua107;
#if defined(__ANDROID__) && defined(__aarch64__)
    if (IsNeonProvider(config->provider)) {
      using namespace sherpa_onnx::neon;
      if (!ValidNumThreads(config->num_threads)) {
        delete handle;
        return static_cast<const SherpaOnnxVoxLingua107 *>(nullptr);
      }
      auto affinity = cpu ? CpuAffinity::Resolve(cpu->cpu_cores, cpu->num_cpu_cores, config->num_threads)
                          : CpuAffinity{};
      ScopedCpuAffinityRequest placement(affinity);
      handle->backend = VoxLinguaBackend::kNeon;
      handle->neon = std::make_unique<transai::voxlingua107::Runtime>(config->model, threads);
      placement.ApplyWorkers();
      handle->affinity = affinity;
      return static_cast<const SherpaOnnxVoxLingua107 *>(handle);
    }
#endif
    if (!config->lang_map || !*config->lang_map) {
      delete handle;
      return static_cast<const SherpaOnnxVoxLingua107 *>(nullptr);
    }
    handle->backend = VoxLinguaBackend::kCpu;
    handle->cpu = std::make_unique<transai::voxlingua107::cpu::Runtime>(
        config->model, config->lang_map, threads);
    return static_cast<const SherpaOnnxVoxLingua107 *>(handle);
  });
}

void SherpaOnnxDestroyVoxLingua107(const SherpaOnnxVoxLingua107 *identifier) {
  delete identifier;
}

const SherpaOnnxVoxLingua107Result *SherpaOnnxVoxLingua107Compute(
    const SherpaOnnxVoxLingua107 *identifier, const float *samples, int32_t n) {
  if (!identifier || !samples || n <= 0) return nullptr;
  return Call(__func__, [&] {
    auto result = std::make_unique<VoxResult>();
#if defined(__ANDROID__) && defined(__aarch64__)
    if (identifier->backend == VoxLinguaBackend::kNeon && identifier->neon) {
      const sherpa_onnx::neon::ScopedInferenceCore affinity(identifier->affinity);
      FillResult(identifier->neon->Process(samples, n), result.get());
      return static_cast<const SherpaOnnxVoxLingua107Result *>(result.release());
    }
#endif
    if (!identifier->cpu) return static_cast<const SherpaOnnxVoxLingua107Result *>(nullptr);
    FillResult(identifier->cpu->Process(samples, n), result.get());
    return static_cast<const SherpaOnnxVoxLingua107Result *>(result.release());
  });
}

void SherpaOnnxDestroyVoxLingua107Result(const SherpaOnnxVoxLingua107Result *result) {
  delete static_cast<const VoxResult *>(result);
}