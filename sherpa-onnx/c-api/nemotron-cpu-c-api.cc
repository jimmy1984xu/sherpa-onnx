#include "sherpa-onnx/c-api/c-api.h"

#include <cmath>
#include <cstring>
#include <memory>
#include <new>
#include <vector>

#include "sherpa-onnx/csrc/nemotron-cpu/session.h"

#if defined(__ANDROID__) && defined(__aarch64__)
#include "sherpa-onnx/csrc/nemotron-neon/runtime.h"
#endif

enum class NemotronSessionBackend { kCpu, kNeon };

struct SherpaOnnxNemotronSession {
  NemotronSessionBackend backend = NemotronSessionBackend::kCpu;
  std::unique_ptr<transai::nemotron::cpu::NemotronCpuRuntime> cpu;
#if defined(__ANDROID__) && defined(__aarch64__)
  std::unique_ptr<transai::nemotron::Runtime> neon;
#endif
};

static bool IsNeonProvider(const char *provider) {
  return provider && std::strcmp(provider, "neon") == 0;
}

SherpaOnnxNemotronSession *SherpaOnnxCreateNemotronSession(
    const SherpaOnnxNemotronSessionConfig *config) {
  if (!config) return nullptr;
  try {
    auto *handle = new SherpaOnnxNemotronSession;
    const int32_t threads = config->num_threads > 0 ? config->num_threads : 1;
#if defined(__ANDROID__) && defined(__aarch64__)
    if (IsNeonProvider(config->provider)) {
      if (!config->model || !*config->model) {
        delete handle;
        return nullptr;
      }
      handle->backend = NemotronSessionBackend::kNeon;
      handle->neon = std::make_unique<transai::nemotron::Runtime>(config->model, threads);
      return handle;
    }
#endif
    if (!config->embed_onnx || !config->step_onnx || !config->mel_filters ||
        !config->silence_embeds) {
      delete handle;
      return nullptr;
    }
    transai::nemotron::cpu::SessionConfig cfg;
    cfg.embed_onnx = config->embed_onnx;
    cfg.step_onnx = config->step_onnx;
    cfg.mel_filters = config->mel_filters;
    cfg.silence_embeds = config->silence_embeds;
    cfg.num_threads = threads;
    handle->cpu = std::make_unique<transai::nemotron::cpu::NemotronCpuRuntime>(cfg);
    return handle;
  } catch (...) {
    return nullptr;
  }
}

void SherpaOnnxDestroyNemotronSession(const SherpaOnnxNemotronSession *session) {
  delete session;
}

void SherpaOnnxNemotronSessionResetContinuity(const SherpaOnnxNemotronSession *session) {
  if (!session) return;
#if defined(__ANDROID__) && defined(__aarch64__)
  if (session->backend == NemotronSessionBackend::kNeon && session->neon) {
    session->neon->ResetContinuity();
    return;
  }
#endif
  if (session->cpu) session->cpu->ResetContinuity();
}

int32_t SherpaOnnxNemotronSessionAcceptWaveform(const SherpaOnnxNemotronSession *session,
                                                const float *samples, int32_t n,
                                                int64_t offset_samples,
                                                SherpaOnnxNemotronTimelineView *out) {
  if (!session || !samples || n <= 0 || !out) return 0;
#if defined(__ANDROID__) && defined(__aarch64__)
  if (session->backend == NemotronSessionBackend::kNeon && session->neon) {
    thread_local std::vector<int16_t> pcm_scratch;
    pcm_scratch.resize(static_cast<size_t>(n));
    for (int32_t i = 0; i < n; ++i) {
      const float value = std::fmin(1.f, std::fmax(-1.f, samples[i]));
      pcm_scratch[static_cast<size_t>(i)] = static_cast<int16_t>(value * 32767.f);
    }
    transai::nemotron::NeonTimelineView view;
    if (!session->neon->AcceptWaveform(pcm_scratch.data(), n, offset_samples, &view)) {
      return -1;
    }
    out->frame_count = 0;
    out->num_speakers = view.num_speakers;
    out->timeline = nullptr;
    out->rttm = view.rttm;
    if (!view.rttm || !view.rttm[0]) return 0;
    return 1;
  }
#endif
  if (!session->cpu) return -1;
  transai::nemotron::cpu::TimelineView view;
  const bool ok = session->cpu->AcceptWaveform(samples, n, offset_samples, &view);
  if (!ok) return -1;
  out->frame_count = view.frame_count;
  out->num_speakers = view.num_speakers;
  out->timeline = view.timeline;
  out->rttm = nullptr;
  if (view.frame_count <= 0) return 0;
  return 1;
}