#pragma once

#include <cstdint>
#include <memory>
#include <string>

namespace transai::nemotron::cpu {

struct SessionConfig {
  std::string embed_onnx;
  std::string step_onnx;
  std::string mel_filters;
  std::string silence_embeds;
  int32_t num_threads = 1;
};

struct TimelineView {
  int32_t frame_count = 0;
  int32_t num_speakers = 8;
  const float *timeline = nullptr;
};

class NemotronCpuRuntime {
 public:
  explicit NemotronCpuRuntime(const SessionConfig &cfg);
  ~NemotronCpuRuntime();

  NemotronCpuRuntime(const NemotronCpuRuntime &) = delete;
  NemotronCpuRuntime &operator=(const NemotronCpuRuntime &) = delete;

  void ResetContinuity();
  bool AcceptWaveform(const float *samples, int32_t sample_count, int64_t offset_samples, TimelineView *out);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace transai::nemotron::cpu