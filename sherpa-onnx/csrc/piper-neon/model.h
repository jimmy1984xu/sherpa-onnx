// sherpa-onnx/csrc/piper-neon/model.h
//
// Copyright (c)  2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_PIPER_NEON_MODEL_H_
#define SHERPA_ONNX_CSRC_PIPER_NEON_MODEL_H_

#include <memory>
#include <functional>
#include <string>

#include "onnxruntime_cxx_api.h"  // NOLINT
#include "sherpa-onnx/csrc/offline-tts-model-config.h"
#include "sherpa-onnx/csrc/offline-tts-vits-model-meta-data.h"

namespace sherpa_onnx {

class PiperNeonModel {
 public:
  ~PiperNeonModel();

  explicit PiperNeonModel(const OfflineTtsModelConfig &config);

  Ort::Value RunStreaming(
      Ort::Value input, float speed,
      const std::function<int32_t(const float *, int32_t, float)> &callback,
      bool emit_prefix) const;
  const OfflineTtsVitsModelMetaData &GetMetaData() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_PIPER_NEON_MODEL_H_
