// sherpa-onnx/csrc/speaker-segmentation.h
//
// Copyright (c) 2026

#ifndef SHERPA_ONNX_CSRC_SPEAKER_SEGMENTATION_H_
#define SHERPA_ONNX_CSRC_SPEAKER_SEGMENTATION_H_

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "sherpa-onnx/csrc/offline-speaker-segmentation-model-config.h"
#include "sherpa-onnx/csrc/offline-speaker-segmentation-pyannote-model-meta-data.h"

namespace sherpa_onnx {

enum SpeakerSegmentationSpanFlag : int32_t {
  kSpeakerSegmentationContinue = 0,
  kSpeakerSegmentationSpeakerCountChanged = 1 << 0,
  kSpeakerSegmentationSingleSpeakerChanged = 1 << 1,
  kSpeakerSegmentationInputFinished = 1 << 2,
};

struct SpeakerSegmentationConfig {
  OfflineSpeakerSegmentationModelConfig model;
  float min_duration_on = 0.30F;
  float min_duration_off = 0.50F;
  float change_vote_threshold = 0.50F;

  bool Validate() const;
  std::string ToString() const;
};

struct SpeakerSegmentationSpan {
  float start;
  float end;
  int32_t speaker_count;
  int32_t flag;

  // Fused session-local pyannote track activity. The low three bits are
  // valid; this is deliberately not a global speaker ID.
  uint8_t local_speaker_mask;
  // Fused winning powerset probability for local_speaker_mask, in [0, 1].
  float local_speaker_mask_confidence;
};

// This callback is a test seam for the object-level streaming and fusion
// logic. It receives one zero-padded model window and returns per-frame,
// three-bit local masks after powerset decoding.
using SpeakerSegmentationTestForward =
    std::function<std::vector<uint8_t>(const std::vector<float> &window)>;

// An object-level streaming runner for a pyannote segmentation model. It does
// not compute embeddings, cluster speakers, or expose local speaker IDs.
class SpeakerSegmentation {
 public:
  explicit SpeakerSegmentation(const SpeakerSegmentationConfig &config);
  ~SpeakerSegmentation();

  SpeakerSegmentation(const SpeakerSegmentation &) = delete;
  SpeakerSegmentation &operator=(const SpeakerSegmentation &) = delete;

  // Creates a runner with an injected local-mask forward function. This is
  // intended for unit testing the streaming state machine without ONNX.
  static std::unique_ptr<SpeakerSegmentation> CreateForTesting(
      const SpeakerSegmentationConfig &config,
      const OfflineSpeakerSegmentationPyannoteModelMetaData &meta_data,
      SpeakerSegmentationTestForward forward);

  int32_t SampleRate() const;

  // n must not be negative. Calling this after InputFinished() is rejected
  // until Reset() is called.
  void AcceptWaveform(const float *samples, int32_t n);

  // Pads all remaining shift-grid windows on the right, finalizes their spans,
  // and marks the final returned span with kSpeakerSegmentationInputFinished.
  // This method is idempotent.
  void InputFinished();

  bool Empty() const;

  // It is an error to call Front() if Empty() returns true. The returned
  // reference is owned by this object and remains valid until the next call to
  // a non-const method on the same object.
  const SpeakerSegmentationSpan &Front() const;

  void Pop();
  void Reset();

 private:
  SpeakerSegmentation(
      const SpeakerSegmentationConfig &config,
      const OfflineSpeakerSegmentationPyannoteModelMetaData &meta_data,
      SpeakerSegmentationTestForward forward);

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_SPEAKER_SEGMENTATION_H_
