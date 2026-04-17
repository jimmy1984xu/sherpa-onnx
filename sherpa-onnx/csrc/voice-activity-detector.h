// sherpa-onnx/csrc/voice-activity-detector.h
//
// Copyright (c)  2023  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_VOICE_ACTIVITY_DETECTOR_H_
#define SHERPA_ONNX_CSRC_VOICE_ACTIVITY_DETECTOR_H_

#include <memory>
#include <vector>

#include "sherpa-onnx/csrc/vad-model-config.h"

namespace sherpa_onnx {

struct SpeechSegment {
  int32_t start;  // in samples
  std::vector<float> samples;
};

class VoiceActivityDetector {
 public:
  explicit VoiceActivityDetector(const VadModelConfig &config,
                                 float buffer_size_in_seconds = 60);

  template <typename Manager>
  VoiceActivityDetector(Manager *mgr, const VadModelConfig &config,
                        float buffer_size_in_seconds = 60);

  ~VoiceActivityDetector();

  void AcceptWaveform(const float *samples, int32_t n);
  float Compute(const float *samples, int32_t n);

  bool Empty() const;
  void Pop();
  void Clear();

  // It is an error to call Front() if Empty() returns true.
  //
  // The returned reference is valid until the next call to any
  // methods of VoiceActivityDetector.
  const SpeechSegment &Front() const;

  bool IsSpeechDetected() const;

  // It is empty if IsSpeechDetected() returns false
  SpeechSegment CurrentSpeechSegment() const;

  int32_t CurrentSegmentStart() const;

  int32_t BufferHead() const;

  int32_t BufferTail() const;

  bool CopyBufferRange(int32_t start, int32_t end, float *out,
                       int32_t n) const;

  // Dynamically increase the max utterance length used by VAD force-cut logic.
  //
  // Return true on success. Return false if max_duration_seconds is invalid,
  // smaller than the current value, or larger than the buffer capacity given
  // at construction time.
  bool SetMaxUtteranceLength(float max_duration_seconds) const;

  void Reset() const;

  // At the end of the utterance, you can invoke this method so that
  // the last speech segment can be detected.
  void Flush() const;

  const VadModelConfig &GetConfig() const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_VOICE_ACTIVITY_DETECTOR_H_
