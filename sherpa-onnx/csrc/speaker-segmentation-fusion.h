// sherpa-onnx/csrc/speaker-segmentation-fusion.h
//
// Copyright (c) 2026

#ifndef SHERPA_ONNX_CSRC_SPEAKER_SEGMENTATION_FUSION_H_
#define SHERPA_ONNX_CSRC_SPEAKER_SEGMENTATION_FUSION_H_

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

namespace sherpa_onnx {

struct FinalizedSpeakerFrame {
  int64_t frame_index;
  int32_t speaker_count;  // 0, 1, or 2
  // Low three bits identify session-local tracks after cross-window alignment.
  uint8_t local_speaker_mask;
  // Winning fused powerset-class probability in [0, 1].
  float local_speaker_mask_confidence;
  bool single_speaker_changed_before;
  std::array<float, 7> fused_probabilities{};
  int32_t coverage_count = 0;
  bool has_change_candidate_cluster = false;
  int32_t change_vote_count = 0;
  int32_t change_vote_coverage = 0;
  float change_vote_ratio = 0.0F;
  bool change_vote_threshold_passed = false;
};

struct SpeakerSegmentationWindowTrace {
  int32_t window_id = 0;
  int64_t start_frame = 0;
  std::array<int32_t, 3> track_permutation{0, 1, 2};
  std::vector<float> input_probabilities;
  std::vector<float> aligned_probabilities;
  std::vector<uint8_t> raw_masks;
  std::vector<uint8_t> stabilized_masks;
  std::vector<int64_t> change_candidate_frames;
};

struct SpeakerSegmentationFusionConfig {
  int32_t frame_shift_samples;
  int32_t window_shift_samples;
  int32_t sample_rate;
  float min_duration_on;
  float min_duration_off;
  float change_vote_threshold;
  std::function<void(const SpeakerSegmentationWindowTrace &)> on_window_trace;
  std::function<void(const FinalizedSpeakerFrame &)> on_frame_trace;
};

// Fuses independent local segmentation windows without assigning global
// speaker identities. Raw local-mask population counts are accumulated for the
// speaker-count result. Local-track transitions contribute identity-free
// single-speaker-change candidates; overlapping windows confirm a change by
// majority of covering windows after 100 ms clustering.
class SpeakerSegmentationFusion {
 public:
  explicit SpeakerSegmentationFusion(
      const SpeakerSegmentationFusionConfig &config);
  ~SpeakerSegmentationFusion();

  SpeakerSegmentationFusion(const SpeakerSegmentationFusion &) = delete;
  SpeakerSegmentationFusion &operator=(
      const SpeakerSegmentationFusion &) = delete;

  // Adds masks for one window. Each mask uses the low three bits for its
  // window-local speaker tracks. This compatibility overload is equivalent to
  // one-hot powerset probabilities.
  void AddWindow(int64_t start_frame, const std::vector<uint8_t> &raw_masks);

  // Adds frame-major, normalized 7-class pyannote powerset probabilities.
  // The fuser aligns each window's three local tracks to its session-local
  // canonical tracks before aggregating probabilities across overlap.
  void AddWindowProbabilities(int64_t start_frame,
                              const std::vector<float> &probabilities);

  // Finalizes and removes frames in [earliest pending frame, frame_exclusive).
  // The caller must only request frames that cannot be covered by a later
  // window.
  std::vector<FinalizedSpeakerFrame> FinalizeBefore(
      int64_t frame_exclusive);

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_SPEAKER_SEGMENTATION_FUSION_H_
