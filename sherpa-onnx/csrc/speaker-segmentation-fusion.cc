// sherpa-onnx/csrc/speaker-segmentation-fusion.cc
//
// Copyright (c) 2026

#include "sherpa-onnx/csrc/speaker-segmentation-fusion.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>

namespace sherpa_onnx {
namespace {

int32_t PopCount3(uint8_t mask) {
  return static_cast<int32_t>((mask & 1) + ((mask >> 1) & 1) +
                              ((mask >> 2) & 1));
}

int32_t DurationToFrames(float seconds,
                         const SpeakerSegmentationFusionConfig &config) {
  if (seconds <= 0) {
    return 0;
  }

  return static_cast<int32_t>(std::ceil(
      seconds * static_cast<float>(config.sample_rate) /
      static_cast<float>(config.frame_shift_samples)));
}

void RemoveShortActiveIslands(std::vector<uint8_t> *active,
                              int32_t min_on_frames) {
  if (min_on_frames <= 1) {
    return;
  }

  const int32_t n = static_cast<int32_t>(active->size());
  for (int32_t begin = 0; begin < n;) {
    if (!(*active)[begin]) {
      ++begin;
      continue;
    }

    int32_t end = begin + 1;
    while (end < n && (*active)[end]) {
      ++end;
    }

    if (end - begin < min_on_frames) {
      std::fill(active->begin() + begin, active->begin() + end, 0);
    }
    begin = end;
  }
}

void CloseShortInactiveGaps(std::vector<uint8_t> *active,
                            int32_t min_off_frames) {
  if (min_off_frames <= 0) {
    return;
  }

  const int32_t n = static_cast<int32_t>(active->size());
  for (int32_t begin = 0; begin < n;) {
    if ((*active)[begin]) {
      ++begin;
      continue;
    }

    int32_t end = begin + 1;
    while (end < n && !(*active)[end]) {
      ++end;
    }

    const bool bounded_by_active = begin > 0 && end < n &&
                                   (*active)[begin - 1] && (*active)[end];
    if (bounded_by_active && end - begin <= min_off_frames) {
      std::fill(active->begin() + begin, active->begin() + end, 1);
    }
    begin = end;
  }
}

std::vector<uint8_t> StabilizeMasks(
    const std::vector<uint8_t> &raw_masks, int32_t min_on_frames,
    int32_t min_off_frames) {
  std::vector<uint8_t> masks(raw_masks.size());
  for (int32_t bit = 0; bit != 3; ++bit) {
    std::vector<uint8_t> active(raw_masks.size());
    for (size_t i = 0; i != raw_masks.size(); ++i) {
      active[i] = (raw_masks[i] >> bit) & 1;
    }

    RemoveShortActiveIslands(&active, min_on_frames);
    CloseShortInactiveGaps(&active, min_off_frames);

    for (size_t i = 0; i != active.size(); ++i) {
      masks[i] |= static_cast<uint8_t>(active[i] << bit);
    }
  }
  return masks;
}

}  // namespace

class SpeakerSegmentationFusion::Impl {
 public:
  explicit Impl(const SpeakerSegmentationFusionConfig &config)
      : config_(config), min_on_frames_(0), min_off_frames_(0) {
    if (config.frame_shift_samples <= 0 || config.window_shift_samples <= 0 ||
        config.sample_rate <= 0) {
      throw std::invalid_argument("Speaker segmentation frame configuration "
                                  "must be positive");
    }
    if (config.min_duration_on < 0 || config.min_duration_off < 0) {
      throw std::invalid_argument("Speaker segmentation minimum durations "
                                  "must not be negative");
    }
    if (config.change_vote_threshold < 0 ||
        config.change_vote_threshold > 1) {
      throw std::invalid_argument("Speaker segmentation change vote threshold "
                                  "must be in [0, 1]");
    }

    min_on_frames_ = DurationToFrames(config.min_duration_on, config);
    min_off_frames_ = DurationToFrames(config.min_duration_off, config);
  }

  void AddWindow(int64_t start_frame, const std::vector<uint8_t> &raw_masks) {
    if (start_frame < 0) {
      throw std::invalid_argument("Speaker segmentation window start frame "
                                  "must not be negative");
    }
    if (raw_masks.empty()) {
      return;
    }

    for (size_t i = 0; i != raw_masks.size(); ++i) {
      const int64_t frame = start_frame + static_cast<int64_t>(i);
      FrameAccumulator &acc = frames_[frame];
      acc.count_sum += std::min(PopCount3(raw_masks[i]), 2);
      ++acc.count_coverage;
      ++acc.change_coverage;
    }

    const std::vector<uint8_t> masks =
        StabilizeMasks(raw_masks, min_on_frames_, min_off_frames_);
    for (size_t i = 1; i != masks.size(); ++i) {
      const int32_t previous_count = PopCount3(masks[i - 1]);
      const int32_t current_count = PopCount3(masks[i]);
      if (previous_count != 1 || current_count != 1 ||
          masks[i - 1] == masks[i]) {
        continue;
      }

      const int64_t center = start_frame + static_cast<int64_t>(i);
      // A center vote is stronger than its one-frame neighborhood. The
      // neighborhood tolerates small frame alignment differences, while peak
      // selection below still emits one boundary at the actual transition.
      AddChangeVote(center - 1, 0.5F);
      AddChangeVote(center, 1.0F);
      AddChangeVote(center + 1, 0.5F);
    }
  }

  std::vector<FinalizedSpeakerFrame> FinalizeBefore(int64_t frame_exclusive) {
    std::vector<FinalizedSpeakerFrame> ans;
    if (frames_.empty()) {
      return ans;
    }

    std::vector<int64_t> confirmed_changes;
    for (const auto &p : frames_) {
      if (p.first >= frame_exclusive) {
        break;
      }
      if (p.second.change_coverage > 0 &&
          p.second.change_vote / p.second.change_coverage >=
              config_.change_vote_threshold) {
        confirmed_changes.push_back(p.first);
      }
    }

    const std::vector<int64_t> change_peaks =
        SelectChangePeaks(confirmed_changes);
    size_t next_change_peak = 0;

    for (auto iter = frames_.begin();
         iter != frames_.end() && iter->first < frame_exclusive;) {
      const int64_t frame = iter->first;
      while (next_change_peak < change_peaks.size() &&
             change_peaks[next_change_peak] < frame) {
        ++next_change_peak;
      }

      const FrameAccumulator &acc = iter->second;
      FinalizedSpeakerFrame result;
      result.frame_index = frame;
      result.speaker_count = std::max(
          0, std::min(2, static_cast<int32_t>(std::round(
                             static_cast<float>(acc.count_sum) /
                             static_cast<float>(acc.count_coverage)))));
      result.single_speaker_changed_before =
          next_change_peak < change_peaks.size() &&
          change_peaks[next_change_peak] == frame;
      ans.push_back(result);
      iter = frames_.erase(iter);
    }
    return ans;
  }

 private:
  struct FrameAccumulator {
    int32_t count_sum = 0;
    int32_t count_coverage = 0;
    float change_vote = 0;
    float change_coverage = 0;
  };

  void AddChangeVote(int64_t frame, float weight) {
    auto iter = frames_.find(frame);
    if (iter != frames_.end()) {
      iter->second.change_vote += weight;
    }
  }

  float ChangeVoteRatio(int64_t frame) const {
    const auto iter = frames_.find(frame);
    if (iter == frames_.end() || iter->second.change_coverage == 0) {
      return 0;
    }
    return iter->second.change_vote / iter->second.change_coverage;
  }

  std::vector<int64_t> SelectChangePeaks(
      const std::vector<int64_t> &confirmed) const {
    std::vector<int64_t> peaks;
    for (size_t begin = 0; begin < confirmed.size();) {
      size_t end = begin + 1;
      while (end < confirmed.size() &&
             confirmed[end] == confirmed[end - 1] + 1) {
        ++end;
      }

      int64_t best_frame = confirmed[begin];
      float best_ratio = ChangeVoteRatio(best_frame);
      for (size_t i = begin + 1; i < end; ++i) {
        const float ratio = ChangeVoteRatio(confirmed[i]);
        if (ratio > best_ratio) {
          best_frame = confirmed[i];
          best_ratio = ratio;
        }
      }
      peaks.push_back(best_frame);
      begin = end;
    }

    if (min_on_frames_ <= 1 || peaks.size() < 2) {
      return peaks;
    }

    std::vector<int64_t> suppressed;
    for (int64_t peak : peaks) {
      if (suppressed.empty() || peak - suppressed.back() >= min_on_frames_) {
        suppressed.push_back(peak);
        continue;
      }

      const float previous_ratio = ChangeVoteRatio(suppressed.back());
      const float ratio = ChangeVoteRatio(peak);
      if (ratio > previous_ratio) {
        suppressed.back() = peak;
      }
    }
    return suppressed;
  }

 private:
  SpeakerSegmentationFusionConfig config_;
  int32_t min_on_frames_;
  int32_t min_off_frames_;
  std::map<int64_t, FrameAccumulator> frames_;
};

SpeakerSegmentationFusion::SpeakerSegmentationFusion(
    const SpeakerSegmentationFusionConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

SpeakerSegmentationFusion::~SpeakerSegmentationFusion() = default;

void SpeakerSegmentationFusion::AddWindow(
    int64_t start_frame, const std::vector<uint8_t> &raw_masks) {
  impl_->AddWindow(start_frame, raw_masks);
}

std::vector<FinalizedSpeakerFrame> SpeakerSegmentationFusion::FinalizeBefore(
    int64_t frame_exclusive) {
  return impl_->FinalizeBefore(frame_exclusive);
}

}  // namespace sherpa_onnx
