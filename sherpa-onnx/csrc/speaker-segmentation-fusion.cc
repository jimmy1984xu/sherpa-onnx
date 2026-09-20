// sherpa-onnx/csrc/speaker-segmentation-fusion.cc
//
// Copyright (c) 2026

#include "sherpa-onnx/csrc/speaker-segmentation-fusion.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

namespace sherpa_onnx {
namespace {

int32_t PopCount3(uint8_t mask) {
  return static_cast<int32_t>((mask & 1) + ((mask >> 1) & 1) +
                              ((mask >> 2) & 1));
}

constexpr int32_t kNumPowersetClasses = 7;
constexpr std::array<uint8_t, kNumPowersetClasses> kPowersetMasks = {
    0b000, 0b001, 0b010, 0b100, 0b011, 0b101, 0b110};

int32_t ClassForMask(uint8_t mask) {
  mask &= 0b111;
  for (int32_t i = 0; i != kNumPowersetClasses; ++i) {
    if (kPowersetMasks[i] == mask) {
      return i;
    }
  }
  throw std::invalid_argument("Unsupported powerset mask");
}

int32_t BestClass(const std::array<float, kNumPowersetClasses> &values) {
  int32_t best = 0;
  for (int32_t i = 1; i != kNumPowersetClasses; ++i) {
    if (values[i] > values[best]) {
      best = i;
    }
  }
  return best;
}

uint8_t ApplyPermutation(uint8_t mask, const std::array<int32_t, 3> &mapping) {
  uint8_t ans = 0;
  for (int32_t bit = 0; bit != 3; ++bit) {
    if ((mask >> bit) & 1) {
      ans |= static_cast<uint8_t>(1 << mapping[bit]);
    }
  }
  return ans;
}

std::array<float, kNumPowersetClasses> RemapProbabilities(
    const float *raw, const std::array<int32_t, 3> &mapping) {
  std::array<float, kNumPowersetClasses> ans{};
  for (int32_t c = 0; c != kNumPowersetClasses; ++c) {
    const int32_t mapped_class = ClassForMask(ApplyPermutation(kPowersetMasks[c], mapping));
    ans[mapped_class] += raw[c];
  }
  return ans;
}

float ActivityAgreement(uint8_t reference, uint8_t candidate) {
  const int32_t shared_active = PopCount3(reference & candidate);
  const int32_t disagreement = PopCount3(reference ^ candidate);
  const int32_t shared_inactive = 3 - PopCount3(reference | candidate);
  return 3.0F * shared_active + 0.25F * shared_inactive -
         2.0F * disagreement;
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

constexpr float kChangeClusterRadiusSeconds = 0.10F;
constexpr float kShortGapMaxSeconds = 0.10F;

}  // namespace

class SpeakerSegmentationFusion::Impl {
 public:
  explicit Impl(const SpeakerSegmentationFusionConfig &config)
      : config_(config),
        min_on_frames_(0),
        min_off_frames_(0),
        cluster_radius_frames_(0),
        short_gap_frames_(0),
        next_window_id_(0) {
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
    cluster_radius_frames_ =
        DurationToFrames(kChangeClusterRadiusSeconds, config);
    short_gap_frames_ = DurationToFrames(kShortGapMaxSeconds, config);
  }

  void AddWindow(int64_t start_frame, const std::vector<uint8_t> &raw_masks) {
    std::vector<float> probabilities(raw_masks.size() * kNumPowersetClasses,
                                     0.0F);
    for (size_t i = 0; i != raw_masks.size(); ++i) {
      probabilities[i * kNumPowersetClasses + ClassForMask(raw_masks[i])] =
          1.0F;
    }
    AddWindowProbabilities(start_frame, probabilities);
  }

  void AddWindowProbabilities(int64_t start_frame,
                              const std::vector<float> &probabilities) {
    if (start_frame < 0) {
      throw std::invalid_argument("Speaker segmentation window start frame "
                                  "must not be negative");
    }
    if (probabilities.empty()) {
      return;
    }
    if (probabilities.size() % kNumPowersetClasses != 0) {
      throw std::invalid_argument("Speaker segmentation probabilities must be "
                                  "frame-major 7-class values");
    }
    const size_t num_frames = probabilities.size() / kNumPowersetClasses;
    std::vector<float> normalized_probabilities = probabilities;
    for (size_t i = 0; i != num_frames; ++i) {
      float sum = 0.0F;
      float *row = normalized_probabilities.data() +
                   i * kNumPowersetClasses;
      for (int32_t c = 0; c != kNumPowersetClasses; ++c) {
        const float probability = row[c];
        if (!std::isfinite(probability) || probability < 0) {
          throw std::invalid_argument("Speaker segmentation probabilities "
                                      "must be finite and non-negative");
        }
        sum += probability;
      }
      if (!std::isfinite(sum) || sum <= 0.0F) {
        throw std::invalid_argument(
            "Each speaker segmentation probability frame must have "
            "positive mass");
      }
      for (int32_t c = 0; c != kNumPowersetClasses; ++c) {
        row[c] /= sum;
      }
    }

    const std::array<int32_t, 3> permutation =
        BestTrackPermutation(start_frame, normalized_probabilities);
    std::vector<uint8_t> aligned_masks(num_frames);
    for (size_t i = 0; i != num_frames; ++i) {
      const float *raw = normalized_probabilities.data() +
                         i * kNumPowersetClasses;
      const std::array<float, kNumPowersetClasses> aligned =
          RemapProbabilities(raw, permutation);
      const int32_t best_class = BestClass(aligned);
      aligned_masks[i] = kPowersetMasks[best_class];
      const int64_t frame = start_frame + static_cast<int64_t>(i);
      FrameAccumulator &acc = frames_[frame];
      for (int32_t c = 0; c != kNumPowersetClasses; ++c) {
        acc.probability_sum[c] += aligned[c];
      }
      ++acc.count_coverage;
    }

    const int32_t window_id = next_window_id_++;
    const std::vector<uint8_t> masks =
        StabilizeMasks(aligned_masks, min_on_frames_, min_off_frames_);
    for (int64_t local : ExtractChangeFrames(masks)) {
      candidates_.push_back(ChangeCandidate{start_frame + local, window_id});
    }
  }

  std::vector<FinalizedSpeakerFrame> FinalizeBefore(int64_t frame_exclusive) {
    std::vector<FinalizedSpeakerFrame> ans;
    if (frames_.empty()) {
      return ans;
    }

    const std::set<int64_t> change_frames = ConfirmReadyClusters(frame_exclusive);
    int64_t emit_exclusive = EmitExclusive(frame_exclusive);

    for (auto iter = frames_.begin();
         iter != frames_.end() && iter->first < emit_exclusive;) {
      const int64_t frame = iter->first;
      const FrameAccumulator &acc = iter->second;
      FinalizedSpeakerFrame result;
      result.frame_index = frame;
      std::array<float, kNumPowersetClasses> averaged{};
      for (int32_t c = 0; c != kNumPowersetClasses; ++c) {
        averaged[c] = acc.probability_sum[c] /
                      static_cast<float>(acc.count_coverage);
      }
      const int32_t best_class = BestClass(averaged);
      result.speaker_count = PopCount3(kPowersetMasks[best_class]);
      result.local_speaker_mask = kPowersetMasks[best_class];
      result.local_speaker_mask_confidence = averaged[best_class];
      result.single_speaker_changed_before =
          change_frames.find(frame) != change_frames.end();
      ans.push_back(result);
      iter = frames_.erase(iter);
    }

    candidates_.erase(std::remove_if(candidates_.begin(), candidates_.end(),
                                     [emit_exclusive](const ChangeCandidate &c) {
                                       return c.frame < emit_exclusive;
                                     }),
                      candidates_.end());
    return ans;
  }

 private:
  struct FrameAccumulator {
    std::array<float, kNumPowersetClasses> probability_sum{};
    int32_t count_coverage = 0;
  };

  std::array<int32_t, 3> BestTrackPermutation(
      int64_t start_frame, const std::vector<float> &probabilities) const {
    static constexpr std::array<std::array<int32_t, 3>, 6> kPermutations = {
        std::array<int32_t, 3>{0, 1, 2}, std::array<int32_t, 3>{0, 2, 1},
        std::array<int32_t, 3>{1, 0, 2}, std::array<int32_t, 3>{1, 2, 0},
        std::array<int32_t, 3>{2, 0, 1}, std::array<int32_t, 3>{2, 1, 0}};
    std::array<int32_t, 3> best = kPermutations[0];
    float best_score = -std::numeric_limits<float>::infinity();
    const size_t num_frames = probabilities.size() / kNumPowersetClasses;
    for (const auto &candidate : kPermutations) {
      float score = 0.0F;
      bool has_overlap = false;
      for (size_t i = 0; i != num_frames; ++i) {
        const auto iter = frames_.find(start_frame + static_cast<int64_t>(i));
        if (iter == frames_.end() || iter->second.count_coverage == 0) {
          continue;
        }
        has_overlap = true;
        std::array<float, kNumPowersetClasses> reference{};
        for (int32_t c = 0; c != kNumPowersetClasses; ++c) {
          reference[c] = iter->second.probability_sum[c] /
                         static_cast<float>(iter->second.count_coverage);
        }
        const float *raw = probabilities.data() + i * kNumPowersetClasses;
        const std::array<float, kNumPowersetClasses> remapped =
            RemapProbabilities(raw, candidate);
        const int32_t reference_class = BestClass(reference);
        const int32_t candidate_class = BestClass(remapped);
        score += ActivityAgreement(kPowersetMasks[reference_class],
                                   kPowersetMasks[candidate_class]) *
                 reference[reference_class] * remapped[candidate_class];
      }
      if (has_overlap && score > best_score) {
        best_score = score;
        best = candidate;
      }
    }
    return best;
  }


  struct ChangeCandidate {
    int64_t frame = 0;
    int32_t window_id = 0;
  };

  std::vector<int64_t> ExtractChangeFrames(
      const std::vector<uint8_t> &masks) const {
    std::vector<int64_t> times;
    const int32_t n = static_cast<int32_t>(masks.size());
    int32_t i = 0;
    while (i < n) {
      if (PopCount3(masks[i]) != 1) {
        ++i;
        continue;
      }

      const uint8_t speaker = masks[i];
      int32_t j = i + 1;
      while (j < n && masks[j] == speaker) {
        ++j;
      }
      if (j >= n) {
        break;
      }

      if (PopCount3(masks[j]) == 1) {
        times.push_back(j);
        i = j;
        continue;
      }

      if (PopCount3(masks[j]) == 0 && short_gap_frames_ > 0) {
        int32_t k = j;
        while (k < n && PopCount3(masks[k]) == 0) {
          ++k;
        }
        const int32_t gap = k - j;
        if (k < n && PopCount3(masks[k]) == 1 && masks[k] != speaker &&
            gap <= short_gap_frames_) {
          times.push_back(k);
          i = k;
          continue;
        }
      }
      i = j;
    }
    return times;
  }

  std::vector<std::vector<ChangeCandidate>> ClusterCandidates() const {
    std::vector<ChangeCandidate> sorted = candidates_;
    std::sort(sorted.begin(), sorted.end(),
              [](const ChangeCandidate &a, const ChangeCandidate &b) {
                if (a.frame != b.frame) {
                  return a.frame < b.frame;
                }
                return a.window_id < b.window_id;
              });

    std::vector<std::vector<ChangeCandidate>> clusters;
    for (const auto &candidate : sorted) {
      if (clusters.empty() ||
          candidate.frame - clusters.back().back().frame >
              cluster_radius_frames_) {
        clusters.push_back({candidate});
        continue;
      }
      clusters.back().push_back(candidate);
    }

    for (auto &cluster : clusters) {
      std::vector<ChangeCandidate> unique_windows;
      std::set<int32_t> seen;
      for (const auto &candidate : cluster) {
        if (seen.insert(candidate.window_id).second) {
          unique_windows.push_back(candidate);
        }
      }
      cluster.swap(unique_windows);
    }
    return clusters;
  }

  int64_t RepresentativeFrame(
      const std::vector<ChangeCandidate> &cluster) const {
    const int32_t n_vote = static_cast<int32_t>(cluster.size());
    const int32_t needed = (n_vote + 1) / 2;
    return cluster[static_cast<size_t>(needed - 1)].frame;
  }

  int64_t EmitExclusive(int64_t frame_exclusive) const {
    int64_t emit_exclusive = frame_exclusive;
    for (const auto &cluster : ClusterCandidates()) {
      if (cluster.empty()) {
        continue;
      }
      const int64_t min_frame = cluster.front().frame;
      const int64_t max_frame = cluster.back().frame;
      if (min_frame < frame_exclusive && max_frame >= frame_exclusive) {
        emit_exclusive = std::min(emit_exclusive, min_frame);
      }
    }
    return emit_exclusive;
  }

  std::set<int64_t> ConfirmReadyClusters(int64_t frame_exclusive) const {
    const int64_t emit_exclusive = EmitExclusive(frame_exclusive);
    struct Peak {
      int64_t frame = 0;
      float ratio = 0;
      int32_t n_vote = 0;
    };
    std::vector<Peak> peaks;
    for (const auto &cluster : ClusterCandidates()) {
      if (cluster.empty()) {
        continue;
      }
      const int64_t max_frame = cluster.back().frame;
      if (max_frame >= emit_exclusive) {
        continue;
      }

      const int32_t n_vote = static_cast<int32_t>(cluster.size());
      const int64_t rep = RepresentativeFrame(cluster);
      const auto iter = frames_.find(rep);
      if (iter == frames_.end() || iter->second.count_coverage <= 0) {
        continue;
      }
      const float ratio = static_cast<float>(n_vote) /
                          static_cast<float>(iter->second.count_coverage);
      if (ratio < config_.change_vote_threshold) {
        continue;
      }
      peaks.push_back(Peak{rep, ratio, n_vote});
    }

    std::sort(peaks.begin(), peaks.end(),
              [](const Peak &a, const Peak &b) { return a.frame < b.frame; });

    std::vector<Peak> suppressed;
    for (const auto &peak : peaks) {
      if (suppressed.empty() || min_on_frames_ <= 1 ||
          peak.frame - suppressed.back().frame >= min_on_frames_) {
        suppressed.push_back(peak);
        continue;
      }
      if (peak.ratio > suppressed.back().ratio) {
        suppressed.back() = peak;
      }
    }

    std::set<int64_t> frames;
    for (const auto &peak : suppressed) {
      frames.insert(peak.frame);
    }
    return frames;
  }

  SpeakerSegmentationFusionConfig config_;
  int32_t min_on_frames_;
  int32_t min_off_frames_;
  int32_t cluster_radius_frames_;
  int32_t short_gap_frames_;
  int32_t next_window_id_;
  std::map<int64_t, FrameAccumulator> frames_;
  std::vector<ChangeCandidate> candidates_;
};

SpeakerSegmentationFusion::SpeakerSegmentationFusion(
    const SpeakerSegmentationFusionConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

SpeakerSegmentationFusion::~SpeakerSegmentationFusion() = default;

void SpeakerSegmentationFusion::AddWindow(
    int64_t start_frame, const std::vector<uint8_t> &raw_masks) {
  impl_->AddWindow(start_frame, raw_masks);
}

void SpeakerSegmentationFusion::AddWindowProbabilities(
    int64_t start_frame, const std::vector<float> &probabilities) {
  impl_->AddWindowProbabilities(start_frame, probabilities);
}

std::vector<FinalizedSpeakerFrame> SpeakerSegmentationFusion::FinalizeBefore(
    int64_t frame_exclusive) {
  return impl_->FinalizeBefore(frame_exclusive);
}

}  // namespace sherpa_onnx
