// sherpa-onnx/csrc/speaker-segmentation.cc
//
// Copyright (c) 2026

#include "sherpa-onnx/csrc/speaker-segmentation.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <deque>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "onnxruntime_cxx_api.h"  // NOLINT
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/offline-speaker-segmentation-pyannote-model.h"
#include "sherpa-onnx/csrc/speaker-segmentation-fusion.h"

namespace sherpa_onnx {
namespace {

bool ValidateMetaData(
    const OfflineSpeakerSegmentationPyannoteModelMetaData &meta_data) {
  if (meta_data.num_speakers != 3) {
    SHERPA_ONNX_LOGE("Expected num_speakers=3. Given %d",
                     meta_data.num_speakers);
    return false;
  }

  if (meta_data.num_classes != 7) {
    SHERPA_ONNX_LOGE("Expected num_classes=7. Given %d",
                     meta_data.num_classes);
    return false;
  }

  if (meta_data.powerset_max_classes != 2) {
    SHERPA_ONNX_LOGE("Expected powerset_max_classes=2. Given %d",
                     meta_data.powerset_max_classes);
    return false;
  }

  if (meta_data.sample_rate <= 0 || meta_data.window_size <= 0 ||
      meta_data.window_shift <= 0 || meta_data.receptive_field_size <= 0 ||
      meta_data.receptive_field_shift <= 0) {
    SHERPA_ONNX_LOGE(
        "Invalid segmentation metadata: sample_rate=%d, window_size=%d, "
        "window_shift=%d, receptive_field_size=%d, "
        "receptive_field_shift=%d",
        meta_data.sample_rate, meta_data.window_size, meta_data.window_shift,
        meta_data.receptive_field_size, meta_data.receptive_field_shift);
    return false;
  }

  return true;
}

int64_t FrameIndexForSample(
    int64_t sample,
    const OfflineSpeakerSegmentationPyannoteModelMetaData &meta_data) {
  return static_cast<int64_t>(
      static_cast<double>(sample) / meta_data.receptive_field_shift + 0.5);
}

uint8_t PowersetClassToMask(int32_t class_index) {
  static constexpr std::array<uint8_t, 7> kMasks = {
      0b000, 0b001, 0b010, 0b100, 0b011, 0b101, 0b110};
  return kMasks[class_index];
}

}  // namespace

bool SpeakerSegmentationConfig::Validate() const {
  if (!model.Validate()) {
    return false;
  }

  if (min_duration_on < 0) {
    SHERPA_ONNX_LOGE("min_duration_on %.3f is negative", min_duration_on);
    return false;
  }

  if (min_duration_off < 0) {
    SHERPA_ONNX_LOGE("min_duration_off %.3f is negative", min_duration_off);
    return false;
  }

  if (change_vote_threshold <= 0 || change_vote_threshold > 1) {
    SHERPA_ONNX_LOGE("change_vote_threshold %.3f is outside (0, 1]",
                     change_vote_threshold);
    return false;
  }

  return true;
}

std::string SpeakerSegmentationConfig::ToString() const {
  std::ostringstream os;
  os << "SpeakerSegmentationConfig(";
  os << "model=" << model.ToString() << ", ";
  os << "min_duration_on=" << min_duration_on << ", ";
  os << "min_duration_off=" << min_duration_off << ", ";
  os << "change_vote_threshold=" << change_vote_threshold << ")";
  return os.str();
}

class SpeakerSegmentation::Impl {
 public:
  explicit Impl(const SpeakerSegmentationConfig &config)
      : config_(config), model_(std::make_unique<
                            OfflineSpeakerSegmentationPyannoteModel>(
                            config.model)),
        meta_data_(model_->GetModelMetaData()),
        fusion_config_{meta_data_.receptive_field_shift,
                       meta_data_.window_shift,
                       meta_data_.sample_rate,
                       config_.min_duration_on,
                       config_.min_duration_off,
                       config_.change_vote_threshold} {
    if (!ValidateMetaData(meta_data_)) {
      SHERPA_ONNX_LOGE("Unsupported pyannote segmentation model metadata");
      SHERPA_ONNX_EXIT(-1);
    }
    ResetFusion();
  }

  Impl(const SpeakerSegmentationConfig &config,
       const OfflineSpeakerSegmentationPyannoteModelMetaData &meta_data,
       SpeakerSegmentationTestForward forward)
      : config_(config),
        meta_data_(meta_data),
        test_forward_(std::move(forward)),
        fusion_config_{meta_data_.receptive_field_shift,
                       meta_data_.window_shift,
                       meta_data_.sample_rate,
                       config_.min_duration_on,
                       config_.min_duration_off,
                       config_.change_vote_threshold} {
    if (!ValidateMetaData(meta_data_) || !test_forward_) {
      SHERPA_ONNX_LOGE("Invalid speaker segmentation test runner setup");
      SHERPA_ONNX_EXIT(-1);
    }
    ResetFusion();
  }

  int32_t SampleRate() const { return meta_data_.sample_rate; }

  void AcceptWaveform(const float *samples, int32_t n) {
    if (n < 0) {
      SHERPA_ONNX_LOGE("n must not be negative. Given %d", n);
      return;
    }
    if (input_finished_) {
      SHERPA_ONNX_LOGE("AcceptWaveform() after InputFinished() requires "
                       "Reset()");
      return;
    }
    if (n == 0) {
      return;
    }
    if (!samples) {
      SHERPA_ONNX_LOGE("samples is null while n is %d", n);
      return;
    }

    audio_buffer_.insert(audio_buffer_.end(), samples, samples + n);
    received_samples_ += n;
    RunCompleteWindows();
  }

  void InputFinished() {
    if (input_finished_) {
      return;
    }
    input_finished_ = true;

    while (next_window_start_ < received_samples_) {
      ProcessWindow(next_window_start_, /*pad_right=*/true);
      next_window_start_ += meta_data_.window_shift;
      // The final right-padded window may consume less than one window shift.
      // No more audio will be accepted after InputFinished(), so retaining that
      // short tail is safe and avoids dropping samples that are not buffered.
      if (next_window_start_ <= received_samples_) {
        TrimAudioBuffer();
      }
    }

    if (!spans_.empty()) {
      spans_.back().flag |= kSpeakerSegmentationInputFinished;
    }
  }

  bool Empty() const { return spans_.empty(); }

  const SpeakerSegmentationSpan &Front() const { return spans_.front(); }

  void Pop() {
    if (!spans_.empty()) {
      spans_.pop_front();
    }
  }

  void Reset() {
    audio_buffer_.clear();
    spans_.clear();
    received_samples_ = 0;
    buffer_start_sample_ = 0;
    next_window_start_ = 0;
    published_until_sample_ = 0;
    input_finished_ = false;
    ResetFusion();
  }

 private:
  void ResetFusion() {
    fusion_ = std::make_unique<SpeakerSegmentationFusion>(fusion_config_);
  }

  void RunCompleteWindows() {
    while (next_window_start_ + meta_data_.window_size <= received_samples_) {
      ProcessWindow(next_window_start_, /*pad_right=*/false);
      next_window_start_ += meta_data_.window_shift;
      TrimAudioBuffer();
    }
  }

  std::vector<float> GetWindow(int64_t start_sample, bool pad_right) const {
    std::vector<float> window(meta_data_.window_size);
    const int64_t available_samples = received_samples_ - start_sample;
    const int64_t copy_samples =
        std::max<int64_t>(0, std::min<int64_t>(available_samples,
                                               meta_data_.window_size));
    if (!pad_right && copy_samples != meta_data_.window_size) {
      SHERPA_ONNX_LOGE("Incomplete window without padding: start=%lld, "
                       "available=%lld, window=%d",
                       static_cast<long long>(start_sample),
                       static_cast<long long>(copy_samples),
                       meta_data_.window_size);
      SHERPA_ONNX_EXIT(-1);
    }

    const int64_t buffer_offset = start_sample - buffer_start_sample_;
    if (buffer_offset < 0 || buffer_offset + copy_samples >
                                 static_cast<int64_t>(audio_buffer_.size())) {
      SHERPA_ONNX_LOGE("Audio buffer does not contain window [%lld, %lld)",
                       static_cast<long long>(start_sample),
                       static_cast<long long>(start_sample + copy_samples));
      SHERPA_ONNX_EXIT(-1);
    }

    std::copy(audio_buffer_.begin() + buffer_offset,
              audio_buffer_.begin() + buffer_offset + copy_samples,
              window.begin());
    return window;
  }

  std::vector<uint8_t> RunForward(const std::vector<float> &window) const {
    if (test_forward_) {
      return test_forward_(window);
    }

    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
    std::array<int64_t, 3> shape = {1, 1, meta_data_.window_size};
    Ort::Value input = Ort::Value::CreateTensor(
        memory_info, const_cast<float *>(window.data()), window.size(),
        shape.data(), shape.size());
    Ort::Value output = model_->Forward(std::move(input));
    const auto output_shape = output.GetTensorTypeAndShapeInfo().GetShape();
    if (output_shape.size() != 3 || output_shape[0] != 1 ||
        output_shape[2] != meta_data_.num_classes) {
      SHERPA_ONNX_LOGE("Unexpected segmentation output shape");
      SHERPA_ONNX_EXIT(-1);
    }

    const int32_t num_frames = static_cast<int32_t>(output_shape[1]);
    const float *scores = output.GetTensorData<float>();
    std::vector<uint8_t> masks(num_frames);
    for (int32_t frame = 0; frame != num_frames; ++frame) {
      int32_t best_class = 0;
      const float *row = scores + frame * meta_data_.num_classes;
      for (int32_t c = 1; c != meta_data_.num_classes; ++c) {
        if (row[c] > row[best_class]) {
          best_class = c;
        }
      }
      masks[frame] = PowersetClassToMask(best_class);
    }
    return masks;
  }

  void ProcessWindow(int64_t start_sample, bool pad_right) {
    const auto masks = RunForward(GetWindow(start_sample, pad_right));
    if (masks.empty()) {
      SHERPA_ONNX_LOGE("Segmentation model returned no frames");
      SHERPA_ONNX_EXIT(-1);
    }

    fusion_->AddWindow(FrameIndexForSample(start_sample, meta_data_), masks);

    const int64_t finalize_sample =
        std::min(start_sample + meta_data_.window_shift, received_samples_);
    const int64_t finalize_frame =
        FrameIndexForSample(start_sample + meta_data_.window_shift, meta_data_);
    EmitFinalizedFrames(fusion_->FinalizeBefore(finalize_frame),
                        finalize_sample);
  }

  void EmitFinalizedFrames(const std::vector<FinalizedSpeakerFrame> &frames,
                           int64_t end_sample) {
    if (frames.empty() || end_sample <= published_until_sample_) {
      return;
    }

    int64_t run_start = published_until_sample_;
    int32_t speaker_count = frames.front().speaker_count;
    if (!spans_.empty()) {
      speaker_count = spans_.back().speaker_count;
    }
    for (size_t i = 0; i != frames.size(); ++i) {
      const bool count_changed = frames[i].speaker_count != speaker_count;
      const bool single_speaker_changed =
          frames[i].single_speaker_changed_before;
      if (!count_changed && !single_speaker_changed) {
        continue;
      }

      const int64_t frame_sample =
          frames[i].frame_index * meta_data_.receptive_field_shift;
      const int64_t boundary = std::max(
          run_start, std::min(frame_sample, end_sample));
      int32_t flag = kSpeakerSegmentationContinue;
      if (count_changed) {
        flag |= kSpeakerSegmentationSpeakerCountChanged;
      }
      if (single_speaker_changed) {
        flag |= kSpeakerSegmentationSingleSpeakerChanged;
      }
      if (boundary <= run_start && !spans_.empty()) {
        // The change sits on an already published checkpoint. Attach the flag
        // to the span that ends there; do not emit a zero-length span.
        spans_.back().flag |= flag;
      } else {
        AppendSpan(run_start, boundary, speaker_count, flag);
        run_start = boundary;
      }
      speaker_count = frames[i].speaker_count;
    }

    AppendSpan(run_start, end_sample, speaker_count,
               kSpeakerSegmentationContinue);
    published_until_sample_ = end_sample;
  }

  void AppendSpan(int64_t start_sample, int64_t end_sample,
                  int32_t speaker_count, int32_t flag) {
    if (end_sample <= start_sample) {
      return;
    }

    SpeakerSegmentationSpan span;
    span.start = static_cast<float>(start_sample) / meta_data_.sample_rate;
    span.end = static_cast<float>(end_sample) / meta_data_.sample_rate;
    span.speaker_count = std::max(0, std::min(2, speaker_count));
    span.flag = flag;
    spans_.push_back(span);
  }

  void TrimAudioBuffer() {
    const int64_t drop = next_window_start_ - buffer_start_sample_;
    if (drop <= 0) {
      return;
    }
    if (drop > static_cast<int64_t>(audio_buffer_.size())) {
      SHERPA_ONNX_LOGE("Cannot drop %lld samples from a buffer of %lld",
                       static_cast<long long>(drop),
                       static_cast<long long>(audio_buffer_.size()));
      SHERPA_ONNX_EXIT(-1);
    }

    audio_buffer_.erase(audio_buffer_.begin(),
                        audio_buffer_.begin() + drop);
    buffer_start_sample_ = next_window_start_;
  }

 private:
  SpeakerSegmentationConfig config_;
  std::unique_ptr<OfflineSpeakerSegmentationPyannoteModel> model_;
  OfflineSpeakerSegmentationPyannoteModelMetaData meta_data_;
  SpeakerSegmentationTestForward test_forward_;
  SpeakerSegmentationFusionConfig fusion_config_;
  std::unique_ptr<SpeakerSegmentationFusion> fusion_;

  std::vector<float> audio_buffer_;
  std::deque<SpeakerSegmentationSpan> spans_;
  int64_t received_samples_ = 0;
  int64_t buffer_start_sample_ = 0;
  int64_t next_window_start_ = 0;
  int64_t published_until_sample_ = 0;
  bool input_finished_ = false;
};

SpeakerSegmentation::SpeakerSegmentation(
    const SpeakerSegmentationConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

SpeakerSegmentation::SpeakerSegmentation(
    const SpeakerSegmentationConfig &config,
    const OfflineSpeakerSegmentationPyannoteModelMetaData &meta_data,
    SpeakerSegmentationTestForward forward)
    : impl_(std::make_unique<Impl>(config, meta_data, std::move(forward))) {}

SpeakerSegmentation::~SpeakerSegmentation() = default;

std::unique_ptr<SpeakerSegmentation> SpeakerSegmentation::CreateForTesting(
    const SpeakerSegmentationConfig &config,
    const OfflineSpeakerSegmentationPyannoteModelMetaData &meta_data,
    SpeakerSegmentationTestForward forward) {
  return std::unique_ptr<SpeakerSegmentation>(
      new SpeakerSegmentation(config, meta_data, std::move(forward)));
}

int32_t SpeakerSegmentation::SampleRate() const { return impl_->SampleRate(); }

void SpeakerSegmentation::AcceptWaveform(const float *samples, int32_t n) {
  impl_->AcceptWaveform(samples, n);
}

void SpeakerSegmentation::InputFinished() { impl_->InputFinished(); }

bool SpeakerSegmentation::Empty() const { return impl_->Empty(); }

const SpeakerSegmentationSpan &SpeakerSegmentation::Front() const {
  return impl_->Front();
}

void SpeakerSegmentation::Pop() { impl_->Pop(); }

void SpeakerSegmentation::Reset() { impl_->Reset(); }

}  // namespace sherpa_onnx
