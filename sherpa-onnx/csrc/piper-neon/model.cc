// Copyright (c) 2023 Xiaomi Corporation
#include "sherpa-onnx/csrc/piper-neon/model.h"

#include <algorithm>
#include <array>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/neon-tts-load-profile.h"
#include "sherpa-onnx/csrc/piper-neon/session.h"

namespace sherpa_onnx {

class PiperNeonModel::Impl {
 public:
  explicit Impl(const OfflineTtsModelConfig &config)
      : config_(config),
        env_(ORT_LOGGING_LEVEL_ERROR, "piper"),
        sess_opts_(transai::piper_neon::Options(config.num_threads)) {
    {
      transai::piper_neon::LoadProfile profile("encoder");
      encoder_map_ = std::make_unique<transai::aot::Mapping>(config.vits.model);
      encoder_ = std::make_unique<Ort::Session>(
          env_, encoder_map_->data(), encoder_map_->size(), sess_opts_);
    }
    const auto asset = config.vits.model.rfind(".aot/");
    const auto path = config.vits.model.substr(0, asset + 5) + "tts/decoder.ort";
    transai::piper_neon::LoadProfile profile("decoder");
    decoder_map_ = std::make_unique<transai::aot::Mapping>(path);
    decoder_ = std::make_unique<Ort::Session>(
        env_, decoder_map_->data(), decoder_map_->size(), sess_opts_);
    ReadMetadata();
  }

  Ort::Value RunStreaming(
      Ort::Value x, float speed,
      const std::function<int32_t(const float *, int32_t, float)> &callback,
      bool emit_prefix) {
    auto latent = Encode(std::move(x), speed);
    const auto shape = latent.GetTensorTypeAndShapeInfo().GetShape();
    const int64_t frames = shape.back(), channels = shape[1];
    constexpr int64_t hop = 256;
    auto mem = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
    const char *input_names[] = {"latent"};
    const char *output_names[] = {"audio"};
    int64_t committed = 0;
    std::vector<float> prefix_samples;
    const int prefix_frames = kFirstFrames + context_frames_;
    if (callback && emit_prefix && frames > prefix_frames) {
      std::vector<float> prefix(channels * prefix_frames);
      const auto *data = latent.GetTensorData<float>();
      for (int64_t channel = 0; channel < channels; ++channel) {
        std::copy_n(data + channel * frames, prefix_frames,
                    prefix.data() + channel * prefix_frames);
      }
      const int64_t dims[] = {1, channels, prefix_frames};
      auto input = Ort::Value::CreateTensor<float>(
          mem, prefix.data(), prefix.size(), dims, 3);
      auto audio = decoder_->Run({}, input_names, &input, 1, output_names, 1);
      committed = kFirstFrames * hop;
      const auto prefix_count =
          audio[0].GetTensorTypeAndShapeInfo().GetElementCount();
      const auto *samples = audio[0].GetTensorData<float>();
      prefix_samples.assign(samples, samples + prefix_count);
      if (!callback(samples, committed, float(committed) / float(frames * hop))) {
        Ort::AllocatorWithDefaultOptions allocator;
        const int64_t truncated_dims[] = {1, 1, 1, committed};
        auto truncated = Ort::Value::CreateTensor<float>(
            allocator, truncated_dims, 4);
        std::copy_n(samples, committed, truncated.GetTensorMutableData<float>());
        return truncated;
      }
    }
    auto audio = decoder_->Run({}, input_names, &latent, 1, output_names, 1);
    const int64_t count = audio[0].GetTensorTypeAndShapeInfo().GetElementCount();
    if (committed) {
      auto *full = audio[0].GetTensorMutableData<float>();
      std::copy_n(prefix_samples.data(), committed, full);
      const int64_t fade = std::min<int64_t>(
          {128, int64_t(prefix_samples.size()) - committed, count - committed});
      for (int64_t i = 0; i < fade; ++i) {
        const float weight = float(fade - i) / float(fade);
        full[committed + i] = prefix_samples[committed + i] * weight +
                              full[committed + i] * (1 - weight);
      }
    }
    if (callback) {
      callback(audio[0].GetTensorData<float>() + committed, count - committed, 1.f);
    }
    return std::move(audio[0]);
  }

  const OfflineTtsVitsModelMetaData &GetMetaData() const { return meta_data_; }

 private:
  static constexpr int kFirstFrames = 4;

  void ReadMetadata() {
    const auto validate_io = [](const Ort::Session &session,
                                std::initializer_list<const char *> inputs,
                                const char *output) {
      Ort::AllocatorWithDefaultOptions allocator;
      if (session.GetInputCount() != inputs.size() || session.GetOutputCount() != 1)
        throw std::runtime_error("Unexpected Piper AOT graph input or output count");
      size_t i = 0;
      for (auto name : inputs) {
        if (std::string(session.GetInputNameAllocated(i++, allocator).get()) != name)
          throw std::runtime_error("Unexpected Piper AOT input name");
      }
      if (std::string(session.GetOutputNameAllocated(0, allocator).get()) != output)
        throw std::runtime_error("Unexpected Piper AOT output name");
    };
    validate_io(*encoder_, {"input", "input_lengths", "scales"}, "latent");
    validate_io(*decoder_, {"latent"}, "audio");

    const auto meta = encoder_->GetModelMetadata();
    Ort::AllocatorWithDefaultOptions allocator;
    const auto field = [&](const char *name) {
      auto value = meta.LookupCustomMetadataMapAllocated(name, allocator);
      if (!value || !value.get()[0])
        throw std::runtime_error(std::string("Missing Piper AOT metadata: ") + name);
      return std::string(value.get());
    };
    meta_data_.sample_rate = std::stoi(field("sample_rate"));
    meta_data_.voice = field("voice");
    meta_data_.language = field("language");
    context_frames_ = std::stoi(field("piper_context_frames"));
    if (field("comment") != "piper" || field("has_espeak") != "1" ||
        field("n_speakers") != "1" || field("piper_first_frames") != "4" ||
        field("piper_max_tokens") != "0" ||
        !((meta_data_.sample_rate == 16000 && context_frames_ == 12) ||
          (meta_data_.sample_rate == 22050 && context_frames_ == 48)))
      throw std::runtime_error("Unexpected Piper AOT synthesis metadata");
    meta_data_.is_piper = true;
    meta_data_.num_speakers = 1;
  }

  Ort::Value Encode(Ort::Value input, float speed) {
    auto mem = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
    const auto shape = input.GetTensorTypeAndShapeInfo().GetShape();
    if (shape.size() != 2 || shape.front() != 1)
      throw std::invalid_argument("Piper AOT expects batch size 1");
    int64_t length = shape[1], one = 1, three = 3;
    float length_scale = config_.vits.length_scale;
    if (speed != 1 && speed > 0) length_scale = 1. / speed;
    std::array<float, 3> scales{
        config_.vits.noise_scale, length_scale, config_.vits.noise_scale_w};
    std::array<Ort::Value, 3> inputs{
        std::move(input),
        Ort::Value::CreateTensor(mem, &length, 1, &one, 1),
        Ort::Value::CreateTensor(mem, scales.data(), scales.size(), &three, 1)};
    const char *input_names[] = {"input", "input_lengths", "scales"};
    const char *output_names[] = {"latent"};
    auto results = encoder_->Run({}, input_names, inputs.data(), inputs.size(), output_names, 1);
    return std::move(results[0]);
  }

  OfflineTtsModelConfig config_;
  Ort::Env env_;
  Ort::SessionOptions sess_opts_;
  std::unique_ptr<transai::aot::Mapping> encoder_map_, decoder_map_;
  std::unique_ptr<Ort::Session> encoder_, decoder_;
  int context_frames_ = 0;
  OfflineTtsVitsModelMetaData meta_data_;
};

PiperNeonModel::PiperNeonModel(const OfflineTtsModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

PiperNeonModel::~PiperNeonModel() = default;

Ort::Value PiperNeonModel::RunStreaming(
    Ort::Value input, float speed,
    const std::function<int32_t(const float *, int32_t, float)> &callback,
    bool emit_prefix) const {
  return impl_->RunStreaming(std::move(input), speed, callback, emit_prefix);
}

const OfflineTtsVitsModelMetaData &PiperNeonModel::GetMetaData() const {
  return impl_->GetMetaData();
}

}  // namespace sherpa_onnx
