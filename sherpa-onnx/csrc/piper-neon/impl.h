// Copyright (c) 2023 Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_PIPER_NEON_IMPL_H_
#define SHERPA_ONNX_CSRC_PIPER_NEON_IMPL_H_

#include <algorithm>
#include <array>
#include <future>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/offline-tts-impl.h"
#include "sherpa-onnx/csrc/piper-neon/bryce_text_normalizer.h"
#include "sherpa-onnx/csrc/piper-neon/model.h"
#include "sherpa-onnx/csrc/neon-tts-load-profile.h"
#include "sherpa-onnx/csrc/piper-phonemize-lexicon.h"

namespace sherpa_onnx {

class PiperNeonImpl : public OfflineTtsImpl {
 public:
  PiperNeonImpl(const OfflineTtsConfig &config,
                std::future<std::string> frontend_ready, bool bryce)
      : config_(config), bryce_(bryce), model_(config.model) {
    if (frontend_ready.valid()) frontend_ready.get();
    transai::piper_neon::LoadProfile profile("frontend-init");
    frontend_ = std::make_unique<PiperPhonemizeLexicon>(
        config_.model.vits.tokens, config_.model.vits.data_dir, model_.GetMetaData());
  }

  int32_t SampleRate() const override { return model_.GetMetaData().sample_rate; }
  int32_t NumSpeakers() const override { return 1; }

  GeneratedAudio Generate(
      const std::string &input, const GenerationConfig &generation,
      GeneratedAudioCallback callback = nullptr) const override {
    if (config_.model.debug)
      SHERPA_ONNX_LOGE("%s", generation.ToString().c_str());
    if (generation.speed <= 0) {
      SHERPA_ONNX_LOGE("Piper synthesis speed must be positive.");
      return {};
    }
    if (generation.sid != 0)
      SHERPA_ONNX_LOGE("Piper is single-speaker. Use speaker ID 0.");

    const std::string text = bryce_ ? bryce_text::NormalizeEnglishNumbers(input) : input;
    auto sentences = frontend_->ConvertTextToTokenIds(text, model_.GetMetaData().voice);
    if (sentences.empty() || (sentences.size() == 1 && sentences.front().tokens.empty())) {
      SHERPA_ONNX_LOGE("Failed to convert Piper text to token IDs.");
      return {};
    }

    const size_t batch_size = config_.max_num_sentences > 0
                                  ? size_t(config_.max_num_sentences)
                                  : sentences.size();
    GeneratedAudio result;
    result.sample_rate = SampleRate();
    int batches_done = 0;
    bool emitted = false;
    bool cancelled = false;
    GeneratedAudioCallback stream;
    if (callback) {
      stream = [&](const float *p, int32_t n, float progress) {
        emitted = true;
        const auto keep = callback(p, n, (batches_done + progress) / sentences.size());
        if (progress >= 1) ++batches_done;
        cancelled = !keep;
        return keep;
      };
    }

    for (size_t begin = 0; begin < sentences.size() && !cancelled; begin += batch_size) {
      const size_t end = std::min(begin + batch_size, sentences.size());
      size_t num_tokens = 0;
      for (size_t i = begin; i < end; ++i) num_tokens += sentences[i].tokens.size();
      std::vector<int64_t> tokens;
      tokens.reserve(num_tokens);
      for (size_t i = begin; i < end; ++i) {
        tokens.insert(tokens.end(), sentences[i].tokens.begin(), sentences[i].tokens.end());
      }
      auto mem = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
      const std::array<int64_t, 2> shape{1, static_cast<int64_t>(tokens.size())};
      auto tensor = Ort::Value::CreateTensor(
          mem, tokens.data(), tokens.size(), shape.data(), shape.size());
      auto audio = model_.RunStreaming(std::move(tensor), generation.speed, stream, !emitted);
      const auto *data = audio.GetTensorData<float>();
      const auto count = audio.GetTensorTypeAndShapeInfo().GetElementCount();
      if (generation.silence_scale == 1) {
        result.samples.insert(result.samples.end(), data, data + count);
      } else {
        GeneratedAudio segment;
        segment.sample_rate = result.sample_rate;
        segment.samples.assign(data, data + count);
        segment = segment.ScaleSilence(generation.silence_scale);
        result.samples.insert(result.samples.end(), segment.samples.begin(), segment.samples.end());
      }
    }
    return result;
  }

  GeneratedAudio Generate(
      const std::string &text, int64_t sid = 0, float speed = 1.0,
      GeneratedAudioCallback callback = nullptr) const override {
    GenerationConfig generation;
    generation.sid = sid;
    generation.speed = speed;
    generation.silence_scale = config_.silence_scale;
    return Generate(text, generation, std::move(callback));
  }

 private:
  OfflineTtsConfig config_;
  bool bryce_;
  PiperNeonModel model_;
  std::unique_ptr<PiperPhonemizeLexicon> frontend_;
};

}  // namespace sherpa_onnx
#endif  // SHERPA_ONNX_CSRC_PIPER_NEON_IMPL_H_
