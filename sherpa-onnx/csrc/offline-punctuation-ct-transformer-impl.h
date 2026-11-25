// sherpa-onnx/csrc/offline-punctuation-ct-transformer-impl.h
//
// Copyright (c)  2024  Xiaomi Corporation
#ifndef SHERPA_ONNX_CSRC_OFFLINE_PUNCTUATION_CT_TRANSFORMER_IMPL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_PUNCTUATION_CT_TRANSFORMER_IMPL_H_

#include <math.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/math.h"
#include "sherpa-onnx/csrc/offline-ct-transformer-model.h"
#include "sherpa-onnx/csrc/offline-punctuation-impl.h"
#include "sherpa-onnx/csrc/offline-punctuation.h"
#include "sherpa-onnx/csrc/text-utils.h"

namespace sherpa_onnx {

class OfflinePunctuationCtTransformerImpl : public OfflinePunctuationImpl {
 public:
  explicit OfflinePunctuationCtTransformerImpl(
      const OfflinePunctuationConfig &config)
      : config_(config), model_(config.model) {}

  template <typename Manager>
  OfflinePunctuationCtTransformerImpl(Manager *mgr,
                                      const OfflinePunctuationConfig &config)
      : config_(config), model_(mgr, config.model) {}

  std::string AddPunctuation(const std::string &text) const override {
    if (text.empty()) {
      return {};
    }

    std::vector<std::string> tokens = SplitUtf8(text);
    std::vector<int32_t> token_ids;
    token_ids.reserve(tokens.size());

    const auto &meta_data = model_.GetModelMetadata();

    for (const auto &t : tokens) {
      std::string token = ToLowerCase(t);
      if (meta_data.token2id.count(token)) {
        token_ids.push_back(meta_data.token2id.at(token));
      } else {
        token_ids.push_back(meta_data.unk_id);
      }
    }

    if (token_ids.empty()) return text + meta_data.id2punct[meta_data.dot_id];

    const bool neon = config_.model.provider == "neon";
    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    int32_t segment_size = 20;
    int32_t max_len = 200;
    int32_t num_segments =
        ceil((static_cast<float>(token_ids.size()) + segment_size - 1) /
             segment_size);

    std::vector<int32_t> punctuations;
    std::vector<int32_t> cached;
    std::vector<int32_t> this_punctuations;
    std::vector<float> logits;
    int32_t previous_start = -1, previous_end = -1;
    int32_t last = -1;
    for (int32_t i = 0; i != num_segments; ++i) {
      int32_t this_start = i * segment_size;         // included
      int32_t this_end = this_start + segment_size;  // not included
      if (this_end > static_cast<int32_t>(token_ids.size())) {
        this_end = token_ids.size();
      }

      if (last != -1) {
        this_start = last;
      }
      // Cache raw argmax for repeated NEON windows before boundary edits.
      const int32_t len = this_end - this_start;
      if (len <= 0) break;
      if (!neon || this_start != previous_start || this_end != previous_end) {
        const std::array<int64_t, 2> shape = {1, len};
        Ort::Value x = Ort::Value::CreateTensor(
            memory_info, token_ids.data() + this_start, len, shape.data(),
            shape.size());
        const int64_t len_shape = 1;
        int32_t input_len = len;
        Ort::Value x_len = Ort::Value::CreateTensor(
            memory_info, &input_len, 1, &len_shape, 1);
        Ort::Value out{nullptr};
        if (neon) {
          const std::array<int64_t, 3> logit_shape = {
              1, len, meta_data.num_punctuations};
          logits.resize(static_cast<size_t>(len) * meta_data.num_punctuations);
          out = Ort::Value::CreateTensor(memory_info, logits.data(), logits.size(),
                                         logit_shape.data(), logit_shape.size());
          model_.ForwardInto(std::move(x), std::move(x_len), &out);
        } else {
          out = model_.Forward(std::move(x), std::move(x_len));
          const auto out_shape = out.GetTensorTypeAndShapeInfo().GetShape();
          assert(out_shape[0] == 1);
          assert(out_shape[1] == len);
          assert(out_shape[2] == meta_data.num_punctuations);
        }
        const float *p = out.GetTensorData<float>();
        cached.resize(len);
        for (int32_t k = 0; k != len; ++k, p += meta_data.num_punctuations) {
          cached[k] = static_cast<int32_t>(
              std::max_element(p, p + meta_data.num_punctuations) - p);
        }
        previous_start = this_start;
        previous_end = this_end;
      }
      this_punctuations = cached;

      int32_t dot_index = -1;
      int32_t comma_index = -1;

      for (int32_t m = static_cast<int32_t>(this_punctuations.size()) - 2;
           m >= 1; --m) {
        int32_t punct_id = this_punctuations[m];

        if (punct_id == meta_data.dot_id || punct_id == meta_data.quest_id) {
          dot_index = m;
          break;
        }

        if (comma_index == -1 && punct_id == meta_data.comma_id) {
          comma_index = m;
        }
      }  // for (int32_t k = this_punctuations.size() - 1; k >= 1; --k)

      if (dot_index == -1 && len >= max_len && comma_index != -1) {
        dot_index = comma_index;
        this_punctuations[dot_index] = meta_data.dot_id;
      }

      if (dot_index == -1) {
        if (last == -1) {
          last = this_start;
        }

        if (i == num_segments - 1) {
          dot_index = static_cast<int32_t>(this_punctuations.size()) - 1;
        }
      } else {
        last = this_start + dot_index + 1;
      }

      if (dot_index != -1) {
        punctuations.insert(punctuations.end(), this_punctuations.begin(),
                            this_punctuations.begin() + (dot_index + 1));
      }
    }  // for (int32_t i = 0; i != num_segments; ++i)

    if (punctuations.empty()) {
      return text + meta_data.id2punct[meta_data.dot_id];
    }
    std::vector<std::string> words_punct;

    for (int32_t i = 0; i != static_cast<int32_t>(punctuations.size()); ++i) {
      if (i >= static_cast<int32_t>(tokens.size())) {
        break;
      }
      std::string &w = tokens[i];
      if (i > 0 && !(words_punct.back()[0] & 0x80) && !(w[0] & 0x80)) {
        words_punct.push_back(" ");
      }
      words_punct.push_back(std::move(w));

      if (punctuations[i] != meta_data.underline_id) {
        words_punct.push_back(meta_data.id2punct[punctuations[i]]);
      }
    }

    if (words_punct.back() == meta_data.id2punct[meta_data.comma_id] ||
        words_punct.back() == meta_data.id2punct[meta_data.pause_id]) {
      words_punct.back() = meta_data.id2punct[meta_data.dot_id];
    }

    if (words_punct.back() != meta_data.id2punct[meta_data.dot_id] &&
        words_punct.back() != meta_data.id2punct[meta_data.quest_id]) {
      words_punct.push_back(meta_data.id2punct[meta_data.dot_id]);
    }

    std::string ans;
    for (const auto &w : words_punct) {
      ans.append(w);
    }
    return ans;
  }

 private:
  OfflinePunctuationConfig config_;
  OfflineCtTransformerModel model_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_PUNCTUATION_CT_TRANSFORMER_IMPL_H_
