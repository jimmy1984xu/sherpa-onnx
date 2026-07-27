// sherpa-onnx/csrc/offline-ctc-greedy-search-decoder.h
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-ctc-greedy-search-decoder.h"

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/macros.h"

namespace sherpa_onnx {

namespace {

float LogSumExp(const float *logits, int32_t vocab_size) {
  const float max_logit = *std::max_element(logits, logits + vocab_size);
  float sum = 0.0f;
  for (int32_t i = 0; i != vocab_size; ++i) {
    sum += std::exp(logits[i] - max_logit);
  }

  return max_logit + std::log(sum);
}

}  // namespace

std::vector<OfflineCtcDecoderResult> OfflineCtcGreedySearchDecoder::Decode(
    Ort::Value log_probs, Ort::Value log_probs_length) {
  std::vector<int64_t> shape = log_probs.GetTensorTypeAndShapeInfo().GetShape();
  int32_t batch_size = static_cast<int32_t>(shape[0]);
  int32_t num_frames = static_cast<int32_t>(shape[1]);
  int32_t vocab_size = static_cast<int32_t>(shape[2]);

  const int64_t *p_log_probs_length = log_probs_length.GetTensorData<int64_t>();

  std::vector<OfflineCtcDecoderResult> ans;
  ans.reserve(batch_size);

  for (int32_t b = 0; b != batch_size; ++b) {
    const float *p_log_probs =
        log_probs.GetTensorData<float>() + b * num_frames * vocab_size;

    OfflineCtcDecoderResult r;
    int64_t prev_id = -1;
    float raw_log_prob_sum = 0.0f;
    float normalized_log_prob_sum = 0.0f;

    for (int32_t t = 0; t != static_cast<int32_t>(p_log_probs_length[b]); ++t) {
      auto y = static_cast<int64_t>(std::distance(
          static_cast<const float *>(p_log_probs),
          std::max_element(
              static_cast<const float *>(p_log_probs),
              static_cast<const float *>(p_log_probs) + vocab_size)));

      const float raw_log_prob = p_log_probs[y];
      const float log_sum_exp = LogSumExp(p_log_probs, vocab_size);
      const float normalized_log_prob = raw_log_prob - log_sum_exp;
      p_log_probs += vocab_size;

      if (y != blank_id_ && y != prev_id) {
        r.tokens.push_back(y);
        r.timestamps.push_back(t);
        r.ys_log_probs.push_back(raw_log_prob);
        raw_log_prob_sum += raw_log_prob;
        normalized_log_prob_sum += normalized_log_prob;
        SHERPA_ONNX_LOGE(
            "CTC confidence token: batch=%d frame=%d token=%lld raw_log_prob=%.6f "
            "log_sum_exp=%.6f normalized_log_prob=%.6f posterior=%.6f",
            b, t, static_cast<long long>(y), raw_log_prob, log_sum_exp,
            normalized_log_prob, std::exp(normalized_log_prob));
      }
      prev_id = y;
    }  // for (int32_t t = 0; ...)

    if (!r.tokens.empty()) {
      const float token_count = static_cast<float>(r.tokens.size());
      SHERPA_ONNX_LOGE(
          "CTC confidence summary: batch=%d emitted_tokens=%d "
          "raw_confidence=%.6f normalized_confidence=%.6f",
          b, static_cast<int32_t>(r.tokens.size()),
          std::exp(raw_log_prob_sum / token_count),
          std::exp(normalized_log_prob_sum / token_count));
    } else {
      SHERPA_ONNX_LOGE("CTC confidence summary: batch=%d emitted_tokens=0", b);
    }

    ans.push_back(std::move(r));
  }
  return ans;
}

}  // namespace sherpa_onnx
