// sherpa-onnx/csrc/confidence-utils.h
//
// Copyright (c)  2025  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_CONFIDENCE_UTILS_H_
#define SHERPA_ONNX_CSRC_CONFIDENCE_UTILS_H_

#include <cmath>
#include <vector>

namespace sherpa_onnx {

// Calculate average confidence from log probabilities
// Formula: confidence = (1/N) * Σ exp(log_prob_i)
// where N is the number of tokens and log_prob_i is the log probability
// of the i-th token.
//
// Args:
//   token_log_probs: A vector of log probabilities for each token
//
// Returns:
//   Average confidence value in range [0.0, 1.0]
//   Returns 0.0 if token_log_probs is empty (no tokens)
//
// Note:
//   This function computes the arithmetic mean of probabilities:
//   mean(exp(log_prob)) rather than exp(mean(log_prob))
inline float CalculateAverageConfidence(
    const std::vector<float> &token_log_probs) {
  if (token_log_probs.empty()) {
    // Empty text: set confidence to 0.0 (no tokens means no confidence)
    return 0.0f;
  }

  float sum_exp = 0.0f;
  for (float log_prob : token_log_probs) {
    // exp(log_prob) converts log probability to probability
    sum_exp += std::exp(log_prob);
  }
  return sum_exp / static_cast<float>(token_log_probs.size());
}

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_CONFIDENCE_UTILS_H_

