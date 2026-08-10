// sherpa-onnx/csrc/offline-paraformer-modified-beam-search-decoder.cc
//
// Copyright (c) 2026

#include "sherpa-onnx/csrc/offline-paraformer-modified-beam-search-decoder.h"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/math.h"

namespace sherpa_onnx {
namespace {

struct Hypothesis {
  std::vector<int64_t> tokens;
  std::vector<float> ys_log_probs;
  float log_prob = 0.0F;
  const ContextState *context_state = nullptr;
};

bool CompareHypotheses(const Hypothesis &a, const Hypothesis &b) {
  return a.log_prob > b.log_prob;
}

void SetTimestamps(OfflineParaformerDecoderResult *result,
                   const float *peak, int32_t dim) {
  std::vector<float> timestamps;
  timestamps.reserve(result->tokens.size());

  constexpr float kFrameShiftInMilliseconds = 10.0F;
  constexpr float kLfrWindowSize = 6.0F;
  constexpr float kUpsampleFactor = 3.0F;
  constexpr float kMillisecondsPerSecond = 1000.0F;
  const float scale = kFrameShiftInMilliseconds * kLfrWindowSize /
                      kUpsampleFactor / kMillisecondsPerSecond;

  for (int32_t k = 0; k != dim; ++k) {
    if (peak[k] > 1.0F - 1e-4F) {
      timestamps.push_back(k * scale);
    }
  }

  if (!timestamps.empty()) {
    timestamps.pop_back();
  }

  if (timestamps.size() == result->tokens.size()) {
    result->timestamps = std::move(timestamps);
  }
}

}  // namespace

std::vector<OfflineParaformerDecoderResult>
OfflineParaformerModifiedBeamSearchDecoder::Decode(
    Ort::Value log_probs, Ort::Value token_num,
    Ort::Value us_cif_peak /*=Ort::Value(nullptr)*/) {
  const std::vector<int64_t> shape =
      log_probs.GetTensorTypeAndShapeInfo().GetShape();
  const int32_t batch_size = static_cast<int32_t>(shape[0]);
  const int32_t num_tokens = static_cast<int32_t>(shape[1]);
  const int32_t vocab_size = static_cast<int32_t>(shape[2]);
  const float *logits = log_probs.GetTensorData<float>();

  std::vector<int32_t> token_counts(batch_size, num_tokens);
  const auto token_num_info = token_num.GetTensorTypeAndShapeInfo();
  if (token_num_info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
    const int32_t *p = token_num.GetTensorData<int32_t>();
    std::copy(p, p + batch_size, token_counts.begin());
  } else {
    const int64_t *p = token_num.GetTensorData<int64_t>();
    for (int32_t i = 0; i != batch_size; ++i) {
      token_counts[i] = static_cast<int32_t>(p[i]);
    }
  }

  std::vector<OfflineParaformerDecoderResult> results(batch_size);
  for (int32_t i = 0; i != batch_size; ++i) {
    const int32_t this_num_tokens =
        std::max(0, std::min(token_counts[i], num_tokens));
    results[i] = DecodeOne(logits + i * num_tokens * vocab_size,
                           this_num_tokens, vocab_size);

    if (us_cif_peak) {
      const int32_t dim = static_cast<int32_t>(
          us_cif_peak.GetTensorTypeAndShapeInfo().GetShape().back());
      const float *peak = us_cif_peak.GetTensorData<float>() + i * dim;
      SetTimestamps(&results[i], peak, dim);
    }
  }

  return results;
}

OfflineParaformerDecoderResult
OfflineParaformerModifiedBeamSearchDecoder::DecodeOne(
    const float *logits, int32_t num_tokens, int32_t vocab_size) const {
  const bool preserve_greedy_baseline = context_graph_ != nullptr;

  // Reserve one active-path slot for the un-biased greedy result. Otherwise,
  // partial hotword prefixes can prune the normal result before their score is
  // cancelled by ContextGraph::Finalize().
  std::vector<Hypothesis> active;
  if (!preserve_greedy_baseline || max_active_paths_ > 1) {
    active.resize(1);
    active[0].context_state =
        context_graph_ == nullptr ? nullptr : context_graph_->Root();
  }

  Hypothesis greedy_baseline;
  bool greedy_baseline_finished = false;
  std::vector<Hypothesis> completed;

  for (int32_t t = 0;
       t != num_tokens && (preserve_greedy_baseline || !active.empty()); ++t) {
    std::vector<Hypothesis> candidates;
    candidates.reserve(active.size() * vocab_size);
    const float *frame_logits = logits + t * vocab_size;
    std::vector<float> frame_log_probs(frame_logits,
                                       frame_logits + vocab_size);
    LogSoftmax(frame_log_probs.data(), vocab_size);

    if (preserve_greedy_baseline && !greedy_baseline_finished) {
      const int32_t token = static_cast<int32_t>(std::distance(
          frame_log_probs.begin(),
          std::max_element(frame_log_probs.begin(), frame_log_probs.end())));
      const float token_log_prob = frame_log_probs[token];
      greedy_baseline.log_prob += token_log_prob;
      if (token == eos_id_) {
        greedy_baseline_finished = true;
      } else {
        greedy_baseline.tokens.push_back(token);
        greedy_baseline.ys_log_probs.push_back(token_log_prob);
      }
    }

    if (active.empty()) {
      continue;
    }

    for (const auto &hyp : active) {
      for (int32_t token = 0; token != vocab_size; ++token) {
        const float token_log_prob = frame_log_probs[token];
        if (token == eos_id_) {
          Hypothesis ended = hyp;
          ended.log_prob += token_log_prob;
          completed.push_back(std::move(ended));
          continue;
        }

        Hypothesis next = hyp;
        next.tokens.push_back(token);
        next.ys_log_probs.push_back(token_log_prob);
        next.log_prob += token_log_prob;

        if (context_graph_ != nullptr) {
          auto context_result = context_graph_->ForwardOneStep(
              next.context_state, token, /*strict_mode=*/false);
          next.log_prob += std::get<0>(context_result);
          next.context_state = std::get<1>(context_result);
          const auto *matched_node = std::get<2>(context_result);
          if (matched_node != nullptr) {
            SHERPA_ONNX_LOGE(
                "[debug] Paraformer hotword candidate matched: level=%d, "
                "bonus=%.2f, phrase='%s'",
                matched_node->level, std::get<0>(context_result),
                matched_node->phrase.c_str());
          }
        }
        candidates.push_back(std::move(next));
      }
    }

    const int32_t biased_capacity =
        max_active_paths_ - (preserve_greedy_baseline ? 1 : 0);
    const int32_t keep = std::min<int32_t>(biased_capacity, candidates.size());
    if (keep == 0) {
      active.clear();
      break;
    }
    std::partial_sort(candidates.begin(), candidates.begin() + keep,
                      candidates.end(), CompareHypotheses);
    candidates.resize(keep);
    active = std::move(candidates);
  }

  active.insert(active.end(), std::make_move_iterator(completed.begin()),
                std::make_move_iterator(completed.end()));
  OfflineParaformerDecoderResult result;
  if (active.empty() && !preserve_greedy_baseline) {
    return result;
  }

  for (auto &hyp : active) {
    if (context_graph_ != nullptr) {
      auto finalize_result = context_graph_->Finalize(hyp.context_state);
      hyp.log_prob += finalize_result.first;
    }
  }

  const Hypothesis *best =
      preserve_greedy_baseline ? &greedy_baseline : nullptr;
  if (!active.empty()) {
    const auto best_biased =
        std::max_element(active.begin(), active.end(), CompareHypotheses);
    if (best == nullptr || best_biased->log_prob > best->log_prob) {
      best = &*best_biased;
    }
  }

  if (best == nullptr) {
    return result;
  }
  result.tokens = best->tokens;
  result.ys_log_probs = best->ys_log_probs;
  return result;
}

}  // namespace sherpa_onnx
