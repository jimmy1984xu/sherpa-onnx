// sherpa-onnx/csrc/offline-paraformer-beam-search-decoder.cc
#include "sherpa-onnx/csrc/offline-paraformer-beam-search-decoder.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <tuple>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/context-graph.h"
#include "sherpa-onnx/csrc/log.h"
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/macros.h"
namespace sherpa_onnx {

namespace {

// 对 logits 做 log_softmax，返回 (T, vocab) 的 log 概率矩阵
std::vector<std::vector<float>> LogSoftmax(const Ort::Value& logits) {
  auto info = logits.GetTensorTypeAndShapeInfo();
  auto shape = info.GetShape();
  int64_t T = shape[0];
  int64_t vocab = shape[1];

  const float* data = logits.GetTensorData<float>();
  std::vector<std::vector<float>> log_probs(T, std::vector<float>(vocab));

  for (int64_t t = 0; t < T; ++t) {
    const float* row = data + t * vocab;
    float max_val = *std::max_element(row, row + vocab);
    float sum = 0.0f;
    for (int64_t v = 0; v < vocab; ++v) {
      sum += std::exp(row[v] - max_val);
    }
    float log_sum = std::log(sum) + max_val;
    for (int64_t v = 0; v < vocab; ++v) {
      log_probs[t][v] = row[v] - log_sum;
    }
  }
  return log_probs;
}

// Beam 条目，使用 const ContextState* 存储当前热词图状态
struct BeamEntry {
  std::vector<int64_t> tokens;          // 已输出的 token ID 序列
  float score;                          // 累积 log 概率 + 热词偏置
  const ContextState* graph_state;      // 当前热词图状态（nullptr 表示无图）

  BeamEntry(const std::vector<int64_t>& tok, float s, const ContextState* gs)
      : tokens(tok), score(s), graph_state(gs) {}

  // 按分数降序（用于优先队列）
  bool operator<(const BeamEntry& other) const {
    return score < other.score;
  }
};

}  // namespace

OfflineParaformerBeamSearchDecoder::OfflineParaformerBeamSearchDecoder(
    int32_t eos_id, int32_t max_active_paths, float lm_scale, int32_t unk_id,
    float hotwords_score, float blank_penalty)
    : eos_id_(eos_id),
      max_active_paths_(max_active_paths),
      lm_scale_(lm_scale),
      unk_id_(unk_id),
      hotwords_score_(hotwords_score),
      blank_penalty_(blank_penalty) {}

ContextGraphPtr OfflineParaformerBeamSearchDecoder::GetContextGraph(
    OfflineStream* s) const {
  if (!s) return nullptr;
  // 假设 OfflineStream 有 GetContextGraph() 方法
  return s->GetContextGraph();
}

std::vector<OfflineParaformerDecoderResult>
OfflineParaformerBeamSearchDecoder::Decode(
    Ort::Value logits, Ort::Value logits_length,
    Ort::Value /*us_cif_peak*/, OfflineStream **ss, int32_t n) {
  auto logits_info = logits.GetTensorTypeAndShapeInfo();
  auto shape = logits_info.GetShape();
  int64_t B = shape[0];
  int64_t T_max = shape[1];
  int64_t vocab_size = shape[2];

  // ---------- 安全读取长度 ----------
  std::vector<int64_t> safe_len(B, T_max);  // 默认使用 T_max
  bool use_fallback = true;

  try {
    auto len_info = logits_length.GetTensorTypeAndShapeInfo();
    auto len_shape = len_info.GetShape();
    auto len_type = len_info.GetElementType();

    if (len_shape.size() == 1 && len_shape[0] == B) {
      if (len_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
        const int64_t* raw_len = logits_length.GetTensorData<int64_t>();
        bool all_valid = true;
        for (int64_t i = 0; i < B; ++i) {
          if (raw_len[i] <= 0 || raw_len[i] > T_max) { all_valid = false; break; }
        }
        if (all_valid) {
          for (int64_t i = 0; i < B; ++i) safe_len[i] = raw_len[i];
          use_fallback = false;
        }
      } else if (len_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
        const int32_t* raw_len = logits_length.GetTensorData<int32_t>();
        bool all_valid = true;
        for (int64_t i = 0; i < B; ++i) {
          if (raw_len[i] <= 0 || raw_len[i] > T_max) { all_valid = false; break; }
        }
        if (all_valid) {
          for (int64_t i = 0; i < B; ++i) safe_len[i] = static_cast<int64_t>(raw_len[i]);
          use_fallback = false;
        }
      } else {
        SHERPA_ONNX_LOGE("logits_length has unsupported type (%d), using T_max fallback.", len_type);
      }
    } else {
      SHERPA_ONNX_LOGE("logits_length shape is not (B,), using T_max fallback.");
    }
  } catch (const std::exception& e) {
    SHERPA_ONNX_LOGE("Exception when reading logits_length: %s, using T_max fallback.", e.what());
  }

  if (use_fallback) {
    SHERPA_ONNX_LOGE("Using T_max as length for all utterances due to invalid logits_length.");
  }
  // ---------------------------------

  std::vector<OfflineParaformerDecoderResult> results(B);

  for (int64_t b = 0; b < B; ++b) {
    int64_t T = safe_len[b];  // 使用安全的长度
    if (T <= 0) {
      results[b].tokens.clear();
      results[b].timestamps.clear();
      continue;
    }

    const float* utt_logits = logits.GetTensorData<float>() + b * T_max * vocab_size;

    // 计算 log softmax per frame (只计算有效长度)
    std::vector<std::vector<float>> frame_log_probs(T, std::vector<float>(vocab_size));
    for (int64_t t = 0; t < T; ++t) {
      const float* row = utt_logits + t * vocab_size;
      float max_val = *std::max_element(row, row + vocab_size);
      float sum = 0.0f;
      for (int64_t v = 0; v < vocab_size; ++v) {
        sum += std::exp(row[v] - max_val);
      }
      float log_sum = std::log(sum) + max_val;
      for (int64_t v = 0; v < vocab_size; ++v) {
        frame_log_probs[t][v] = row[v] - log_sum;
      }
    }

    // 获取热词图
    ContextGraphPtr graph = nullptr;
    if (ss && ss[b]) graph = ss[b]->GetContextGraph();

    // 初始化 beam
    const ContextState* start_state = graph ? graph->Root() : nullptr;
    std::vector<BeamEntry> beam;
    beam.emplace_back(std::vector<int64_t>(), 0.0f, start_state);

    for (int64_t t = 0; t < T; ++t) {
      std::priority_queue<BeamEntry> new_beam_queue;
      const auto& frame_probs = frame_log_probs[t];
      for (const auto& entry : beam) {
        for (int64_t v = 0; v < vocab_size; ++v) {
          float score = entry.score + frame_probs[v];
          const ContextState* next_state = entry.graph_state;
          if (graph && entry.graph_state) {
            float fw_score = 0.0f;
            const ContextState* out_state = nullptr;
            std::tie(fw_score, next_state, out_state) =
                graph->ForwardOneStep(entry.graph_state, static_cast<int32_t>(v), true);
            score += fw_score;
          } else {
            next_state = entry.graph_state;
          }
          std::vector<int64_t> new_tokens = entry.tokens;
          new_tokens.push_back(v);
          new_beam_queue.emplace(new_tokens, score, next_state);
        }
      }
      // 裁剪 beam
      beam.clear();
      int kept = 0;
      while (!new_beam_queue.empty() && kept < max_active_paths_) {
        beam.push_back(new_beam_queue.top());
        new_beam_queue.pop();
        ++kept;
      }
    }

    // 选择最佳路径
    if (!beam.empty()) {
      auto best = std::max_element(beam.begin(), beam.end(),
          [](const BeamEntry& a, const BeamEntry& b) { return a.score < b.score; });
      results[b].tokens = best->tokens;
      // EOS is a termination marker, not a user-visible token. The beam
      // search keeps fixed-width paths and may continue expanding after EOS,
      // so truncate at its first occurrence before converting token IDs.
      auto eos_it = std::find(results[b].tokens.begin(),
                              results[b].tokens.end(),
                              static_cast<int64_t>(eos_id_));
      if (eos_it != results[b].tokens.end()) {
        results[b].tokens.erase(eos_it, results[b].tokens.end());
      }
      results[b].timestamps.clear();
    }
  }
  return results;
}

}  // namespace sherpa_onnx
