// sherpa-onnx/csrc/offline-paraformer-beam-search-decoder.cc
#include "sherpa-onnx/csrc/offline-paraformer-beam-search-decoder.h"

#include <algorithm>
#include <cmath>
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

// A beam entry stores only the token emitted at this frame and its parent.
// Keeping parent indices avoids copying the complete prefix for every vocab
// candidate. The selected path is reconstructed after the final frame.
struct BeamEntry {
  float score = 0.0f;
  float acoustic_log_prob = 0.0f;
  const ContextState* graph_state = nullptr;
  int32_t parent_index = -1;
  int64_t token = -1;
  uint64_t order = 0;
};

bool IsBetter(const BeamEntry& lhs, const BeamEntry& rhs) {
  if (lhs.score != rhs.score) return lhs.score > rhs.score;
  return lhs.order < rhs.order;
}

// priority_queue::top() is the worst retained candidate, so replacement is
// O(log K) and the queue never grows to K * vocab_size entries.
struct WorseFirst {
  bool operator()(const BeamEntry& lhs, const BeamEntry& rhs) const {
    return IsBetter(lhs, rhs);
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

    // 获取热词图
    ContextGraphPtr graph = nullptr;
    if (ss && ss[b]) graph = ss[b]->GetContextGraph();
    // CreateStream() may provide an empty graph when no hotwords are active.
    // Treat it as no graph so the hotword transition lookup is removed from
    // the innermost vocab loop.
    if (graph && graph->Root()->next.empty()) graph.reset();

    // 初始化 beam
    const ContextState* start_state = graph ? graph->Root() : nullptr;
    std::vector<BeamEntry> beam;
    beam.push_back(BeamEntry{0.0f, 0.0f, start_state, -1, -1, 0});
    std::vector<std::vector<BeamEntry>> history;
    history.reserve(T);
    const int32_t beam_size = std::max<int32_t>(1, max_active_paths_);
    uint64_t order = 1;

    for (int64_t t = 0; t < T; ++t) {
      const float* row = utt_logits + t * vocab_size;
      const float max_val = *std::max_element(row, row + vocab_size);
      float sum = 0.0f;
      for (int64_t v = 0; v < vocab_size; ++v) {
        sum += std::exp(row[v] - max_val);
      }
      const float log_sum = std::log(sum) + max_val;
      std::priority_queue<BeamEntry, std::vector<BeamEntry>, WorseFirst>
          new_beam_queue;
      for (int32_t parent = 0; parent < static_cast<int32_t>(beam.size());
           ++parent) {
        const auto& entry = beam[parent];
        for (int64_t v = 0; v < vocab_size; ++v) {
          const float acoustic_log_prob = row[v] - log_sum;
          float score = entry.score + acoustic_log_prob;
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
          BeamEntry candidate{score, acoustic_log_prob, next_state, parent, v,
                              order++};
          if (new_beam_queue.size() < static_cast<size_t>(beam_size)) {
            new_beam_queue.push(std::move(candidate));
          } else if (IsBetter(candidate, new_beam_queue.top())) {
            new_beam_queue.pop();
            new_beam_queue.push(std::move(candidate));
          }
        }
      }
      std::vector<BeamEntry> next_beam;
      next_beam.reserve(new_beam_queue.size());
      while (!new_beam_queue.empty()) {
        next_beam.push_back(std::move(new_beam_queue.top()));
        new_beam_queue.pop();
      }
      std::sort(next_beam.begin(), next_beam.end(), IsBetter);
      history.push_back(next_beam);
      beam = std::move(next_beam);
    }

    // 选择最佳路径
    if (!beam.empty()) {
      std::vector<std::pair<int64_t, float>> reversed_path;
      reversed_path.reserve(T);
      int32_t path_index = 0;  // next_beam is sorted from best to worst
      for (int64_t t = T - 1; t >= 0; --t) {
        const auto& entry = history[t][path_index];
        reversed_path.emplace_back(entry.token, entry.acoustic_log_prob);
        path_index = entry.parent_index;
      }
      std::reverse(reversed_path.begin(), reversed_path.end());
      // The beam score includes ContextGraph hotword bonuses, so confidence
      // must come from the acoustic log probability for each selected token.
      // EOS is a termination marker and is not returned as a user token.
      for (const auto& [token, acoustic_log_prob] : reversed_path) {
        if (token == eos_id_) {
          break;
        }
        results[b].tokens.push_back(token);
        results[b].ys_log_probs.push_back(acoustic_log_prob);
      }
      results[b].timestamps.clear();
    }
  }
  return results;
}

}  // namespace sherpa_onnx
