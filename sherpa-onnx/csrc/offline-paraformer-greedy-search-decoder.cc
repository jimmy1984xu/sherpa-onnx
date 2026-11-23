// sherpa-onnx/csrc/offline-paraformer-greedy-search-decoder.cc
//
// Copyright (c)  2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-paraformer-greedy-search-decoder.h"

#include <algorithm>
#include <sstream>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/math.h"

namespace sherpa_onnx {

std::vector<OfflineParaformerDecoderResult>
OfflineParaformerGreedySearchDecoder::Decode(
    Ort::Value log_probs, Ort::Value /*token_num*/,
    Ort::Value us_cif_peak,
    OfflineStream ** /*ss*/, int32_t /*n*/
) {
  std::vector<int64_t> shape = log_probs.GetTensorTypeAndShapeInfo().GetShape();
  if (shape.size() != 3 || shape[0] <= 0 || shape[1] <= 0 || shape[2] <= 0) {
    SHERPA_ONNX_LOGE("ParaformerGreedyDecode: invalid log_probs shape rank=%zu",
                     shape.size());
    return {};
  }
  int32_t batch_size = static_cast<int32_t>(shape[0]);
  int32_t num_tokens = static_cast<int32_t>(shape[1]);
  int32_t vocab_size = static_cast<int32_t>(shape[2]);
  const size_t row_stride = static_cast<size_t>(num_tokens) * vocab_size;
  const float *base = log_probs.GetTensorData<float>();
  if (!base) {
    SHERPA_ONNX_LOGE("ParaformerGreedyDecode: null log_probs data");
    return {};
  }
  SHERPA_ONNX_LOGI(
      "ParaformerGreedyDecode: batch=%d tokens=%d vocab=%d stride=%zu",
      batch_size, num_tokens, vocab_size, row_stride);

  std::vector<OfflineParaformerDecoderResult> results(batch_size);

  for (int32_t i = 0; i != batch_size; ++i) {
    const float *p = base + static_cast<size_t>(i) * row_stride;
    for (int32_t k = 0; k != num_tokens; ++k) {
      auto max_idx = static_cast<int64_t>(
          std::distance(p, std::max_element(p, p + vocab_size)));
      if (max_idx == eos_id_) {
        break;
      }

      results[i].tokens.push_back(max_idx);
      results[i].ys_log_probs.push_back(
          ComputeLogSoftmaxScore(p, vocab_size,
                                 static_cast<int32_t>(max_idx)));

      p += vocab_size;
    }

    if (!results[i].tokens.empty()) {
      std::ostringstream os;
      os << "token_ids(" << results[i].tokens.size() << ")=[" << results[i].tokens[0];
      for (size_t ti = 1; ti < results[i].tokens.size() && ti < 32; ++ti) {
        os << "," << results[i].tokens[ti];
      }
      if (results[i].tokens.size() > 32) os << ",...";
      os << "]";
      SHERPA_ONNX_LOGI("ParaformerGreedyDecode: stream=%d %s", i,
                       os.str().c_str());
    }

    if (us_cif_peak) {
      const auto peak_shape =
          us_cif_peak.GetTensorTypeAndShapeInfo().GetShape();
      if (!peak_shape.empty()) {
        const int32_t dim = static_cast<int32_t>(peak_shape.back());
        const float *peak = us_cif_peak.GetTensorData<float>() + i * dim;
        std::vector<float> timestamps;
        timestamps.reserve(results[i].tokens.size());
        // Paraformer CIF peaks are upsampled by 3 and use a 10 ms frame shift.
        const float scale = 10.0f * 6.0f / 3.0f / 1000.0f;
        for (int32_t k = 0; k != dim; ++k) {
          if (peak[k] > 1.0f - 1e-4f) timestamps.push_back(k * scale);
        }
        if (!timestamps.empty()) timestamps.pop_back();
        if (timestamps.size() == results[i].tokens.size()) {
          results[i].timestamps = std::move(timestamps);
        }
      }
    }
  }

  return results;
}

}  // namespace sherpa_onnx
