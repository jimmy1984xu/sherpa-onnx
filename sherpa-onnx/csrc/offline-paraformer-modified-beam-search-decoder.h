// sherpa-onnx/csrc/offline-paraformer-modified-beam-search-decoder.h
//
// Copyright (c) 2026

#ifndef SHERPA_ONNX_CSRC_OFFLINE_PARAFORMER_MODIFIED_BEAM_SEARCH_DECODER_H_
#define SHERPA_ONNX_CSRC_OFFLINE_PARAFORMER_MODIFIED_BEAM_SEARCH_DECODER_H_

#include <cstdint>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/context-graph.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/offline-paraformer-decoder.h"

namespace sherpa_onnx {

class OfflineParaformerModifiedBeamSearchDecoder
    : public OfflineParaformerDecoder {
 public:
  OfflineParaformerModifiedBeamSearchDecoder(int32_t eos_id,
                                             int32_t max_active_paths,
                                             ContextGraphPtr context_graph)
      : eos_id_(eos_id),
        max_active_paths_(max_active_paths),
        context_graph_(std::move(context_graph)) {
    SHERPA_ONNX_CHECK_GT(max_active_paths_, 0);
  }

  std::vector<OfflineParaformerDecoderResult> Decode(
      Ort::Value log_probs, Ort::Value token_num,
      Ort::Value us_cif_peak = Ort::Value(nullptr)) override;

  OfflineParaformerDecoderResult DecodeOne(const float *logits,
                                           int32_t num_tokens,
                                           int32_t vocab_size) const;

 private:
  int32_t eos_id_;
  int32_t max_active_paths_;
  ContextGraphPtr context_graph_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_PARAFORMER_MODIFIED_BEAM_SEARCH_DECODER_H_
