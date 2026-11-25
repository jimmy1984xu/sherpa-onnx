// sherpa-onnx/csrc/offline-paraformer-beam-search-decoder.h
#ifndef SHERPA_ONNX_CSRC_OFFLINE_PARAFORMER_BEAM_SEARCH_DECODER_H_
#define SHERPA_ONNX_CSRC_OFFLINE_PARAFORMER_BEAM_SEARCH_DECODER_H_

#include <memory>
#include <vector>

#include "sherpa-onnx/csrc/offline-paraformer-decoder.h"
#include "sherpa-onnx/csrc/offline-paraformer-model.h"
#include "sherpa-onnx/csrc/offline-stream.h"
#include "sherpa-onnx/csrc/context-graph.h"

namespace sherpa_onnx {

class OfflineParaformerBeamSearchDecoder : public OfflineParaformerDecoder {
 public:
  OfflineParaformerBeamSearchDecoder(int32_t eos_id,
                                     int32_t max_active_paths,
                                     float lm_scale,
                                     int32_t unk_id,
                                     float hotwords_score,
                                     float blank_penalty);

  std::vector<OfflineParaformerDecoderResult> Decode(
      Ort::Value logits,
      Ort::Value logits_length,
      Ort::Value us_cif_peak = Ort::Value(nullptr),
      OfflineStream **ss = nullptr,
      int32_t n = 0) override;

 private:
  int32_t eos_id_;
  int32_t max_active_paths_;
  float lm_scale_;
  int32_t unk_id_;
  float hotwords_score_;
  float blank_penalty_;

  // 辅助函数：从 OfflineStream 中获取 ContextGraph（若存在）
  ContextGraphPtr GetContextGraph(OfflineStream* s) const;
};

}  // namespace sherpa_onnx

#endif
