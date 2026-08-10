// sherpa-onnx/csrc/offline-recognizer-paraformer-tpl-impl.h
//
// Copyright (c)  2025  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_RECOGNIZER_PARAFORMER_TPL_IMPL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_RECOGNIZER_PARAFORMER_TPL_IMPL_H_

#include <fstream>
#include <memory>
#include <sstream>
#include <utility>
#include <vector>

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/math.h"
#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/offline-model-config.h"
#include "sherpa-onnx/csrc/offline-paraformer-hotwords.h"
#include "sherpa-onnx/csrc/offline-paraformer-modified-beam-search-decoder.h"
#include "sherpa-onnx/csrc/offline-recognizer-impl.h"
#include "sherpa-onnx/csrc/offline-recognizer.h"
#include "sherpa-onnx/csrc/symbol-table.h"
#include "sherpa-onnx/csrc/utils.h"
#include "ssentencepiece/csrc/ssentencepiece.h"

namespace sherpa_onnx {

// defined in ../offline-recognizer-paraformer-impl.h
OfflineRecognitionResult Convert(const OfflineParaformerDecoderResult &src,
                                 const SymbolTable &sym_table);

template <typename ParaformerModel>
class OfflineRecognizerParaformerTplImpl : public OfflineRecognizerImpl {
 public:
  explicit OfflineRecognizerParaformerTplImpl(
      const OfflineRecognizerConfig &config)
      : OfflineRecognizerImpl(config),
        config_(config),
        symbol_table_(config_.model_config.tokens),
        model_(std::make_unique<ParaformerModel>(config.model_config)) {
    if (config_.decoding_method == "modified_beam_search") {
      InitBpeEncoder();
      if (!config_.hotwords_file.empty()) {
        InitHotwords();
      }
      decoder_ = std::make_unique<OfflineParaformerModifiedBeamSearchDecoder>(
          symbol_table_["</s>"], config_.max_active_paths, hotwords_graph_);
    } else if (config_.decoding_method != "greedy_search") {
      SHERPA_ONNX_LOGE(
          "Only greedy_search and modified_beam_search are supported. Given %s",
          config_.decoding_method.c_str());
      SHERPA_ONNX_EXIT(-1);
    }

    InitFeatConfig();
  }

  template <typename Manager>
  OfflineRecognizerParaformerTplImpl(Manager *mgr,
                                     const OfflineRecognizerConfig &config)
      : OfflineRecognizerImpl(mgr, config),
        config_(config),
        symbol_table_(mgr, config_.model_config.tokens),
        model_(std::make_unique<ParaformerModel>(mgr, config.model_config)) {
    if (config_.decoding_method == "modified_beam_search") {
      InitBpeEncoder(mgr);
      if (!config_.hotwords_file.empty()) {
        InitHotwords(mgr);
      }
      decoder_ = std::make_unique<OfflineParaformerModifiedBeamSearchDecoder>(
          symbol_table_["</s>"], config_.max_active_paths, hotwords_graph_);
    } else if (config_.decoding_method != "greedy_search") {
      SHERPA_ONNX_LOGE(
          "Only greedy_search and modified_beam_search are supported. Given %s",
          config_.decoding_method.c_str());
      SHERPA_ONNX_EXIT(-1);
    }

    InitFeatConfig();
  }

  std::unique_ptr<OfflineStream> CreateStream() const override {
    return std::make_unique<OfflineStream>(config_.feat_config);
  }

  void DecodeStreams(OfflineStream **ss, int32_t n) const override {
    for (int32_t i = 0; i < n; ++i) {
      DecodeOneStream(ss[i]);
    }
  }

  OfflineRecognizerConfig GetConfig() const override { return config_; }

 private:
  void InitBpeEncoder() {
    if (!config_.model_config.bpe_vocab.empty()) {
      bpe_encoder_ = std::make_unique<ssentencepiece::Ssentencepiece>(
          config_.model_config.bpe_vocab);
    }
  }

  template <typename Manager>
  void InitBpeEncoder(Manager *mgr) {
    if (!config_.model_config.bpe_vocab.empty()) {
      auto buf = ReadFile(mgr, config_.model_config.bpe_vocab);
      std::istringstream is(std::string(buf.begin(), buf.end()));
      bpe_encoder_ = std::make_unique<ssentencepiece::Ssentencepiece>(is);
    }
  }

  void InitHotwords() {
    std::ifstream is(config_.hotwords_file);
    if (!is) {
      SHERPA_ONNX_LOGE("Open hotwords file failed: '%s'",
                       config_.hotwords_file.c_str());
      SHERPA_ONNX_EXIT(-1);
    }

    if (!EncodeParaformerHotwords(is, config_.model_config.tokens,
                                  symbol_table_, &hotwords_,
                                  &boost_scores_)) {
      SHERPA_ONNX_LOGE(
          "Some hotwords failed to encode and were skipped. See above for "
          "details.");
    }
    hotwords_graph_ = std::make_shared<ContextGraph>(
        hotwords_, config_.hotwords_score, boost_scores_);
  }

  template <typename Manager>
  void InitHotwords(Manager *mgr) {
    auto buf = ReadFile(mgr, config_.hotwords_file);
    std::istringstream is(std::string(buf.begin(), buf.end()));

    if (!EncodeParaformerHotwords(is, symbol_table_, &hotwords_,
                                  &boost_scores_)) {
      SHERPA_ONNX_LOGE(
          "Some hotwords failed to encode and were skipped. See above for "
          "details.");
    }
    hotwords_graph_ = std::make_shared<ContextGraph>(
        hotwords_, config_.hotwords_score, boost_scores_);
  }

  void InitFeatConfig() {
    config_.feat_config.normalize_samples = false;
    config_.feat_config.window_type = "hamming";
    config_.feat_config.high_freq = 0;
    config_.feat_config.snip_edges = true;
  }

  void DecodeOneStream(OfflineStream *s) const {
    std::vector<float> f = s->GetFrames();

    std::vector<float> logits = model_->Run(std::move(f));
    if (logits.empty()) {
      SHERPA_ONNX_LOGE("No speech detected");
      return;
    }

    int32_t vocab_size = model_->VocabSize();
    int32_t num_tokens = logits.size() / vocab_size;

    OfflineParaformerDecoderResult r;
    if (decoder_ != nullptr) {
      r = decoder_->DecodeOne(logits.data(), num_tokens, vocab_size);
    } else {
      const int32_t eos_id = symbol_table_["</s>"];
      const float *p = logits.data();
      for (int32_t i = 0; i < num_tokens; ++i) {
        auto max_idx = static_cast<int64_t>(
            std::distance(p, std::max_element(p, p + vocab_size)));

        if (max_idx == eos_id) {
          break;
        }
        r.tokens.push_back(max_idx);
        r.ys_log_probs.push_back(ComputeLogSoftmaxScore(
            p, vocab_size, static_cast<int32_t>(max_idx)));
        p += vocab_size;
      }
    }

    auto result = Convert(r, symbol_table_);
    result.text = ApplyInverseTextNormalization(std::move(result.text));
    result.text = ApplyHomophoneReplacer(std::move(result.text));
    s->SetResult(result);
  }

 private:
  OfflineRecognizerConfig config_;
  SymbolTable symbol_table_;
  std::vector<std::vector<int32_t>> hotwords_;
  std::vector<float> boost_scores_;
  ContextGraphPtr hotwords_graph_;
  std::unique_ptr<ssentencepiece::Ssentencepiece> bpe_encoder_;
  std::unique_ptr<ParaformerModel> model_;
  std::unique_ptr<OfflineParaformerModifiedBeamSearchDecoder> decoder_;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_RECOGNIZER_PARAFORMER_TPL_IMPL_H_
