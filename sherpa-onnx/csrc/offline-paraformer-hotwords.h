// sherpa-onnx/csrc/offline-paraformer-hotwords.h
//
// Copyright (c)  2026  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_PARAFORMER_HOTWORDS_H_
#define SHERPA_ONNX_CSRC_OFFLINE_PARAFORMER_HOTWORDS_H_

#include <istream>
#include <string>
#include <vector>

#include "sherpa-onnx/csrc/symbol-table.h"

namespace sherpa_onnx {

bool EncodeParaformerHotwords(std::istream &is, const std::string &tokens,
                              const SymbolTable &symbol_table,
                              std::vector<std::vector<int32_t>> *hotwords,
                              std::vector<float> *boost_scores);

// This overload is retained for Paraformer implementations that do not have
// access to seg_dict through their resource manager.
bool EncodeParaformerHotwords(std::istream &is,
                              const SymbolTable &symbol_table,
                              std::vector<std::vector<int32_t>> *hotwords,
                              std::vector<float> *boost_scores);

// This overload is used when model files are accessed through a resource
// manager. The caller provides seg_dict through the same manager.
bool EncodeParaformerHotwords(std::istream &is, std::istream &seg_dict,
                              const SymbolTable &symbol_table,
                              std::vector<std::vector<int32_t>> *hotwords,
                              std::vector<float> *boost_scores);

std::string GetParaformerSegDictPath(const std::string &tokens);

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_PARAFORMER_HOTWORDS_H_
