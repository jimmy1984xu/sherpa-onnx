#ifndef SHERPA_ONNX_CSRC_NEON_ORT_SESSION_H_
#define SHERPA_ONNX_CSRC_NEON_ORT_SESSION_H_

#include <stdexcept>
#include <string>

#include "onnxruntime_cxx_api.h"
#include "sherpa-onnx/csrc/mali-g720/aot-file.h"

namespace transai::aot {

inline void RequireMappedOrtGraph(const File& file) {
  const auto& aot = file.manifest().at("aot");
  const auto format = aot.at("graph_format").get<std::string>();
  if (format != "ort" || aot.at("ort_runtime") != OrtGetApiBase()->GetVersionString())
    throw std::runtime_error("Unsupported NEON AOT graph format or ORT runtime");
}

// The mapping must outlive every session using these read-only bytes.
inline void SetMappedOrtGraphOptions(Ort::SessionOptions& options) {
  options.SetGraphOptimizationLevel(ORT_DISABLE_ALL)
      .AddConfigEntry("session.load_model_format", "ORT")
      .AddConfigEntry("session.use_ort_model_bytes_directly", "1")
      .AddConfigEntry("session.use_ort_model_bytes_for_initializers", "1");
}

}  // namespace transai::aot
#endif
