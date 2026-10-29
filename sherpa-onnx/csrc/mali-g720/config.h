#ifndef SHERPA_ONNX_CSRC_MALI_G720_CONFIG_H_
#define SHERPA_ONNX_CSRC_MALI_G720_CONFIG_H_
#include <algorithm>
#include <cctype>
#include <fstream>
#include <string>
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/offline-model-config.h"

namespace sherpa_onnx {
inline bool IsMaliG720(std::string provider) {
  std::transform(provider.begin(), provider.end(), provider.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return provider == "malig720";
}
inline int MaliG720Version(const OfflineModelConfig &c) {
  if (c.model_type == "parakeet_v2_mali_g720") return 2;
  if (c.model_type == "parakeet_v3_mali_g720") return 3;
  return 0;
}
inline std::string MaliG720Parent(const std::string &p) {
  auto pos = p.find_last_of("/\\");
  return pos == std::string::npos ? "." : p.substr(0, pos);
}
inline bool ValidateMaliG720(const OfflineModelConfig &c) {
#if !defined(SHERPA_ONNX_ENABLE_MALI_G720)
  SHERPA_ONNX_LOGE("MaliG720 backend was not enabled in this build; no fallback");
  return false;
#else
  if (!MaliG720Version(c) || c.num_threads < 1 ||
      !c.transducer.joiner_filename.empty()) return false;
  for (const auto &p : {c.transducer.encoder_filename,
                         c.transducer.decoder_filename, c.tokens}) {
    std::ifstream f(p, std::ios::binary);
    if (!f) { SHERPA_ONNX_LOGE("MaliG720 missing file: %s", p.c_str()); return false; }
  }
  // Other model fields must not silently select an unrelated backend.
  if (!c.paraformer.model.empty() || !c.nemo_ctc.model.empty() ||
      !c.whisper.encoder.empty()) return false;
  return true;
#endif
}
}  // namespace sherpa_onnx
#endif
