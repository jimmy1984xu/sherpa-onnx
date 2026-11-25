#ifndef SHERPA_ONNX_MATCHA_NEON_SESSION_H_
#define SHERPA_ONNX_MATCHA_NEON_SESSION_H_
#include "sherpa-onnx/csrc/session.h"
#include "sherpa-onnx/csrc/neon-tts-load-profile.h"
#include "sherpa-onnx/csrc/text-utils.h"
#include "sherpa-onnx/csrc/neon-tts-frontend.h"
#include "sherpa-onnx/csrc/offline-tts-model-config.h"
#ifdef SHERPA_ONNX_ENABLE_MATCHA_NEON
#include <sys/auxv.h>
#include <asm/hwcap.h>
#include "sherpa-onnx/csrc/neon-ort-session.h"
#include "sherpa-onnx/csrc/neon-threads.h"
namespace sherpa_onnx { void RegisterMatchaNeonOps(Ort::SessionOptions&); }
namespace transai::matcha_neon {
inline void Validate(const std::string& path) {
  auto file = aot::File::Open(path);
  const auto& m=file->manifest();const auto& a=m.at("aot");
  if(m.at("architecture").at("family")!="matcha-tts" || m.at("architecture").at("version")!=1 ||
     m.at("architecture").at("precision")!="qkv-last2 W8A8/BF16/FP32" ||
     a.at("packed_linear_abi")!="matcha-kai-v1" || a.at("alignment")!=65536 || a.at("packed_rhs_alignment")!=64 ||
     m.at("generation")!=nlohmann::json{{"sample_rate",16000},{"noise_scale",.667},{"length_scale",1.},{"ode_steps",3},{"noise_seed",42}})
    throw std::runtime_error("Unsupported Matcha NEON v1.0 contract");
  aot::RequireMappedOrtGraph(*file);
  if(!(getauxval(AT_HWCAP)&HWCAP_ASIMDHP) || !(getauxval(AT_HWCAP2)&HWCAP2_I8MM) || !(getauxval(AT_HWCAP2)&HWCAP2_BF16))
    throw std::runtime_error("Matcha NEON requires ASIMDHP, I8MM and BF16");
}
inline Ort::SessionOptions Options(const sherpa_onnx::OfflineTtsModelConfig& config) {
  if (config.provider != "neon") throw std::invalid_argument("Matcha AOT requires neon");
  Ort::SessionOptions options;
  options.SetIntraOpNumThreads(sherpa_onnx::neon::ResolveNumThreads(config.num_threads))
      .SetInterOpNumThreads(1);
  sherpa_onnx::RegisterMatchaNeonOps(options);
  aot::SetMappedOrtGraphOptions(options);
  options.AddConfigEntry("session.intra_op.allow_spinning","1");
  options.AddConfigEntry("session.force_spinning_stop","1");
  return options;
}
inline std::unique_ptr<Ort::Session> Session(Ort::Env& env,const std::string& path,
    Ort::SessionOptions& options, std::unique_ptr<aot::Mapping>& mapped) {
  LoadProfile profile(path.find("vocoder") != std::string::npos ? "vocoder-session" : "acoustic-session");
  mapped = std::make_unique<aot::Mapping>(path);
  return std::make_unique<Ort::Session>(env,mapped->data(),mapped->size(),options);
}
}
#endif
#endif
