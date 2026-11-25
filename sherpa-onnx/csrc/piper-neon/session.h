#pragma once
#ifdef SHERPA_ONNX_ENABLE_MATCHA_NEON
#include "sherpa-onnx/csrc/neon-tts-frontend.h"
#include "sherpa-onnx/csrc/neon-ort-session.h"
#include "sherpa-onnx/csrc/neon-threads.h"
#include <sys/auxv.h>
#include <asm/hwcap.h>
namespace sherpa_onnx {void RegisterPiperNeonOps(Ort::SessionOptions&);}
namespace transai::piper_neon {
inline void Validate(const std::string&path){
 auto f=aot::File::Open(path);const auto&m=f->manifest();
 const auto family=m.at("architecture").at("family").get<std::string>();
 const bool amy=family=="piper-en-amy-tts",bryce=family=="piper-en-bryce-tts";
 if((family!="piper-es-carlfm-tts"&&!amy&&!bryce)||m.at("version")!="1.0"||m.at("aot").at("kernel_abi")!=(bryce?"piper-bryce-u8-bf16-v1":amy?"piper-amy-u8-bf16-v1":"piper-u8-bf16-v1"))throw std::runtime_error("Unsupported Piper NEON v1.0 contract");
 if (family == "piper-es-carlfm-tts" &&
     m.at("aot").at("load_optimization") != "duration-acoustic-attention-loop-packed-u8-v1")
   throw std::invalid_argument("Carlfm requires the released prepacked RHS graph.");
 aot::RequireMappedOrtGraph(*f);
 if(!(getauxval(AT_HWCAP)&HWCAP_ASIMDHP)||!(getauxval(AT_HWCAP2)&HWCAP2_I8MM)||!(getauxval(AT_HWCAP2)&HWCAP2_BF16))throw std::runtime_error("Piper requires FP16/I8MM/BF16");
}
inline Ort::SessionOptions Options(int num_threads) {
  Ort::SessionOptions options;
  options.SetIntraOpNumThreads(sherpa_onnx::neon::ResolveNumThreads(num_threads))
      .SetInterOpNumThreads(1)
      .AddConfigEntry("session.force_spinning_stop", "1");
  sherpa_onnx::RegisterPiperNeonOps(options);
  aot::SetMappedOrtGraphOptions(options);
  return options;
}
}
#endif
