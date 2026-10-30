#ifndef SHERPA_ONNX_CSRC_MALI_G720_MODEL_MANIFEST_H_
#define SHERPA_ONNX_CSRC_MALI_G720_MODEL_MANIFEST_H_
#include <map>
#include "sherpa-onnx/csrc/mali-g720/init-timer.h"
#include <set>
#include "nlohmann/json.hpp"
#include "sherpa-onnx/csrc/mali-g720/config.h"

namespace sherpa_onnx {
// Bundled assets are hashed by the offline delivery verifier, not at recognizer initialization.
inline void ValidateMaliG720Manifest(const OfflineModelConfig &c, int version) {
  MaliG720InitTimer timer("manifest");
  const std::string root = MaliG720Parent(MaliG720Parent(c.transducer.encoder_filename));
  std::ifstream stream(root + "/model.json");
  if (!stream) throw std::invalid_argument("MaliG720 missing model.json: " + root);
  const auto manifest = nlohmann::json::parse(stream);
  const auto &a = manifest.at("architecture");
  const int vocab = version == 2 ? 1024 : 8192;
  if (manifest.at("format") != 2 || a.at("family") != "parakeet-tdt" ||
      a.at("version") != version || a.at("encoder_layers") != 24 ||
      a.at("encoder_dim") != 1024 || a.at("attention_heads") != 8 ||
      a.at("mel_bins") != 128 || a.at("predictor_hidden") != 640 ||
      a.at("predictor_layers") != 2 || a.at("vocab_size") != vocab ||
      a.at("blank") != vocab || a.at("sample_rate") != 16000 ||
      a.at("max_samples") != 480000 || a.at("durations") != nlohmann::json({0,1,2,3,4}))
    throw std::invalid_argument("MaliG720 incompatible model architecture: " + root);
  if (manifest.at("aot").at("format") != 1)
    throw std::invalid_argument("MaliG720 compiled executable manifest required");
  const std::set<std::string> required = {
    "encoder/encoder.full.nemo.plan.taig720", "encoder/encoder.weights.bin",
    "encoder/encoder_primary.clbin", "encoder/encoder_alternate.clbin", "encoder/encoder_tiny.clbin",
    "gpu/frontend.bin", "gpu/tdt_weights.bin", "gpu/frontend_tdt.clbin",
    "gpu/frontend_tdt_trace.clbin", "tokens/tokens.txt"};
  const std::set<std::string> programs = {"encoder/encoder_primary.clbin", "encoder/encoder_alternate.clbin",
    "encoder/encoder_tiny.clbin", "gpu/frontend_tdt.clbin", "gpu/frontend_tdt_trace.clbin"};
  std::set<std::string> compiled;
  std::set<std::pair<std::string,std::string>> options;
  for (const auto& program : manifest.at("aot").at("programs")) {
    const std::string role=program.at("role"),name=program.at("name"),build_options=program.at("options");
    const std::string key=role+"/"+name,source=program.at("source_sha256");
    if (!programs.count(key) || !compiled.insert(key).second || !options.emplace(role,build_options).second ||
        build_options.empty() || source.size()!=64 || source.find_first_not_of("0123456789abcdef")!=std::string::npos)
      throw std::invalid_argument("MaliG720 invalid AOT executable description: "+key);
  }
  if (compiled!=programs) throw std::invalid_argument("MaliG720 incomplete AOT executable set");
  timer.Mark("parse_contract");
  std::set<std::string> observed;
  for (const auto &asset : manifest.at("assets")) {
    const std::string role = asset.at("role"), name = asset.at("name");
    const std::string key = role + "/" + name;
    if (!required.count(key) || !observed.insert(key).second)
      throw std::invalid_argument("MaliG720 invalid/duplicate model asset: " + key);
    const std::string path = role == "tokens" ? c.tokens :
        MaliG720Parent(role == "encoder" ? c.transducer.encoder_filename :
                      c.transducer.decoder_filename) + "/" + name;
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file || file.tellg() < 0 || static_cast<uint64_t>(file.tellg()) != asset.at("bytes").get<uint64_t>())
      throw std::invalid_argument("MaliG720 model asset size mismatch: " + path);
    timer.Mark("asset/"+key);
  }
  if (observed != required) throw std::invalid_argument("MaliG720 incomplete model manifest");
}
}
#endif
