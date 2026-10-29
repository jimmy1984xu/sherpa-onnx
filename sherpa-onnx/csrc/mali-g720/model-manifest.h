#ifndef SHERPA_ONNX_CSRC_MALI_G720_MODEL_MANIFEST_H_
#define SHERPA_ONNX_CSRC_MALI_G720_MODEL_MANIFEST_H_
#include <map>
#include <set>
#include "nlohmann/json.hpp"
#include "sherpa-onnx/csrc/mali-g720/config.h"
#include "sherpa-onnx/csrc/mali-g720/sha256.h"

namespace sherpa_onnx {
// The runtime binds to an architecture; each model carries its own asset identities.
inline void ValidateMaliG720Manifest(const OfflineModelConfig &c, int version) {
  const std::string root = MaliG720Parent(MaliG720Parent(c.transducer.encoder_filename));
  std::ifstream stream(root + "/model.json");
  if (!stream) throw std::invalid_argument("MaliG720 missing model.json: " + root);
  const auto manifest = nlohmann::json::parse(stream);
  const auto &a = manifest.at("architecture");
  const int vocab = version == 2 ? 1024 : 8192;
  if (manifest.at("format") != 1 || a.at("family") != "parakeet-tdt" ||
      a.at("version") != version || a.at("encoder_layers") != 24 ||
      a.at("encoder_dim") != 1024 || a.at("attention_heads") != 8 ||
      a.at("mel_bins") != 128 || a.at("predictor_hidden") != 640 ||
      a.at("predictor_layers") != 2 || a.at("vocab_size") != vocab ||
      a.at("blank") != vocab || a.at("sample_rate") != 16000 ||
      a.at("max_samples") != 480000 || a.at("durations") != nlohmann::json({0,1,2,3,4}))
    throw std::invalid_argument("MaliG720 incompatible model architecture: " + root);
  const std::set<std::string> required = {
    "encoder/encoder.full.nemo.plan.taig720", "encoder/encoder.weights.bin",
    "encoder/teacher_kernels.cl", "gpu/frontend.bin", "gpu/frontend.cl",
    "gpu/frontend_layout.clh", "gpu/tdt_weights.bin", "gpu/tdt_layout.clh",
    "gpu/tdt.cl", "gpu/multi.cl", "tokens/tokens.txt"};
  std::set<std::string> observed;
  for (const auto &asset : manifest.at("assets")) {
    const std::string role = asset.at("role"), name = asset.at("name");
    const std::string key = role + "/" + name;
    const std::string digest = asset.at("sha256");
    if (!required.count(key) || !observed.insert(key).second || digest.size() != 64 ||
        digest.find_first_not_of("0123456789abcdef") != std::string::npos)
      throw std::invalid_argument("MaliG720 invalid/duplicate model asset: " + key);
    const std::string path = role == "tokens" ? c.tokens :
        MaliG720Parent(role == "encoder" ? c.transducer.encoder_filename :
                      c.transducer.decoder_filename) + "/" + name;
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file || file.tellg() < 0 || static_cast<uint64_t>(file.tellg()) != asset.at("bytes").get<uint64_t>() ||
        MaliG720Sha256(path) != digest)
      throw std::invalid_argument("MaliG720 model asset size/SHA256 mismatch: " + path);
  }
  if (observed != required) throw std::invalid_argument("MaliG720 incomplete model manifest");
}
}
#endif
