#include "sherpa-onnx/csrc/mali-g720/offline-recognizer.h"
#include "sherpa-onnx/csrc/mali-g720/config.h"
#include "sherpa-onnx/csrc/mali-g720/gpu-abi.h"
#include "sherpa-onnx/csrc/mali-g720/model-manifest.h"
#include "sherpa-onnx/csrc/mali-g720/sha256.h"
#include "sherpa-onnx/csrc/symbol-table.h"
#include <dlfcn.h>
#include <cmath>
#include <mutex>
#include <stdexcept>

namespace sherpa_onnx {
namespace {
class MaliG720Recognizer final : public OfflineRecognizerImpl {
 public:
  explicit MaliG720Recognizer(const OfflineRecognizerConfig &c)
      : OfflineRecognizerImpl(c) {
    if (!ValidateMaliG720(c.model_config) || c.decoding_method != "greedy_search" ||
        c.blank_penalty != 0 || !c.hotwords_file.empty() || !c.lm_config.model.empty() ||
        !c.ctc_fst_decoder_config.graph.empty() || c.feat_config.sampling_rate != 16000)
      throw std::invalid_argument("MaliG720 supports frozen16k greedy TDT only; no fallback");
    int version = MaliG720Version(c.model_config);
    blank_ = version == 2 ? 1024 : 8192;
    ValidateMaliG720Manifest(c.model_config, version);
    symbols_ = SymbolTable(c.model_config.tokens);
    if (symbols_.NumSymbols() != blank_ + 1 || !symbols_.Contains("<blk>") ||
        symbols_["<blk>"] != blank_) throw std::invalid_argument("MaliG720 vocabulary mismatch");
    const std::string name = "libtransai_parakeet_gpu_v" + std::to_string(version) + ".so";
    library_ = dlopen(name.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!library_) throw std::runtime_error(std::string("MaliG720 dlopen: ") + dlerror());
    try {
      auto create = Load<int (*)(const char *, const char *, void **, char *, size_t)>(
          "transai_parakeet_gpu_create");
      recognize_ = Load<Recognize>("transai_parakeet_gpu_recognize_ex");
      destroy_ = Load<void (*)(void *)>("transai_parakeet_gpu_destroy");
      char error[2048]{};
      if (create(MaliG720Parent(c.model_config.transducer.encoder_filename).c_str(),
                 MaliG720Parent(c.model_config.transducer.decoder_filename).c_str(),
                 &handle_, error, sizeof(error))) throw std::runtime_error(error);
    } catch (...) { dlclose(library_); library_ = nullptr; throw; }
  }
  ~MaliG720Recognizer() override {
    if (handle_) destroy_(handle_);
    if (library_) dlclose(library_);
  }
  std::unique_ptr<OfflineStream> CreateStream() const override {
    auto s = std::make_unique<OfflineStream>(MaliG720Tag{});
    s->SetOption("mali_g720.stream", "1");
    return s;
  }
  std::unique_ptr<OfflineStream> CreateStream(const std::string &hotwords) const override {
    if (!hotwords.empty()) throw std::invalid_argument("MaliG720 hotwords unsupported");
    return CreateStream();
  }
  void DecodeStreams(OfflineStream **ss, int32_t n) const override {
    if (n < 0 || (n && !ss)) throw std::invalid_argument("invalid stream array");
    std::lock_guard<std::mutex> lock(mutex_);
    for (int32_t i = 0; i < n; ++i) {
      if (!ss[i] || ss[i]->GetOption("mali_g720.stream") != "1")
        throw std::invalid_argument("MaliG720 requires its own raw stream");
      auto s = ss[i];
      s->SetResult({});
      if (!s->GetOption("mali_g720.error").empty())
        throw std::runtime_error(s->GetOption("mali_g720.error"));
      auto floats = s->GetFrames();
      if (floats.empty() || floats.size() > 480000) throw std::invalid_argument("invalid PCM length");
      std::vector<int16_t> pcm(floats.size());
      for (size_t j = 0; j < floats.size(); ++j) {
        float x = floats[j];
        if (!std::isfinite(x) || x < -1 || x > 1) throw std::invalid_argument("invalid PCM float");
        pcm[j] = static_cast<int16_t>(std::min(32767L, std::lround(x * 32768.0f)));
      }
      MaliG720GpuResultEx extra{1, sizeof(MaliG720GpuResultEx), {}, nullptr, 0};
      char error[2048]{};
      if (recognize_(handle_, pcm.data(), pcm.size(), &extra, error, sizeof(error)))
        throw std::runtime_error(error);
      const auto &gpu = extra.base;
      if (extra.abi_version != 1 || extra.struct_bytes != sizeof(extra) ||
          extra.score_count != gpu.token_count || (extra.score_count && !extra.ys_log_probs))
        throw std::runtime_error("MaliG720 invalid score ABI/count");
      if (gpu.token_count < 0 || gpu.token_count > 3750 || gpu.utf8_bytes > 131072 ||
          !gpu.utf8 || (gpu.token_count && !gpu.triples)) throw std::runtime_error("invalid GPU result");
      OfflineRecognitionResult r;
      r.text.assign(gpu.utf8, gpu.utf8_bytes);
      for (int j = 0; j < gpu.token_count; ++j) {
        int y = gpu.triples[3*j], t = gpu.triples[3*j+1], d = gpu.triples[3*j+2];
        if (y < 0 || y >= blank_ || t < 0 || t >= 375 || d < 0 || d > 4)
          throw std::runtime_error("invalid GPU token tuple");
        r.tokens.push_back(symbols_[y]);
        r.timestamps.push_back(t * 0.08f);
        r.durations.push_back(d * 0.08f);
        float score = extra.ys_log_probs[j];
        if (!std::isfinite(score) || score > 2e-5f) throw std::runtime_error("invalid GPU score");
        r.ys_log_probs.push_back(score);
      }
      r.text = ApplyInverseTextNormalization(std::move(r.text));
      r.text = ApplyHomophoneReplacer(std::move(r.text));
      s->SetResult(r);  // Copy while plugin pointers are protected by mutex_.
    }
  }
  OfflineRecognizerConfig GetConfig() const override { return config_; }
  void SetConfig(const OfflineRecognizerConfig &) override {
    throw std::invalid_argument("MaliG720 model/config is immutable; create a new recognizer");
  }
 private:
  template <class T> T Load(const char *name) {
    auto p = dlsym(library_, name);
    if (!p) throw std::runtime_error(std::string("MaliG720 missing symbol: ") + name);
    return reinterpret_cast<T>(p);
  }
  using Recognize = int (*)(void *, const int16_t *, size_t, MaliG720GpuResultEx *, char *, size_t);
  void *library_ = nullptr, *handle_ = nullptr;
  Recognize recognize_ = nullptr;
  void (*destroy_)(void *) = nullptr;
  int blank_ = 0;
  SymbolTable symbols_;
  mutable std::mutex mutex_;
};
}
std::unique_ptr<OfflineRecognizerImpl> CreateMaliG720Recognizer(const OfflineRecognizerConfig &c) {
  return std::make_unique<MaliG720Recognizer>(c);
}
}
