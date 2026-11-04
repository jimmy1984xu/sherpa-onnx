#include "sherpa-onnx/csrc/mali-g720/offline-recognizer.h"
#include "sherpa-onnx/csrc/mali-g720/config.h"
#include "sherpa-onnx/csrc/mali-g720/gpu-abi.h"
#include "sherpa-onnx/csrc/mali-g720/model-manifest.h"
#include "sherpa-onnx/csrc/symbol-table.h"
#include "sherpa-onnx/csrc/mali-g720/hotword-graph.h"
#include "sherpa-onnx/csrc/mali-g720/pcm.h"
#include "sherpa-onnx/csrc/utils.h"
#include <fstream>
#include <sstream>
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
    if (!ValidateMaliG720(c.model_config) ||
        (c.decoding_method != "greedy_search" && c.decoding_method != "modified_beam_search") ||
        !c.lm_config.model.empty() ||
        !c.ctc_fst_decoder_config.graph.empty() || c.feat_config.sampling_rate != 16000)
      throw std::invalid_argument("MaliG720 requires frozen16k TDT greedy/modified_beam_search; no fallback");
    if (!std::isfinite(c.hotwords_score) || c.hotwords_score < 0 ||
        !std::isfinite(c.blank_penalty) || c.blank_penalty < 0 ||
        c.max_active_paths < 1 || c.max_active_paths > 64)
      throw std::invalid_argument("invalid MaliG720 hotword score, blank penalty or beam paths (1..64)");
    if (c.decoding_method == "greedy_search" && (!c.hotwords_file.empty() || c.blank_penalty != 0))
      throw std::invalid_argument("MaliG720 hotwords/blank penalty require modified_beam_search");
    MaliG720InitTimer timer("recognizer");
    int version = MaliG720Version(c.model_config);
    blank_ = version == 2 ? 1024 : 8192;
    aot_file_=transai::aot::File::Open(c.model_config.transducer.encoder_filename);
    ValidateMaliG720Manifest(c.model_config, version);
    timer.Mark("manifest");
    symbols_ = SymbolTable(aot_file_->Text("tokens.txt"),false);
    if (symbols_.NumSymbols() != blank_ + 1 || !symbols_.Contains("<blk>") ||
        symbols_["<blk>"] != blank_) throw std::invalid_argument("MaliG720 vocabulary mismatch");
    if (!c.model_config.bpe_vocab.empty()) {
      std::ifstream vocab(c.model_config.bpe_vocab);
      if (!vocab) throw std::invalid_argument("cannot open MaliG720 bpe_vocab");
      std::string line; int id=0;
      while (std::getline(vocab,line)) {
        auto tab=line.find('\t');
        if (tab==std::string::npos || !symbols_.Contains(line.substr(0,tab)) ||
            symbols_[line.substr(0,tab)]!=id++)
          throw std::invalid_argument("MaliG720 bpe_vocab does not match model tokens");
        std::istringstream score(line.substr(tab+1));float value;
        if (!(score>>value) || !std::isfinite(value))
          throw std::invalid_argument("invalid MaliG720 bpe_vocab score");
        score>>std::ws;
        if (!score.eof()) throw std::invalid_argument("invalid MaliG720 bpe_vocab record");
      }
      if (id!=blank_) throw std::invalid_argument("MaliG720 bpe_vocab size mismatch");
      vocab.clear();vocab.seekg(0);
      bpe_ = std::make_unique<ssentencepiece::Ssentencepiece>(vocab);
    }
    if (!c.hotwords_file.empty()) {
      std::ifstream file(c.hotwords_file);
      if (!file) throw std::invalid_argument("cannot open MaliG720 hotwords_file");
      EncodeWords(file, &hotwords_, &boost_scores_);
      hotwords_graph_ = MakeGraph(hotwords_, boost_scores_);
    }
    timer.Mark("vocabulary");
    const std::string name = "libtransai_parakeet_gpu_v" + std::to_string(version) + ".so";
    library_ = dlopen(name.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!library_) throw std::runtime_error(std::string("MaliG720 dlopen: ") + dlerror());
    timer.Mark("dlopen");
    try {
      auto create = Load<int (*)(const char *, const char *, void **, char *, size_t)>(
          "transai_parakeet_gpu_create");
      recognize_ = Load<Recognize>("transai_parakeet_gpu_recognize_ex");
      destroy_ = Load<void (*)(void *)>("transai_parakeet_gpu_destroy");
      if (c.decoding_method == "modified_beam_search") {
        recognize_beam_ = Load<RecognizeBeam>("transai_parakeet_gpu_recognize_beam");
      }
      char error[2048]{};
      const std::string encoder=c.model_config.transducer.encoder_filename+"/encoder";
      const std::string decoder=c.model_config.transducer.encoder_filename+"/gpu";
      if (create(encoder.c_str(),decoder.c_str(),
                 &handle_, error, sizeof(error))) throw std::runtime_error(error);
      timer.Mark("gpu_create");
    } catch (...) { dlclose(library_); library_ = nullptr; throw; }
  }
  ~MaliG720Recognizer() override {
    if (handle_) destroy_(handle_);
    if (library_) dlclose(library_);
  }
  std::unique_ptr<OfflineStream> CreateStream() const override {
    return NewStream(hotwords_graph_);
  }
  std::unique_ptr<OfflineStream> CreateStream(const std::string &hotwords) const override {
    if (hotwords.empty()) return CreateStream();
    if (config_.decoding_method != "modified_beam_search")
      throw std::invalid_argument("MaliG720 hotwords require modified_beam_search");
    std::lock_guard<std::mutex> lock(hotword_cache_mutex_);
    if(hotwords==cached_hotwords_) return NewStream(cached_hotword_graph_);
    std::string lines = hotwords;
    std::replace(lines.begin(), lines.end(), '/', '\n');
    std::istringstream input(lines);
    auto words = hotwords_;
    auto scores = boost_scores_;
    std::vector<std::vector<int32_t>> current;
    std::vector<float> current_scores;
    EncodeWords(input, &current, &current_scores);
    words.insert(words.end(), current.begin(), current.end());
    scores.insert(scores.end(), current_scores.begin(), current_scores.end());
    auto graph=MakeGraph(words,scores);
    cached_hotwords_=hotwords;cached_hotword_graph_=graph;
    return NewStream(std::move(graph));
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
      MaliG720Pcm16(floats.data(),pcm.data(),floats.size());
      MaliG720GpuResultEx extra{1, sizeof(MaliG720GpuResultEx), {}, nullptr, 0};
      char error[2048]{};
      int status;
      if (config_.decoding_method == "modified_beam_search") {
        const auto context=s->GetContextGraph();
        if(cached_gpu_graph_.empty() || context!=cached_decode_graph_) {
          cached_gpu_graph_=MaliHotwordGraph(context,blank_+1);
          cached_decode_graph_=context;
        }
        const auto &graph=cached_gpu_graph_;
        status = recognize_beam_(handle_, pcm.data(), pcm.size(), config_.max_active_paths,
            config_.blank_penalty, graph.data(), graph.size(), &extra, error, sizeof(error));
      } else {
        status = recognize_(handle_, pcm.data(), pcm.size(), &extra, error, sizeof(error));
      }
      if (status) throw std::runtime_error(error);
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
  std::unique_ptr<OfflineStream> NewStream(ContextGraphPtr graph) const {
    auto stream = std::make_unique<OfflineStream>(MaliG720Tag{}, std::move(graph));
    stream->SetOption("mali_g720.stream", "1");
    return stream;
  }
  void EncodeWords(std::istream &input, std::vector<std::vector<int32_t>> *words,
                   std::vector<float> *scores) const {
    bool ok;
    const auto &unit = config_.model_config.modeling_unit;
    if (unit.empty()) {
      ok = EncodeKeywords(input, symbols_, words, nullptr, scores, nullptr);
    } else {
      if (unit != "cjkchar" && unit != "bpe" && unit != "bbpe" && unit != "cjkchar+bpe")
        throw std::invalid_argument("unsupported MaliG720 modeling_unit");
      if (unit != "cjkchar" && !bpe_) throw std::invalid_argument("MaliG720 text hotwords require bpe_vocab");
      ok = EncodeHotwords(input, unit, symbols_, bpe_.get(), words, scores);
    }
    if (!ok || words->empty()) throw std::invalid_argument("invalid or out-of-vocabulary MaliG720 hotwords");
    if (scores->empty()) scores->resize(words->size(), config_.hotwords_score);
    for (size_t i = 0; i < words->size(); ++i) {
      if ((*words)[i].empty() || !std::isfinite((*scores)[i]) || (*scores)[i] < 0)
        throw std::invalid_argument("invalid MaliG720 hotword phrase/score");
      for (int id : (*words)[i]) if (id < 0 || id >= blank_ ||
          (symbols_.Contains("<unk>") && id == symbols_["<unk>"]))
        throw std::invalid_argument("blank/control token in MaliG720 hotwords");
    }
  }
  ContextGraphPtr MakeGraph(const std::vector<std::vector<int32_t>> &words,
                            const std::vector<float> &scores) const {
    return words.empty() ? nullptr : std::make_shared<ContextGraph>(words, config_.hotwords_score, scores);
  }
  template <class T> T Load(const char *name) {
    auto p = dlsym(library_, name);
    if (!p) throw std::runtime_error(std::string("MaliG720 missing symbol: ") + name);
    return reinterpret_cast<T>(p);
  }
  using Recognize = int (*)(void *, const int16_t *, size_t, MaliG720GpuResultEx *, char *, size_t);
  using RecognizeBeam = int (*)(void *, const int16_t *, size_t, int32_t, float,
                               const int32_t *, size_t, MaliG720GpuResultEx *, char *, size_t);
  RecognizeBeam recognize_beam_ = nullptr;
  std::unique_ptr<ssentencepiece::Ssentencepiece> bpe_;
  std::vector<std::vector<int32_t>> hotwords_;
  std::vector<float> boost_scores_;
  ContextGraphPtr hotwords_graph_;
  void *library_ = nullptr, *handle_ = nullptr;
  Recognize recognize_ = nullptr;
  void (*destroy_)(void *) = nullptr;
  int blank_ = 0;
  SymbolTable symbols_;
  std::shared_ptr<transai::aot::File> aot_file_;
  mutable std::mutex hotword_cache_mutex_;
  mutable std::string cached_hotwords_;
  mutable ContextGraphPtr cached_hotword_graph_,cached_decode_graph_;
  mutable std::vector<int32_t> cached_gpu_graph_;
  mutable std::mutex mutex_;
};
}
std::unique_ptr<OfflineRecognizerImpl> CreateMaliG720Recognizer(const OfflineRecognizerConfig &c) {
  return std::make_unique<MaliG720Recognizer>(c);
}
}
