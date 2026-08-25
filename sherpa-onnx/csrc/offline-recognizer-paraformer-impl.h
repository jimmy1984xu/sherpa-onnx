// sherpa-onnx/csrc/offline-recognizer-paraformer-impl.h
//
// Copyright (c)  2022-2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_OFFLINE_RECOGNIZER_PARAFORMER_IMPL_H_
#define SHERPA_ONNX_CSRC_OFFLINE_RECOGNIZER_PARAFORMER_IMPL_H_

#include <algorithm>
#include <cinttypes>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <regex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <cstring>

#include "Eigen/Dense"
#include "sherpa-onnx/csrc/offline-model-config.h"
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/offline-paraformer-decoder.h"
#include "sherpa-onnx/csrc/offline-paraformer-greedy-search-decoder.h"
#include "sherpa-onnx/csrc/offline-paraformer-beam-search-decoder.h"
#include "sherpa-onnx/csrc/offline-paraformer-hotword-embedding.h"
#include "sherpa-onnx/csrc/offline-paraformer-model.h"
#include "sherpa-onnx/csrc/offline-recognizer-impl.h"
#include "sherpa-onnx/csrc/offline-recognizer.h"
#include "sherpa-onnx/csrc/pad-sequence.h"
#include "sherpa-onnx/csrc/symbol-table.h"
#include "sherpa-onnx/csrc/context-graph.h"
#include "sherpa-onnx/csrc/file-utils.h"
namespace sherpa_onnx {

// ========== 热词编译器（seaco-paraformer），复用主类的 seg_map ==========
class OfflineParaformerHotwordCompiler {
 public:
  OfflineParaformerHotwordCompiler(const std::string& hw_model_path, int thread_num = 1)
      : env_(ORT_LOGGING_LEVEL_ERROR, "ParaformerHotword") {
    session_options_.SetIntraOpNumThreads(thread_num);
    session_options_.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
    session_options_.DisableCpuMemArena();

    try {
       session_ = std::make_unique<Ort::Session>(
           env_, hw_model_path.c_str(), session_options_);
      SHERPA_ONNX_LOGE("Loaded hotword compiler from %s", hw_model_path.c_str());
    } catch (std::exception const &e) {
      SHERPA_ONNX_LOGE("Failed to load hotword compiler: %s", e.what());
      exit(-1);
    }

    Ort::AllocatorWithDefaultOptions allocator;
    size_t num_inputs = session_->GetInputCount();
    for (size_t i = 0; i < num_inputs; ++i) {
      auto name = session_->GetInputNameAllocated(i, allocator);
      input_names_.push_back(name.get());
    }
    size_t num_outputs = session_->GetOutputCount();
    for (size_t i = 0; i < num_outputs; ++i) {
      auto name = session_->GetOutputNameAllocated(i, allocator);
      output_names_.push_back(name.get());
    }
  }

  // 输入：已编码的 token ID 序列列表（每个序列长度可变，有效 token）
  // 输出：每个输入序列对应的 embedding 向量，最后一个始终是 dummy/default
  // embedding。seaco-paraformer 的导出模型依赖这个额外的 embedding，即使
  // 当前没有任何热词。
  OfflineParaformerHotwordEmbedding Compile(
      const std::vector<std::vector<int32_t>>& token_ids_list) {
    const int max_len = 10;
    // FunASR 的参考实现总是追加一个 dummy 序列：首 token 为 1，长度为 1，
    // 其余位置为 padding。不要使用全零长度 10 的序列，它会从错误的时间
    // 索引提取 embedding。
    const int num_hotwords = static_cast<int>(token_ids_list.size()) + 1;
    std::vector<int> lengths;
    std::vector<int32_t> matrix;
    matrix.reserve(num_hotwords * max_len);

    for (const auto& ids : token_ids_list) {
      int len = static_cast<int>(ids.size());
      if (len > max_len) len = max_len;
      if (len <= 0) {
        // Empty entries are not valid hotwords. Keep the tensor well-formed
        // and use the same one-token default representation as the dummy.
        lengths.push_back(1);
        matrix.push_back(1);
        for (int i = 1; i < max_len; ++i) {
          matrix.push_back(0);
        }
        continue;
      }
      lengths.push_back(len);
      for (int i = 0; i < len; ++i) {
        matrix.push_back(ids[i]);
      }
      // 补齐到 max_len
      for (int i = len; i < max_len; ++i) {
        matrix.push_back(0);
      }
    }

    lengths.push_back(1);
    matrix.push_back(1);
    for (int i = 1; i < max_len; ++i) {
      matrix.push_back(0);
    }

    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::array<int64_t, 2> shape{num_hotwords, max_len};
    Ort::Value input = Ort::Value::CreateTensor<int32_t>(
        mem_info, matrix.data(), matrix.size(), shape.data(), shape.size());

    std::vector<Ort::Value> inputs;
    inputs.push_back(std::move(input));
    //std::vector<const char*> in_names(input_names_.begin(), input_names_.end());
    //std::vector<const char*> out_names(output_names_.begin(), output_names_.end());
    std::vector<const char*> in_names;
    for (const auto& s : input_names_) in_names.push_back(s.c_str());
    std::vector<const char*> out_names;
    for (const auto& s : output_names_) out_names.push_back(s.c_str());
    auto outputs = session_->Run(Ort::RunOptions{nullptr},
		    in_names.data(), inputs.data(), inputs.size(),
		    out_names.data(), out_names.size());

    //auto outputs = session_->Run(Ort::RunOptions{nullptr},
    //                             in_names.data(), inputs.data(), inputs.size(),
    //                             out_names.data(), out_names.size());
    if (outputs.empty()) {
      SHERPA_ONNX_LOGE("Hotword compiler returned no output.");
      return {};
    }

    auto& output = outputs[0];
    auto info = output.GetTensorTypeAndShapeInfo();
    auto out_shape = info.GetShape();
    if (out_shape.size() != 3 || out_shape[0] != max_len || out_shape[1] != num_hotwords) {
      SHERPA_ONNX_LOGE("Unexpected output shape from hotword compiler.");
      return {};
    }
    int64_t emb_dim = out_shape[2];
    const float* data = output.GetTensorData<float>();

    std::vector<float> flattened(static_cast<size_t>(num_hotwords) * emb_dim);
    for (int i = 0; i < num_hotwords; ++i) {
      int t = lengths[i] - 1;  // 有效长度-1即为最后一帧索引
      int64_t offset = (t * num_hotwords + i) * emb_dim;
      std::copy(data + offset, data + offset + emb_dim,
                flattened.begin() + static_cast<size_t>(i) * emb_dim);
    }
    SHERPA_ONNX_LOGE(
        "Compiled Paraformer hotword embedding: %d hotwords + 1 dummy, "
        "embedding_dim=%" PRId64,
        num_hotwords - 1, emb_dim);
    auto storage = std::make_shared<const std::vector<float>>(
        std::move(flattened));
    return OfflineParaformerHotwordEmbedding(std::move(storage), num_hotwords,
                                             static_cast<int32_t>(emb_dim));
  }

 private:
  Ort::Env env_;
  Ort::SessionOptions session_options_;
  std::unique_ptr<Ort::Session> session_;
  std::vector<std::string> input_names_, output_names_;
};


// ========== 扩展流，支持存储热词 embedding ==========
class OfflineParaformerStream : public OfflineStream {
 public:
  OfflineParaformerStream(const FeatureExtractorConfig& feat_config,
                          ContextGraphPtr graph = nullptr)
      : OfflineStream(feat_config, graph) {}

  void SetHotwordEmbedding(OfflineParaformerHotwordEmbedding emb) {
    hw_emb_ = std::move(emb);
  }
  const OfflineParaformerHotwordEmbedding& GetHotwordEmbedding() const {
    return hw_emb_;
  }
  bool HasHotwordEmbedding() const { return !hw_emb_.empty(); }

 private:
  OfflineParaformerHotwordEmbedding hw_emb_;
};
OfflineRecognitionResult Convert(const OfflineParaformerDecoderResult &src,
                                 const SymbolTable &sym_table) {
  OfflineRecognitionResult r;
  r.tokens.reserve(src.tokens.size());
  r.ys_log_probs = src.ys_log_probs;
  r.timestamps = src.timestamps;

  std::string text;

  // When the current token ends with "@@" we set mergeable to true
  bool mergeable = false;

  for (int32_t i = 0; i != src.tokens.size(); ++i) {
    auto sym = sym_table[src.tokens[i]];
    r.tokens.push_back(sym);

    if ((sym.back() != '@') || (sym.size() > 2 && sym[sym.size() - 2] != '@')) {
      // sym does not end with "@@"
      const uint8_t *p = reinterpret_cast<const uint8_t *>(sym.c_str());
      if (p[0] < 0x80) {
        // an ascii
        if (mergeable) {
          mergeable = false;
          text.append(sym);
        } else {
          text.append(" ");
          text.append(sym);
        }
      } else {
        // not an ascii
        mergeable = false;

        if (i > 0) {
          const uint8_t p = reinterpret_cast<const uint8_t *>(
              sym_table[src.tokens[i - 1]].c_str())[0];
          if (p < 0x80) {
            // put a space between ascii and non-ascii
            text.append(" ");
          }
        }
        text.append(sym);
      }
    } else {
      // this sym ends with @@
      sym = std::string(sym.data(), sym.size() - 2);
      if (mergeable) {
        text.append(sym);
      } else {
        text.append(" ");
        text.append(sym);
        mergeable = true;
      }
    }
  }
  r.text = std::move(text);

  return r;
}

class OfflineRecognizerParaformerImpl : public OfflineRecognizerImpl {
 public:
  explicit OfflineRecognizerParaformerImpl(
      const OfflineRecognizerConfig &config)
      : OfflineRecognizerImpl(config),
        config_(config),
        symbol_table_(config_.model_config.tokens),
        model_(std::make_unique<OfflineParaformerModel>(config.model_config)) {
    if (config.decoding_method == "greedy_search") {
      int32_t eos_id = symbol_table_["</s>"];
      decoder_ = std::make_unique<OfflineParaformerGreedySearchDecoder>(eos_id);
    } else if (config_.decoding_method == "modified_beam_search") {
      LoadSegMap();  // 加载 seg_dict 到 seg_map_
      // 不再需要 BPE，已移除相关代码
      if (!config_.hotwords_file.empty()) {
        InitHotwords();
      }
      int32_t eos_id = symbol_table_["</s>"];
      int32_t unk_id = symbol_table_["<unk>"];
      decoder_ = std::make_unique<OfflineParaformerBeamSearchDecoder>(
          eos_id, config_.max_active_paths, 0.0, unk_id, config_.hotwords_score, 0.0);
    } else {
      SHERPA_ONNX_LOGE("Only greedy_search and modified_beam_search are supported. Given %s",
                       config.decoding_method.c_str());
      SHERPA_ONNX_EXIT(-1);
    }

    InitFeatConfig();

    // 如果存在 seaco-paraformer 热词编译器，使用已加载的 seg_map_
    if (!config_.model_config.paraformer.hw_compiler.empty()) {
      hw_compiler_ = std::make_unique<OfflineParaformerHotwordCompiler>(
          config_.model_config.paraformer.hw_compiler,
          config_.model_config.num_threads);
      use_hw_compiler_ = true;
      SHERPA_ONNX_LOGE("seaco-paraformer hotword compiler enabled.");
      if (!hotwords_.empty()) {
        hotword_embedding_ = hw_compiler_->Compile(hotwords_);
      }
    }
  }

  template <typename Manager>
  OfflineRecognizerParaformerImpl(Manager *mgr,
                                  const OfflineRecognizerConfig &config)
      : OfflineRecognizerImpl(mgr, config),
        config_(config),
        symbol_table_(mgr, config_.model_config.tokens),
        model_(std::make_unique<OfflineParaformerModel>(mgr,
                                                        config.model_config)) {
    if (config.decoding_method == "greedy_search") {
      int32_t eos_id = symbol_table_["</s>"];
      decoder_ = std::make_unique<OfflineParaformerGreedySearchDecoder>(eos_id);
    } else if (config_.decoding_method == "modified_beam_search") {
	  LoadSegMap(mgr);
      if (!config_.hotwords_file.empty()) {
        InitHotwords(mgr);
      }
      int32_t eos_id = symbol_table_["</s>"];
      int32_t unk_id = symbol_table_["<unk>"];
      decoder_ = std::make_unique<OfflineParaformerBeamSearchDecoder>(
          eos_id, config_.max_active_paths, 0.0, unk_id, config_.hotwords_score, 0.0);
    } else {
      SHERPA_ONNX_LOGE("Only greedy_search and modified_beam_search are supported. Given %s",
                       config.decoding_method.c_str());
      SHERPA_ONNX_EXIT(-1);
    }

    InitFeatConfig();

    if (!config_.model_config.paraformer.hw_compiler.empty()) {
      hw_compiler_ = std::make_unique<OfflineParaformerHotwordCompiler>(
          config_.model_config.paraformer.hw_compiler,
          config_.model_config.num_threads);
      use_hw_compiler_ = true;
      SHERPA_ONNX_LOGE("seaco-paraformer hotword compiler enabled.");
      if (!hotwords_.empty()) {
        hotword_embedding_ = hw_compiler_->Compile(hotwords_);
      }
    }
  }

  // 带热词的 CreateStream（统一构建 ContextGraph + 可选 embedding）
  std::unique_ptr<OfflineStream> CreateStream(const std::string &hotwords) const override {
    // 1. 构建 ContextGraph（解码器偏置）
    auto hws = std::regex_replace(hotwords, std::regex("/"), "\n");
    std::istringstream is(hws);
    std::vector<std::vector<int32_t>> current;
    std::vector<float> current_scores;

    std::string line;
    while (std::getline(is, line)) {
      if (line.empty()) continue;
      std::vector<int32_t> tokens;
      float weight = 1.0f;
      if (EncodeHotwordWithSegMap(line, &tokens, &weight)) {
        current.push_back(tokens);
        current_scores.push_back(weight);
      }
    }

    // 合并默认热词
    int32_t num_default_hws = static_cast<int32_t>(hotwords_.size());
    int32_t num_hws = static_cast<int32_t>(current.size());

    current.insert(current.end(), hotwords_.begin(), hotwords_.end());

    if (!current_scores.empty() && !boost_scores_.empty()) {
      current_scores.insert(current_scores.end(), boost_scores_.begin(),
                            boost_scores_.end());
    } else if (!current_scores.empty() && boost_scores_.empty()) {
      current_scores.insert(current_scores.end(), num_default_hws,
                            config_.hotwords_score);
    } else if (current_scores.empty() && !boost_scores_.empty()) {
      current_scores.insert(current_scores.end(), num_hws,
                            config_.hotwords_score);
      current_scores.insert(current_scores.end(), boost_scores_.begin(),
                            boost_scores_.end());
    } else {
      // Do nothing.
    }

    auto context_graph = std::make_shared<ContextGraph>(
        current, config_.hotwords_score, current_scores);

    // 2. 创建流对象
    auto stream = std::make_unique<OfflineParaformerStream>(config_.feat_config, context_graph);
    // 3. 如果使用 seaco 编译器，生成 embedding
    if (use_hw_compiler_ && hw_compiler_ && !current.empty()) {
      auto emb = hw_compiler_->Compile(current);
      if (!emb.empty()) {
        stream->SetHotwordEmbedding(std::move(emb));
      }
    }

    return stream;
  }

  std::unique_ptr<OfflineStream> CreateStream() const override {
    auto stream = std::make_unique<OfflineParaformerStream>(config_.feat_config, hotwords_graph_);
    if (!hotword_embedding_.empty()) {
      stream->SetHotwordEmbedding(hotword_embedding_);
    }
    return stream;
  }

  void DecodeStreams(OfflineStream **ss, int32_t n) const override {
    // A seaco embedding tensor has one variable-length hotword list per
    // stream, so it cannot be represented by the shared batch tensor when
    // streams have different hotwords. Decode those requests individually
    // instead of silently replacing their embeddings with zeros.
    if (use_hw_compiler_ && n > 1) {
      bool has_hotword_embedding = false;
      for (int32_t i = 0; i != n; ++i) {
        auto *pstream = static_cast<OfflineParaformerStream *>(ss[i]);
        if (pstream && pstream->HasHotwordEmbedding()) {
          has_hotword_embedding = true;
          break;
        }
      }
      if (has_hotword_embedding) {
        for (int32_t i = 0; i != n; ++i) {
          DecodeStreams(ss + i, 1);
        }
        return;
      }
    }

    // 1. Apply LFR
    // 2. Apply CMVN
    //
    // Please refer to
    // https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/45555.pdf
    // for what LFR means
    //
    // "Lower Frame Rate Neural Network Acoustic Models"
    auto memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);

    std::vector<Ort::Value> features;
    features.reserve(n);

    int32_t feat_dim =
        config_.feat_config.feature_dim * model_->LfrWindowSize();

    std::vector<std::vector<float>> features_vec(n);
    std::vector<int32_t> features_length_vec(n);
    for (int32_t i = 0; i != n; ++i) {
      std::vector<float> f = ss[i]->GetFrames();

      f = ApplyLFR(f);
      ApplyCMVN(&f);

      int32_t num_frames = static_cast<int32_t>(f.size()) / feat_dim;
      features_vec[i] = std::move(f);

      features_length_vec[i] = num_frames;

      std::array<int64_t, 2> shape = {num_frames, feat_dim};

      Ort::Value x = Ort::Value::CreateTensor(
          memory_info, features_vec[i].data(), features_vec[i].size(),
          shape.data(), shape.size());
      features.push_back(std::move(x));
    }

    std::vector<const Ort::Value *> features_pointer(n);
    for (int32_t i = 0; i != n; ++i) {
      features_pointer[i] = &features[i];
    }

    std::array<int64_t, 1> features_length_shape = {n};
    Ort::Value x_length = Ort::Value::CreateTensor(
        memory_info, features_length_vec.data(), n,
        features_length_shape.data(), features_length_shape.size());

    // Caution(fangjun): We cannot pad it with log(eps),
    // i.e., -23.025850929940457f
    Ort::Value x = PadSequence(model_->Allocator(), features_pointer, 0);

    // ---------- 模型输入准备 ----------
    std::vector<Ort::Value> model_inputs;
    model_inputs.push_back(std::move(x));
    model_inputs.push_back(std::move(x_length));

    // seaco 热词 embedding（仅 batch=1）
    bool has_hw_emb = false;
    // Ort::Value created with a user buffer does not copy that buffer. Keep
    // the backing storage alive until model_->Forward() has completed.
    std::vector<float> bias_embed_data;
    if (use_hw_compiler_ && n == 1) {
      auto* pstream = static_cast<OfflineParaformerStream*>(ss[0]);
      if (pstream && pstream->HasHotwordEmbedding()) {
        const auto& emb = pstream->GetHotwordEmbedding();
        if (!emb.empty()) {
          int hw_count = emb.num_embeddings();
          int emb_dim = emb.embedding_dim();
          int expected_emb_dim = model_->BiasEmbedDim();
          if (expected_emb_dim > 0 && emb_dim != expected_emb_dim) {
            SHERPA_ONNX_LOGE(
                "Hotword embedding dimension %d does not match the model "
                "bias_embed dimension %d.",
                emb_dim, expected_emb_dim);
            return;
          }
          if (emb.size() != static_cast<size_t>(hw_count) * emb_dim) {
            SHERPA_ONNX_LOGE("Inconsistent hotword embedding dimensions.");
            return;
          }
          std::array<int64_t, 3> hw_shape{1, hw_count, emb_dim};
          Ort::Value hw_emb_tensor = Ort::Value::CreateTensor<float>(
              memory_info, const_cast<float *>(emb.data()), emb.size(),
              hw_shape.data(), hw_shape.size());
          model_inputs.push_back(std::move(hw_emb_tensor));
          has_hw_emb = true;
	}
      }
    }

    // Seaco Paraformer requires bias_embed even when no hotwords are given.
    // In the no-hotword mode the FunASR implementation uses a zero/default
    // embedding. The compiler's dummy row is only part of a non-empty hotword
    // batch and must not be used as the no-hotword bias.
    if (model_->HasBiasEmbedInput() && !has_hw_emb) {
      int32_t emb_dim = model_->BiasEmbedDim();
      if (emb_dim <= 0) {
        // The seaco export uses a symbolic last dimension. Its compiler and
        // trained model contract use a 512-dimensional bias embedding.
        constexpr int32_t kSeacoBiasEmbedDim = 512;
        emb_dim = kSeacoBiasEmbedDim;
        SHERPA_ONNX_LOGE(
            "bias_embed dimension is dynamic; using seaco default dimension %d.",
            emb_dim);
      }

      bias_embed_data.assign(static_cast<size_t>(n) * emb_dim, 0.0f);
      std::array<int64_t, 3> hw_shape{n, 1, emb_dim};
      Ort::Value hw_emb_tensor = Ort::Value::CreateTensor<float>(
          memory_info, bias_embed_data.data(), bias_embed_data.size(),
          hw_shape.data(), hw_shape.size());
      model_inputs.push_back(std::move(hw_emb_tensor));
    }

    // ---------- 模型前向 ----------
    std::vector<Ort::Value> t;
    try {
      t = model_->Forward(std::move(model_inputs));
    } catch (const Ort::Exception &ex) {
      SHERPA_ONNX_LOGE("\n\nCaught exception:\n\n%s\n\nReturn an empty result",
                       ex.what());
      return;
    }

    if (t.size() < 2) {
      SHERPA_ONNX_LOGE("Paraformer forward returned fewer than 2 outputs.");
      return;
    }

    std::vector<OfflineParaformerDecoderResult> results;
    if (t.size() > 3) {
      results = decoder_->Decode(std::move(t[0]), std::move(t[1]),
                                 std::move(t[3]), ss, n);
    } else {
      results = decoder_->Decode(std::move(t[0]), std::move(t[1]),
                                 Ort::Value(nullptr), ss, n);
    }
    if (results.size() != static_cast<size_t>(n)) {
      SHERPA_ONNX_LOGE("Paraformer decoder returned %zu results for %d streams.",
                       results.size(), n);
      return;
    }

    for (int32_t i = 0; i != n; ++i) {
      auto r = Convert(results[i], symbol_table_);
      r.text = ApplyInverseTextNormalization(std::move(r.text));
      r.text = ApplyHomophoneReplacer(std::move(r.text));
      ss[i]->SetResult(r);
    }
  }

  OfflineRecognizerConfig GetConfig() const override { return config_; }

 private:
  void InitFeatConfig() {
    // Paraformer models assume input samples are in the range
    // [-32768, 32767], so we set normalize_samples to false
    config_.feat_config.normalize_samples = false;
    config_.feat_config.window_type = "hamming";
    config_.feat_config.high_freq = 0;
    config_.feat_config.snip_edges = true;
  }

  std::vector<float> ApplyLFR(const std::vector<float> &in) const {
    int32_t lfr_window_size = model_->LfrWindowSize();
    int32_t lfr_window_shift = model_->LfrWindowShift();
    int32_t in_feat_dim = config_.feat_config.feature_dim;

    int32_t in_num_frames = static_cast<int32_t>(in.size()) / in_feat_dim;
    int32_t out_num_frames =
        (in_num_frames - lfr_window_size) / lfr_window_shift + 1;
    int32_t out_feat_dim = in_feat_dim * lfr_window_size;

    std::vector<float> out(out_num_frames * out_feat_dim);

    const float *p_in = in.data();
    float *p_out = out.data();

    for (int32_t i = 0; i != out_num_frames; ++i) {
      std::copy(p_in, p_in + out_feat_dim, p_out);

      p_out += out_feat_dim;
      p_in += lfr_window_shift * in_feat_dim;
    }

    return out;
  }

  void ApplyCMVN(std::vector<float> *v) const {
    const std::vector<float> &neg_mean = model_->NegativeMean();
    const std::vector<float> &inv_stddev = model_->InverseStdDev();
    int32_t dim = static_cast<int32_t>(neg_mean.size());
    int32_t num_frames = static_cast<int32_t>(v->size()) / dim;

    Eigen::Map<
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        mat(v->data(), num_frames, dim);

    Eigen::Map<const Eigen::RowVectorXf> neg_mean_vec(neg_mean.data(), dim);
    Eigen::Map<const Eigen::RowVectorXf> inv_stddev_vec(inv_stddev.data(), dim);

    mat.array() = (mat.array().rowwise() + neg_mean_vec.array()).rowwise() *
                  inv_stddev_vec.array();
  }
  // 加载 seg_dict，将符号 token 转换为 ID 序列，存入 seg_map_
  void LoadSegMap() {
    std::string seg_dict_path = config_.model_config.paraformer.seg_dict;
    if (seg_dict_path.empty()) {
      SHERPA_ONNX_LOGE("seg_dict path is empty, cannot load seg_map.");
      return;
    }
    std::ifstream seg_file(seg_dict_path);
    if (!seg_file.is_open()) {
      SHERPA_ONNX_LOGE("Failed to open seg_dict: %s", seg_dict_path.c_str());
      return;
    }
    std::string line;
    while (std::getline(seg_file, line)) {
      AddSegMapEntry(line);
    }
  }

  template <typename Manager>
  void LoadSegMap(Manager *mgr) {
    std::string seg_dict_path = config_.model_config.paraformer.seg_dict;
    if (seg_dict_path.empty()) {
      SHERPA_ONNX_LOGE("seg_dict path is empty, cannot load seg_map.");
      return;
    }
    auto buf = ReadFile(mgr, seg_dict_path);
    if (buf.empty()) {
      SHERPA_ONNX_LOGE("Failed to read seg_dict: %s", seg_dict_path.c_str());
      return;
    }
    std::istringstream seg_is(std::string(buf.begin(), buf.end()));
    std::string line;
    while (std::getline(seg_is, line)) {
      AddSegMapEntry(line);
    }
  }

  void AddSegMapEntry(const std::string& line) {
    if (line.empty()) {
      return;
    }

    std::istringstream iss(line);
    std::string word;
    iss >> word;
    if (word.empty()) {
      return;
    }

    std::vector<int32_t> token_ids;
    std::vector<std::pair<std::string, int32_t>> token_entries;
    std::string token_str;
    while (iss >> token_str) {
      if (!symbol_table_.Contains(token_str)) {
        SHERPA_ONNX_LOGE(
            "Token '%s' from seg_dict is not present in tokens.txt; "
            "skip word '%s'",
            token_str.c_str(), word.c_str());
        return;
      }
      int32_t id = symbol_table_[token_str];
      token_ids.push_back(id);
      token_entries.emplace_back(token_str, id);
    }

    if (token_ids.empty()) {
      return;
    }

    seg_map_[word] = std::move(token_ids);
    for (const auto& entry : token_entries) {
      const auto& piece = entry.first;
      const bool is_ascii = std::all_of(
          piece.begin(), piece.end(),
          [](unsigned char c) { return c < 0x80; });
      if (!is_ascii || piece.empty()) {
        continue;
      }
      english_subword_to_id_.emplace(piece, entry.second);
    }
  }

  bool FindEnglishToken(const std::string& piece, bool continuation,
                        int32_t* token_id) const {
    *token_id = -1;

    if (continuation) {
      const std::string continuation_piece = piece + "@@";
      auto it = english_subword_to_id_.find(continuation_piece);
      if (it != english_subword_to_id_.end()) {
        *token_id = it->second;
        return true;
      }
      if (symbol_table_.Contains(continuation_piece)) {
        *token_id = symbol_table_[continuation_piece];
        return true;
      }
    }

    auto it = english_subword_to_id_.find(piece);
    if (it != english_subword_to_id_.end()) {
      *token_id = it->second;
      return true;
    }
    if (symbol_table_.Contains(piece)) {
      *token_id = symbol_table_[piece];
      return true;
    }
    return false;
  }

  bool EncodeEnglishByLetters(const std::string& word,
                              std::vector<int32_t>* token_ids) const {
    token_ids->clear();
    for (size_t i = 0; i < word.size(); ++i) {
      int32_t token_id = -1;
      if (!FindEnglishToken(std::string(1, word[i]), i + 1 < word.size(),
                            &token_id)) {
        token_ids->clear();
        return false;
      }
      token_ids->push_back(token_id);
    }
    return !token_ids->empty();
  }

  bool EncodeEnglishOovWithDp(const std::string& word,
                               std::vector<int32_t>* token_ids) const {
    if (word.empty()) {
      return false;
    }

    const size_t n = word.size();
    const int kUnreachable = static_cast<int>(n) + 1;
    std::vector<int> best_num_tokens(n + 1, kUnreachable);
    std::vector<int64_t> previous(n + 1, -1);
    std::vector<int32_t> previous_token(n + 1, -1);
    best_num_tokens[0] = 0;

    for (size_t begin = 0; begin < n; ++begin) {
      if (best_num_tokens[begin] == kUnreachable) {
        continue;
      }

      // Try all substrings so this also works when the seg_dict does not
      // contain an entry for a newly introduced English word, but the token
      // table still contains usable ASCII pieces.
      const size_t max_len = n - begin;
      for (size_t len = 1; len <= max_len; ++len) {
        const size_t end = begin + len;
        const std::string piece = word.substr(begin, len);
        int32_t token_id = -1;
        if (!FindEnglishToken(piece, end < n, &token_id)) {
          continue;
        }

        const int candidate = best_num_tokens[begin] + 1;
        if (candidate < best_num_tokens[end]) {
          best_num_tokens[end] = candidate;
          previous[end] = static_cast<int64_t>(begin);
          previous_token[end] = token_id;
        }
      }
    }

    if (best_num_tokens[n] == kUnreachable) {
      return false;
    }

    std::vector<int32_t> reversed;
    for (size_t pos = n; pos != 0;) {
      reversed.push_back(previous_token[pos]);
      pos = static_cast<size_t>(previous[pos]);
    }
    token_ids->assign(reversed.rbegin(), reversed.rend());
    return true;
  }

  bool EncodeEnglishWord(const std::string& word,
                         std::vector<int32_t>* token_ids) const {
    bool has_letter = false;
    bool all_upper = true;
    for (unsigned char c : word) {
      if (c >= 'A' && c <= 'Z') {
        has_letter = true;
      } else if (c >= 'a' && c <= 'z') {
        has_letter = true;
        all_upper = false;
      }
    }

    std::string lookup_word = word;
    for (char& c : lookup_word) {
      if (c >= 'A' && c <= 'Z') {
        c = static_cast<char>(c - 'A' + 'a');
      }
    }

    // Acronyms are looked up letter by letter after lowercasing, e.g. GPT5
    // becomes g, p, t, 5. This matches the case-insensitive model vocabulary.
    if (has_letter && all_upper &&
        EncodeEnglishByLetters(lookup_word, token_ids)) {
      return true;
    }

    auto it = seg_map_.find(lookup_word);
    if (it != seg_map_.end()) {
      *token_ids = it->second;
      return true;
    }

    if (EncodeEnglishOovWithDp(lookup_word, token_ids)) {
      return true;
    }

    // Preserve compatibility with an explicitly case-sensitive seg_dict.
    if (lookup_word != word) {
      it = seg_map_.find(word);
      if (it != seg_map_.end()) {
        *token_ids = it->second;
        return true;
      }
      if (EncodeEnglishOovWithDp(word, token_ids)) {
        return true;
      }
    }
    return false;
  }

  // 使用 seg_map_ 将热词字符串（格式 "热词 [权重]"）编码为 ID 序列和权重。
  bool EncodeHotwordWithSegMap(const std::string& hotword,
                               std::vector<int32_t>* tokens,
                               float* weight) const {
  // Parse all fields first. Reading a non-numeric English word directly into
  // a float sets failbit and used to discard the rest of a multi-word phrase.
  std::istringstream full_iss(hotword);
  std::vector<std::string> words;
  std::string field;
  while (full_iss >> field) {
    words.push_back(field);
  }
  if (words.empty()) {
    SHERPA_ONNX_LOGE("Empty hotword string.");
    return false;
  }

  // A missing weight must fall back to config_.hotwords_score in ContextGraph.
  *weight = 0.0f;
  if (words.size() > 1) {
    char* endptr = nullptr;
    const float parsed_weight = std::strtof(words.back().c_str(), &endptr);
    if (endptr != words.back().c_str() && *endptr == '\0') {
      *weight = parsed_weight;
      words.pop_back();
    }
  }

  std::vector<int32_t> result_tokens;
  auto append_english = [&](const std::string& english) {
    std::vector<int32_t> english_tokens;
    if (!EncodeEnglishWord(english, &english_tokens)) {
      SHERPA_ONNX_LOGE(
          "English OOV '%s' cannot be segmented using seg_dict subwords.",
          english.c_str());
      return false;
    }
    result_tokens.insert(result_tokens.end(), english_tokens.begin(),
                         english_tokens.end());
    return true;
  };

  for (const auto& wd : words) {
    bool all_ascii = std::all_of(wd.begin(), wd.end(),
                                 [](unsigned char c) { return c < 0x80; });
    if (all_ascii) {
      if (!append_english(wd)) {
        return false;
      }
    } else {
      const char* p = wd.c_str();
      std::string ascii_run;
      auto flush_ascii = [&]() {
        if (ascii_run.empty()) {
          return true;
        }
        const bool ok = append_english(ascii_run);
        ascii_run.clear();
        return ok;
      };

      while (*p) {
        unsigned char c = static_cast<unsigned char>(*p);
        if (c < 0x80) {
          ascii_run.push_back(*p++);
          continue;
        }

        if (!flush_ascii()) {
          return false;
        }

        int32_t len = 0;
        if ((c & 0xE0) == 0xC0) {
          len = 2;
        } else if ((c & 0xF0) == 0xE0) {
          len = 3;
        } else if ((c & 0xF8) == 0xF0) {
          len = 4;
        } else {
          SHERPA_ONNX_LOGE("Invalid UTF-8 character in hotword '%s'.", wd.c_str());
          return false;
        }

        const size_t offset = static_cast<size_t>(p - wd.c_str());
        const size_t remaining = wd.size() - offset;
        if (static_cast<size_t>(len) > remaining) {
          SHERPA_ONNX_LOGE("Truncated UTF-8 character in hotword '%s'.",
                           wd.c_str());
          return false;
        }

        const auto *bytes = reinterpret_cast<const unsigned char *>(p);
        for (int32_t i = 1; i < len; ++i) {
          if ((bytes[i] & 0xC0) != 0x80) {
            SHERPA_ONNX_LOGE("Invalid UTF-8 continuation byte in hotword '%s'.",
                             wd.c_str());
            return false;
          }
        }

        // Reject overlong encodings, UTF-16 surrogate code points, and values
        // above U+10FFFF in addition to malformed byte sequences.
        if ((len == 2 && c < 0xC2) ||
            (len == 3 && ((c == 0xE0 && bytes[1] < 0xA0) ||
                          (c == 0xED && bytes[1] >= 0xA0))) ||
            (len == 4 && ((c == 0xF0 && bytes[1] < 0x90) ||
                          (c == 0xF4 && bytes[1] >= 0x90) || c > 0xF4))) {
          SHERPA_ONNX_LOGE("Invalid UTF-8 code point in hotword '%s'.",
                           wd.c_str());
          return false;
        }

        std::string char_str(p, len);
	if (!symbol_table_.Contains(char_str)) {
          SHERPA_ONNX_LOGE("Character '%s' not found in token table, cannot encode word '%s'.",
			                             char_str.c_str(), wd.c_str());
	  return false;
	}
        int32_t id = symbol_table_[char_str];
        result_tokens.push_back(id);
        p += len;
      }

      if (!flush_ascii()) {
        return false;
      }
    }
  }

  constexpr size_t kMaxHotwordTokens = 10;
  if (result_tokens.empty() || result_tokens.size() > kMaxHotwordTokens) {
    SHERPA_ONNX_LOGE(
        "Hotword '%s' has %zu tokens; the hotword compiler supports 1 to %zu. "
        "Ignoring it.",
        hotword.c_str(), result_tokens.size(), kMaxHotwordTokens);
    return false;
  }

  *tokens = std::move(result_tokens);
  return true;
  }

  void InitHotwords() {
    std::ifstream is(config_.hotwords_file);
    if (!is) {
      SHERPA_ONNX_LOGE("Open hotwords file failed: '%s'", config_.hotwords_file.c_str());
      SHERPA_ONNX_EXIT(-1);
    }
    std::string line;
    while (std::getline(is, line)) {
      if (line.empty()) continue;
      std::vector<int32_t> tokens;
      float weight = 1.0f;
      if (EncodeHotwordWithSegMap(line, &tokens, &weight)) {
        hotwords_.push_back(tokens);
        boost_scores_.push_back(weight);
      }
    }
    hotwords_graph_ = std::make_shared<ContextGraph>(
        hotwords_, config_.hotwords_score, boost_scores_);
  }

  template <typename Manager>
  void InitHotwords(Manager *mgr) {
    auto buf = ReadFile(mgr, config_.hotwords_file);
    if (buf.empty()) {
      SHERPA_ONNX_LOGE("Failed to read hotwords file: '%s'", config_.hotwords_file.c_str());
      SHERPA_ONNX_EXIT(-1);
    }
    std::istringstream is(std::string(buf.begin(), buf.end()));
    std::string line;
    while (std::getline(is, line)) {
      if (line.empty()) continue;
      std::vector<int32_t> tokens;
      float weight = config_.hotwords_score;
      if (EncodeHotwordWithSegMap(line, &tokens, &weight)) {
        hotwords_.push_back(tokens);
        boost_scores_.push_back(weight);
      }
    }
    hotwords_graph_ = std::make_shared<ContextGraph>(
        hotwords_, config_.hotwords_score, boost_scores_);
  }


  OfflineRecognizerConfig config_;
  SymbolTable symbol_table_;
  std::vector<std::vector<int32_t>> hotwords_;
  std::vector<float> boost_scores_;
  ContextGraphPtr hotwords_graph_;
  std::unordered_map<std::string, std::vector<int32_t>> seg_map_;  // 缓存 seg_dict 映射
  std::unordered_map<std::string, int32_t> english_subword_to_id_;
  std::unique_ptr<OfflineParaformerModel> model_;
  std::unique_ptr<OfflineParaformerDecoder> decoder_;
  std::unique_ptr<OfflineParaformerHotwordCompiler> hw_compiler_;
  OfflineParaformerHotwordEmbedding hotword_embedding_;
  bool use_hw_compiler_ = false;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_RECOGNIZER_PARAFORMER_IMPL_H_
