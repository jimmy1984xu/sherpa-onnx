// sherpa-onnx/csrc/offline-ct-transformer-model.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-ct-transformer-model.h"
#include "sherpa-onnx/csrc/ort-env.h"
#include "sherpa-onnx/csrc/macros.h"

#include <memory>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#if __ANDROID_API__ >= 9
#include "android/asset_manager.h"
#include "android/asset_manager_jni.h"
#endif

#if __OHOS__
#include "rawfile/raw_file_manager.h"
#endif

#include "sherpa-onnx/csrc/file-utils.h"
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/session.h"
#include "sherpa-onnx/csrc/text-utils.h"

#ifdef SHERPA_ONNX_ENABLE_CT_TRANSFORMER_NEON
#include <asm/hwcap.h>
#include <sys/auxv.h>

#include "sherpa-onnx/csrc/mali-g720/aot-file.h"
#include "sherpa-onnx/csrc/neon-ort-session.h"
#endif

namespace sherpa_onnx {
#ifdef SHERPA_ONNX_ENABLE_CT_TRANSFORMER_NEON
void RegisterCtTransformerNeonOps(Ort::SessionOptions &options);
#endif

static Ort::SessionOptions CtSessionOptions(
    const OfflinePunctuationModelConfig &config) {
  if (config.provider != "neon") {
    return GetSessionOptions(config);
  }
#ifdef SHERPA_ONNX_ENABLE_CT_TRANSFORMER_NEON
  if (!(getauxval(AT_HWCAP2) & HWCAP2_I8MM)) {
    throw std::runtime_error("CT-Transformer NEON requires Arm I8MM");
  }
  auto options = GetSessionOptions(config.num_threads, "cpu");
  RegisterCtTransformerNeonOps(options);
  return options;
#else
  throw std::runtime_error("CT-Transformer NEON requires Android arm64");
#endif
}

class OfflineCtTransformerModel::Impl {
 public:
  explicit Impl(const OfflinePunctuationModelConfig &config)
      : config_(config),
        env_(CreateOrtEnv()),
        sess_opts_(CtSessionOptions(config)),
        allocator_{} {
#ifdef SHERPA_ONNX_ENABLE_CT_TRANSFORMER_NEON
    if (config.provider == "neon") {
      const auto file = transai::aot::File::Open(config.ct_transformer);
      const auto &manifest = file->manifest();
      const auto &architecture = manifest.at("architecture");
      const auto &aot = manifest.at("aot");
      const auto &target = aot.at("target");
      if (architecture.at("family") != "ct-transformer" ||
          architecture.at("version") != 1 ||
          architecture.at("precision") != "w8a8-u8-embedding" ||
          aot.at("format") != 1 || !aot.at("programs").empty() ||
          target.at("provider") != "neon" ||
          target.at("abi") != "arm64-v8a" || target.at("isa") != "i8mm" ||
          target.at("input") != "utf8-lines" ||
          file->Asset("punctuation/model.onnx").bytes > 256 * 1024 * 1024) {
        throw std::runtime_error("Expected a CT-Transformer NEON v1.0 AOT");
      }
      transai::aot::RequireMappedOrtGraph(*file);
      transai::aot::SetMappedOrtGraphOptions(sess_opts_);
      model_mapping_ = std::make_unique<transai::aot::Mapping>(
          config.ct_transformer + "/punctuation/model.onnx");
      Init(model_mapping_->data(), model_mapping_->size());
      return;
    }
#endif
    sess_ = std::make_unique<Ort::Session>(
        env_, SHERPA_ONNX_TO_ORT_PATH(config_.ct_transformer), sess_opts_);
    Init(nullptr, 0);
  }

  template <typename Manager>
  Impl(Manager *mgr, const OfflinePunctuationModelConfig &config)
      : config_(config),
        env_(CreateOrtEnv()),
        sess_opts_(CtSessionOptions(config)),
        allocator_{} {
    if (config.provider == "neon") {
      throw std::runtime_error("CT-Transformer NEON requires a filesystem AOT");
    }
    auto buf = ReadFile(mgr, config_.ct_transformer);
    Init(buf.data(), buf.size());
  }

  Ort::Value Forward(Ort::Value text, Ort::Value text_len) {
    std::array<Ort::Value, 2> inputs = {std::move(text), std::move(text_len)};

    auto ans =
        sess_->Run({}, input_names_ptr_.data(), inputs.data(), inputs.size(),
                   output_names_ptr_.data(), output_names_ptr_.size());
    return std::move(ans[0]);
  }

  void ForwardInto(Ort::Value text, Ort::Value text_len, Ort::Value *output) {
    std::array<Ort::Value, 2> inputs = {std::move(text), std::move(text_len)};
    sess_->Run({}, input_names_ptr_.data(), inputs.data(), inputs.size(),
               output_names_ptr_.data(), output, 1);
  }

  OrtAllocator *Allocator() { return allocator_; }

  const OfflineCtTransformerModelMetaData &GetModelMetadata() const {
    return meta_data_;
  }

 private:
  void Init(const void *model_data, size_t model_data_length) {
    if (model_data) {
      sess_ = std::make_unique<Ort::Session>(
          env_, model_data, model_data_length, sess_opts_);
    } else if (!sess_) {
      SHERPA_ONNX_LOGE(
          "Please pass model data or initialize the session outside of "
          "this function");
      SHERPA_ONNX_EXIT(-1);
    }

    GetInputNames(sess_.get(), &input_names_, &input_names_ptr_);

    GetOutputNames(sess_.get(), &output_names_, &output_names_ptr_);

    // get meta data
    Ort::ModelMetadata meta_data = sess_->GetModelMetadata();

    Ort::AllocatorWithDefaultOptions allocator;  // used in the macro below

    int32_t vocab_size = 0;
    SHERPA_ONNX_READ_META_DATA(vocab_size, "vocab_size");
    // Populate the final lookup directly, avoiding the temporary 272k-token
    // vector and an additional copy of the full metadata string.
    auto tokens = meta_data.LookupCustomMetadataMapAllocated("tokens", allocator);
    if (!tokens || !tokens.get()[0] || vocab_size <= 0) {
      throw std::runtime_error("Invalid CT-Transformer token metadata");
    }
    meta_data_.token2id.reserve(vocab_size);
    int32_t token_count = 0;
    for (const char *begin = tokens.get();;) {
      if (token_count >= vocab_size) {
        throw std::runtime_error("CT-Transformer vocabulary size mismatch");
      }
      const char *end = std::strchr(begin, '|');
      meta_data_.token2id[std::string(begin, end ? end - begin : std::strlen(begin))] = token_count++;
      if (!end) break;
      begin = end + 1;
    }
    if (token_count != vocab_size) {
      throw std::runtime_error("CT-Transformer vocabulary size mismatch");
    }

    SHERPA_ONNX_READ_META_DATA_VEC_STRING_SEP(meta_data_.id2punct,
                                              "punctuations", "|");

    std::string unk_symbol;
    SHERPA_ONNX_READ_META_DATA_STR(unk_symbol, "unk_symbol");

    // output shape is (N, T, num_punctuations)
    meta_data_.num_punctuations =
        sess_->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape()[2];

    int32_t i = 0;
    for (const auto &p : meta_data_.id2punct) {
      meta_data_.punct2id[p] = i;
      i += 1;
    }

    meta_data_.unk_id = meta_data_.token2id.at(unk_symbol);

    meta_data_.dot_id = meta_data_.punct2id.at("。");
    meta_data_.comma_id = meta_data_.punct2id.at("，");
    meta_data_.quest_id = meta_data_.punct2id.at("？");
    meta_data_.pause_id = meta_data_.punct2id.at("、");
    meta_data_.underline_id = meta_data_.punct2id.at("_");

    if (config_.debug) {
      std::ostringstream os;
      os << "vocab_size: " << meta_data_.token2id.size() << "\n";
      os << "num_punctuations: " << meta_data_.num_punctuations << "\n";
      os << "punctuations: ";
      for (const auto &s : meta_data_.id2punct) {
        os << s << " ";
      }
      os << "\n";
      SHERPA_ONNX_LOGE("\n%s\n", os.str().c_str());
    }
  }

 private:
  OfflinePunctuationModelConfig config_;
  Ort::Env env_;
  Ort::SessionOptions sess_opts_;
  Ort::AllocatorWithDefaultOptions allocator_;

#ifdef SHERPA_ONNX_ENABLE_CT_TRANSFORMER_NEON
  // ORT initializers refer to this mapping until the session is destroyed.
  std::unique_ptr<transai::aot::Mapping> model_mapping_;
#endif
  std::unique_ptr<Ort::Session> sess_;

  std::vector<std::string> input_names_;
  std::vector<const char *> input_names_ptr_;

  std::vector<std::string> output_names_;
  std::vector<const char *> output_names_ptr_;

  OfflineCtTransformerModelMetaData meta_data_;
};

OfflineCtTransformerModel::OfflineCtTransformerModel(
    const OfflinePunctuationModelConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

template <typename Manager>
OfflineCtTransformerModel::OfflineCtTransformerModel(
    Manager *mgr, const OfflinePunctuationModelConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}

#if __ANDROID_API__ >= 9
template OfflineCtTransformerModel::OfflineCtTransformerModel(
    AAssetManager *mgr, const OfflinePunctuationModelConfig &config);
#endif

#if __OHOS__
template OfflineCtTransformerModel::OfflineCtTransformerModel(
    NativeResourceManager *mgr, const OfflinePunctuationModelConfig &config);
#endif

OfflineCtTransformerModel::~OfflineCtTransformerModel() = default;

Ort::Value OfflineCtTransformerModel::Forward(Ort::Value text,
                                              Ort::Value text_len) const {
  return impl_->Forward(std::move(text), std::move(text_len));
}

void OfflineCtTransformerModel::ForwardInto(Ort::Value text, Ort::Value text_len,
                                           Ort::Value *output) const {
  impl_->ForwardInto(std::move(text), std::move(text_len), output);
}

OrtAllocator *OfflineCtTransformerModel::Allocator() const {
  return impl_->Allocator();
}

const OfflineCtTransformerModelMetaData &
OfflineCtTransformerModel::GetModelMetadata() const {
  return impl_->GetModelMetadata();
}

}  // namespace sherpa_onnx
