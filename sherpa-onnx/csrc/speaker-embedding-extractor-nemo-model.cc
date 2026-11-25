// sherpa-onnx/csrc/speaker-embedding-extractor-nemo-model.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include "sherpa-onnx/csrc/speaker-embedding-extractor-nemo-model.h"
#include "sherpa-onnx/csrc/ort-env.h"

#include <memory>
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
#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/onnx-utils.h"
#include "sherpa-onnx/csrc/session.h"
#include "sherpa-onnx/csrc/speaker-embedding-extractor-nemo-model-meta-data.h"
#include "sherpa-onnx/csrc/text-utils.h"

#ifdef SHERPA_ONNX_ENABLE_TITANET_NEON
#include <asm/hwcap.h>
#include <sys/auxv.h>

#include "sherpa-onnx/csrc/neon-ort-session.h"
#endif

namespace sherpa_onnx {

#ifdef SHERPA_ONNX_ENABLE_TITANET_NEON
void RegisterTitaNetNeonOps(Ort::SessionOptions &options);
#endif

class SpeakerEmbeddingExtractorNeMoModel::Impl {
 public:
  explicit Impl(const SpeakerEmbeddingExtractorConfig &config)
      : config_(config),
        env_(CreateOrtEnv()),
        sess_opts_(MakeSessionOptions(config)),
        allocator_{} {
#ifdef SHERPA_ONNX_ENABLE_TITANET_NEON
    if (config.provider == "neon") {
      const auto file = transai::aot::File::Open(config.model);
      const auto &manifest = file->manifest();
      const auto &architecture = manifest.at("architecture");
      if (architecture.at("family") != "titanet-large" ||
          architecture.at("version") != 1 ||
          architecture.at("precision") != "w8a8" ||
          manifest.at("aot").at("format") != 1 ||
          file->Asset("speaker/model.onnx").bytes > 128 * 1024 * 1024) {
        throw std::runtime_error("Expected a TitaNet-Large NEON v1.0 AOT");
      }
      transai::aot::RequireMappedOrtGraph(*file);
      transai::aot::SetMappedOrtGraphOptions(sess_opts_);
      model_mapping_ = std::make_unique<transai::aot::Mapping>(config.model + "/speaker/model.onnx");
      Init(model_mapping_->data(), model_mapping_->size());
      return;
    }
#endif
    sess_ = std::make_unique<Ort::Session>(
        env_, SHERPA_ONNX_TO_ORT_PATH(config.model), sess_opts_);
    Init(nullptr, 0);
  }

  template <typename Manager>
  Impl(Manager *mgr, const SpeakerEmbeddingExtractorConfig &config)
      : config_(config),
        env_(CreateOrtEnv()),
        sess_opts_(GetSessionOptions(config)),
        allocator_{} {
    {
      auto buf = ReadFile(mgr, config.model);
      Init(buf.data(), buf.size());
    }
  }

  Ort::Value Compute(Ort::Value x, Ort::Value x_lens) const {
    std::array<Ort::Value, 2> inputs = {std::move(x), std::move(x_lens)};

    // output_names_ptr_[0] is logits
    // output_names_ptr_[1] is embeddings
    // so we use output_names_ptr_.data() + 1 here to extract only the
    // embeddings
    auto outputs = sess_->Run({}, input_names_ptr_.data(), inputs.data(),
                              inputs.size(), output_names_ptr_.data() + 1, 1);
    return std::move(outputs[0]);
  }

  OrtAllocator *Allocator() { return allocator_; }

  const SpeakerEmbeddingExtractorNeMoModelMetaData &GetMetaData() const {
    return meta_data_;
  }

 private:
  static Ort::SessionOptions MakeSessionOptions(
      const SpeakerEmbeddingExtractorConfig &config) {
    if (config.provider != "neon") {
      return GetSessionOptions(config);
    }
#ifdef SHERPA_ONNX_ENABLE_TITANET_NEON
    if (!(getauxval(AT_HWCAP2) & HWCAP2_I8MM)) {
      throw std::runtime_error("TitaNet NEON requires Arm I8MM");
    }
    auto options = GetSessionOptions(config.num_threads, "cpu");
    RegisterTitaNetNeonOps(options);
    return options;
#else
    throw std::runtime_error("TitaNet NEON requires an Android arm64 build");
#endif
  }

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
    if (config_.provider == "neon" &&
        (input_names_.size() != 2 || output_names_.size() != 2 ||
         sess_->GetOutputTypeInfo(1).GetTensorTypeAndShapeInfo()
                 .GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)) {
      throw std::runtime_error("Invalid TitaNet NEON tensor interface");
    }

    // get meta data
    Ort::ModelMetadata meta_data = sess_->GetModelMetadata();
    if (config_.debug) {
      std::ostringstream os;
      PrintModelMetadata(os, meta_data);
#if __OHOS__
      SHERPA_ONNX_LOGE("%{public}s", os.str().c_str());
#else
      SHERPA_ONNX_LOGE("%s", os.str().c_str());
#endif
    }

    Ort::AllocatorWithDefaultOptions allocator;  // used in the macro below
    SHERPA_ONNX_READ_META_DATA(meta_data_.output_dim, "output_dim");
    SHERPA_ONNX_READ_META_DATA(meta_data_.feat_dim, "feat_dim");
    SHERPA_ONNX_READ_META_DATA(meta_data_.sample_rate, "sample_rate");
    SHERPA_ONNX_READ_META_DATA(meta_data_.window_size_ms, "window_size_ms");
    SHERPA_ONNX_READ_META_DATA(meta_data_.window_stride_ms, "window_stride_ms");
    SHERPA_ONNX_READ_META_DATA_STR(meta_data_.language, "language");

    SHERPA_ONNX_READ_META_DATA_STR_WITH_DEFAULT(
        meta_data_.feature_normalize_type, "feature_normalize_type", "");

    SHERPA_ONNX_READ_META_DATA_STR_WITH_DEFAULT(meta_data_.window_type,
                                                "window_type", "povey");

    std::string framework;
    SHERPA_ONNX_READ_META_DATA_STR(framework, "framework");
    if (framework != "nemo") {
#if __OHOS__
      SHERPA_ONNX_LOGE("Expect a NeMo model, given: %{public}s",
                       framework.c_str());
#else
      SHERPA_ONNX_LOGE("Expect a NeMo model, given: %s", framework.c_str());
#endif
      SHERPA_ONNX_EXIT(-1);
    }
  }

 private:
  SpeakerEmbeddingExtractorConfig config_;
  Ort::Env env_;
  Ort::SessionOptions sess_opts_;
  Ort::AllocatorWithDefaultOptions allocator_;

#ifdef SHERPA_ONNX_ENABLE_TITANET_NEON
  std::unique_ptr<transai::aot::Mapping> model_mapping_;
#endif
  std::unique_ptr<Ort::Session> sess_;

  std::vector<std::string> input_names_;
  std::vector<const char *> input_names_ptr_;

  std::vector<std::string> output_names_;
  std::vector<const char *> output_names_ptr_;

  SpeakerEmbeddingExtractorNeMoModelMetaData meta_data_;
};

SpeakerEmbeddingExtractorNeMoModel::SpeakerEmbeddingExtractorNeMoModel(
    const SpeakerEmbeddingExtractorConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

template <typename Manager>
SpeakerEmbeddingExtractorNeMoModel::SpeakerEmbeddingExtractorNeMoModel(
    Manager *mgr, const SpeakerEmbeddingExtractorConfig &config)
    : impl_(std::make_unique<Impl>(mgr, config)) {}

SpeakerEmbeddingExtractorNeMoModel::~SpeakerEmbeddingExtractorNeMoModel() =
    default;

const SpeakerEmbeddingExtractorNeMoModelMetaData &
SpeakerEmbeddingExtractorNeMoModel::GetMetaData() const {
  return impl_->GetMetaData();
}

Ort::Value SpeakerEmbeddingExtractorNeMoModel::Compute(
    Ort::Value x, Ort::Value x_lens) const {
  return impl_->Compute(std::move(x), std::move(x_lens));
}

OrtAllocator *SpeakerEmbeddingExtractorNeMoModel::Allocator() const {
  return impl_->Allocator();
}

#if __ANDROID_API__ >= 9
template SpeakerEmbeddingExtractorNeMoModel::SpeakerEmbeddingExtractorNeMoModel(
    AAssetManager *mgr, const SpeakerEmbeddingExtractorConfig &config);
#endif

#if __OHOS__
template SpeakerEmbeddingExtractorNeMoModel::SpeakerEmbeddingExtractorNeMoModel(
    NativeResourceManager *mgr, const SpeakerEmbeddingExtractorConfig &config);
#endif

}  // namespace sherpa_onnx
