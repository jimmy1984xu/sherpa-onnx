#include "runtime.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include "onnxruntime_cxx_api.h"
#include "sherpa-onnx/csrc/mali-g720/aot-file.h"
#include "sherpa-onnx/csrc/neon-ort-session.h"

namespace transai::voxlingua107 {
void RegisterOps(Ort::SessionOptions& options);

struct Runtime::Impl {
  Impl(const std::string &path, int num_threads) {
    const auto file = aot::File::Open(path);
    const auto &manifest = file->manifest();
    const auto &architecture = manifest.at("architecture");
    const auto &aot = manifest.at("aot");
    const nlohmann::json target = {{"provider", "neon"}, {"abi", "arm64-v8a"},
                                   {"isa", "i8mm"}, {"sample_rate", 16000},
                                   {"classes", 107}, {"embedding_dim", 256}};
    if (architecture.at("family") != "voxlingua107" ||
        architecture.at("version") != 1 || architecture.at("precision") != "w8a8-fp32" ||
        aot.at("format") != 1 || !aot.at("programs").empty() || aot.at("target") != target ||
        file->Asset("language/model.onnx").bytes > 256 * 1024 * 1024) {
      throw std::runtime_error("Expected a VoxLingua107 NEON v1.0 AOT");
    }
    const auto language_map = nlohmann::json::parse(file->Text("language/lang_map.json"));
    if (language_map.size() != codes.size()) {
      throw std::runtime_error("Invalid VoxLingua107 language map");
    }
    for (size_t i = 0; i < codes.size(); ++i) {
      codes[i] = language_map.at(std::to_string(i)).at("iso").get<std::string>();
    }
    Ort::SessionOptions options;
    options.SetIntraOpNumThreads(num_threads);
    options.SetInterOpNumThreads(1);
    options.SetExecutionMode(ORT_SEQUENTIAL);
    options.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
    RegisterOps(options);
    aot::RequireMappedOrtGraph(*file);
    aot::SetMappedOrtGraphOptions(options);
    model_mapping = std::make_unique<aot::Mapping>(path + "/language/model.onnx");
    session = Ort::Session(env, model_mapping->data(), model_mapping->size(), options);
  }

  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "voxlingua107"};
  // Keep read-only ORT initializers mapped until after session destruction.
  std::unique_ptr<aot::Mapping> model_mapping;
  Ort::Session session{nullptr};
  Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  std::array<std::string, 107> codes;
};

Runtime::Runtime(const std::string &path, int num_threads)
    : impl_(std::make_unique<Impl>(path, num_threads)) {}
Runtime::~Runtime() = default;

Result Runtime::Process(const float *samples, size_t count) const {
  if (!samples || count == 0 || count > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
    throw std::invalid_argument("Invalid VoxLingua107 sample count");
  }
  Result result;
  result.language_id = 0;
  const int64_t input_shape[] = {1, static_cast<int64_t>(count)};
  const int64_t logits_shape[] = {1, static_cast<int64_t>(result.log_probabilities.size())};
  const int64_t embedding_shape[] = {1, static_cast<int64_t>(result.embedding.size())};
  auto input = Ort::Value::CreateTensor<float>(
      impl_->memory, const_cast<float *>(samples), count, input_shape, 2);
  Ort::Value outputs[] = {
      Ort::Value::CreateTensor<float>(impl_->memory, result.log_probabilities.data(),
          result.log_probabilities.size(), logits_shape, 2),
      Ort::Value::CreateTensor<float>(impl_->memory, result.embedding.data(),
          result.embedding.size(), embedding_shape, 2),
  };
  const char *inputs[] = {"audio"};
  const char *names[] = {"logits", "embedding"};
  impl_->session.Run(Ort::RunOptions{nullptr}, inputs, &input, 1, names, outputs, 2);
  float best_log_probability = -std::numeric_limits<float>::infinity();
  for (size_t i = 0; i < result.log_probabilities.size(); ++i) {
    const auto value = result.log_probabilities[i];
    if (!std::isfinite(value)) throw std::runtime_error("Non-finite VoxLingua107 log probability");
    if (value > best_log_probability) {
      best_log_probability = value;
      result.language_id = i;
    }
  }
  for (const auto value : result.embedding) {
    if (!std::isfinite(value)) throw std::runtime_error("Non-finite VoxLingua107 embedding");
  }
  result.language = impl_->codes[result.language_id];
  result.probability = std::exp(result.log_probabilities[result.language_id]);
  return result;
}

}  // namespace transai::voxlingua107
