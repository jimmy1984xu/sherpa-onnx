#include "runtime.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "onnxruntime_cxx_api.h"
#include "nlohmann/json.hpp"

namespace transai::voxlingua107::cpu {
namespace {

constexpr int kClasses = 107;
constexpr int kEmbeddingDim = 256;
constexpr float kLogFloor = 1e-30f;

std::string ReadFile(const std::string &path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("Failed to read file: " + path);
  std::ostringstream buffer;
  buffer << in.rdbuf();
  return buffer.str();
}

std::array<std::string, kClasses> LoadLanguageCodes(const std::string &lang_map_path) {
  const auto text = ReadFile(lang_map_path);
  const auto json = nlohmann::json::parse(text);
  std::array<std::string, kClasses> codes{};
  for (int i = 0; i < kClasses; ++i) {
    const auto key = std::to_string(i);
    if (!json.contains(key) || !json.at(key).contains("iso")) {
      throw std::runtime_error("Invalid VoxLingua107 lang_map at index " + key);
    }
    codes[static_cast<size_t>(i)] = json.at(key).at("iso").get<std::string>();
  }
  return codes;
}

std::vector<float> SoftmaxLogits(const float *logits, size_t count) {
  if (count == 0) return {};
  float max_logit = logits[0];
  for (size_t i = 1; i < count; ++i) max_logit = std::max(max_logit, logits[i]);
  std::vector<float> probs(count, 0.f);
  double sum = 0.0;
  for (size_t i = 0; i < count; ++i) {
    const double value = std::exp(static_cast<double>(logits[i] - max_logit));
    probs[i] = static_cast<float>(value);
    sum += value;
  }
  if (sum <= std::numeric_limits<double>::min()) return probs;
  for (float &p : probs) p = static_cast<float>(p / sum);
  return probs;
}

}  // namespace

struct Runtime::Impl {
  explicit Impl(const std::string &model_onnx, const std::string &lang_map_json, int num_threads)
      : codes(LoadLanguageCodes(lang_map_json)) {
    const int threads = std::max(1, num_threads);
    Ort::SessionOptions options;
    options.SetIntraOpNumThreads(threads);
    options.SetInterOpNumThreads(1);
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session = std::make_unique<Ort::Session>(env, model_onnx.c_str(), options);
    Ort::AllocatorWithDefaultOptions allocator;
    const size_t input_count = session->GetInputCount();
    const size_t output_count = session->GetOutputCount();
    if (input_count < 1 || output_count < 1) {
      throw std::runtime_error("VoxLingua107 CPU model missing IO tensors");
    }
    for (size_t i = 0; i < input_count; ++i) {
      input_names.push_back(session->GetInputNameAllocated(i, allocator).get());
    }
    for (size_t i = 0; i < output_count; ++i) {
      output_names.push_back(session->GetOutputNameAllocated(i, allocator).get());
    }
  }

  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "voxlingua107_cpu"};
  std::unique_ptr<Ort::Session> session;
  std::vector<std::string> input_names;
  std::vector<std::string> output_names;
  std::array<std::string, kClasses> codes;
};

Runtime::Runtime(const std::string &model_onnx, const std::string &lang_map_json, int num_threads)
    : impl_(std::make_unique<Impl>(model_onnx, lang_map_json, num_threads)) {}

Runtime::~Runtime() = default;

Result Runtime::Process(const float *samples, size_t count) const {
  if (!samples || count == 0 || count > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
    throw std::invalid_argument("Invalid VoxLingua107 sample count");
  }
  auto &s = *impl_;
  Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  const std::array<int64_t, 2> input_shape{1, static_cast<int64_t>(count)};
  Ort::Value input = Ort::Value::CreateTensor<float>(
      memory, const_cast<float *>(samples), count, input_shape.data(), input_shape.size());
  std::vector<const char *> input_name_ptrs;
  input_name_ptrs.reserve(s.input_names.size());
  for (const auto &name : s.input_names) input_name_ptrs.push_back(name.c_str());
  std::vector<const char *> output_name_ptrs;
  output_name_ptrs.reserve(s.output_names.size());
  for (const auto &name : s.output_names) output_name_ptrs.push_back(name.c_str());
  auto outputs = s.session->Run(Ort::RunOptions{nullptr}, input_name_ptrs.data(), &input,
                                input_name_ptrs.size(), output_name_ptrs.data(),
                                output_name_ptrs.size());
  if (outputs.empty() || !outputs.front().IsTensor()) {
    throw std::runtime_error("VoxLingua107 CPU run returned empty logits");
  }
  const auto &info = outputs.front().GetTensorTypeAndShapeInfo();
  const auto shape = info.GetShape();
  if (shape.size() != 2 || shape[0] != 1 || shape[1] <= 0) {
    throw std::runtime_error("VoxLingua107 CPU unexpected logits shape");
  }
  const size_t logit_count = static_cast<size_t>(shape[1]);
  const float *logits = outputs.front().GetTensorData<float>();
  const auto probs = SoftmaxLogits(logits, logit_count);
  Result result;
  const size_t class_count = std::min(logit_count, static_cast<size_t>(kClasses));
  size_t best = 0;
  float best_prob = -1.f;
  for (size_t i = 0; i < class_count; ++i) {
    const float p = probs[i];
    result.log_probabilities[i] = std::log(std::max(p, kLogFloor));
    if (p > best_prob) {
      best_prob = p;
      best = i;
    }
  }
  result.language_id = best;
  result.language = s.codes[best];
  result.probability = best_prob;

  if (outputs.size() > 1 && outputs[1].IsTensor()) {
    const auto embed_shape = outputs[1].GetTensorTypeAndShapeInfo().GetShape();
    const size_t embed_count = embed_shape.size() == 2 && embed_shape[0] == 1
                                   ? static_cast<size_t>(embed_shape[1])
                                   : outputs[1].GetTensorTypeAndShapeInfo().GetElementCount();
    const float *embed = outputs[1].GetTensorData<float>();
    const size_t n = std::min(embed_count, static_cast<size_t>(kEmbeddingDim));
    std::copy_n(embed, n, result.embedding.begin());
  }
  return result;
}

}  // namespace transai::voxlingua107::cpu