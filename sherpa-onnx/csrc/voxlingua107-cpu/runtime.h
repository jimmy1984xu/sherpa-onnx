#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <string>

namespace transai::voxlingua107::cpu {

struct Result {
  std::array<float, 107> log_probabilities{};
  std::array<float, 256> embedding{};
  size_t language_id = 0;
  std::string language;
  float probability = 0.f;
};

class Runtime {
 public:
  Runtime(const std::string &model_onnx, const std::string &lang_map_json, int num_threads);
  ~Runtime();

  Result Process(const float *samples, size_t count) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace transai::voxlingua107::cpu