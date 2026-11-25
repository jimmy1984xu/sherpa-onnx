#ifndef TRANSAI_VOXLINGUA107_RUNTIME_H_
#define TRANSAI_VOXLINGUA107_RUNTIME_H_

#include <array>
#include <cstddef>
#include <memory>
#include <string>

namespace transai::voxlingua107 {

struct Result {
  std::array<float, 107> log_probabilities{};
  std::array<float, 256> embedding{};
  size_t language_id = 0;
  std::string language;
  float probability = 0;
};

class Runtime {
 public:
  Runtime(const std::string &path, int num_threads);
  ~Runtime();
  Result Process(const float *samples, size_t count) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace transai::voxlingua107

#endif
