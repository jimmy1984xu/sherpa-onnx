#ifndef SHERPA_ONNX_CSRC_NEON_THREADS_H_
#define SHERPA_ONNX_CSRC_NEON_THREADS_H_

#include <cstdint>

namespace sherpa_onnx::neon {

inline constexpr bool ValidNumThreads(int32_t num_threads) {
  return num_threads >= 0 && num_threads <= 8;
}

inline constexpr int32_t ResolveNumThreads(int32_t num_threads) {
  return num_threads == 0 ? 1 : num_threads;
}

}  // namespace sherpa_onnx::neon

#endif  // SHERPA_ONNX_CSRC_NEON_THREADS_H_
