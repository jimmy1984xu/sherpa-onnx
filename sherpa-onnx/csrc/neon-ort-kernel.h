#ifndef SHERPA_ONNX_CSRC_NEON_ORT_KERNEL_H_
#define SHERPA_ONNX_CSRC_NEON_ORT_KERNEL_H_

#include <cstddef>
#include <exception>

#include "onnxruntime_cxx_api.h"

namespace sherpa_onnx::neon {

template <bool kPreserveErrorCode = true, typename Function>
OrtStatusPtr Guard(Function fn) noexcept {
  try {
    fn();
    return nullptr;
  } catch (const Ort::Exception& error) {
    return Ort::GetApi().CreateStatus(
        kPreserveErrorCode ? error.GetOrtErrorCode() : ORT_FAIL, error.what());
  } catch (const std::exception& error) {
    return Ort::GetApi().CreateStatus(ORT_FAIL, error.what());
  }
}

template <typename Function>
void ParallelFor(const Ort::KernelContext& context, std::size_t tasks,
                 std::size_t batches, Function fn) {
  context.ParallelFor([](void* arg, std::size_t index) {
    (*static_cast<Function*>(arg))(index);
  }, tasks, batches, &fn);
}

}  // namespace sherpa_onnx::neon
#endif  // SHERPA_ONNX_CSRC_NEON_ORT_KERNEL_H_