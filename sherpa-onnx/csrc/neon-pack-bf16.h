#pragma once

#include <arm_neon.h>

#include <cstddef>
#include <cstdint>

namespace sherpa_onnx {

// Pack FP32 directly into the KleidiAI BF16 layout (MR=8, KR=4).
// The caller supplies a K dimension padded to a multiple of four.
inline void PackBf16F32(int m, int k, const float* x, uint8_t* packed) {
  auto* out = reinterpret_cast<uint16_t*>(packed);
  for (int block = 0; block < m; block += 8) {
    for (int column = 0; column < k; column += 4) {
      for (int row = 0; row < 8; ++row) {
        const auto value = block + row < m
            ? vld1q_f32(x + std::size_t(block + row) * k + column)
            : vdupq_n_f32(0);
        const auto bf16 = vreinterpret_u16_bf16(vcvt_bf16_f32(value));
        vst1_u16(out + std::size_t(block) * k + std::size_t(column) * 8 + row * 4, bf16);
      }
    }
  }
}

}  // namespace sherpa_onnx

