// Shared arithmetic for the Piper unsigned I8MM convolutions.
inline void ScanU8Channels(const float* x, int channels, int frames,
                           float& minimum, float& maximum) {
  for (int channel = 0; channel < channels; ++channel) {
    const auto* data = x + size_t(channel) * frames;
    auto low_vec = vdupq_n_f32(0);
    auto high_vec = low_vec;
    int t = 0;
    for (; t + 4 <= frames; t += 4) {
      auto value = vld1q_f32(data + t);
      low_vec = vminq_f32(low_vec, value);
      high_vec = vmaxq_f32(high_vec, value);
    }
    float low = vminvq_f32(low_vec);
    float high = vmaxvq_f32(high_vec);
    for (; t < frames; ++t) {
      low = std::min(low, data[t]);
      high = std::max(high, data[t]);
    }
    minimum = std::min(minimum, low);
    maximum = std::max(maximum, high);
  }
}

inline void ScanU8Block(const float* x, size_t begin, size_t end,
                        float& low, float& high) {
  float32x4_t lo0 = vdupq_n_f32(0), lo1 = lo0, lo2 = lo0, lo3 = lo0;
  float32x4_t hi0 = lo0, hi1 = lo0, hi2 = lo0, hi3 = lo0;
  size_t j = begin;
  for (; j + 16 <= end; j += 16) {
    auto a = vld1q_f32(x + j), b = vld1q_f32(x + j + 4);
    auto c = vld1q_f32(x + j + 8), d = vld1q_f32(x + j + 12);
    lo0 = vminq_f32(lo0, a); lo1 = vminq_f32(lo1, b);
    lo2 = vminq_f32(lo2, c); lo3 = vminq_f32(lo3, d);
    hi0 = vmaxq_f32(hi0, a); hi1 = vmaxq_f32(hi1, b);
    hi2 = vmaxq_f32(hi2, c); hi3 = vmaxq_f32(hi3, d);
  }
  low = vminvq_f32(vminq_f32(vminq_f32(lo0, lo1), vminq_f32(lo2, lo3)));
  high = vmaxvq_f32(vmaxq_f32(vmaxq_f32(hi0, hi1), vmaxq_f32(hi2, hi3)));
  for (; j < end; ++j) {
    low = std::min(low, x[j]);
    high = std::max(high, x[j]);
  }
}

inline void QuantizeU8Activation(const Ort::KernelContext& ctx,
                                 const float* x, size_t count, float minimum,
                                 float maximum, uint8_t* output, float& scale,
                                 int32_t& zero) {
  scale = maximum == minimum ? 1.f : (maximum - minimum) / 255.f;
  zero = std::max(0, std::min(255, int32_t(std::nearbyint(-minimum / scale))));
  const float inverse = 1.f / scale;
  parallel(ctx, 4, [&](size_t task) {
    size_t i = count * task / 4;
    const size_t end = count * (task + 1) / 4;
    const auto zero_vec = vdupq_n_s32(zero);
    for (; i + 8 <= end; i += 8) {
      auto a = vcvtnq_s32_f32(vmulq_n_f32(vld1q_f32(x + i), inverse));
      auto b = vcvtnq_s32_f32(vmulq_n_f32(vld1q_f32(x + i + 4), inverse));
      a = vaddq_s32(a, zero_vec);
      b = vaddq_s32(b, zero_vec);
      auto u16 = vcombine_u16(vqmovun_s32(a), vqmovun_s32(b));
      vst1_u8(output + i, vqmovn_u16(u16));
    }
    for (; i < end; ++i) {
      output[i] = std::max(0, std::min(255, int(std::nearbyint(x[i] * inverse)) + zero));
    }
  });
}

template <typename Load>
inline void PackU8Tile(Load load, int row, int column, uint8_t* output,
                       uint8x8_t (&packed)[8]) {
  const auto t0 = vtrn_u8(load(column, row), load(column + 1, row));
  const auto t1 = vtrn_u8(load(column + 2, row), load(column + 3, row));
  const auto t2 = vtrn_u8(load(column + 4, row), load(column + 5, row));
  const auto t3 = vtrn_u8(load(column + 6, row), load(column + 7, row));
  const auto u0 = vtrn_u16(vreinterpret_u16_u8(t0.val[0]), vreinterpret_u16_u8(t1.val[0]));
  const auto u1 = vtrn_u16(vreinterpret_u16_u8(t0.val[1]), vreinterpret_u16_u8(t1.val[1]));
  const auto u2 = vtrn_u16(vreinterpret_u16_u8(t2.val[0]), vreinterpret_u16_u8(t3.val[0]));
  const auto u3 = vtrn_u16(vreinterpret_u16_u8(t2.val[1]), vreinterpret_u16_u8(t3.val[1]));
  const auto v0 = vtrn_u32(vreinterpret_u32_u16(u0.val[0]), vreinterpret_u32_u16(u2.val[0]));
  const auto v1 = vtrn_u32(vreinterpret_u32_u16(u1.val[0]), vreinterpret_u32_u16(u3.val[0]));
  const auto v2 = vtrn_u32(vreinterpret_u32_u16(u0.val[1]), vreinterpret_u32_u16(u2.val[1]));
  const auto v3 = vtrn_u32(vreinterpret_u32_u16(u1.val[1]), vreinterpret_u32_u16(u3.val[1]));
  packed[0] = vreinterpret_u8_u32(v0.val[0]);
  packed[1] = vreinterpret_u8_u32(v1.val[0]);
  packed[2] = vreinterpret_u8_u32(v2.val[0]);
  packed[3] = vreinterpret_u8_u32(v3.val[0]);
  packed[4] = vreinterpret_u8_u32(v0.val[1]);
  packed[5] = vreinterpret_u8_u32(v1.val[1]);
  packed[6] = vreinterpret_u8_u32(v2.val[1]);
  packed[7] = vreinterpret_u8_u32(v3.val[1]);
  for (int lane = 0; lane < 8; ++lane) {
    vst1_u8(output + lane * 8, packed[lane]);
  }
}

inline void AccumulateU8Tile(const uint8_t* lhs, const uint8_t* rhs,
                             int columns, uint32x4_t (&acc)[16]) {
  for (auto& value : acc) value = vdupq_n_u32(0);
  for (int column = 0; column < columns; column += 8) {
    const auto a0 = vld1q_u8(lhs);
    const auto a1 = vld1q_u8(lhs + 16);
    const auto a2 = vld1q_u8(lhs + 32);
    const auto a3 = vld1q_u8(lhs + 48);
    const auto b0 = vld1q_u8(rhs);
    const auto b1 = vld1q_u8(rhs + 16);
    const auto b2 = vld1q_u8(rhs + 32);
    const auto b3 = vld1q_u8(rhs + 48);
    acc[0] = vmmlaq_u32(acc[0], a0, b0);
    acc[1] = vmmlaq_u32(acc[1], a0, b1);
    acc[2] = vmmlaq_u32(acc[2], a0, b2);
    acc[3] = vmmlaq_u32(acc[3], a0, b3);
    acc[4] = vmmlaq_u32(acc[4], a1, b0);
    acc[5] = vmmlaq_u32(acc[5], a1, b1);
    acc[6] = vmmlaq_u32(acc[6], a1, b2);
    acc[7] = vmmlaq_u32(acc[7], a1, b3);
    acc[8] = vmmlaq_u32(acc[8], a2, b0);
    acc[9] = vmmlaq_u32(acc[9], a2, b1);
    acc[10] = vmmlaq_u32(acc[10], a2, b2);
    acc[11] = vmmlaq_u32(acc[11], a2, b3);
    acc[12] = vmmlaq_u32(acc[12], a3, b0);
    acc[13] = vmmlaq_u32(acc[13], a3, b1);
    acc[14] = vmmlaq_u32(acc[14], a3, b2);
    acc[15] = vmmlaq_u32(acc[15], a3, b3);
    lhs += 64;
    rhs += 64;
  }
}

inline void EmitU8Tile(const uint32x4_t (&acc)[16], int columns, int rows, int width,
                       int row, int col, int start, int output_frames,
                       const int32_t* row_sums, const int32_t* column_sums,
                       const uint8_t* zeros, const float* scales, const float* bias,
                       int32_t input_zero, float input_scale, float* output) {
  uint32_t rawsum[16][4];
  for (int i = 0; i < 16; ++i) vst1q_u32(rawsum[i], acc[i]);
  for (int c = 0; c < std::min(8, columns - col); ++c) {
    int32_t dots[8];
    for (int r = 0; r < 8; ++r) {
      dots[r] = int32_t(rawsum[(r / 2) * 4 + c / 2][(r % 2) * 2 + c % 2]);
    }
    const int channel = col + c;
    const int32_t zero = zeros[channel];
    const int32_t correction = -input_zero * column_sums[channel] +
                               width * input_zero * zero;
    const float scale = scales[channel];
    auto* dst = output + size_t(channel) * output_frames + start + row;
    int r = 0;
    for (; r + 4 <= 8 && row + r + 4 <= rows; r += 4) {
      auto sum = vsubq_s32(vaddq_s32(vld1q_s32(dots + r), vdupq_n_s32(correction)),
                           vmulq_n_s32(vld1q_s32(row_sums + row + r), zero));
      auto value = vmulq_n_f32(vcvtq_f32_s32(sum), scale);
      value = vaddq_f32(vmulq_n_f32(value, input_scale), vdupq_n_f32(bias[channel]));
      vst1q_f32(dst + r, value);
    }
    for (; r < 8 && row + r < rows; ++r) {
      const int32_t sum = dots[r] + correction - zero * row_sums[row + r];
      const float value = float(sum) * scale;
      dst[r] = value * input_scale + bias[channel];
    }
  }
}

inline void EmitU8TileInterleaved(const uint32x4_t (&acc)[16], int n, int rows, int kp,
                                  int r, int col, int start, int tout, const int32_t* sumx,
                                  const int32_t* sums, const uint8_t* zeros, const float* scales,
                                  const float* bias, int32_t xz, float xs, float* y) {
  auto emit_pair=[&](int pair){
   auto a0=vreinterpretq_s32_u32(acc[pair]),a1=vreinterpretq_s32_u32(acc[4+pair]);
   auto a2=vreinterpretq_s32_u32(acc[8+pair]),a3=vreinterpretq_s32_u32(acc[12+pair]);
   auto write=[&](int cc,int32x4_t d0,int32x4_t d1){if(col+cc>=n)return;
    int32_t z=zeros[col+cc],correction=-xz*sums[col+cc]+kp*xz*z;float scale=scales[col+cc];
    auto s0=vsubq_s32(vaddq_s32(d0,vdupq_n_s32(correction)),vmulq_n_s32(vld1q_s32(sumx+r),z));
    auto s1=vsubq_s32(vaddq_s32(d1,vdupq_n_s32(correction)),vmulq_n_s32(vld1q_s32(sumx+r+4),z));
    auto f0=vaddq_f32(vmulq_n_f32(vmulq_n_f32(vcvtq_f32_s32(s0),scale),xs),vdupq_n_f32(bias[col+cc]));
    auto f1=vaddq_f32(vmulq_n_f32(vmulq_n_f32(vcvtq_f32_s32(s1),scale),xs),vdupq_n_f32(bias[col+cc]));
    auto*dst=y+size_t(col+cc)*tout+start+r;
    if(r+8<=rows){vst1q_f32(dst,f0);vst1q_f32(dst+4,f1);}else{float values[8];vst1q_f32(values,f0);vst1q_f32(values+4,f1);std::copy_n(values,rows-r,dst);}
   };
   write(pair*2,vuzp1q_s32(a0,a1),vuzp1q_s32(a2,a3));
   write(pair*2+1,vuzp2q_s32(a0,a1),vuzp2q_s32(a2,a3));
  };
  emit_pair(0);emit_pair(1);emit_pair(2);emit_pair(3);
}
