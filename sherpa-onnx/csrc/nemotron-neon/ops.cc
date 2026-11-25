#include <onnxruntime_cxx_api.h>

#include <arm_neon.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#include "kai/kai_common.h"
#include "pack/kai_lhs_quant_pack_qai8dxp_f32.h"
#include "pack/kai_rhs_pack_nxk_qsi8cxp_qsi8cx_neon.h"
#include "pack/kai_lhs_pack_bf16p8x4_f16_neon.h"
#include "pack/kai_rhs_pack_kxn_bf16p12x4biasf32_f16_neon.h"
#include "matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm.h"
#include "matmul_clamp_f32_bf16p_bf16p/kai_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla.h"

#include "sherpa-onnx/csrc/neon-ort-kernel.h"

namespace {
constexpr auto low = std::numeric_limits<float>::lowest();
constexpr auto high = std::numeric_limits<float>::max();

using sherpa_onnx::neon::Guard;
template <typename Function>
void parallel(const Ort::KernelContext& ctx, size_t tasks, Function fn) {
  sherpa_onnx::neon::ParallelFor(ctx, tasks, 0, fn);
}

float32x4_t exp_nonpositive(float32x4_t x);

void gelu(float* data, size_t columns) {
    for (size_t c = 0; c < columns; c += 4) {
        auto x = vld1q_f32(data + c);
        auto z = vmulq_n_f32(vmulq_f32(x, vfmaq_n_f32(vdupq_n_f32(1), vmulq_f32(x, x), 0.044715f)), 0.7978845608f);
        const auto e = exp_nonpositive(vmulq_n_f32(vabsq_f32(z), -2));
        const auto one = vdupq_n_f32(1);
        const auto tanh = vdivq_f32(vsubq_f32(one, e), vaddq_f32(one, e));
        const auto signed_tanh = vbslq_f32(vcltq_f32(z, vdupq_n_f32(0)), vnegq_f32(tanh), tanh);
        vst1q_f32(data + c, vmulq_f32(vmulq_n_f32(x, 0.5f), vaddq_f32(one, signed_tanh)));
    }
}

// The step graph has K aligned to 32 and consumes 4x8 I8MM LHS
// blocks. The final saturating narrow performs the int8 clamp.
void pack_lhs_i8mm(size_t rows, size_t width, const float* input, uint8_t* packed) {
    constexpr size_t mr = 4, kr = 8;
    for (size_t row = 0; row < rows; ++row) {
        const auto* source = input + row * width;
        auto vmax = vdupq_n_f32(0);
        auto vmin = vdupq_n_f32(0);
        for (size_t k = 0; k < width; k += 8) {
            const auto a = vld1q_f32(source + k);
            const auto b = vld1q_f32(source + k + 4);
            vmax = vmaxq_f32(vmax, vmaxq_f32(a, b));
            vmin = vminq_f32(vmin, vminq_f32(a, b));
        }
        constexpr float qmin = -128, qmax = 127;
        const float rmin = vminvq_f32(vmin), rmax = vmaxvq_f32(vmax);
        const float scale = rmin == rmax ? 1.0f : (qmax - qmin) / (rmax - rmin);
        const float scaled_min = rmin * scale, scaled_max = rmax * scale;
        const float min_error = qmin + scaled_min, max_error = qmax + scaled_max;
        const float zero = min_error + max_error > 0 ? qmin - scaled_min : qmax - scaled_max;
        const auto zero_point = static_cast<int32_t>(std::rint(std::fmin(qmax, std::fmax(qmin, zero))));
        const auto bias = vdupq_n_s16(static_cast<int16_t>(zero_point));
        const auto scale_v = vdupq_n_f32(scale);
        auto* tile = packed + (row / mr) * mr * (width + sizeof(int32_t) + sizeof(float));
        auto* dst = tile + row % mr * kr;
        for (size_t k = 0; k < width; k += kr) {
            const auto lo = vcvtaq_s32_f32(vmulq_f32(vld1q_f32(source + k), scale_v));
            const auto hi = vcvtaq_s32_f32(vmulq_f32(vld1q_f32(source + k + 4), scale_v));
            const auto combined = vqmovn_high_s32(vqmovn_s32(lo), hi);
            vst1_s8(reinterpret_cast<int8_t*>(dst), vqmovn_s16(vaddq_s16(combined, bias)));
            dst += mr * kr;
        }
        const int32_t offset = -zero_point;
        const float reciprocal = scale ? 1.0f / scale : 0.0f;
        std::memcpy(tile + mr * width + row % mr * sizeof(offset), &offset, sizeof(offset));
        std::memcpy(tile + mr * (width + sizeof(offset)) + row % mr * sizeof(reciprocal),
                    &reciprocal, sizeof(reciprocal));
    }
}

// Same packed W8A8 format and epilogue as the existing 16x4 I8MM kernel,
// with an 8x8 result tile to reduce loads for each group of 64 outputs.
struct W8RowPair {
    int32x4_t c0 = vdupq_n_s32(0), c1 = c0, c2 = c0, c3 = c0;

    void Accumulate(int8x16_t lhs, int8x16_t rhs0, int8x16_t rhs1,
                    int8x16_t rhs2, int8x16_t rhs3) {
        c0 = vmmlaq_s32(c0, lhs, rhs0);
        c1 = vmmlaq_s32(c1, lhs, rhs1);
        c2 = vmmlaq_s32(c2, lhs, rhs2);
        c3 = vmmlaq_s32(c3, lhs, rhs3);
    }
};

struct W8ColumnScale {
    int32x4_t sums;
    float32x4_t scales, biases;
};

W8ColumnScale LoadW8ColumnScale(const uint8_t* end) {
    W8ColumnScale col;
    std::memcpy(&col.sums, end, sizeof(col.sums));
    std::memcpy(&col.scales, end + 16, sizeof(col.scales));
    std::memcpy(&col.biases, end + 32, sizeof(col.biases));
    return col;
}

void StoreW8Row(int32x4_t product, const W8ColumnScale& rhs, int32_t offset,
                float scale, float* output) {
#pragma clang fp contract(off)
    const auto adjusted = vmlaq_n_s32(product, rhs.sums, offset);
    const auto point_scale = vmulq_n_f32(rhs.scales, scale);
    const auto scaled = vmulq_f32(vcvtq_f32_s32(adjusted), point_scale);
    const auto value = vaddq_f32(scaled, rhs.biases);
    vst1q_f32(output, vminq_f32(vmaxq_f32(value, vdupq_n_f32(low)), vdupq_n_f32(high)));
}

template<int Pair>
void StoreW8RowPair(const W8RowPair& accum, const uint8_t* lhs_meta,
                    const W8ColumnScale& left, const W8ColumnScale& right,
                    float* dst, size_t stride) {
    int32_t offsets[2];
    float inverse_scales[2];
    std::memcpy(offsets, lhs_meta + Pair*8, sizeof(offsets));
    std::memcpy(inverse_scales, lhs_meta + 16 + Pair*8, sizeof(inverse_scales));
    const auto a0 = vreinterpretq_s64_s32(accum.c0), a1 = vreinterpretq_s64_s32(accum.c1);
    const auto b0 = vreinterpretq_s64_s32(accum.c2), b1 = vreinterpretq_s64_s32(accum.c3);
    StoreW8Row(vreinterpretq_s32_s64(vuzp1q_s64(a0, a1)), left, offsets[0], inverse_scales[0], dst);
    StoreW8Row(vreinterpretq_s32_s64(vuzp1q_s64(b0, b1)), right, offsets[0], inverse_scales[0], dst + 4);
    StoreW8Row(vreinterpretq_s32_s64(vuzp2q_s64(a0, a1)), left, offsets[1], inverse_scales[1], dst + stride);
    StoreW8Row(vreinterpretq_s32_s64(vuzp2q_s64(b0, b1)), right, offsets[1], inverse_scales[1], dst + stride + 4);
}

void RunW8Matrix(size_t m, size_t n, size_t k, const uint8_t* lhs, const uint8_t* rhs,
                 float* dst, size_t stride) {
    constexpr size_t rows = 8, columns = 8;
    const size_t packed_depth = 4*k;
    const size_t lhs_stride = packed_depth + 32, rhs_stride = packed_depth + 48;
    const size_t full_rows = m / rows * rows, full_columns = n / columns * columns;
    auto load = [](const uint8_t* p) { return vreinterpretq_s8_u8(vld1q_u8(p)); };
    for (size_t row = 0; row < full_rows; row += rows) {
        const auto* lhs0 = lhs + row / 4 * lhs_stride;
        const auto* lhs1 = lhs0 + lhs_stride;
        for (size_t col = 0; col < full_columns; col += columns) {
            const auto* rhs0 = rhs + col / 4 * rhs_stride;
            const auto* rhs1 = rhs0 + rhs_stride;
            W8RowPair r0, r1, r2, r3;
            for (size_t d = 0; d < packed_depth; d += 32) {
                const auto b0 = load(rhs0 + d), b1 = load(rhs0 + d + 16);
                const auto b2 = load(rhs1 + d), b3 = load(rhs1 + d + 16);
                r0.Accumulate(load(lhs0 + d),      b0, b1, b2, b3);
                r1.Accumulate(load(lhs0 + d + 16), b0, b1, b2, b3);
                r2.Accumulate(load(lhs1 + d),      b0, b1, b2, b3);
                r3.Accumulate(load(lhs1 + d + 16), b0, b1, b2, b3);
            }
            const auto left = LoadW8ColumnScale(rhs0 + packed_depth);
            const auto right = LoadW8ColumnScale(rhs1 + packed_depth);
            auto* output = dst + row*stride + col;
            StoreW8RowPair<0>(r0, lhs0 + packed_depth, left, right, output, stride);
            StoreW8RowPair<1>(r1, lhs0 + packed_depth, left, right, output + 2*stride, stride);
            StoreW8RowPair<0>(r2, lhs1 + packed_depth, left, right, output + 4*stride, stride);
            StoreW8RowPair<1>(r3, lhs1 + packed_depth, left, right, output + 6*stride, stride);
        }
    }
    auto tail = [&](size_t row, size_t col, size_t rows_remaining, size_t columns_remaining) {
        kai_run_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm(
            rows_remaining, columns_remaining, k, lhs + row / 4 * lhs_stride,
            rhs + col / 4 * rhs_stride, dst + row*stride + col,
            stride*sizeof(float), sizeof(float), low, high);
    };
    if (full_columns < n && full_rows) tail(0, full_columns, full_rows, n - full_columns);
    if (full_rows < m) tail(full_rows, 0, m - full_rows, n);
}



template<bool Activate> class Gemm {
    size_t n_, k_;
    const uint8_t* packed_ = nullptr;
    std::unique_ptr<uint8_t, decltype(&std::free)> storage_{nullptr, &std::free};
    static constexpr bool column_tiles_ = true;
    size_t tile_;

    static Ort::ConstValue constant(Ort::ConstKernelInfo info, size_t index) {
        int is_constant = 0;
        auto value = info.GetTensorConstantInput(index, &is_constant);
        if (!is_constant) throw std::invalid_argument("NEON GEMM requires constant weights, scales and bias");
        return value;
    }

public:
    explicit Gemm(const OrtKernelInfo* raw) {
        const Ort::ConstKernelInfo info{raw};
        const auto weight = constant(info, 1);
        const auto shape = weight.GetTensorTypeAndShapeInfo().GetShape();
        const auto n = info.GetAttribute<int64_t>("n"), k = info.GetAttribute<int64_t>("k");
        if (n <= 0 || n % 4 || k <= 0 || k % 32 || shape.size() != 1 ||
            size_t(shape[0]) != kai_get_rhs_packed_size_rhs_pack_nxk_qsi8cxp_qsi8cx_neon(n, k, 4, 8, 1)) {
            throw std::invalid_argument("Invalid offline I8MM RHS constants");
        }
        n_ = n;
        k_ = k;
        tile_ = k_ == 2048 ? 64 : 128;
        void* aligned = nullptr;
        if (posix_memalign(&aligned, 64, size_t(shape[0])) != 0)
            throw std::bad_alloc();
        storage_.reset(static_cast<uint8_t*>(aligned));
        std::memcpy(storage_.get(), weight.GetTensorData<uint8_t>(), size_t(shape[0]));
        packed_ = storage_.get();
    }

    OrtStatusPtr ComputeV2(OrtKernelContext* raw) noexcept {
        return Guard([&] {
            const Ort::KernelContext context(raw);
            const auto input = context.GetInput(0);
            auto shape = input.GetTensorTypeAndShapeInfo().GetShape();
            if (shape.size() != 3 || shape[0] != 1 || shape[1] < 1 || size_t(shape[2]) != k_) {
                throw std::invalid_argument("Invalid NEON GEMM input");
            }
            const size_t m = shape[1];
            shape[2] = n_;
            auto output = context.GetOutput(0, shape);
            const float* x = input.GetTensorData<float>();
            float* y = output.GetTensorMutableData<float>();
            if (column_tiles_) {
                const size_t row_tile = ((m + 15) / 16) * 4;
                static thread_local std::vector<uint8_t> lhs;
                lhs.resize(kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32(m, k_, 4, 8, 1));
                auto* const packed_lhs = lhs.data();
                parallel(context, (m + row_tile - 1) / row_tile, [&](size_t task) {
                    const size_t start = task * row_tile;
                    const size_t rows = std::min(row_tile, m - start);
                    const size_t offset = kai_get_lhs_packed_offset_lhs_quant_pack_qai8dxp_f32(start, k_, 4, 8, 1);
                    pack_lhs_i8mm(rows, k_, x + start*k_, packed_lhs + offset);
                });
                const size_t tile = tile_;
                parallel(context, (n_ + tile - 1) / tile, [&](size_t task) {
                    const size_t start = task * tile;
                    const size_t columns = std::min(tile, n_ - start);
                    const size_t offset = kai_get_rhs_packed_offset_rhs_pack_nxk_qsi8cxp_qsi8cx_neon(start, k_, 4, 8, 1);
                    if (k_ == 2048) {
                        RunW8Matrix(m, columns, k_, packed_lhs, packed_ + offset, y + start, n_);
                    } else {
                        kai_run_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm(
                            m, columns, k_, packed_lhs, packed_ + offset, y + start,
                            n_*sizeof(float), sizeof(float), low, high);
                    }
                    if constexpr (Activate) for (size_t row = 0; row < m; ++row) gelu(y + row*n_ + start, columns);
                });
                return;
            }
        });
    }
};

float32x4_t exp_nonpositive(float32x4_t x) {
    x = vmaxq_f32(x, vdupq_n_f32(-80.0f));
    const auto n = vrndnq_f32(vmulq_n_f32(x, 1.44269504089f));
    const auto r = vfmsq_n_f32(x, n, 0.69314718056f);
    auto p = vfmaq_n_f32(vdupq_n_f32(1.0f / 6), r, 1.0f / 24);
    p = vfmaq_f32(vdupq_n_f32(0.5f), r, p);
    p = vfmaq_f32(vdupq_n_f32(1), r, p);
    p = vfmaq_f32(vdupq_n_f32(1), r, p);
    const auto bits = vshlq_n_s32(vaddq_s32(vcvtq_s32_f32(n), vdupq_n_s32(127)), 23);
    return vmulq_f32(p, vreinterpretq_f32_s32(bits));
}

struct AttentionWorkspace {
    std::vector<__fp16> query, value, probability;
    std::vector<uint8_t> lhs, rhs, value_rhs;
    std::vector<float> scores, zero_bias;

    void pack_rhs(size_t n, size_t k, const __fp16* b, std::vector<uint8_t>& packed) {
        packed.resize(kai_get_rhs_packed_size_rhs_pack_kxn_bf16p12x4biasf32_f16_neon(n, k));
        zero_bias.resize(n, 0.0f);
        kai_run_rhs_pack_kxn_bf16p12x4biasf32_f16_neon(1, n, k, 12, 4, 1, n*sizeof(__fp16), b,
                                                       zero_bias.data(), nullptr, packed.data(), 0, nullptr);
    }

    void multiply(size_t m, size_t n, size_t k, const __fp16* a, const std::vector<uint8_t>& packed,
                  float* output, size_t out_stride) {
        lhs.resize(kai_get_lhs_packed_size_lhs_pack_bf16p8x4_f16_neon(m, k, 8, 4, 1));
        kai_run_lhs_pack_bf16p8x4_f16_neon(m, k, 8, 4, 1, 0, a, k*sizeof(__fp16), lhs.data());
        kai_run_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla(
            m, n, k, lhs.data(), packed.data(), output, out_stride*sizeof(float), sizeof(float), low, high);
    }
};

void softmax(float* score, __fp16* prob, size_t n) {
    auto vmax = vdupq_n_f32(low);
    size_t end = n & ~size_t(3);
    for (size_t i = 0; i < end; i += 4) vmax = vmaxq_f32(vmax, vld1q_f32(score + i));
    float max = vmaxvq_f32(vmax);
    for (size_t i = end; i < n; ++i) max = std::max(max, score[i]);
    auto vsum = vdupq_n_f32(0);
    for (size_t i = 0; i < end; i += 4) {
        auto z = vmulq_n_f32(vsubq_f32(vld1q_f32(score + i), vdupq_n_f32(max)), 0.125f);
        auto e = exp_nonpositive(z);
        vsum = vaddq_f32(vsum, e);
        vst1q_f32(score + i, e);
    }
    float sum = vaddvq_f32(vsum);
    for (size_t i = end; i < n; ++i) sum += (score[i] = std::exp((score[i] - max)*0.125f));
    const float inverse = 1 / sum;
    for (size_t i = 0; i < end; i += 4) {
        vst1_f16(prob + i, vcvt_f16_f32(vmulq_n_f32(vld1q_f32(score + i), inverse)));
    }
    for (size_t i = end; i < n; ++i) prob[i] = __fp16(score[i] * inverse);
}

// Kleidiai RHS blocks hold 12 columns, with four BF16 values per
// column in each depth slice. Emit that layout directly from key rows.
constexpr size_t key_tile = 12;
template<size_t Width>
void StorePackedKeyBlock(const std::array<std::array<__fp16, Width>, key_tile>& keys,
                         uint8_t* packed) {
    constexpr size_t bias_bytes = key_tile * sizeof(float);
    std::memset(packed, 0, bias_bytes);
    packed += bias_bytes;
    for (size_t depth = 0; depth < Width; depth += 4) {
        for (size_t row = 0; row < key_tile; row += 2) {
            const auto low = vcvt_f32_f16(vld1_f16(keys[row].data() + depth));
            const auto high = vcvt_f32_f16(vld1_f16(keys[row + 1].data() + depth));
            const auto pair = vcvtq_high_bf16_f32(vcvtq_low_bf16_f32(low), high);
            std::memcpy(packed, &pair, sizeof(pair));
            packed += sizeof(pair);
        }
    }
}

template<bool Packed, bool Cached> class Attention {
    static_assert(Packed && Cached, "The Medium graph always uses packed attention with a prefix");
    const size_t tile_ = 64;
public:
    explicit Attention(const OrtKernelInfo*) {}

    OrtStatusPtr ComputeV2(OrtKernelContext* raw) noexcept {
        return Guard([&] {
            constexpr size_t heads = 8, depth = 64, hidden = heads*depth;
            const Ort::KernelContext context(raw);
            const auto q = context.GetInput(0), k = context.GetInput(1), v = context.GetInput(2);
            auto shape = q.GetTensorTypeAndShapeInfo().GetShape();
            if (shape.size() != 3 || shape[0] != 1 || shape[1] < 1 || shape[2] != (Packed ? 3*hidden : hidden)) {
                throw std::invalid_argument("Invalid NEON attention shape");
            }
            const size_t t = shape[1];
            size_t prefix = 0, capture = 0;
            const __fp16* saved = nullptr;
            __fp16* next = nullptr;
            if constexpr (Cached) {
                const auto past = context.GetInput(3), requested = context.GetInput(4);
                const auto dims = past.GetTensorTypeAndShapeInfo().GetShape();
                const auto keep = *requested.GetTensorData<int64_t>();
                if (dims.size() != 3 || dims[0] != 1 || dims[1] < 0 || dims[2] != hidden*2 ||
                    keep < 0 || keep > int64_t(t) || (dims[1] > 0 && keep > 0)) {
                    throw std::invalid_argument("Invalid explicit speaker-prefix state");
                }
                prefix = dims[1];
                capture = keep;
                saved = reinterpret_cast<const __fp16*>(past.GetTensorRawData());
                const std::array<int64_t, 3> next_shape{1, keep, hidden*2};
                next = reinterpret_cast<__fp16*>(context.GetOutput(1, next_shape.data(), next_shape.size()).GetTensorMutableRawData());
            }
            const size_t total = prefix + t;
            if constexpr (Packed) {
                for (const auto& table : {k, v}) {
                    const auto dims = table.GetTensorTypeAndShapeInfo().GetShape();
                    if (dims.size() != 2 || dims[0] < int64_t(total) || dims[1] != depth/2)
                        throw std::invalid_argument("Invalid rotary table");
                }
            }
            shape[2] = hidden;
            auto output = context.GetOutput(0, shape);
            const float* qp = q.GetTensorData<float>();
            const float* kp = k.GetTensorData<float>();
            const float* vp = v.GetTensorData<float>();
            float* y = output.GetTensorMutableData<float>();
            parallel(context, heads, [&](size_t head) {
                static thread_local AttentionWorkspace work;
                work.query.resize(t*depth);
                work.value.resize(total*depth);
                work.probability.resize(std::min(tile_, t)*total);
                work.scores.resize(std::min(tile_, t)*total);
                const size_t offset = head*depth;
                constexpr size_t key_tile_bytes = key_tile * (sizeof(float) + depth * sizeof(__fp16));
                work.rhs.resize(kai_get_rhs_packed_size_rhs_pack_kxn_bf16p12x4biasf32_f16_neon(total, depth));
                std::array<std::array<__fp16, depth>, key_tile> key_block;
                auto flush_key = [&](size_t position) {
                    if (position % key_tile == key_tile - 1) {
                        StorePackedKeyBlock(key_block, work.rhs.data() + (position / key_tile) * key_tile_bytes);
                    }
                };
                for (size_t row = 0; row < prefix; ++row) {
                    const auto* key = saved + row*2*hidden + offset;
                    std::copy_n(key + hidden, depth, work.value.data() + row*depth);
                    std::copy_n(key, depth, key_block[row % key_tile].data());
                    flush_key(row);
                }
                for (size_t row = 0; row < t; ++row) {
                    const size_t position = prefix + row;
                    if constexpr (Packed) {
                        const float* qrow = qp + row*3*hidden + offset;
                        const float* krow = qrow + hidden;
                        auto& key = key_block[position % key_tile];
                        for (size_t i = 0; i < depth/2; i += 4) {
                            const auto c = vld1q_f32(kp + position*(depth/2) + i);
                            const auto s = vld1q_f32(vp + position*(depth/2) + i);
                            const auto rotate = [&](const float* input, __fp16* dst) {
                                const auto a = vld1q_f32(input + i), b = vld1q_f32(input + i + depth/2);
                                vst1_f16(dst + i, vcvt_f16_f32(vfmsq_f32(vmulq_f32(a, c), b, s)));
                                vst1_f16(dst + i + depth/2, vcvt_f16_f32(vfmaq_f32(vmulq_f32(b, c), a, s)));
                            };
                            rotate(qrow, work.query.data() + row*depth);
                            rotate(krow, key.data());
                        }
                        for (size_t i = 0; i < depth; i += 4)
                            vst1_f16(work.value.data() + position*depth + i, vcvt_f16_f32(vld1q_f32(qrow + 2*hidden + i)));
                        if (row < capture) {
                            auto* output_key = next + row*2*hidden + offset;
                            std::copy_n(key.data(), depth, output_key);
                            std::copy_n(work.value.data() + position*depth, depth, output_key + hidden);
                        }
                    }
                    flush_key(position);
                }
                if (total % key_tile) {
                    for (size_t row = total % key_tile; row < key_tile; ++row) key_block[row].fill(0);
                    StorePackedKeyBlock(key_block, work.rhs.data() + (total / key_tile) * key_tile_bytes);
                }
                work.pack_rhs(depth, total, work.value.data(), work.value_rhs);
                for (size_t first = 0; first < t; first += tile_) {
                    const size_t rows = std::min(tile_, t - first);
                    work.multiply(rows, total, depth, work.query.data() + first*depth, work.rhs, work.scores.data(), total);
                    for (size_t row = 0; row < rows; ++row)
                        softmax(work.scores.data() + row*total, work.probability.data() + row*total, total);
                    work.multiply(rows, depth, total, work.probability.data(), work.value_rhs, y + first*hidden + offset, hidden);
                }
            });
        });
    }
};

class SkipNorm {
    const float epsilon_;

public:
    explicit SkipNorm(const OrtKernelInfo* raw) : epsilon_(Ort::ConstKernelInfo{raw}.GetAttribute<float>("epsilon")) {}

    OrtStatusPtr ComputeV2(OrtKernelContext* raw) noexcept {
        return Guard([&] {
            constexpr size_t width = 512;
            const Ort::KernelContext context(raw);
            const auto x = context.GetInput(0), skip = context.GetInput(1);
            const auto shape = x.GetTensorTypeAndShapeInfo().GetShape();
            if (shape.size() != 3 || shape[0] != 1 || shape[1] < 1 || shape[2] != width ||
                skip.GetTensorTypeAndShapeInfo().GetShape() != shape) throw std::invalid_argument("Invalid skip norm input");
            const auto scale = context.GetInput(2), bias = context.GetInput(3);
            if (scale.GetTensorTypeAndShapeInfo().GetElementCount() != width || bias.GetTensorTypeAndShapeInfo().GetElementCount() != width)
                throw std::invalid_argument("Invalid skip norm scale or bias");
            const float* input = x.GetTensorData<float>();
            const float* residual = skip.GetTensorData<float>();
            const float* weight = scale.GetTensorData<float>();
            const float* shift = bias.GetTensorData<float>();
            float* out = context.GetOutput(0, shape).GetTensorMutableData<float>();
            float* sum = context.GetOutput(1, shape).GetTensorMutableData<float>();
            const size_t rows = shape[1];
            constexpr size_t tile = 16;
            parallel(context, (rows + tile - 1) / tile, [&](size_t task) {
                for (size_t row = task*tile; row < std::min(rows, (task+1)*tile); ++row) {
                    const size_t begin = row*width;
                    auto mean = vdupq_n_f32(0), squares = vdupq_n_f32(0);
                    for (size_t col = 0; col < width; col += 4) {
                        const auto a = vaddq_f32(vld1q_f32(input + begin + col), vld1q_f32(residual + begin + col));
                        mean = vaddq_f32(mean, a);
                        squares = vfmaq_f32(squares, a, a);
                        vst1q_f32(sum + begin + col, a);
                    }
                    const float mu = vaddvq_f32(mean) / width;
                    const float variance = std::max(0.0f, vaddvq_f32(squares) / width - mu*mu);
                    const float inverse = 1 / std::sqrt(variance + epsilon_);
                    for (size_t col = 0; col < width; col += 4) {
                        const auto centered = vsubq_f32(vld1q_f32(sum + begin + col), vdupq_n_f32(mu));
                        const auto scaled = vmulq_n_f32(centered, inverse);
                        vst1q_f32(out + begin + col, vfmaq_f32(vld1q_f32(shift + col), scaled, vld1q_f32(weight + col)));
                    }
                }
            });
        });
    }
};

template<class K, size_t Inputs, bool Quantized = false, size_t Outputs = 1>
struct Op : Ort::CustomOpBase<Op<K, Inputs, Quantized, Outputs>, K, true> {
    const char* name;
    explicit Op(const char* op_name) : name(op_name) {}
    const char* GetName() const { return name; }
    size_t GetInputTypeCount() const { return Inputs; }
    ONNXTensorElementDataType GetInputType(size_t index) const {
        if constexpr (Inputs == 5) {
            if (index == 3) return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
            if (index == 4) return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
        }
        return Quantized && index == 1 ? ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 : ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }
    size_t GetOutputTypeCount() const { return Outputs; }
    ONNXTensorElementDataType GetOutputType(size_t index) const {
        return Inputs == 5 && index == 1 ? ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 : ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }
    OrtStatusPtr CreateKernelV2(const OrtApi&, const OrtKernelInfo* info, void** kernel) const noexcept {
        return Guard([&] { *kernel = new K(info); });
    }
};
}  // namespace

namespace transai::nemotron {
void RegisterOps(Ort::SessionOptions &options) {
    static const Op<Gemm<false>, 2, true> gemm("NeonW8Gemm");
    static const Op<Gemm<true>, 2, true> gelu("NeonW8Gelu");
    static const Op<Attention<true, true>, 5, false, 2> attention("NeonCachedAttention");
    static const Op<SkipNorm, 4, false, 2> norm("NeonSkipNorm");
    static Ort::CustomOpDomain domain = [] {
        Ort::CustomOpDomain registered("transai.neon");
        registered.Add(&gemm);
        registered.Add(&gelu);
        registered.Add(&attention);
        registered.Add(&norm);
        return registered;
    }();
    options.Add(domain);
}
}  // namespace transai::nemotron
