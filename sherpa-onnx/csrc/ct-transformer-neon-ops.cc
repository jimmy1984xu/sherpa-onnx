#include "onnxruntime_cxx_api.h"

#include <arm_neon.h>
#include <asm/hwcap.h>
#include <sys/auxv.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm.h"
#include "pack/kai_lhs_quant_pack_qai8dxp_f32.h"
#include "pack/kai_rhs_pack_nxk_qsi8cxp_qsi8cx_neon.h"

namespace {
constexpr size_t kWidth = 256;
constexpr size_t kHeads = 8;
constexpr size_t kHeadWidth = kWidth / kHeads;
constexpr size_t kQkv = 3 * kWidth;
constexpr size_t kFfn = 1024;
constexpr size_t kTaps = 11;
constexpr size_t kTasks = 4;
constexpr float kEpsilon = 1e-12f;
constexpr float kAttentionScale = 0.1767766922712326f;

template <class F>
OrtStatusPtr Guard(F fn) noexcept {
  try {
    fn();
    return nullptr;
  } catch (const Ort::Exception &error) {
    return Ort::GetApi().CreateStatus(error.GetOrtErrorCode(), error.what());
  } catch (const std::exception &error) {
    return Ort::GetApi().CreateStatus(ORT_FAIL, error.what());
  }
}

template <class F>
void Parallel(const Ort::KernelContext &context, size_t count, F fn) {
  context.ParallelFor([](void *opaque, size_t index) {
    (*static_cast<F *>(opaque))(index);
  }, count, count, &fn);
}

template <class F>
void Rows(const Ort::KernelContext &context, size_t rows, F fn) {
  const size_t tasks = std::min(rows, kTasks);
  Parallel(context, tasks, [&](size_t task) {
    fn(rows * task / tasks, rows * (task + 1) / tasks, task);
  });
}

Ort::ConstValue Constant(const Ort::ConstKernelInfo &info, size_t index, size_t count) {
  int is_constant = 0;
  auto value = info.GetTensorConstantInput(index, &is_constant);
  if (!is_constant || value.GetTensorTypeAndShapeInfo().GetElementCount() != count) {
    throw std::invalid_argument("Unexpected CT-Transformer constant shape");
  }
  return value;
}

template <size_t N>
std::array<float, N> Read(const Ort::ConstKernelInfo &info, size_t index) {
  const auto value = Constant(info, index, N);
  std::array<float, N> result;
  std::copy_n(value.GetTensorData<float>(), N, result.data());
  return result;
}

class Gemm {
 public:
  Gemm(const Ort::ConstKernelInfo &info, size_t index, size_t n, size_t k)
      : n_(n), k_(k), packed_(kai_get_rhs_packed_size_rhs_pack_nxk_qsi8cxp_qsi8cx_neon(n, k, 4, 8, 1)) {
    const auto w = Constant(info, index, n * k);
    const auto scale = Constant(info, index + 1, n);
    const auto bias = Constant(info, index + 2, n);
    const kai_rhs_pack_qsi8cx_params params{1, 1.0f};
    kai_run_rhs_pack_nxk_qsi8cxp_qsi8cx_neon(
        1, n, k, 4, 8, 1, w.GetTensorData<int8_t>(), bias.GetTensorData<float>(),
        scale.GetTensorData<float>(), packed_.data(), 0, &params);
  }

  void Run(const Ort::KernelContext &context, size_t m, const float *input,
           float *output, std::vector<uint8_t> &lhs, bool relu = false) const {
    lhs.resize(kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32(m, k_, 4, 8, 1));
    kai_run_lhs_quant_pack_qai8dxp_f32(m, k_, 4, 8, 1, 0, input, k_ * sizeof(float), lhs.data());
    const size_t columns = n_ / kTasks;
    Parallel(context, kTasks, [&](size_t task) {
      const size_t start = task * columns;
      const size_t offset = kai_get_rhs_packed_offset_rhs_pack_nxk_qsi8cxp_qsi8cx_neon(start, k_, 4, 8, 1);
      kai_run_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm(
          m, columns, k_, lhs.data(), packed_.data() + offset, output + start,
          n_ * sizeof(float), sizeof(float), relu ? 0.0f : std::numeric_limits<float>::lowest(),
          std::numeric_limits<float>::max());
    });
  }

 private:
  size_t n_, k_;
  std::vector<uint8_t> packed_;
};

void Normalize(const float *source, const float *scale, const float *bias, float *out) {
  auto sum = vdupq_n_f32(0);
  for (size_t c = 0; c < kWidth; c += 4) sum = vaddq_f32(sum, vld1q_f32(source + c));
  const auto average = vdupq_n_f32(vaddvq_f32(sum) / kWidth);
  auto squares = vdupq_n_f32(0);
  for (size_t c = 0; c < kWidth; c += 4) {
    const auto diff = vsubq_f32(vld1q_f32(source + c), average);
    squares = vfmaq_f32(squares, diff, diff);
  }
  const auto inverse = vdupq_n_f32(1.0f / std::sqrt(vaddvq_f32(squares) / kWidth + kEpsilon));
  for (size_t c = 0; c < kWidth; c += 4) {
    const auto centered = vsubq_f32(vld1q_f32(source + c), average);
    const auto scaled = vmulq_f32(vmulq_f32(centered, inverse), vld1q_f32(scale + c));
    vst1q_f32(out + c, vaddq_f32(scaled, vld1q_f32(bias + c)));
  }
}

void Attention(const float *qkv, const float *kv, size_t rows, size_t query,
               size_t head, float *scores, float *output) {
  const size_t channel = head * kHeadWidth;
  float32x4_t q[kHeadWidth / 4];
  for (size_t j = 0; j < kHeadWidth / 4; ++j) {
    q[j] = vmulq_n_f32(vld1q_f32(qkv + query * kQkv + channel + j * 4), kAttentionScale);
  }
  float maximum = std::numeric_limits<float>::lowest();
  for (size_t row = 0; row < rows; ++row) {
    const float *key = kv + row * (2 * kHeadWidth);
    auto a = vdupq_n_f32(0), b = a;
    for (size_t j = 0; j < kHeadWidth / 4; j += 2) {
      a = vfmaq_f32(a, q[j], vld1q_f32(key + 4 * j));
      b = vfmaq_f32(b, q[j + 1], vld1q_f32(key + 4 * (j + 1)));
    }
    scores[row] = vaddvq_f32(vaddq_f32(a, b));
    maximum = std::max(maximum, scores[row]);
  }
  // Range-reduced vector exp, as used by the Nemotron attention kernel.
  // Softmax inputs are nonpositive after subtracting the row maximum.
  auto vsum = vdupq_n_f32(0);
  const size_t vector_rows = rows & ~size_t(3);
  for (size_t row = 0; row < vector_rows; row += 4) {
    const auto x = vmaxq_f32(vsubq_f32(vld1q_f32(scores + row), vdupq_n_f32(maximum)), vdupq_n_f32(-80));
    const auto n = vrndnq_f32(vmulq_n_f32(x, 1.44269504089f));
    const auto r = vfmsq_n_f32(x, n, 0.69314718056f);
    auto p = vfmaq_n_f32(vdupq_n_f32(1.0f / 6), r, 1.0f / 24);
    p = vfmaq_f32(vdupq_n_f32(0.5f), r, p);
    p = vfmaq_f32(vdupq_n_f32(1), r, p);
    p = vfmaq_f32(vdupq_n_f32(1), r, p);
    const auto bits = vshlq_n_s32(vaddq_s32(vcvtq_s32_f32(n), vdupq_n_s32(127)), 23);
    const auto e = vmulq_f32(p, vreinterpretq_f32_s32(bits));
    vst1q_f32(scores + row, e);
    vsum = vaddq_f32(vsum, e);
  }
  float total = vaddvq_f32(vsum);
  for (size_t row = vector_rows; row < rows; ++row) {
    scores[row] = std::exp(scores[row] - maximum);
    total += scores[row];
  }
  const float inverse = 1.0f / total;
  float32x4_t out[kHeadWidth / 4] = {};
  for (size_t row = 0; row < rows; ++row) {
    const float *value = kv + row * (2 * kHeadWidth) + kHeadWidth;
    const float probability = scores[row] * inverse;
    for (size_t j = 0; j < kHeadWidth / 4; ++j) {
      out[j] = vfmaq_n_f32(out[j], vld1q_f32(value + j * 4), probability);
    }
  }
  for (size_t j = 0; j < kHeadWidth / 4; ++j) vst1q_f32(output + channel + j * 4, out[j]);
}

void Fsmn(const float *qkv, size_t rows, size_t row, const float *weights, float *out) {
  const size_t radius = kTaps / 2;
  const size_t begin = row < radius ? radius - row : 0;
  const size_t end = std::min(kTaps, rows - row + radius);
  const float *value = qkv + row * kQkv + 2 * kWidth;
  for (size_t c = 0; c < kWidth; c += 16) {
    auto a = vdupq_n_f32(0), b = a, d = a, e = a;
    for (size_t tap = begin; tap < end; ++tap) {
      const auto *x = qkv + (row + tap - radius) * kQkv + 2 * kWidth + c;
      const auto *w = weights + tap * kWidth + c;
      a = vfmaq_f32(a, vld1q_f32(x), vld1q_f32(w));
      b = vfmaq_f32(b, vld1q_f32(x + 4), vld1q_f32(w + 4));
      d = vfmaq_f32(d, vld1q_f32(x + 8), vld1q_f32(w + 8));
      e = vfmaq_f32(e, vld1q_f32(x + 12), vld1q_f32(w + 12));
    }
    vst1q_f32(out + c, vaddq_f32(a, vld1q_f32(value + c)));
    vst1q_f32(out + c + 4, vaddq_f32(b, vld1q_f32(value + c + 4)));
    vst1q_f32(out + c + 8, vaddq_f32(d, vld1q_f32(value + c + 8)));
    vst1q_f32(out + c + 12, vaddq_f32(e, vld1q_f32(value + c + 12)));
  }
}

struct Workspace {
  std::vector<float> norm, qkv, kv, attention, memory, projection, skip, ff, scores;
  std::vector<uint8_t> lhs;

  void Resize(size_t rows) {
    norm.resize(rows * kWidth);
    qkv.resize(rows * kQkv);
    kv.resize(rows * 2 * kWidth);
    attention.resize(rows * kWidth);
    memory.resize(rows * kWidth);
    projection.resize(rows * kWidth);
    skip.resize(rows * kWidth);
    ff.resize(rows * kFfn);
    scores.resize(kTasks * rows);
  }
};

class Sanm {
 public:
  explicit Sanm(const OrtKernelInfo *raw) : Sanm(Ort::ConstKernelInfo{raw}) {}

  OrtStatusPtr ComputeV2(OrtKernelContext *raw) noexcept {
    return Guard([&] {
      const Ort::KernelContext context(raw);
      const auto input = context.GetInput(0);
      const auto shape = input.GetTensorTypeAndShapeInfo().GetShape();
      if (shape.size() != 3 || shape[0] != 1 || shape[1] <= 0 || shape[2] != kWidth) {
        throw std::invalid_argument("CT SANM expects [1,T,256]");
      }
      const size_t rows = shape[1];
      const float *x = input.GetTensorData<float>();
      auto output = context.GetOutput(0, shape);
      float *y = output.GetTensorMutableData<float>();
      static thread_local Workspace local;
      auto &work = local;
      work.Resize(rows);

      Rows(context, rows, [&](size_t start, size_t end, size_t) {
        for (size_t row = start; row < end; ++row) {
          Normalize(x + row * kWidth, norm1_scale_.data(), norm1_bias_.data(), work.norm.data() + row * kWidth);
        }
      });
      qkv_.Run(context, rows, work.norm.data(), work.qkv.data(), work.lhs);
      Parallel(context, kTasks, [&](size_t task) {
        float *scores = work.scores.data() + task * rows;
        // Keep the K/V rows for each head contiguous throughout its QK and PV.
        constexpr size_t heads_per_task = kHeads / kTasks;
        for (size_t head = task * heads_per_task; head < (task + 1) * heads_per_task; ++head) {
          float *kv = work.kv.data() + head * rows * (2 * kHeadWidth);
          for (size_t row = 0; row < rows; ++row) {
            const float *packed_qkv = work.qkv.data() + row * kQkv + head * kHeadWidth;
            std::copy_n(packed_qkv + kWidth, kHeadWidth, kv + row * (2 * kHeadWidth));
            std::copy_n(packed_qkv + 2 * kWidth, kHeadWidth, kv + row * (2 * kHeadWidth) + kHeadWidth);
          }
          for (size_t row = 0; row < rows; ++row) {
            Attention(work.qkv.data(), kv, rows, row, head, scores, work.attention.data() + row * kWidth);
          }
        }
        for (size_t row = rows * task / kTasks; row < rows * (task + 1) / kTasks; ++row) {
          Fsmn(work.qkv.data(), rows, row, fsmn_.data(), work.memory.data() + row * kWidth);
        }
      });
      projection_.Run(context, rows, work.attention.data(), work.projection.data(), work.lhs);
      Rows(context, rows, [&](size_t start, size_t end, size_t) {
        for (size_t row = start; row < end; ++row) {
          const size_t base = row * kWidth;
          for (size_t i = base; i < base + kWidth; i += 4) {
            const auto residual = vaddq_f32(vld1q_f32(work.projection.data() + i),
                                           vld1q_f32(work.memory.data() + i));
            vst1q_f32(work.skip.data() + i, vaddq_f32(residual, vld1q_f32(x + i)));
          }
          Normalize(work.skip.data() + base, norm2_scale_.data(), norm2_bias_.data(), work.norm.data() + base);
        }
      });
      ff1_.Run(context, rows, work.norm.data(), work.ff.data(), work.lhs, true);
      ff2_.Run(context, rows, work.ff.data(), y, work.lhs);
      for (size_t i = 0; i < rows * kWidth; i += 4) {
        vst1q_f32(y + i, vaddq_f32(vld1q_f32(y + i), vld1q_f32(work.skip.data() + i)));
      }
    });
  }

 private:
  explicit Sanm(const Ort::ConstKernelInfo &info)
      : norm1_scale_(Read<kWidth>(info, 1)), norm1_bias_(Read<kWidth>(info, 2)),
        norm2_scale_(Read<kWidth>(info, 3)), norm2_bias_(Read<kWidth>(info, 4)),
        fsmn_(Read<kTaps * kWidth>(info, 5)), qkv_(info, 6, kQkv, kWidth),
        projection_(info, 9, kWidth, kWidth), ff1_(info, 12, kFfn, kWidth),
        ff2_(info, 15, kWidth, kFfn) {}

  std::array<float, kWidth> norm1_scale_, norm1_bias_, norm2_scale_, norm2_bias_;
  std::array<float, kTaps * kWidth> fsmn_;
  Gemm qkv_, projection_, ff1_, ff2_;
};

class Embedding {
 public:
  explicit Embedding(const OrtKernelInfo *) {}

  OrtStatusPtr ComputeV2(OrtKernelContext *raw) noexcept {
    return Guard([&] {
      const Ort::KernelContext context(raw);
      const auto input = context.GetInput(0), length = context.GetInput(1), weights = context.GetInput(2);
      const auto shape = input.GetTensorTypeAndShapeInfo().GetShape();
      const auto weight_shape = weights.GetTensorTypeAndShapeInfo().GetShape();
      if (shape.size() != 2 || shape[0] != 1 || shape[1] <= 0 ||
          length.GetTensorTypeAndShapeInfo().GetElementCount() != 1 ||
          *length.GetTensorData<int32_t>() != shape[1] || weight_shape.size() != 2 ||
          weight_shape[1] != kWidth) {
        throw std::invalid_argument("CT embedding expects an unpadded [1,T] input");
      }
      const int64_t rows = shape[1], vocab = weight_shape[0];
      const auto *ids = input.GetTensorData<int32_t>();
      const auto scales = context.GetInput(3);
      const auto positions = context.GetInput(4);
      const auto frequencies = context.GetInput(5);
      const auto position_shape = positions.GetTensorTypeAndShapeInfo().GetShape();
      if (vocab <= 0 || scales.GetTensorTypeAndShapeInfo().GetElementCount() !=
                            3u ||
          position_shape.size() != 2 || position_shape[1] != kWidth ||
          frequencies.GetTensorTypeAndShapeInfo().GetElementCount() != kWidth / 2) {
        throw std::invalid_argument("Unexpected CT embedding constants");
      }
      const auto *scale = scales.GetTensorData<float>();
      const auto *table = positions.GetTensorData<float>();
      const auto table_rows = position_shape[0];
      const auto *frequency = frequencies.GetTensorData<float>();
      const int64_t output_shape[] = {1, rows, kWidth};
      auto output = context.GetOutput(0, output_shape, 3);
      auto *y = output.GetTensorMutableData<float>();
      for (int64_t i = 0; i < rows; ++i) {
        const int64_t id = ids[i] < 0 ? vocab + ids[i] : ids[i];
        if (id < 0 || id >= vocab) throw std::invalid_argument("CT token index out of range");
        std::array<float, kWidth> generated;
        const float *position;
        if (i < table_rows) {
          position = table + i * kWidth;
        } else {
          for (size_t c = 0; c < kWidth / 2; ++c) {
            const float angle = float(i + 1) * frequency[c];
            generated[c] = std::sin(angle);
            generated[c + kWidth / 2] = std::cos(angle);
          }
          position = generated.data();
        }
        for (size_t c = 0; c < kWidth; c += 8) {
          float32x4_t lo, hi;
          const auto *src = weights.GetTensorRawData();
          const auto value = vmovl_u8(vld1_u8(static_cast<const uint8_t *>(src) + id * kWidth + c));
          const auto zero = vdupq_n_f32(scale[2]);
          lo = vmulq_n_f32(vmulq_n_f32(vsubq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(value))), zero), scale[1]), scale[0]);
          hi = vmulq_n_f32(vmulq_n_f32(vsubq_f32(vcvtq_f32_u32(vmovl_high_u16(value)), zero), scale[1]), scale[0]);
          vst1q_f32(y + i * kWidth + c, vaddq_f32(lo, vld1q_f32(position + c)));
          vst1q_f32(y + i * kWidth + c + 4, vaddq_f32(hi, vld1q_f32(position + c + 4)));
        }
      }
    });
  }
};

template <class Kernel, size_t Count>
struct Op : Ort::CustomOpBase<Op<Kernel, Count>, Kernel, true> {
  using Type = ONNXTensorElementDataType;
  const char *name;
  std::array<Type, Count> inputs;
  Op(const char *op_name, std::array<Type, Count> types) : name(op_name), inputs(types) {}
  const char *GetName() const { return name; }
  size_t GetInputTypeCount() const { return Count; }
  Type GetInputType(size_t index) const { return inputs[index]; }
  size_t GetOutputTypeCount() const { return 1; }
  Type GetOutputType(size_t) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; }
  OrtStatusPtr CreateKernelV2(const OrtApi &, const OrtKernelInfo *info, void **out) const noexcept {
    return Guard([&] { *out = new Kernel(info); });
  }
};
}  // namespace

namespace sherpa_onnx {

void RegisterCtTransformerNeonOps(Ort::SessionOptions &options) {
  constexpr auto f = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  constexpr auto h = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
  constexpr auto q = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
  constexpr auto i = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
  static const Op<Embedding, 6> embed_u8("EmbedU8", {i, i, h, f, f, f});
  static const Op<Sanm, 18> sanm("SanmW8A8", {f, f, f, f, f, f, q, f, f, q, f, f, q, f, f, q, f, f});
  static Ort::CustomOpDomain domain = [] {
    Ort::CustomOpDomain result("transai.ct");
    result.Add(&embed_u8);
    result.Add(&sanm);
    return result;
  }();
  options.Add(domain);
}

}  // namespace sherpa_onnx
