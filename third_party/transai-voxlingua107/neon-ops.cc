#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"

#include <arm_neon.h>
#include <asm/hwcap.h>
#include <sys/auxv.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

#include "matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm.h"
#include "pack/kai_lhs_quant_pack_qai8dxp_f32.h"
#include "pack/kai_rhs_pack_nxk_qsi8cxp_qsi8cx_neon.h"

namespace {
constexpr size_t kTasks = 4;
constexpr size_t kMr = 4, kNr = 4, kKr = 8, kSr = 1;
constexpr size_t kRows = 16;

template <class F>
void Parallel(const Ort::KernelContext &context, size_t tasks, F fn) {
  if (tasks == 1) fn(0);
  else context.ParallelFor([](void *callback, size_t index) {
    (*static_cast<F *>(callback))(index);
  }, tasks, tasks, &fn);
}

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

Ort::ConstValue Constant(const Ort::ConstKernelInfo &info, size_t index) {
  int constant = 0;
  auto value = info.GetTensorConstantInput(index, &constant);
  if (!constant) throw std::invalid_argument("VoxLingua107 convolution parameters must be constant");
  return value;
}

std::vector<float> ReadVector(const Ort::ConstKernelInfo &info, size_t index, size_t channels) {
  const auto value = Constant(info, index);
  if (value.GetTensorTypeAndShapeInfo().GetElementCount() != channels)
    throw std::invalid_argument("VoxLingua107 convolution parameter size mismatch");
  const float *data = value.GetTensorData<float>();
  return {data, data + channels};
}

std::array<float32x4_t, 4> Transpose4(float32x4_t a, float32x4_t b,
                                    float32x4_t c, float32x4_t d) {
  const auto ab = vtrnq_f32(a, b);
  const auto cd = vtrnq_f32(c, d);
  return {vcombine_f32(vget_low_f32(ab.val[0]), vget_low_f32(cd.val[0])),
          vcombine_f32(vget_low_f32(ab.val[1]), vget_low_f32(cd.val[1])),
          vcombine_f32(vget_high_f32(ab.val[0]), vget_high_f32(cd.val[0])),
          vcombine_f32(vget_high_f32(ab.val[1]), vget_high_f32(cd.val[1]))};
}

struct Scratch {
  std::vector<float> lowered, result, transposed;
  std::vector<uint8_t> lhs;
};

class Conv {
 public:
  explicit Conv(const OrtKernelInfo *info) : Conv(Ort::ConstKernelInfo(info)) {}

  OrtStatusPtr ComputeV2(OrtKernelContext *raw) noexcept {
    return Guard([&] {
      const Ort::KernelContext context(raw);
      const auto input = context.GetInput(0);
      const auto shape = input.GetTensorTypeAndShapeInfo().GetShape();
      if (shape.size() != 3 || shape[0] != 1 || shape[1] != static_cast<int64_t>(inputs_) ||
          shape[2] <= static_cast<int64_t>(padding_)) {
        throw std::invalid_argument("VoxLingua107 expects batch 1 and enough frames for reflect padding");
      }
      const size_t frames = shape[2];
      const size_t padded = frames + 2 * padding_;
      const size_t reduction = dilation_ * (taps_ - 1);
      if (padded <= reduction) throw std::invalid_argument("VoxLingua107 frame window is empty");
      const size_t out_frames = padded - reduction;
      const int64_t out_shape[] = {1, static_cast<int64_t>(channels_), static_cast<int64_t>(out_frames)};
      auto output = context.GetOutput(0, out_shape, 3);
      const float *x = input.GetTensorData<float>();
      float *y = output.GetTensorMutableData<float>();
      const size_t blocks = (out_frames + kRows - 1) / kRows;
      const size_t tasks = std::min(blocks, kTasks);
      // Each task owns its row range and scratch. Packed weights are read-only.
      thread_local std::array<Scratch, kTasks> workspace;
      auto &scratch = workspace;
      const size_t cache_columns = std::max(kNr, (128 * 1024 / k_) / kNr * kNr);
      auto first_row = [&](size_t task) {
        return std::min(out_frames, blocks * task / tasks * kRows);
      };
      for (size_t task = 0; task < tasks; ++task) {
        const size_t rows = first_row(task + 1) - first_row(task);
        auto &s = scratch[task];
        s.lowered.resize(rows * k_);
        if (taps_ > 1) s.transposed.resize((rows + reduction) * inputs_);
        s.result.resize(rows * channels_);
        s.lhs.resize(LhsSize(rows));
      }
      Parallel(context, tasks, [&](size_t task) {
        const size_t first = first_row(task);
        const size_t rows = first_row(task + 1) - first;
        auto &s = scratch[task];
        LowerTiled(x, frames, first, rows, reduction, s);
        kai_run_lhs_quant_pack_qai8dxp_f32(
            rows, k_, kMr, kKr, kSr, 0, s.lowered.data(), k_ * sizeof(float), s.lhs.data());
        const size_t columns = channels_ * k_ >= 512 * 1024 ? cache_columns : channels_;
        for (size_t col = 0; col < channels_; col += columns)
          Gemm(rows, col, std::min(columns, channels_ - col),
               s.lhs.data(), s.result.data() + col, channels_);
        Store(s.result.data(), y, out_frames, first, rows, 0, channels_);
      });
    });
  }

 private:
  explicit Conv(const Ort::ConstKernelInfo &info)
      : taps_(info.GetAttribute<int64_t>("taps")), padding_(info.GetAttribute<int64_t>("padding")),
        dilation_(info.GetAttribute<int64_t>("dilation")), relu_(info.GetAttribute<int64_t>("relu") != 0),
        channels_(info.GetAttribute<int64_t>("channels")), gamma_(ReadVector(info, 4, channels_)),
        offset_(ReadVector(info, 5, channels_)) {
    const auto weights = Constant(info, 1);
    const auto shape = weights.GetTensorTypeAndShapeInfo().GetShape();
    if (shape.size() != 2 || shape[0] <= 0 || shape[1] <= 0 ||
        shape[0] != static_cast<int64_t>(channels_) || channels_ % 4 ||
        !taps_ || !dilation_ || shape[1] % taps_ || (shape[1] / taps_) % 4) {
      throw std::invalid_argument("Invalid VoxLingua107 convolution shape");
    }
    k_ = shape[1];
    inputs_ = k_ / taps_;
    const auto scale = ReadVector(info, 2, channels_);
    const auto bias = ReadVector(info, 3, channels_);
    weights_.resize(kai_get_rhs_packed_size_rhs_pack_nxk_qsi8cxp_qsi8cx_neon(
        channels_, k_, kNr, kKr, kSr));
    const kai_rhs_pack_qsi8cx_params params{1, 1.0f};
    kai_run_rhs_pack_nxk_qsi8cxp_qsi8cx_neon(1, channels_, k_, kNr, kKr, kSr,
        weights.GetTensorData<int8_t>(), bias.data(), scale.data(), weights_.data(), 0, &params);
  }

  static size_t Reflect(int64_t position, int64_t frames) {
    if (position < 0) return -position;
    if (position >= frames) return 2 * frames - 2 - position;
    return position;
  }

  size_t LhsSize(size_t m) const {
    return kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32(m, k_, kMr, kKr, kSr);
  }

  void Gemm(size_t m, size_t first, size_t n, const uint8_t *lhs, float *output, size_t stride) const {
    const size_t offset = kai_get_rhs_packed_offset_rhs_pack_nxk_qsi8cxp_qsi8cx_neon(
        first, k_, kNr, kKr, kSr);
    kai_run_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm(
        m, n, k_, lhs, weights_.data() + offset, output, stride * sizeof(float), sizeof(float),
        relu_ ? 0.f : std::numeric_limits<float>::lowest(), std::numeric_limits<float>::max());
  }

  void TransposeInput(const float *source, size_t frames, int64_t base, size_t rows, float *dest) const {
    for (size_t tile = 0; tile < rows; tile += kRows) {
      const size_t end = std::min(rows, tile + kRows);
      for (size_t block = 0; block < inputs_; block += 32) {
        const size_t end_channel = std::min(inputs_, block + 32);
        for (size_t t = tile; t < end; t += 4) {
          const auto pos = base + static_cast<int64_t>(t);
          if (t + 4 <= end && pos >= 0 && pos + 4 <= static_cast<int64_t>(frames)) {
            for (size_t c = block; c < end_channel; c += 4) {
              const auto *a = source + c * frames + pos;
              const auto v = Transpose4(vld1q_f32(a), vld1q_f32(a + frames),
                                        vld1q_f32(a + 2 * frames), vld1q_f32(a + 3 * frames));
              for (size_t j = 0; j < 4; ++j) vst1q_f32(dest + (t + j) * inputs_ + c, v[j]);
            }
          } else {
            for (size_t j = 0; j < std::min(size_t{4}, end - t); ++j) {
              const size_t idx = Reflect(pos + static_cast<int64_t>(j), frames);
              for (size_t c = block; c < end_channel; ++c)
                dest[(t + j) * inputs_ + c] = source[c * frames + idx];
            }
          }
        }
      }
    }
  }

  void LowerTiled(const float *source, size_t frames, size_t first, size_t rows,
                  size_t reduction, Scratch &scratch) const {
    const auto base = static_cast<int64_t>(first) - static_cast<int64_t>(padding_);
    if (taps_ == 1) {
      TransposeInput(source, frames, base, rows, scratch.lowered.data());
      return;
    }
    TransposeInput(source, frames, base, rows + reduction, scratch.transposed.data());
    // The tap windows share the same time-major input rows. Only the
    // window-specific dynamic INT8 scale is computed again.
    for (size_t t = 0; t < rows; ++t)
      for (size_t tap = 0; tap < taps_; ++tap)
        std::copy_n(scratch.transposed.data() + (t + tap * dilation_) * inputs_, inputs_,
                    scratch.lowered.data() + t * k_ + tap * inputs_);
  }

  void Store(const float *source, float *target, size_t frames, size_t first, size_t rows,
             size_t first_channel, size_t n) const {
    target += first_channel * frames;
    const float *scales = gamma_.data() + first_channel;
    const float *offsets = offset_.data() + first_channel;
    for (size_t t = 0; t < rows; t += 4) {
      const size_t width = std::min(size_t{4}, rows - t);
      const float *row = source + t * n;
      if (width == 4) {
        for (size_t c = 0; c < n; c += 4) {
          const auto gamma = vld1q_f32(scales + c);
          const auto offset = vld1q_f32(offsets + c);
          const auto values = Transpose4(
              vfmaq_f32(offset, vld1q_f32(row + c), gamma),
              vfmaq_f32(offset, vld1q_f32(row + n + c), gamma),
              vfmaq_f32(offset, vld1q_f32(row + 2 * n + c), gamma),
              vfmaq_f32(offset, vld1q_f32(row + 3 * n + c), gamma));
          for (size_t j = 0; j < 4; ++j) vst1q_f32(target + (c + j) * frames + first + t, values[j]);
        }
      } else {
        for (size_t c = 0; c < n; ++c)
          for (size_t j = 0; j < width; ++j)
            target[c * frames + first + t + j] = std::fma(row[j * n + c], scales[c], offsets[c]);
      }
    }
  }

  size_t taps_, padding_, dilation_;
  bool relu_;
  size_t channels_, k_, inputs_;
  std::vector<float> gamma_, offset_;
  std::vector<uint8_t> weights_;
};

struct ConvOp : Ort::CustomOpBase<ConvOp, Conv, true> {
  const char *GetName() const { return "ConvW8A8"; }
  size_t GetInputTypeCount() const { return 6; }
  ONNXTensorElementDataType GetInputType(size_t i) const {
    return i == 1 ? ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8 : ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  }
  size_t GetOutputTypeCount() const { return 1; }
  ONNXTensorElementDataType GetOutputType(size_t) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; }
  OrtStatusPtr CreateKernelV2(const OrtApi &, const OrtKernelInfo *info, void **out) const noexcept {
    return Guard([&] { *out = new Conv(info); });
  }
};

class WeightedStats {
 public:
  explicit WeightedStats(const OrtKernelInfo *) {}
  OrtStatusPtr ComputeV2(OrtKernelContext *raw) noexcept {
    return Guard([&] {
      const Ort::KernelContext context(raw);
      const auto x = context.GetInput(0);
      const auto weights = context.GetInput(1);
      const auto x_shape = x.GetTensorTypeAndShapeInfo().GetShape();
      const auto w_shape = weights.GetTensorTypeAndShapeInfo().GetShape();
      if (x_shape.size() != 3 || w_shape.size() != 3 || x_shape[0] != 1 || w_shape[0] != 1 ||
          x_shape[1] <= 0 || x_shape[2] <= 0 || w_shape[2] != x_shape[2] ||
          (w_shape[1] != 1 && w_shape[1] != x_shape[1]))
        throw std::invalid_argument("VoxLingua107 attention pooling shape mismatch");
      const size_t channels = x_shape[1];
      const size_t frames = x_shape[2];
      const int64_t shape[] = {1, x_shape[1]};
      float *mean = context.GetOutput(0, shape, 2).GetTensorMutableData<float>();
      float *stddev = context.GetOutput(1, shape, 2).GetTensorMutableData<float>();
      const float *all_x = x.GetTensorData<float>();
      const float *all_w = weights.GetTensorData<float>();
      const bool broadcast = w_shape[1] == 1;
      const size_t tasks = std::min(kTasks, channels);
      Parallel(context, tasks, [&](size_t task) {
        for (size_t channel = channels * task / tasks; channel < channels * (task + 1) / tasks; ++channel) {
          const float *values = all_x + channel * frames;
          const float *w = broadcast ? all_w : all_w + channel * frames;
          float32x4_t sum = vdupq_n_f32(0);
          size_t i = 0;
          for (; i + 4 <= frames; i += 4) sum = vfmaq_f32(sum, vld1q_f32(values + i), vld1q_f32(w + i));
          float mu = vaddvq_f32(sum);
          for (; i < frames; ++i) mu = std::fma(values[i], w[i], mu);
          mean[channel] = mu;
          sum = vdupq_n_f32(0);
          i = 0;
          const auto center = vdupq_n_f32(mu);
          for (; i + 4 <= frames; i += 4) {
            const auto d = vsubq_f32(vld1q_f32(values + i), center);
            sum = vfmaq_f32(sum, vmulq_f32(d, d), vld1q_f32(w + i));
          }
          float variance = vaddvq_f32(sum);
          for (; i < frames; ++i) {
            const float d = values[i] - mu;
            variance = std::fma(d * d, w[i], variance);
          }
          stddev[channel] = std::sqrt(std::max(variance, 1e-12f));
        }
      });
    });
  }
};

struct StatsOp : Ort::CustomOpBase<StatsOp, WeightedStats, true> {
  const char *GetName() const { return "WeightedStats"; }
  size_t GetInputTypeCount() const { return 2; }
  size_t GetOutputTypeCount() const { return 2; }
  ONNXTensorElementDataType GetInputType(size_t) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; }
  ONNXTensorElementDataType GetOutputType(size_t) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; }
  OrtStatusPtr CreateKernelV2(const OrtApi &, const OrtKernelInfo *info, void **out) const noexcept {
    return Guard([&] { *out = new WeightedStats(info); });
  }
};
}  // namespace

extern "C" __attribute__((visibility("default"))) OrtStatus *ORT_API_CALL
RegisterCustomOps(OrtSessionOptions *options, const OrtApiBase *api_base) {
  static const OrtApi *api = [api_base] {
    const auto *current = api_base->GetApi(ORT_API_VERSION);
    if (current) Ort::InitApi(current);
    return current;
  }();
  if (!api) return api_base->GetApi(1)->CreateStatus(ORT_FAIL, "Incompatible VoxLingua107 ORT API");
  return Guard([&] {
    if (!(getauxval(AT_HWCAP2) & HWCAP2_I8MM))
      throw std::runtime_error("VoxLingua107 NEON requires AArch64 I8MM");
    static const ConvOp op;
    static const StatsOp stats;
    static Ort::CustomOpDomain domain = [] {
      Ort::CustomOpDomain result("transai.voxlingua");
      result.Add(&op);
      result.Add(&stats);
      return result;
    }();
    Ort::UnownedSessionOptions(options).Add(domain);
  });
}
