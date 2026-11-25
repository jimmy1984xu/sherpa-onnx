#include "onnxruntime_cxx_api.h"

#include <arm_neon.h>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <vector>

#include "kai/kai_common.h"
#include "pack/kai_lhs_quant_pack_qai8dxp_f32.h"
#include "pack/kai_rhs_pack_nxk_qsi8cxp_qsi8cx_neon.h"
#include "matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm.h"

namespace {
template<class F> OrtStatusPtr guarded(F fn) noexcept {
  try {
    fn();
    return nullptr;
  } catch (const Ort::Exception& error) {
    return Ort::GetApi().CreateStatus(error.GetOrtErrorCode(), error.what());
  } catch (const std::exception& error) {
    return Ort::GetApi().CreateStatus(ORT_FAIL, error.what());
  }
}

template<class F> void parallel(const Ort::KernelContext& context, size_t tasks, F fn) {
  context.ParallelFor([](void* arg, size_t index) { (*static_cast<F*>(arg))(index); }, tasks, 0, &fn);
}

Ort::ConstValue constant(const Ort::ConstKernelInfo& info, size_t index) {
  int is_constant = 0;
  auto value = info.GetTensorConstantInput(index, &is_constant);
  if (!is_constant) throw std::invalid_argument("TitaNet NEON requires constant weights");
  return value;
}

class Pointwise {
  size_t input_channels_, output_channels_;
  bool relu_;
  std::vector<uint8_t> packed_;

 public:
  explicit Pointwise(const OrtKernelInfo* raw) {
    const Ort::ConstKernelInfo info{raw};
    relu_ = info.GetAttribute<int64_t>("relu") != 0;
    const auto weight = constant(info, 1), scale = constant(info, 2), bias = constant(info, 3);
    const auto shape = weight.GetTensorTypeAndShapeInfo().GetShape();
    if (shape.size() != 2 || shape[0] <= 0 || shape[1] <= 0 ||
        scale.GetTensorTypeAndShapeInfo().GetElementCount() != size_t(shape[0]) ||
        bias.GetTensorTypeAndShapeInfo().GetElementCount() != size_t(shape[0])) {
      throw std::invalid_argument("Invalid W8 pointwise constants");
    }
    output_channels_ = shape[0];
    input_channels_ = shape[1];
    packed_.resize(kai_get_rhs_packed_size_rhs_pack_nxk_qsi8cxp_qsi8cx_neon(
        output_channels_, input_channels_, 4, 8, 1));
    const kai_rhs_pack_qsi8cx_params params{1, 1.0f};
    kai_run_rhs_pack_nxk_qsi8cxp_qsi8cx_neon(
        1, output_channels_, input_channels_, 4, 8, 1, weight.GetTensorData<int8_t>(),
        bias.GetTensorData<float>(), scale.GetTensorData<float>(), packed_.data(), 0, &params);
  }

  OrtStatusPtr ComputeV2(OrtKernelContext* raw) noexcept {
    return guarded([&] {
      const Ort::KernelContext context(raw);
      const auto input = context.GetInput(0);
      auto shape = input.GetTensorTypeAndShapeInfo().GetShape();
      if (shape.size() != 3 || shape[0] < 1 || shape[1] < 1 ||
          size_t(shape[2]) != input_channels_) {
        throw std::invalid_argument("TitaNet W8 pointwise expects [B,T,C]");
      }
      const size_t rows = size_t(shape[0]) * size_t(shape[1]);
      shape[2] = output_channels_;
      auto output = context.GetOutput(0, shape);
      const auto* x = input.GetTensorData<float>();
      auto* y = output.GetTensorMutableData<float>();
      // Few time tiles leave cores idle. Share one LHS across disjoint columns.
      if (rows <= 96 && output_channels_ >= 512) {
        static thread_local std::vector<uint8_t> shared_lhs;
        shared_lhs.resize(kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32(
            rows, input_channels_, 4, 8, 1));
        kai_run_lhs_quant_pack_qai8dxp_f32(rows, input_channels_, 4, 8, 1, 0,
            x, input_channels_ * sizeof(float), shared_lhs.data());
        const auto* lhs = shared_lhs.data();
        const size_t tile = (output_channels_ + 63) / 64 * 16;
        parallel(context, (output_channels_ + tile - 1) / tile, [&](size_t task) {
          const size_t start = task * tile;
          const size_t count = std::min(tile, output_channels_ - start);
          const size_t offset = kai_get_rhs_packed_offset_rhs_pack_nxk_qsi8cxp_qsi8cx_neon(
              start, input_channels_, 4, 8, 1);
          kai_run_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm(
              rows, count, input_channels_, lhs, packed_.data() + offset, y + start,
              output_channels_ * sizeof(float), sizeof(float),
              relu_ ? 0.0f : std::numeric_limits<float>::lowest(),
              std::numeric_limits<float>::max());
        });
        return;
      }
      constexpr size_t tile = 32;
      parallel(context, (rows + tile - 1) / tile, [&](size_t task) {
        const size_t start = task * tile;
        const size_t count = std::min(tile, rows - start);
        thread_local std::vector<uint8_t> lhs;
        lhs.resize(kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32(
            count, input_channels_, 4, 8, 1));
        kai_run_lhs_quant_pack_qai8dxp_f32(count, input_channels_, 4, 8, 1, 0,
            x + start * input_channels_, input_channels_ * sizeof(float), lhs.data());
        kai_run_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm(
            count, output_channels_, input_channels_, lhs.data(), packed_.data(),
            y + start * output_channels_, output_channels_ * sizeof(float), sizeof(float),
            relu_ ? 0.0f : std::numeric_limits<float>::lowest(),
            std::numeric_limits<float>::max());
      });
    });
  }
};

class Depthwise {
  size_t channels_, taps_;

  template<int Taps, bool Interior>
  static void frame(const float* x, const float* weight, float* y, int t, int frames,
                    size_t channels) {
    constexpr int radius = Taps / 2;
    const int start = Interior ? 0 : std::max(0, radius - t);
    const int end = Interior ? Taps : std::min(Taps, frames + radius - t);
    for (size_t c = 0; c < channels; c += 16) {
      auto a = vdupq_n_f32(0), b = a, d = a, e = a;
      for (int tap = start; tap < end; ++tap) {
        const auto* data = x + size_t(t + tap - radius) * channels + c;
        const auto* w = weight + size_t(tap) * channels + c;
        a = vfmaq_f32(a, vld1q_f32(data), vld1q_f32(w));
        b = vfmaq_f32(b, vld1q_f32(data + 4), vld1q_f32(w + 4));
        d = vfmaq_f32(d, vld1q_f32(data + 8), vld1q_f32(w + 8));
        e = vfmaq_f32(e, vld1q_f32(data + 12), vld1q_f32(w + 12));
      }
      vst1q_f32(y + c, a);
      vst1q_f32(y + c + 4, b);
      vst1q_f32(y + c + 8, d);
      vst1q_f32(y + c + 12, e);
    }
  }

  template<int Taps>
  void frames_in_tile(const float* x, const float* weight, float* y,
                      int first, int last, int frames) const {
    constexpr int radius = Taps / 2;
    int t = first;
    for (; t < std::min(last, radius); ++t)
      frame<Taps, false>(x, weight, y + size_t(t) * channels_, t, frames, channels_);
    for (; t < std::min(last, frames - radius); ++t)
      frame<Taps, true>(x, weight, y + size_t(t) * channels_, t, frames, channels_);
    for (; t < last; ++t)
      frame<Taps, false>(x, weight, y + size_t(t) * channels_, t, frames, channels_);
  }

 public:
  explicit Depthwise(const OrtKernelInfo* raw) {
    const Ort::ConstKernelInfo info{raw};
    const auto shape = constant(info, 1).GetTensorTypeAndShapeInfo().GetShape();
    if (shape.size() != 2 || shape[1] <= 0 || shape[1] % 16 ||
        (shape[0] != 3 && shape[0] != 7 && shape[0] != 11 && shape[0] != 15)) {
      throw std::invalid_argument("Unsupported TitaNet depthwise weight shape");
    }
    taps_ = shape[0];
    channels_ = shape[1];
  }

  OrtStatusPtr ComputeV2(OrtKernelContext* raw) noexcept {
    return guarded([&] {
      const Ort::KernelContext context(raw);
      const auto input = context.GetInput(0);
      const auto shape = input.GetTensorTypeAndShapeInfo().GetShape();
      if (shape.size() != 3 || shape[0] < 1 || shape[1] < 1 ||
          shape[1] > std::numeric_limits<int>::max() || size_t(shape[2]) != channels_) {
        throw std::invalid_argument("TitaNet depthwise expects [B,T,C]");
      }
      const int frames = static_cast<int>(shape[1]);
      const auto* x = input.GetTensorData<float>();
      const auto weight = context.GetInput(1);
      const auto* w = weight.GetTensorData<float>();
      auto output = context.GetOutput(0, shape);
      auto* y = output.GetTensorMutableData<float>();
      constexpr size_t tile = 24;
      const size_t tiles = (size_t(frames) + tile - 1) / tile;
      parallel(context, size_t(shape[0]) * tiles, [&](size_t task) {
        const size_t offset = (task / tiles) * size_t(frames) * channels_;
        const int first = int((task % tiles) * tile);
        const int last = std::min(frames, first + int(tile));
        switch (taps_) {
          case 3: frames_in_tile<3>(x + offset, w, y + offset, first, last, frames); break;
          case 7: frames_in_tile<7>(x + offset, w, y + offset, first, last, frames); break;
          case 11: frames_in_tile<11>(x + offset, w, y + offset, first, last, frames); break;
          case 15: frames_in_tile<15>(x + offset, w, y + offset, first, last, frames); break;
        }
      });
    });
  }
};

template<class K, size_t Inputs>
struct Op : Ort::CustomOpBase<Op<K, Inputs>, K, true> {
  explicit Op(const char* name) : name_(name) {}
  const char* GetName() const { return name_; }
  size_t GetInputTypeCount() const { return Inputs; }
  ONNXTensorElementDataType GetInputType(size_t index) const {
    return Inputs == 4 && index == 1 ? ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8
                                     : ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  }
  size_t GetOutputTypeCount() const { return 1; }
  ONNXTensorElementDataType GetOutputType(size_t) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; }
  OrtStatusPtr CreateKernelV2(const OrtApi&, const OrtKernelInfo* info, void** kernel) const noexcept {
    return guarded([&] { *kernel = new K(info); });
  }
  const char* name_;
};
}  // namespace

namespace sherpa_onnx {

void RegisterTitaNetNeonOps(Ort::SessionOptions &options) {
  static const Op<Pointwise, 4> pointwise("W8Pointwise");
  static const Op<Depthwise, 2> depthwise("Depthwise");
  static Ort::CustomOpDomain domain = [] {
    Ort::CustomOpDomain registered("transai.titanet");
    registered.Add(&pointwise);
    registered.Add(&depthwise);
    return registered;
  }();
  options.Add(domain);
}

}  // namespace sherpa_onnx
