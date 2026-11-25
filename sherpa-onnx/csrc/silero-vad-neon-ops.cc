#include "onnxruntime_cxx_api.h"

#include <arm_neon.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "kiss_fftr.h"

namespace {
constexpr int kHidden = 128;
constexpr int kFft = 256;
constexpr int kBins = kFft / 2 + 1;
constexpr int kWindow = 576;
enum class Kind { Half = 1, Int8 };

template<Kind K> constexpr bool kInt8 = K == Kind::Int8;

template<class F> OrtStatusPtr Guard(F fn) noexcept {
  try {
    fn();
    return nullptr;
  } catch (const Ort::Exception& error) {
    return Ort::GetApi().CreateStatus(error.GetOrtErrorCode(), error.what());
  } catch (const std::exception& error) {
    return Ort::GetApi().CreateStatus(ORT_FAIL, error.what());
  }
}

template<class F> void Parallel(const Ort::KernelContext& context, F fn) {
  context.ParallelFor([](void* arg, size_t index) { (*static_cast<F*>(arg))(index); },
                      4, 4, &fn);
}

Ort::ConstValue Constant(const Ort::ConstKernelInfo& info, size_t index,
                         size_t elements) {
  int is_constant = 0;
  auto value = info.GetTensorConstantInput(index, &is_constant);
  if (!is_constant || value.GetTensorTypeAndShapeInfo().GetElementCount() != elements)
    throw std::invalid_argument("Incorrect Silero NEON constant size");
  return value;
}

template<class T> std::vector<T> Read(const Ort::ConstKernelInfo& info, size_t index,
                                      size_t size) {
  const auto value = Constant(info, index, size);
  const auto* data = value.GetTensorData<T>();
  return std::vector<T>(data, data + size);
}

template<Kind K> using Element = std::conditional_t<kInt8<K>, int8_t, float16_t>;

template<Kind K, int Width> struct Prepared {
  alignas(16) Element<K> data[Width];
  static constexpr int kGroup = Width;
  float scale[Width / kGroup] = {1.0f};

  Prepared() = default;

  explicit Prepared(const float* x) {
    if constexpr (kInt8<K>) {
      for (int offset = 0; offset < Width; offset += kGroup) {
        float32x4_t peak = vdupq_n_f32(0);
        for (int i = offset; i < offset + kGroup; i += 4)
          peak = vmaxq_f32(peak, vabsq_f32(vld1q_f32(x+i)));
        const float max_abs = vmaxvq_f32(peak);
        if (max_abs < 1.0e-20f) {
          std::memset(data + offset, 0, kGroup);
          continue;
        }
        scale[offset / kGroup] = max_abs / 127.0f;
        const float inverse = 127.0f / max_abs;
        for (int i = offset; i < offset + kGroup; i += 16) {
          const auto a = vcvtnq_s32_f32(vmulq_n_f32(vld1q_f32(x+i), inverse));
          const auto b = vcvtnq_s32_f32(vmulq_n_f32(vld1q_f32(x+i+4), inverse));
          const auto c = vcvtnq_s32_f32(vmulq_n_f32(vld1q_f32(x+i+8), inverse));
          const auto d = vcvtnq_s32_f32(vmulq_n_f32(vld1q_f32(x+i+12), inverse));
          const auto lo = vcombine_s16(vqmovn_s32(a), vqmovn_s32(b));
          const auto hi = vcombine_s16(vqmovn_s32(c), vqmovn_s32(d));
          vst1q_s8(data+i, vcombine_s8(vqmovn_s16(lo), vqmovn_s16(hi)));
        }
      }
    } else if constexpr (K == Kind::Half) {
      for (int i = 0; i < Width; i += 8)
        vst1q_f16(data+i, vcombine_f16(vcvt_f16_f32(vld1q_f32(x+i)),
                                      vcvt_f16_f32(vld1q_f32(x+i+4))));
    }
  }
};

template<Kind K, int Width> struct Matrix {
  std::vector<Element<K>> weights;
  std::vector<float> scales;
  std::vector<float> bias;

  Matrix(const Ort::ConstKernelInfo& info, size_t index, size_t rows)
      : scales(Read<float>(info, index+1, kInt8<K> ? rows * (Width / Prepared<K, Width>::kGroup) : 1)),
        bias(Read<float>(info, index+2, rows)) {
    if constexpr (K == Kind::Half) {
      const auto raw = Read<Ort::Float16_t>(info, index, rows * Width);
      weights.resize(raw.size());
      std::memcpy(weights.data(), raw.data(), raw.size() * sizeof(float16_t));
    } else {
      weights = Read<Element<K>>(info, index, rows * Width);
    }
  }

  float Dot(const Prepared<K, Width>& x, int row) const {
    const auto* w = weights.data() + row * Width;
    if constexpr (K == Kind::Int8) {
      auto a = vdupq_n_s32(0), b = a;
      for (int i = 0; i + 32 <= Width; i += 32) {
        a = vdotq_s32(a, vld1q_s8(x.data+i), vld1q_s8(w+i));
        b = vdotq_s32(b, vld1q_s8(x.data+i+16), vld1q_s8(w+i+16));
      }
      if constexpr (Width % 32 == 16)
        a = vdotq_s32(a, vld1q_s8(x.data+Width-16), vld1q_s8(w+Width-16));
      return static_cast<float>(vaddvq_s32(vaddq_s32(a, b))) * (x.scale[0] * scales[row]) + bias[row];
    } else if constexpr (K == Kind::Half) {
      auto a = vdupq_n_f32(0), b = a, c = a, d = a;
      for (int i = 0; i < Width; i += 16) {
        const auto x0 = vld1q_f16(x.data+i), x1 = vld1q_f16(x.data+i+8);
        const auto w0 = vld1q_f16(w+i), w1 = vld1q_f16(w+i+8);
        a = vfmlalq_low_f16(a, x0, w0);
        b = vfmlalq_high_f16(b, x0, w0);
        c = vfmlalq_low_f16(c, x1, w1);
        d = vfmlalq_high_f16(d, x1, w1);
      }
      return vaddvq_f32(vaddq_f32(vaddq_f32(a, b), vaddq_f32(c, d))) + bias[row];
    }
  }
};

template<Kind K, int Width, int In, int Out>
void Conv(const Matrix<K, Width>& matrix, const float* input, int frames, int center,
          float* output) {
  alignas(16) float patch[Width] = {};
  for (int tap = 0; tap < 3; ++tap) {
    const int t = center + tap - 1;
    if (t >= 0 && t < frames)
      std::memcpy(patch + tap * In, input + t * In, In * sizeof(float));
  }
  const Prepared<K, Width> x(patch);
  for (int row = 0; row < Out; ++row)
    output[row] = std::max(0.0f, matrix.Dot(x, row));
}

template<int Count>
void ConvertHalf(const float* input, float16_t* output) {
  int i = 0;
  for (; i + 8 <= Count; i += 8)
    vst1q_f16(output+i, vcombine_f16(vcvt_f16_f32(vld1q_f32(input+i)),
                                    vcvt_f16_f32(vld1q_f32(input+i+4))));
  for (; i < Count; ++i) output[i] = static_cast<float16_t>(input[i]);
}

template<int Width, int In, int Out>
void ConvHalf(const Matrix<Kind::Half, Width>& matrix, const float16_t* input,
              int frames, int center, float* output) {
  Prepared<Kind::Half, Width> patch;
  std::memset(patch.data, 0, sizeof(patch.data));
  for (int tap = 0; tap < 3; ++tap) {
    const int frame = center + tap - 1;
    if (frame >= 0 && frame < frames)
      std::memcpy(patch.data + tap * In, input + frame * In, In * sizeof(float16_t));
  }
  for (int row = 0; row < Out; ++row)
    output[row] = std::max(0.0f, matrix.Dot(patch, row));
}

inline float Sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

template<Kind Encoder, Kind Lstm, Kind Recurrent> class Step {
  static_assert(Encoder == Kind::Half && Lstm == Kind::Half && Recurrent == Kind::Int8);
 public:
  explicit Step(const OrtKernelInfo* raw) : Step(Ort::ConstKernelInfo{raw}) {}

  OrtStatusPtr ComputeV2(OrtKernelContext* raw) noexcept {
    return Guard([&] {
      const Ort::KernelContext context(raw);
      const auto input = context.GetInput(0), state = context.GetInput(1), sr = context.GetInput(2);
      const auto shape = input.GetTensorTypeAndShapeInfo().GetShape();
      if (shape.size() != 2 || shape[0] != 1 || shape[1] != kWindow ||
          state.GetTensorTypeAndShapeInfo().GetElementCount() != 256 ||
          sr.GetTensorTypeAndShapeInfo().GetElementCount() != 1 ||
          *sr.GetTensorData<int64_t>() != 16000)
        throw std::invalid_argument("Silero NEON requires 16 kHz, audio [1,576], state [2,1,128]");
      const auto* wave = input.GetTensorData<float>();
      const auto* previous = state.GetTensorData<float>();
      constexpr int64_t probability_shape[] = {1, 1};
      constexpr int64_t state_shape[] = {2, 1, 128};
      auto state_out = context.GetOutput(1, state_shape, 3);
      float* next = state_out.GetTensorMutableData<float>();

      alignas(16) float features2[2*64], features3[64], features4[128];
      std::array<bool, 4> fft_ready;
      if constexpr (Encoder == Kind::Half) {
        // Each source frame is shared by neighboring windows. Round it once.
        alignas(16) float16_t spectra[4*kBins], features[4*128];
        Parallel(context, [&](size_t index) {
          alignas(16) float frame[kBins];
          fft_ready[index] = Spectrum(wave, index, frame);
          if (fft_ready[index]) ConvertHalf<kBins>(frame, spectra + index*kBins);
        });
        if (!std::all_of(fft_ready.begin(), fft_ready.end(), [](bool ready) { return ready; }))
          throw std::bad_alloc();
        Parallel(context, [&](size_t index) {
          alignas(16) float frame[128];
          ConvHalf<400, 129, 128>(encoder0_, spectra, 4, index, frame);
          ConvertHalf<128>(frame, features + index*128);
        });
        for (int i = 0; i < 2; ++i)
          ConvHalf<384, 128, 64>(encoder1_, features, 4, 2*i, features2 + i*64);
      }
      Conv<Encoder, 192, 64, 64>(encoder2_, features2, 2, 0, features3);
      Conv<Encoder, 192, 64, 128>(encoder3_, features3, 1, 0, features4);
      const Prepared<Lstm, 128> x(features4);
      const Prepared<Recurrent, 128> h(previous);
      Parallel(context, [&](size_t task) {
        const int first = 32*static_cast<int>(task);
        for (int unit = first; unit < first+32; ++unit) {
          float gates[4];
          for (int gate = 0; gate < 4; ++gate) {
            const int row = kHidden*gate + unit;
            gates[gate] = input_projection_.Dot(x, row) + hidden_projection_.Dot(h, row);
          }
          // Source tensors are PyTorch IFGO, before ONNX LSTM's IOFG reordering.
          next[kHidden+unit] = Sigmoid(gates[1]) * previous[kHidden+unit] +
                               Sigmoid(gates[0]) * std::tanh(gates[2]);
          next[unit] = Sigmoid(gates[3]) * std::tanh(next[kHidden+unit]);
        }
      });
      float32x4_t result = vdupq_n_f32(0);
      for (int i = 0; i < kHidden; i += 4) {
        const auto activation = vmaxq_f32(vdupq_n_f32(0), vld1q_f32(next+i));
        result = vfmaq_f32(result, activation, vld1q_f32(head_.data()+i));
      }
      auto prob = context.GetOutput(0, probability_shape, 2);
      *prob.GetTensorMutableData<float>() = Sigmoid(vaddvq_f32(result) + head_bias_);
    });
  }

 private:
  explicit Step(const Ort::ConstKernelInfo& info)
      : window_(Read<float>(info, 3, kFft)),
        encoder0_(info, 4, 128), encoder1_(info, 7, 64),
        encoder2_(info, 10, 64), encoder3_(info, 13, 128),
        input_projection_(info, 16, 512), hidden_projection_(info, 19, 512),
        head_(Read<float>(info, 22, 128)), head_bias_(Read<float>(info, 23, 1)[0]) {}

  bool Spectrum(const float* wave, size_t frame, float* out) const {
    // KissFFT's real plan owns scratch storage, so each ORT worker gets its own.
    thread_local std::unique_ptr<kiss_fftr_state, decltype(&std::free)> fft(
        kiss_fftr_alloc(kFft, 0, nullptr, nullptr), &std::free);
    if (!fft) return false;
    alignas(16) float time[kFft];
    alignas(16) kiss_fft_cpx frequency[kBins];
    const int offset = static_cast<int>(frame) * (kFft/2);
    for (int i = 0; i < kFft; ++i) {
      const int sample = offset + i;
      const int reflected = sample < kWindow ? sample : 2*kWindow-2-sample;
      time[i] = wave[reflected] * window_[i];
    }
    kiss_fftr(fft.get(), time, frequency);
    for (int i = 0; i < kBins; ++i)
      out[i] = std::sqrt(frequency[i].r*frequency[i].r + frequency[i].i*frequency[i].i);
    return true;
  }

  std::vector<float> window_;
  Matrix<Encoder, 400> encoder0_;
  Matrix<Encoder, 384> encoder1_;
  Matrix<Encoder, 192> encoder2_, encoder3_;
  Matrix<Lstm, 128> input_projection_;
  Matrix<Recurrent, 128> hidden_projection_;
  std::vector<float> head_;
  float head_bias_;
};

template<Kind Encoder, Kind Lstm, Kind Recurrent = Lstm> struct Op
    : Ort::CustomOpBase<Op<Encoder, Lstm, Recurrent>, Step<Encoder, Lstm, Recurrent>, true> {
  explicit Op(const char* name) : name_(name) {}
  const char* GetName() const { return name_; }
  size_t GetInputTypeCount() const { return 24; }
  ONNXTensorElementDataType GetInputType(size_t i) const {
    if (i == 2) return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    if (i >= 4 && i <= 19 && (i-4)%3 == 0) {
      const auto kind = i < 16 ? Encoder : (i == 16 ? Lstm : Recurrent);
      if (kind == Kind::Half) return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
      if (kind == Kind::Int8) return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
    }
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  }
  size_t GetOutputTypeCount() const { return 2; }
  ONNXTensorElementDataType GetOutputType(size_t) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; }
  OrtStatusPtr CreateKernelV2(const OrtApi&, const OrtKernelInfo* info, void** kernel) const noexcept {
    return Guard([&] { *kernel = new Step<Encoder, Lstm, Recurrent>(info); });
  }
  const char* name_;
};
}  // namespace

namespace sherpa_onnx {

void RegisterSileroNeonOp(Ort::SessionOptions &options) {
  static const Op<Kind::Half, Kind::Half, Kind::Int8> op(
      "SileroStepw8_lstm_recurrent");
  static Ort::CustomOpDomain domain = [] {
    Ort::CustomOpDomain value("transai.silero");
    value.Add(&op);
    return value;
  }();
  options.Add(domain);
}

}  // namespace sherpa_onnx
