// Ported from agent-sdk Nemotron offline CPU core (iter3).
#include "sherpa-onnx/csrc/nemotron-cpu/session.h"

#include "onnxruntime_cxx_api.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <memory>
#include <utility>
#include <vector>

namespace transai::nemotron::cpu {
namespace {

constexpr int kSampleRate = 16000;
constexpr int kNfft = 512;
constexpr int kWin = 400;
constexpr int kHop = 160;
constexpr int kMels = 128;
constexpr int kSub = 8;
constexpr int kHidden = 512;
constexpr int kSpeakers = 8;
constexpr float kPreemph = 0.97f;
constexpr float kActivityThreshold = 0.5f;
constexpr int kSilenceFrames = 1;
constexpr double kPredThresh = 0.25;
constexpr double kLatestBoost = 0.05;
constexpr double kMinPosRate = 0.5;
constexpr double kStrongRate = 0.75;
constexpr double kWeakRate = 1.5;
constexpr double kLogHalf = -0.6931471805599453;
constexpr double kGuard = 1.0 / 16777216.0;
constexpr int kChunk = 340;
constexpr int kRight = 40;
constexpr int kFifo = 40;
constexpr int kUpdate = 300;
constexpr int kCache = 264;
/** 与最终说话人赋值的局部 mask 继承间隔一致，超过则断开槽位连续性。 */
constexpr int64_t kCacheContinuityMaxGapSamples = 3 * kSampleRate;
/** 8 个模型槽位各占一个 bit。重叠是多个 bit，不能再把第 4 个槽位收成 mask 7。 */
constexpr uint8_t kLocalMasks[] = {1, 2, 4, 8, 16, 32, 64, 128};

struct Preset {
    int chunk = kChunk;
    int right = kRight;
    int fifo = kFifo;
    int update = kUpdate;
    int cache = kCache;
};

float sigmoid(float value) { return 1.f / (1.f + std::exp(-value)); }

std::vector<float> readFloatFile(const std::string& path, size_t count) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("打不开 " + path);
    std::vector<float> data(count);
    input.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(count * sizeof(float)));
    if (static_cast<size_t>(input.gcount()) != count * sizeof(float)) {
        throw std::runtime_error("文件长度不对 " + path);
    }
    return data;
}

void fftRadix2(std::vector<double>& real, std::vector<double>& imag) {
    const int count = static_cast<int>(real.size());
    for (int i = 1, j = 0; i < count; ++i) {
        int bit = count >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) {
            std::swap(real[i], real[j]);
            std::swap(imag[i], imag[j]);
        }
    }
    for (int length = 2; length <= count; length <<= 1) {
        const double angle = -2.0 * 3.14159265358979323846 / length;
        const double stepReal = std::cos(angle);
        const double stepImag = std::sin(angle);
        for (int i = 0; i < count; i += length) {
            double weightReal = 1.0;
            double weightImag = 0.0;
            for (int j = 0; j < length / 2; ++j) {
                const int left = i + j;
                const int right = left + length / 2;
                const double rotatedReal = real[right] * weightReal - imag[right] * weightImag;
                const double rotatedImag = real[right] * weightImag + imag[right] * weightReal;
                real[right] = real[left] - rotatedReal;
                imag[right] = imag[left] - rotatedImag;
                real[left] += rotatedReal;
                imag[left] += rotatedImag;
                const double nextReal = weightReal * stepReal - weightImag * stepImag;
                weightImag = weightReal * stepImag + weightImag * stepReal;
                weightReal = nextReal;
            }
        }
    }
}

std::vector<float> logMel(const float* audio, int length, const std::vector<float>& mel, int* frameCount) {
    const int frames = std::max(0, length / kHop);
    *frameCount = frames;
    std::vector<float> emphasized(static_cast<size_t>(std::max(length, 0)));
    if (length > 0) emphasized[0] = audio[0];
    for (int i = 1; i < length; ++i) emphasized[i] = audio[i] - kPreemph * audio[i - 1];
    std::vector<double> window(kNfft, 0.0);
    const int offset = (kNfft - kWin) / 2;
    for (int i = 0; i < kWin; ++i) {
        window[offset + i] = 0.5 - 0.5 * std::cos(2.0 * 3.14159265358979323846 * i / (kWin - 1));
    }
    const int pad = kNfft / 2;
    std::vector<double> signal(static_cast<size_t>(pad + std::max(length, 0) + kNfft), 0.0);
    for (int i = 0; i < length; ++i) signal[pad + i] = emphasized[i];
    std::vector<float> features(static_cast<size_t>(frames) * kMels);
    std::vector<double> real(kNfft), imag(kNfft), power(kNfft / 2 + 1);
    for (int frame = 0; frame < frames; ++frame) {
        const int start = frame * kHop;
        for (int i = 0; i < kNfft; ++i) {
            real[i] = signal[start + i] * window[i];
            imag[i] = 0.0;
        }
        fftRadix2(real, imag);
        for (int bin = 0; bin <= kNfft / 2; ++bin) power[bin] = real[bin] * real[bin] + imag[bin] * imag[bin];
        for (int melIndex = 0; melIndex < kMels; ++melIndex) {
            double sum = 0.0;
            const float* row = mel.data() + static_cast<size_t>(melIndex) * (kNfft / 2 + 1);
            for (int bin = 0; bin <= kNfft / 2; ++bin) sum += static_cast<double>(row[bin]) * power[bin];
            features[static_cast<size_t>(frame) * kMels + melIndex] = static_cast<float>(std::log(sum + kGuard));
        }
    }
    return features;
}

struct SpeakerCache {
    explicit SpeakerCache(const std::vector<float>& silence, Preset config = {}) : preset(config), silence(silence) {
        const int budget = preset.cache / kSpeakers - kSilenceFrames;
        minPositive = static_cast<int>(budget * kMinPosRate);
        strongCount = static_cast<int>(budget * kStrongRate);
        weakCount = static_cast<int>(budget * kWeakRate);
    }

    std::vector<std::vector<float>> embedsWithFifo() const {
        auto out = embeds;
        out.insert(out.end(), fifo.begin(), fifo.end());
        return out;
    }

    void update(const std::vector<std::vector<float>>& input, const std::vector<std::vector<float>>& frameProbs, int chunkCount) {
        const int cacheCount = static_cast<int>(embeds.size());
        const int chunkStart = cacheCount + static_cast<int>(fifo.size());
        std::vector<std::vector<float>> nextFifo = fifo;
        nextFifo.insert(nextFifo.end(), input.begin() + chunkStart, input.begin() + chunkStart + chunkCount);
        int popped = 0;
        if (static_cast<int>(nextFifo.size()) > preset.fifo) {
            popped = std::min(std::max(preset.update, static_cast<int>(nextFifo.size()) - preset.fifo), static_cast<int>(nextFifo.size()));
        }
        if (popped) {
            std::vector<std::vector<float>> fifoProbs(frameProbs.begin() + cacheCount, frameProbs.begin() + cacheCount + static_cast<int>(nextFifo.size()));
            std::vector<std::vector<float>> stored = compressed ? probs : std::vector<std::vector<float>>(frameProbs.begin(), frameProbs.begin() + cacheCount);
            std::vector<std::vector<float>> cacheEmbeds = embeds;
            cacheEmbeds.insert(cacheEmbeds.end(), nextFifo.begin(), nextFifo.begin() + popped);
            std::vector<std::vector<float>> cacheProbs = stored;
            cacheProbs.insert(cacheProbs.end(), fifoProbs.begin(), fifoProbs.begin() + popped);
            nextFifo.erase(nextFifo.begin(), nextFifo.begin() + popped);
            if (static_cast<int>(cacheEmbeds.size()) > preset.cache) {
                compress(cacheEmbeds, cacheProbs);
                compressed = true;
            }
            embeds = std::move(cacheEmbeds);
            probs = std::move(cacheProbs);
        }
        fifo = std::move(nextFifo);
    }

    void clear() {
        embeds.clear();
        probs.clear();
        fifo.clear();
        compressed = false;
    }

    /** 把两段语音之间的静音补进 FIFO，避免模型把断开的片段听成连续说话。 */
    void insertSilence(int frames) {
        for (int i = 0; i < frames; ++i) fifo.push_back(silence);
    }

    Preset preset;
    std::vector<float> silence;
    std::vector<std::vector<float>> embeds;
    std::vector<std::vector<float>> probs;
    std::vector<std::vector<float>> fifo;
    bool compressed = false;
    int minPositive = 0;
    int strongCount = 0;
    int weakCount = 0;

    std::vector<std::vector<double>> frameScores(const std::vector<std::vector<float>>& frameProbs) const {
        const int count = static_cast<int>(frameProbs.size());
        std::vector<std::vector<double>> scores(count, std::vector<double>(kSpeakers, -INFINITY));
        std::vector<int> positiveCount(kSpeakers, 0);
        for (int time = 0; time < count; ++time) {
            double complement[kSpeakers];
            double total = 0.0;
            for (int speaker = 0; speaker < kSpeakers; ++speaker) {
                complement[speaker] = std::log(std::max(1.0 - frameProbs[time][speaker], kPredThresh));
                total += complement[speaker];
            }
            for (int speaker = 0; speaker < kSpeakers; ++speaker) {
                if (!(frameProbs[time][speaker] > kActivityThreshold)) continue;
                const double value = std::log(std::max(static_cast<double>(frameProbs[time][speaker]), kPredThresh)) - complement[speaker] + total - kLogHalf;
                scores[time][speaker] = value;
                if (value > 0) ++positiveCount[speaker];
            }
        }
        for (int time = 0; time < count; ++time) {
            for (int speaker = 0; speaker < kSpeakers; ++speaker) {
                if (!(scores[time][speaker] > 0) && frameProbs[time][speaker] > kActivityThreshold && positiveCount[speaker] >= minPositive) {
                    scores[time][speaker] = -INFINITY;
                }
            }
        }
        return scores;
    }

    void boost(std::vector<std::vector<double>>& scores, int keep, double amount) const {
        const int count = static_cast<int>(scores.size());
        std::vector<int> order(count);
        for (int speaker = 0; speaker < kSpeakers; ++speaker) {
            for (int i = 0; i < count; ++i) order[i] = i;
            std::sort(order.begin(), order.end(), [&](int left, int right) {
                if (scores[left][speaker] != scores[right][speaker]) return scores[left][speaker] > scores[right][speaker];
                return left < right;
            });
            for (int i = 0; i < std::min(keep, count); ++i) scores[order[i]][speaker] += amount;
        }
    }

    void compress(std::vector<std::vector<float>>& cacheEmbeds, std::vector<std::vector<float>>& cacheProbs) const {
        const int count = static_cast<int>(cacheProbs.size());
        auto scores = frameScores(cacheProbs);
        for (int time = preset.cache; time < count; ++time) {
            for (int speaker = 0; speaker < kSpeakers; ++speaker) scores[time][speaker] += kLatestBoost;
        }
        boost(scores, strongCount, -2 * kLogHalf);
        boost(scores, weakCount, -kLogHalf);
        const int scored = count + kSilenceFrames;
        struct Item { int index; double value; };
        std::vector<Item> flat;
        flat.reserve(static_cast<size_t>(kSpeakers) * scored);
        for (int speaker = 0; speaker < kSpeakers; ++speaker) {
            for (int time = 0; time < scored; ++time) {
                flat.push_back({speaker * scored + time, time >= count ? INFINITY : scores[time][speaker]});
            }
        }
        std::sort(flat.begin(), flat.end(), [](const Item& left, const Item& right) {
            if (left.value != right.value) return left.value > right.value;
            return left.index < right.index;
        });
        const int sentinel = scored * kSpeakers;
        std::vector<int> picked;
        for (int i = 0; i < preset.cache && i < static_cast<int>(flat.size()); ++i) {
            picked.push_back(flat[i].value == -INFINITY ? sentinel : flat[i].index);
        }
        std::sort(picked.begin(), picked.end());
        std::vector<std::vector<float>> keptEmbeds;
        std::vector<std::vector<float>> keptProbs;
        const std::vector<float> zero(kSpeakers, 0.f);
        for (int index : picked) {
            const int frame = index == sentinel ? count : std::min(index % scored, count);
            if (frame == count) {
                keptEmbeds.push_back(silence);
                keptProbs.push_back(zero);
            } else {
                keptEmbeds.push_back(cacheEmbeds[frame]);
                keptProbs.push_back(cacheProbs[frame]);
            }
        }
        cacheEmbeds = std::move(keptEmbeds);
        cacheProbs = std::move(keptProbs);
    }
};

}  // namespace

// --- NemotronCpuRuntime ---

struct NemotronCpuRuntime::Impl {
  std::unique_ptr<Ort::Env> env;
  std::unique_ptr<Ort::Session> embed;
  std::unique_ptr<Ort::Session> step;
  std::vector<float> mel;
  std::vector<float> silence;
  int32_t threads = 1;
  std::unique_ptr<SpeakerCache> cache;
  int64_t last_end_samples = -1;
  Preset preset;
  std::vector<float> timeline_scratch;
};

static Ort::Session MakeSession(Ort::Env& env, const std::string& path, int threads) {
  Ort::SessionOptions options;
  options.SetIntraOpNumThreads(std::max(1, threads));
  options.SetInterOpNumThreads(1);
  options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  return Ort::Session(env, path.c_str(), options);
}

NemotronCpuRuntime::NemotronCpuRuntime(const SessionConfig& cfg) : impl_(std::make_unique<Impl>()) {
  auto& s = *impl_;
  s.threads = std::max(1, cfg.num_threads);
  s.env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "nemotron-cpu");
  s.embed = std::make_unique<Ort::Session>(MakeSession(*s.env, cfg.embed_onnx, s.threads));
  s.step = std::make_unique<Ort::Session>(MakeSession(*s.env, cfg.step_onnx, s.threads));
  s.mel = readFloatFile(cfg.mel_filters, static_cast<size_t>(kMels) * (kNfft / 2 + 1));
  s.silence = readFloatFile(cfg.silence_embeds, kHidden);
  s.preset = Preset{};
  s.cache = std::make_unique<SpeakerCache>(s.silence, s.preset);
}

NemotronCpuRuntime::~NemotronCpuRuntime() = default;

void NemotronCpuRuntime::ResetContinuity() {
  impl_->cache->clear();
  impl_->last_end_samples = -1;
}

bool NemotronCpuRuntime::AcceptWaveform(const float* samples, int32_t sample_count, int64_t offset_samples, TimelineView* out) {
  if (!samples || sample_count <= 0 || !out) return false;
  auto& state = *impl_;
  out->frame_count = 0;
  out->timeline = nullptr;
  try {
    int frame_count = 0;
    auto features = logMel(samples, sample_count, state.mel, &frame_count);
    if (frame_count <= 0) return true;
    const int embed_count = (frame_count + kSub - 1) / kSub;
    const char* feature_inputs[] = {"features"};
    const char* feature_outputs[] = {"embeds"};
    const char* step_inputs[] = {"embeds"};
    const char* step_outputs[] = {"logits"};
    std::vector<int64_t> feature_shape{1, frame_count, kMels};
    auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value feature_tensor = Ort::Value::CreateTensor<float>(
        memory, features.data(), features.size(), feature_shape.data(), feature_shape.size());
    auto embed_output = state.embed->Run(Ort::RunOptions{nullptr}, feature_inputs, &feature_tensor, 1, feature_outputs, 1);
    const auto embed_shape = embed_output[0].GetTensorTypeAndShapeInfo().GetShape();
    if (embed_shape.size() != 3 || embed_shape[1] != embed_count || embed_shape[2] != kHidden) {
      ResetContinuity();
      return false;
    }
    const float* embed_data = embed_output[0].GetTensorData<float>();
    std::vector<std::vector<float>> embed_frames(embed_count, std::vector<float>(kHidden));
    for (int i = 0; i < embed_count; ++i) {
      std::copy(embed_data + static_cast<size_t>(i) * kHidden,
                embed_data + static_cast<size_t>(i + 1) * kHidden, embed_frames[i].begin());
    }
    auto& cache = *state.cache;
    if (state.last_end_samples >= 0) {
      const int64_t gap_samples = offset_samples - state.last_end_samples;
      if (gap_samples < 0 || gap_samples > kCacheContinuityMaxGapSamples) {
        ResetContinuity();
      } else {
        const int64_t frame_samples = static_cast<int64_t>(kHop) * kSub;
        const int silence_frames = static_cast<int>((gap_samples + frame_samples - 1) / frame_samples);
        cache.insertSilence(silence_frames);
      }
    }
    int written = 0;
    state.timeline_scratch.assign(static_cast<size_t>(frame_count) * kSpeakers, 0.f);
    for (int start = 0; start < embed_count; start += kChunk) {
      const int end = std::min(start + kChunk, embed_count);
      const int chunk_count = end - start;
      const int right_end = std::min(end + kRight, embed_count);
      auto cached = cache.embedsWithFifo();
      std::vector<std::vector<float>> incoming = cached;
      incoming.insert(incoming.end(), embed_frames.begin() + start, embed_frames.begin() + right_end);
      const int length = static_cast<int>(incoming.size());
      std::vector<float> buffer(static_cast<size_t>(length) * kHidden);
      for (int i = 0; i < length; ++i) {
        std::copy(incoming[i].begin(), incoming[i].end(), buffer.begin() + static_cast<size_t>(i) * kHidden);
      }
      std::vector<int64_t> step_shape{1, length, kHidden};
      Ort::Value step_tensor = Ort::Value::CreateTensor<float>(
          memory, buffer.data(), buffer.size(), step_shape.data(), step_shape.size());
      auto logits_output = state.step->Run(Ort::RunOptions{nullptr}, step_inputs, &step_tensor, 1, step_outputs, 1);
      const float* logits = logits_output[0].GetTensorData<float>();
      std::vector<std::vector<float>> pooled;
      pooled.reserve(length);
      for (int time = 0; time < length; ++time) {
        std::vector<float> probability(kSpeakers, 0.f);
        for (int sub = 0; sub < kSub; ++sub) {
          const float* row = logits + (static_cast<size_t>(time) * kSub + sub) * kSpeakers;
          for (int speaker = 0; speaker < kSpeakers; ++speaker) probability[speaker] += sigmoid(row[speaker]);
        }
        for (int speaker = 0; speaker < kSpeakers; ++speaker) probability[speaker] /= kSub;
        pooled.push_back(std::move(probability));
      }
      const int base = static_cast<int>(cached.size()) * kSub;
      const int frame_budget = chunk_count * kSub;
      for (int i = 0; i < frame_budget && written < frame_count; ++i) {
        const float* row = logits + static_cast<size_t>(base + i) * kSpeakers;
        for (int speaker = 0; speaker < kSpeakers; ++speaker) {
          state.timeline_scratch[static_cast<size_t>(written) * kSpeakers + speaker] = sigmoid(row[speaker]);
        }
        ++written;
      }
      cache.update(incoming, pooled, chunk_count);
    }
    state.last_end_samples = offset_samples + sample_count;
    out->frame_count = frame_count;
    out->num_speakers = kSpeakers;
    out->timeline = state.timeline_scratch.data();
    return true;
  } catch (const std::exception&) {
    ResetContinuity();
    return false;
  }
}

}  // namespace transai::nemotron::cpu
