#include "runtime.h"

#include <onnxruntime_cxx_api.h>
#include <arm_neon.h>
#include <asm/hwcap.h>
#include <sys/auxv.h>
#include <sys/mman.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <iomanip>
#include <limits>
#include <locale>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "sherpa-onnx/csrc/neon-ort-session.h"

namespace transai::nemotron {
void RegisterOps(Ort::SessionOptions &options);

namespace {
constexpr int rate = 16000, hop = 160, commit_samples = 51200;
constexpr int fft = 512, bins = 257, bands = 128, width = 512, speakers = 8, factor = 8;
// Medium geometry documented in docs/NEMOTRON_NEON.md.
constexpr int right_samples = factor * hop, cache_limit = 188, fifo_limit = 188, update_period = 128;
constexpr int budget = cache_limit / speakers - 1;
constexpr int positive_min = budget / 2, strong_count = budget * 3 / 4, weak_count = budget * 3 / 2;
constexpr int minimum_on = 30, minimum_off = 50;  // 10 ms output frames
using Prob = std::array<float, speakers>;
using Embedding = std::array<float, width>;

void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

template<class Array> void read_floats(const aot::File& file, const char* name, Array& result) {
    const auto region = file.Asset(name);
    require(region.bytes == sizeof(result), "Invalid Nemotron model constant");
    file.Read(region, result.data());
}

class Frontend {
    std::array<float, bands * bins> mel_{};
    std::array<std::pair<int, int>, bands> mel_support_{};
    std::array<float, 400> window_{};
    std::array<int, fft> reverse_{};
    std::array<std::complex<float>, fft - 1> roots_{};

public:
    explicit Frontend(const aot::File& model) {
        read_floats(model, "diarization/mel_filters.bin", mel_);
        for (int band = 0; band < bands; ++band) {
            const auto first = mel_.begin() + band * bins;
            const auto last = first + bins;
            const auto nonzero = [](float value) { return value != 0; };
            const auto start = std::find_if(first, last, nonzero);
            const auto end = std::find_if(std::make_reverse_iterator(last),
                                          std::make_reverse_iterator(start), nonzero).base();
            mel_support_[band] = {int(start - first), int(end - first)};
        }
        const double pi = std::acos(-1.0);
        for (int i = 0; i < 400; ++i) window_[i] = 0.5f - 0.5f * std::cos(float(2 * pi / 399) * i);
        for (int i = 0; i < fft; ++i) {
            reverse_[i] = (reverse_[i >> 1] >> 1) | ((i & 1) << 8);
        }
        for (int half = 1; half < fft; half *= 2) {
            for (int k = 0; k < half; ++k) {
                const int i = k * fft / (2 * half);
                roots_[half - 1 + k] = {float(std::cos(-2*pi*i/fft)), float(std::sin(-2*pi*i/fft))};
            }
        }
    }

    void extract(const int16_t* pcm, int start, int end,
                 std::vector<float>& features) const {
        const int frames = (end - start) / hop;
        features.resize(frames * bands);
        for (int frame = 0; frame < frames; ++frame) {
            std::array<std::complex<float>, fft> x{};
            for (int i = 0; i < 400; ++i) {
                const int j = start + frame * hop + i - 200;
                if (j >= 0 && j < end) {
                    const float now = pcm[j] / 32768.0f;
                    const float previous = j ? pcm[j-1] / 32768.0f : 0;
                    x[reverse_[i + 56]] = (now - 0.97f * previous) * window_[i];
                }
            }
            for (int half = 1; half < 4; half *= 2) {
                for (int begin = 0; begin < fft; begin += 2 * half) {
                    for (int k = 0; k < half; ++k) {
                        const auto a = x[begin + k];
                        const auto b = x[begin + k + half] * roots_[half - 1 + k];
                        x[begin + k] = a + b;
                        x[begin + k + half] = a - b;
                    }
                }
            }
            for (int half = 4; half < fft; half *= 2) {
                const auto* roots = reinterpret_cast<const float*>(roots_.data() + half - 1);
                for (int begin = 0; begin < fft; begin += 2 * half) {
                    auto* lower = reinterpret_cast<float*>(x.data() + begin);
                    auto* upper = reinterpret_cast<float*>(x.data() + begin + half);
                    for (int k = 0; k < half; k += 4) {
                        const auto a = vld2q_f32(lower + 2 * k);
                        const auto b = vld2q_f32(upper + 2 * k);
                        const auto twiddle = vld2q_f32(roots + 2 * k);
                        const auto real = vsubq_f32(vmulq_f32(b.val[0], twiddle.val[0]), vmulq_f32(b.val[1], twiddle.val[1]));
                        const auto imag = vaddq_f32(vmulq_f32(b.val[0], twiddle.val[1]), vmulq_f32(b.val[1], twiddle.val[0]));
                        const float32x4x2_t sum{{vaddq_f32(a.val[0], real), vaddq_f32(a.val[1], imag)}};
                        const float32x4x2_t difference{{vsubq_f32(a.val[0], real), vsubq_f32(a.val[1], imag)}};
                        vst2q_f32(lower + 2 * k, sum);
                        vst2q_f32(upper + 2 * k, difference);
                    }
                }
            }
            std::array<float, bins> power;
            const auto* spectrum = reinterpret_cast<const float*>(x.data());
            for (int k = 0; k < bins - 1; k += 4) {
                const auto bin = vld2q_f32(spectrum + 2 * k);
                vst1q_f32(power.data() + k, vaddq_f32(vmulq_f32(bin.val[0], bin.val[0]),
                                                     vmulq_f32(bin.val[1], bin.val[1])));
            }
            power.back() = std::norm(x[bins - 1]);
            for (int band = 0; band < bands; ++band) {
                const auto begin = mel_.begin() + band * bins;
                const auto [low, high] = mel_support_[band];
                const float energy = std::inner_product(begin + low, begin + high, power.begin() + low, 0.0f);
                features[frame * bands + band] = std::log(energy + 0x1p-24f);
            }
        }
    }
};

struct Cache {
    std::vector<Embedding> embeddings, fifo, input;
    std::vector<Prob> stored;
    bool compressed = false;
    size_t generation = 0, prefix_generation = 0, prefix_reuses = 0;
    std::vector<Ort::Value> layer_prefix;

    using Score = std::pair<float, int>;
    static void top_k(std::vector<Score>& values, int k) {
        const auto greater = [](const Score& a, const Score& b) { return a.first > b.first; };
        if (k * 64 <= int(values.size())) {
            std::partial_sort(values.begin(), values.begin() + k, values.end(), greater);
        } else {
            std::nth_element(values.begin(), values.begin() + k - 1, values.end(), greater);
        }
    }

    void compress(const Embedding& silence) {
        const int n = int(stored.size());
        const float inf = std::numeric_limits<float>::infinity();
        std::vector<Prob> scores(n);
        std::array<int, speakers> positive{};
        for (int frame = 0; frame < n; ++frame) {
            Prob complements;
            for (int s = 0; s < speakers; ++s) complements[s] = std::log(std::max(1 - stored[frame][s], 0.25f));
            const float sum = std::accumulate(complements.begin(), complements.end(), 0.0f);
            for (int s = 0; s < speakers; ++s) {
                const float p = stored[frame][s];
                const float score = p > 0.5f ? std::log(p) - complements[s] + sum - std::log(0.5f) : -inf;
                scores[frame][s] = score;
                positive[s] += score > 0;
            }
        }
        std::vector<Score> flat;
        flat.reserve((n + 1) * speakers);
        for (int s = 0; s < speakers; ++s) {
            std::vector<Score> ranked;
            ranked.reserve(n);
            for (int frame = 0; frame < n; ++frame) {
                float& score = scores[frame][s];
                if (positive[s] >= positive_min && score <= 0) score = -inf;
                if (frame >= cache_limit) score += 0.05f;
                ranked.emplace_back(score, frame);
            }
            top_k(ranked, strong_count);
            for (int i = 0; i < strong_count; ++i) scores[ranked[i].second][s] -= 2 * std::log(0.5f);
            for (auto& value : ranked) value.first = scores[value.second][s];
            std::sort(ranked.begin(), ranked.end(), [](const Score& a, const Score& b) { return a.second < b.second; });
            top_k(ranked, weak_count);
            for (int i = 0; i < weak_count; ++i) scores[ranked[i].second][s] -= std::log(0.5f);
            for (int frame = 0; frame < n; ++frame) flat.emplace_back(scores[frame][s], s*(n+1) + frame);
            flat.emplace_back(inf, s*(n+1) + n);
        }
        top_k(flat, cache_limit);
        flat.resize(cache_limit);
        for (auto& value : flat) if (value.first == -inf) value.second = (n+1) * speakers;
        std::sort(flat.begin(), flat.end(), [](const Score& a, const Score& b) { return a.second < b.second; });
        std::vector<Embedding> next_embeddings;
        std::vector<Prob> next_probs;
        next_embeddings.reserve(cache_limit);
        next_probs.reserve(cache_limit);
        for (const auto& value : flat) {
            const int frame = value.second % (n+1);
            const bool silent = frame == n || value.first == -inf;
            next_embeddings.push_back(silent ? silence : embeddings[frame]);
            next_probs.push_back(silent ? Prob{} : stored[frame]);
        }
        embeddings.swap(next_embeddings);
        stored.swap(next_probs);
        compressed = true;
    }

    void update(int committed, const float* logits, int logit_base,
                const Embedding& silence) {
        const int cached = int(embeddings.size());
        const int fifo_size = int(fifo.size()) + committed;
        const int popped = fifo_size > fifo_limit ? std::min(fifo_size, std::max(update_period, fifo_size - fifo_limit)) : 0;
        if (popped) {
            const int new_size = cached + popped;
            stored.resize(new_size);
            for (int frame = compressed ? cached : 0; frame < new_size; ++frame) {
                for (int s = 0; s < speakers; ++s) {
                    float sum = 0;
                    for (int i = 0; i < factor; ++i)
                        sum += 1 / (1 + std::exp(-logits[((frame-logit_base)*factor+i)*speakers+s]));
                    stored[frame][s] = sum / factor;
                }
            }
            embeddings.insert(embeddings.end(), input.begin() + (cached - logit_base), input.begin() + (new_size - logit_base));
            if (new_size > cache_limit) compress(silence);
            ++generation;
        }
        fifo.assign(input.begin() + (cached + popped - logit_base),
                    input.begin() + (cached + fifo_size - logit_base));
    }
};

struct Turn { int first, last, speaker; };

class TurnCollector {
    std::array<std::vector<Turn>, speakers> tracks_;
    std::array<int, speakers> first_;
    int frames_ = 0;

    void Close(int s) {
        int& first = first_[s];
        if (first < 0) return;
        auto& track = tracks_[s];
        if (!track.empty() && first - track.back().last < minimum_off) {
            track.back().last = frames_;
        } else {
            track.push_back({first, frames_, s});
        }
        first = -1;
    }

public:
    TurnCollector() { first_.fill(-1); }

    void Append(const float* logits, int count) {
        for (int frame = 0; frame < count; ++frame, ++frames_) {
            for (int s = 0; s < speakers; ++s) {
                const float logit = logits[frame * speakers + s];
                require(std::isfinite(logit), "Non-finite logit");
                if (1 / (1 + std::exp(-logit)) > 0.5f) {
                    if (first_[s] < 0) first_[s] = frames_;
                } else {
                    Close(s);
                }
            }
        }
    }

    int Frames() const { return frames_; }

    std::vector<Turn> Finish() {
        std::vector<Turn> turns;
        for (int s = 0; s < speakers; ++s) {
            Close(s);
            for (const auto& t : tracks_[s]) {
                if (t.last - t.first >= minimum_on) turns.push_back(t);
            }
        }
        std::sort(turns.begin(), turns.end(), [](const Turn& a, const Turn& b) {
            return std::tie(a.first, a.last, a.speaker) < std::tie(b.first, b.last, b.speaker);
        });
        return turns;
    }
};

class Model {
    static constexpr size_t layers = 31;
    Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "nemotron_medium"};
    std::unique_ptr<aot::Mapping> embed_mapping_, step_mapping_;
    Ort::Session embed_{nullptr}, step_{nullptr};
    Ort::MemoryInfo memory_ = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Embedding silence_{};
    static constexpr size_t max_prefix_reuse_ = 1;
    std::array<std::string, layers> prefix_inputs_, prefix_outputs_;

    Ort::Value forward(Ort::Session& session, float* input, int rows, int columns,
                       const char* input_name, const char* output_name) {
        const std::array<int64_t, 3> shape{1, rows, columns};
        auto tensor = Ort::Value::CreateTensor<float>(memory_, input, rows * columns, shape.data(), shape.size());
        auto output = session.Run(Ort::RunOptions{nullptr}, &input_name, &tensor, 1, &output_name, 1);
        return std::move(output.front());
    }

    Ort::Value forward_with_prefix(float* embeds, int rows, int capture, bool reuse, Cache& cache) {
        const std::array<int64_t, 3> shape{1, rows, width}, empty_shape{1, 0, 2*width};
        const std::array<int64_t, 1> scalar_shape{1};
        int64_t capture_value = capture;
        Ort::Float16_t dummy{};
        auto input = Ort::Value::CreateTensor<float>(memory_, embeds, rows*width, shape.data(), shape.size());
        auto keep = Ort::Value::CreateTensor<int64_t>(memory_, &capture_value, 1, scalar_shape.data(), scalar_shape.size());
        auto empty = Ort::Value::CreateTensor<Ort::Float16_t>(memory_, &dummy, 0, empty_shape.data(), empty_shape.size());
        std::array<const char*, layers+2> input_names{};
        std::array<const OrtValue*, layers+2> inputs{};
        std::array<const char*, layers+1> output_names{};
        std::array<OrtValue*, layers+1> outputs{};
        input_names[0] = "embeds"; inputs[0] = input;
        input_names[1] = "capture_prefix"; inputs[1] = keep;
        output_names[0] = "logits";
        for (size_t i = 0; i < layers; ++i) {
            input_names[i+2] = prefix_inputs_[i].c_str();
            inputs[i+2] = reuse ? static_cast<const OrtValue*>(cache.layer_prefix[i]) : static_cast<const OrtValue*>(empty);
            output_names[i+1] = prefix_outputs_[i].c_str();
        }
        auto* status = Ort::GetApi().Run(step_, nullptr, input_names.data(), inputs.data(), inputs.size(),
                                        output_names.data(), capture ? outputs.size() : 1, outputs.data());
        if (status) {
            for (auto* value : outputs) if (value) Ort::GetApi().ReleaseValue(value);
            Ort::ThrowOnError(status);
        }
        if (capture) {
            cache.layer_prefix.clear();
            for (size_t i = 0; i < layers; ++i) cache.layer_prefix.emplace_back(outputs[i+1]);
            cache.prefix_generation = cache.generation;
            cache.prefix_reuses = 0;
        } else if (reuse) {
            ++cache.prefix_reuses;
        }
        return Ort::Value(outputs[0]);
    }

public:
    Model(const std::string& path, const aot::File& file, int num_threads) {
        Ort::SessionOptions options;
        options.SetIntraOpNumThreads(num_threads).SetInterOpNumThreads(1)
            .SetExecutionMode(ORT_SEQUENTIAL).SetGraphOptimizationLevel(ORT_ENABLE_ALL)
            .AddConfigEntry("session.force_spinning_stop", "1");
        RegisterOps(options);
        aot::RequireMappedOrtGraph(file);
        aot::SetMappedOrtGraphOptions(options);
        embed_mapping_ = std::make_unique<aot::Mapping>(path + "/diarization/embed.onnx");
        step_mapping_ = std::make_unique<aot::Mapping>(path + "/diarization/step_int8.onnx");
        embed_ = Ort::Session(env_, embed_mapping_->data(), embed_mapping_->size(), options);
        step_ = Ort::Session(env_, step_mapping_->data(), step_mapping_->size(), options);
        // Packed GEMM constants now live in aligned storage. Keep the mapping
        // valid for ORT metadata/other constants, but discard its resident pages.
        require(madvise(const_cast<unsigned char*>(step_mapping_->data()), step_mapping_->size(),
                        MADV_DONTNEED) == 0, "Cannot discard copied Nemotron graph pages");
        require(step_.GetInputCount() == layers + 2 && step_.GetOutputCount() == layers + 1,
                "Invalid Nemotron prefix graph I/O");
        for (size_t i = 0; i < layers; ++i) {
            prefix_inputs_[i] = "prefix_" + std::to_string(i);
            prefix_outputs_[i] = "captured_prefix_" + std::to_string(i);
        }
        read_floats(file, "diarization/silence_embeds.bin", silence_);
    }

    void appendTimeline(const float* logits, int hop_frames, std::vector<float>* timeline) const {
        if (!timeline) return;
        timeline->reserve(timeline->size() + size_t(hop_frames) * speakers);
        for (int frame = 0; frame < hop_frames; ++frame) {
            for (int s = 0; s < speakers; ++s) {
                const float logit = logits[frame * speakers + s];
                timeline->push_back(1.f / (1.f + std::exp(-logit)));
            }
        }
    }

    void submit(std::vector<float>& features, int emit_frames, Cache& cache, TurnCollector* turns,
                std::vector<float>* timeline) {
        const int frames = int(features.size()) / bands;
        const int length = (frames + factor - 1) / factor;
        auto embed = forward(embed_, features.data(), frames, bands, "features", "embeds");
        require(embed.GetTensorTypeAndShapeInfo().GetElementCount() == size_t(length * width), "Invalid embed shape");
        const int prior = int(cache.embeddings.size() + cache.fifo.size());
        const int speaker_frames = int(cache.embeddings.size());
        const int committed = (emit_frames + factor - 1) / factor;
        const bool reuse = speaker_frames && cache.layer_prefix.size() == layers &&
            cache.prefix_generation == cache.generation && cache.prefix_reuses < max_prefix_reuse_ &&
            (cache.compressed || int(cache.fifo.size()) + committed <= fifo_limit);
        const int skip = reuse ? speaker_frames : 0;
        const int step_frames = prior + length - skip;
        auto& input = cache.input;
        input.clear();
        input.reserve(step_frames);
        if (!reuse) input.insert(input.end(), cache.embeddings.begin(), cache.embeddings.end());
        input.insert(input.end(), cache.fifo.begin(), cache.fifo.end());
        const auto* data = embed.GetTensorData<float>();
        for (int frame = 0; frame < length; ++frame) {
            Embedding& e = input.emplace_back();
            std::copy_n(data + frame * width, width, e.begin());
        }
        auto result = forward_with_prefix(input.front().data(), step_frames,
                                          reuse ? 0 : speaker_frames, reuse, cache);
        require(result.GetTensorTypeAndShapeInfo().GetElementCount() == size_t(step_frames*factor*speakers), "Invalid logits shape");
        const auto* logits = result.GetTensorData<float>();
        if (timeline) {
            appendTimeline(logits + (prior - skip) * factor * speakers, emit_frames, timeline);
        }
        cache.update(committed, logits, skip, silence_);
        if (turns) turns->Append(logits + (prior - skip) * factor * speakers, emit_frames);
    }

    const Embedding& silence() const { return silence_; }
};

constexpr int kNeonSampleRate = 16000;
constexpr int kNeonHop = 160;
constexpr int kNeonSub = 8;
constexpr int64_t kNeonCacheContinuityMaxGapSamples = 3 * kNeonSampleRate;

void InsertSilence(Cache& cache, const Embedding& silence, int frames) {
    for (int i = 0; i < frames; ++i) cache.fifo.push_back(silence);
}

}  // namespace

struct Runtime::Impl {
    Impl(const std::string& path, const aot::File& file, int num_threads)
        : frontend(file), model(path, file, num_threads) {}
    Frontend frontend;
    Model model;
};

struct Runtime::SessionStorage {
    Cache cache;
    int64_t last_end_samples = -1;
    std::string rttm_storage;
};

Runtime::Runtime(const std::string& path, int num_threads) {
    const auto hw = getauxval(AT_HWCAP2);
    require((getauxval(AT_HWCAP) & HWCAP_ASIMDHP) && (hw & HWCAP2_I8MM) && (hw & HWCAP2_BF16),
            "Nemotron NEON requires Arm FP16, I8MM and BF16");
    const auto file = aot::File::Open(path);
    const auto& manifest = file->manifest();
    const auto& architecture = manifest.at("architecture");
    const auto& target = manifest.at("aot").at("target");
    require(architecture.at("family") == "nemotron-diarization" &&
            architecture.at("version") == 1 && architecture.at("precision") == "w8a8-bf16" &&
            manifest.at("aot").at("format") == 1 && target.at("preset") == "medium" &&
            target.at("max_prefix_reuses") == 1 &&
            target.at("rhs_packing") == "kai_i8mm_4x8_v1" &&
            file->Asset("diarization/embed.onnx").bytes <= 8 * 1024 * 1024 &&
            file->Asset("diarization/step_int8.onnx").bytes <= 128 * 1024 * 1024,
            "Expected a Nemotron NEON v1.0 AOT");
    impl_ = std::make_unique<Impl>(path, *file, num_threads);
}

Runtime::~Runtime() = default;

void Runtime::ResetContinuity() {
    if (!session_) session_ = std::make_unique<SessionStorage>();
    session_->cache = Cache{};
    session_->last_end_samples = -1;
    session_->rttm_storage.clear();
}

bool Runtime::AcceptWaveform(const int16_t* pcm, int32_t count, int64_t offset_samples, NeonTimelineView* out) {
    if (!pcm || count <= 0 || !out || !impl_) return false;
    if (count > static_cast<int32_t>(kMaxSamples)) return false;
    if (!session_) session_ = std::make_unique<SessionStorage>();
    out->frame_count = 0;
    out->timeline = nullptr;
    out->rttm = nullptr;
    out->num_speakers = 8;
    auto& state = *session_;
    state.rttm_storage.clear();
    try {
        if (state.last_end_samples >= 0) {
            const int64_t gap = offset_samples - state.last_end_samples;
            if (gap < 0 || gap > kNeonCacheContinuityMaxGapSamples) {
                ResetContinuity();
            } else if (gap > 0) {
                const int64_t frame_samples = static_cast<int64_t>(kNeonHop) * kNeonSub;
                const int silence_frames = static_cast<int>((gap + frame_samples - 1) / frame_samples);
                InsertSilence(state.cache, impl_->model.silence(), silence_frames);
            }
        }
        TurnCollector turns;
        std::vector<float> features;
        const int samples = count;
        constexpr int commit = 51200;
        constexpr int right_pad = 1280;
        for (int start = 0; start < samples / kNeonHop * kNeonHop; start += commit) {
            const int end = std::min(start + commit, samples);
            const int available = std::min(end + right_pad, samples);
            impl_->frontend.extract(pcm, start, available, features);
            const int emit_frames = (end - start) / kNeonHop;
            impl_->model.submit(features, emit_frames, state.cache, &turns, nullptr);
        }
        require(turns.Frames() == samples / kNeonHop, "Incomplete frame coverage");
        const std::string recording_id = "seg";
        std::ostringstream rttm;
        rttm.imbue(std::locale::classic());
        rttm << std::fixed << std::setprecision(2);
        for (const auto& t : turns.Finish()) {
            rttm << "SPEAKER " << recording_id << " 1 " << t.first / 100.0 << ' '
                 << (t.last - t.first) / 100.0 << " <NA> <NA> spk" << t.speaker << " <NA> <NA>\n";
        }
        state.last_end_samples = offset_samples + count;
        state.rttm_storage = rttm.str();
        if (!state.rttm_storage.empty()) {
            out->rttm = state.rttm_storage.c_str();
        }
        return true;
    } catch (const std::exception&) {
        ResetContinuity();
        return false;
    }
}

std::string Runtime::Process(const int16_t* pcm, size_t count, const std::string& id) const {
    require(!id.empty() && id.find_first_not_of(
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.-") == std::string::npos,
            "RTTM recording ID must contain only letters, digits, dots, underscores or hyphens");
    require(pcm && count > 0 && count <= kMaxSamples, "Invalid Nemotron PCM length");
    Cache cache;
    TurnCollector turns;
    std::vector<float> features;
    const int samples = static_cast<int>(count);
    constexpr int commit = 51200;
    constexpr int right_pad = 1280;
    for (int start = 0; start < samples / kNeonHop * kNeonHop; start += commit) {
        const int end = std::min(start + commit, samples);
        const int available = std::min(end + right_pad, samples);
        impl_->frontend.extract(pcm, start, available, features);
        impl_->model.submit(features, (end - start) / kNeonHop, cache, &turns, nullptr);
    }
    require(turns.Frames() == samples / kNeonHop, "Incomplete frame coverage");
    std::ostringstream rttm;
    rttm.imbue(std::locale::classic());
    rttm << std::fixed << std::setprecision(2);
    for (const auto& t : turns.Finish()) {
        rttm << "SPEAKER " << id << " 1 " << t.first / 100.0 << ' '
             << (t.last - t.first) / 100.0 << " <NA> <NA> spk" << t.speaker << " <NA> <NA>\n";
    }
    return rttm.str();
}
}  // namespace transai::nemotron
