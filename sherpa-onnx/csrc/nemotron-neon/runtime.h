#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace transai::nemotron {

inline constexpr std::size_t kMaxSamples = 3600 * 16000;

struct NeonTimelineView {
    int32_t frame_count = 0;
    int32_t num_speakers = 8;
    const float *timeline = nullptr;
    /** 段内 RTTM（段相对时间）；直到下次 AcceptWaveform 前有效。 */
    const char *rttm = nullptr;
};

class Runtime {
public:
    Runtime(const std::string& path, int num_threads);
    ~Runtime();
    std::string Process(const int16_t* pcm, size_t count, const std::string& id) const;

    void ResetContinuity();
    bool AcceptWaveform(const int16_t* pcm, int32_t count, int64_t offset_samples, NeonTimelineView* out);

private:
    struct Impl;
    struct SessionStorage;  // defined in runtime.cc (holds per-session Cache)
    std::unique_ptr<Impl> impl_;
    mutable std::unique_ptr<SessionStorage> session_;
};

}  // namespace transai::nemotron
