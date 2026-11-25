#ifndef SHERPA_ONNX_NEON_TTS_LOAD_PROFILE_H_
#define SHERPA_ONNX_NEON_TTS_LOAD_PROFILE_H_

#include <chrono>
#include <cstdio>
#include <cstdlib>

namespace transai::tts_neon {

template <typename Model>
class LoadProfile {
 public:
  explicit LoadProfile(const char* label, bool trace = true)
      : label_(label), enabled_(trace && std::getenv(Model::env) != nullptr) {
    if (enabled_) start_ = Clock::now();
  }

  ~LoadProfile() {
    if (enabled_) {
      const auto elapsed = std::chrono::duration<double, std::milli>(Clock::now() - start_);
      std::fprintf(stderr, "%s %s %.6f ms\n", Model::prefix, label_, elapsed.count());
    }
  }

 private:
  using Clock = std::chrono::steady_clock;
  const char* label_;
  bool enabled_;
  Clock::time_point start_;
};

}  // namespace transai::tts_neon

namespace transai::matcha_neon {
struct LoadProfileConfig {
  static constexpr const char* env = "MATCHA_PROFILE_LOAD";
  static constexpr const char* prefix = "MATCHA_LOAD";
};
using LoadProfile = tts_neon::LoadProfile<LoadProfileConfig>;
}  // namespace transai::matcha_neon

namespace transai::piper_neon {
struct LoadProfileConfig {
  static constexpr const char* env = "PIPER_PROFILE_LOAD";
  static constexpr const char* prefix = "PIPER_LOAD";
};
using LoadProfile = tts_neon::LoadProfile<LoadProfileConfig>;
}  // namespace transai::piper_neon

#endif  // SHERPA_ONNX_NEON_TTS_LOAD_PROFILE_H_
