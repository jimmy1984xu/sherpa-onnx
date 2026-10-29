#ifndef SHERPA_ONNX_CSRC_MALI_G720_OFFLINE_RECOGNIZER_H_
#define SHERPA_ONNX_CSRC_MALI_G720_OFFLINE_RECOGNIZER_H_
#include <memory>
#include "sherpa-onnx/csrc/offline-recognizer-impl.h"
namespace sherpa_onnx {
std::unique_ptr<OfflineRecognizerImpl> CreateMaliG720Recognizer(
    const OfflineRecognizerConfig &config);
}
#endif
