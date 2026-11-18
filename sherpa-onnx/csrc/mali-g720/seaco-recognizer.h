#pragma once
#include "sherpa-onnx/csrc/offline-recognizer-impl.h"
namespace sherpa_onnx {
std::unique_ptr<OfflineRecognizerImpl> CreateSeacoMaliG720Recognizer(const OfflineRecognizerConfig &config);
}
