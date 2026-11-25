#!/usr/bin/env bash
set -e

# TransAI Android arm64-v8a release profile.
# Keep these values explicit so a release does not depend on generic-script defaults.
export BUILD_SHARED_LIBS=ON
export SHERPA_ONNX_ENABLE_TTS=ON
export SHERPA_ONNX_ENABLE_SPEAKER_DIARIZATION=ON
export SHERPA_ONNX_ENABLE_BINARY=OFF
export SHERPA_ONNX_ENABLE_C_API=ON
export SHERPA_ONNX_ENABLE_JNI=ON
export SHERPA_ONNX_ENABLE_RKNN=OFF
export SHERPA_ONNX_ENABLE_QNN=OFF
export SHERPA_ONNX_ENABLE_MALI_G720=ON
export SHERPA_ONNX_ANDROID_PLATFORM=android-21

script_dir="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
cd "$script_dir"
exec ./build-android-arm64-v8a.sh
