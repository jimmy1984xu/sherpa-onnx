#!/usr/bin/env bash
set -eo pipefail
cd /speech_store/jimmy/k2_origin/sherpa-onnx
test "$(git rev-parse HEAD)" = '08a9fe96a74583a776186c1fce8ebe72de7493d5'
test -d /speech_store/jimmy/android/android-ndk-r28c
if test -e build-android-arm64-v8a; then rm -r -- build-android-arm64-v8a; fi
export ANDROID_NDK=/speech_store/jimmy/android/android-ndk-r28c
./build-android-transai-arm64-v8a.sh
lib=build-android-arm64-v8a/install/lib
sha256sum "$lib"/*.so sherpa-onnx/c-api/c-api.h