#!/usr/bin/env bash
set -euo pipefail
remote=jimmy
branch='v1.13.2_transai'
expected_commit='08a9fe96a74583a776186c1fce8ebe72de7493d5'
cd /speech_store/jimmy/k2_origin/sherpa-onnx
test -z "$(git status --porcelain)"
git fetch --prune "$remote" "$branch"
if git show-ref --verify --quiet "refs/heads/$branch"; then
  git checkout "$branch"
else
  git checkout -b "$branch" --track "$remote/$branch"
fi
git pull --ff-only "$remote" "$branch"
test "$(git rev-parse HEAD)" = "$expected_commit"
test -d /speech_store/jimmy/android/android-ndk-r28c
if test -e build-android-arm64-v8a; then rm -r -- build-android-arm64-v8a; fi
export ANDROID_NDK=/speech_store/jimmy/android/android-ndk-r28c
./build-android-transai-arm64-v8a.sh
lib=build-android-arm64-v8a/install/lib
header=sherpa-onnx/c-api/c-api.h
test -d "$lib"
find "$lib" -maxdepth 1 -type f -name '*.so' | sort
sha256sum "$lib"/*.so "$header"