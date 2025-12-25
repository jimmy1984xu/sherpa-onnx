// sherpa-onnx/jni/speaker-embedding-manager.cc
//
// Copyright (c)  2024  Xiaomi Corporation
#include "sherpa-onnx/csrc/speaker-embedding-manager.h"

#include <string>
#include <vector>

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/jni/common.h"

SHERPA_ONNX_EXTERN_C
JNIEXPORT jlong JNICALL
Java_com_k2fsa_sherpa_onnx_SpeakerEmbeddingManager_create(JNIEnv *env,
                                                          jobject /*obj*/,
                                                          jint dim) {
  auto p = new sherpa_onnx::SpeakerEmbeddingManager(dim);
  return (jlong)p;
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT void JNICALL
Java_com_k2fsa_sherpa_onnx_SpeakerEmbeddingManager_delete(JNIEnv * /*env*/,
                                                          jobject /*obj*/,
                                                          jlong ptr) {
  auto manager = reinterpret_cast<sherpa_onnx::SpeakerEmbeddingManager *>(ptr);
  delete manager;
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jboolean JNICALL
Java_com_k2fsa_sherpa_onnx_SpeakerEmbeddingManager_add(JNIEnv *env,
                                                       jobject /*obj*/,
                                                       jlong ptr, jstring name,
                                                       jfloatArray embedding) {
  auto manager = reinterpret_cast<sherpa_onnx::SpeakerEmbeddingManager *>(ptr);

  jfloat *p = env->GetFloatArrayElements(embedding, nullptr);
  jsize n = env->GetArrayLength(embedding);

  if (n != manager->Dim()) {
    SHERPA_ONNX_LOGE("Expected dim %d, given %d", manager->Dim(),
                     static_cast<int32_t>(n));
    exit(-1);
  }

  const char *p_name = env->GetStringUTFChars(name, nullptr);

  jboolean ok = manager->Add(p_name, p);
  env->ReleaseStringUTFChars(name, p_name);
  env->ReleaseFloatArrayElements(embedding, p, JNI_ABORT);

  return ok;
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jboolean JNICALL
Java_com_k2fsa_sherpa_onnx_SpeakerEmbeddingManager_addList(
    JNIEnv *env, jobject /*obj*/, jlong ptr, jstring name,
    jobjectArray embedding_arr) {
  auto manager = reinterpret_cast<sherpa_onnx::SpeakerEmbeddingManager *>(ptr);

  int num_embeddings = env->GetArrayLength(embedding_arr);
  if (num_embeddings == 0) {
    return false;
  }

  std::vector<std::vector<float>> embedding_list;
  embedding_list.reserve(num_embeddings);
  for (int32_t i = 0; i != num_embeddings; ++i) {
    jfloatArray embedding =
        (jfloatArray)env->GetObjectArrayElement(embedding_arr, i);

    jfloat *p = env->GetFloatArrayElements(embedding, nullptr);
    jsize n = env->GetArrayLength(embedding);

    if (n != manager->Dim()) {
      SHERPA_ONNX_LOGE("i: %d. Expected dim %d, given %d", i, manager->Dim(),
                       static_cast<int32_t>(n));
      exit(-1);
    }

    embedding_list.push_back({p, p + n});
    env->ReleaseFloatArrayElements(embedding, p, JNI_ABORT);
  }

  const char *p_name = env->GetStringUTFChars(name, nullptr);

  jboolean ok = manager->Add(p_name, embedding_list);

  env->ReleaseStringUTFChars(name, p_name);

  return ok;
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jboolean JNICALL
Java_com_k2fsa_sherpa_onnx_SpeakerEmbeddingManager_remove(JNIEnv *env,
                                                          jobject /*obj*/,
                                                          jlong ptr,
                                                          jstring name) {
  auto manager = reinterpret_cast<sherpa_onnx::SpeakerEmbeddingManager *>(ptr);

  const char *p_name = env->GetStringUTFChars(name, nullptr);

  jboolean ok = manager->Remove(p_name);

  env->ReleaseStringUTFChars(name, p_name);

  return ok;
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jstring JNICALL
Java_com_k2fsa_sherpa_onnx_SpeakerEmbeddingManager_search(JNIEnv *env,
                                                          jobject /*obj*/,
                                                          jlong ptr,
                                                          jfloatArray embedding,
                                                          jfloat threshold) {
  auto manager = reinterpret_cast<sherpa_onnx::SpeakerEmbeddingManager *>(ptr);

  jfloat *p = env->GetFloatArrayElements(embedding, nullptr);
  jsize n = env->GetArrayLength(embedding);

  if (n != manager->Dim()) {
    SHERPA_ONNX_LOGE("Expected dim %d, given %d", manager->Dim(),
                     static_cast<int32_t>(n));
    exit(-1);
  }

  std::string name = manager->Search(p, threshold);

  env->ReleaseFloatArrayElements(embedding, p, JNI_ABORT);

  return env->NewStringUTF(name.c_str());
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jboolean JNICALL
Java_com_k2fsa_sherpa_onnx_SpeakerEmbeddingManager_verify(
    JNIEnv *env, jobject /*obj*/, jlong ptr, jstring name,
    jfloatArray embedding, jfloat threshold) {
  auto manager = reinterpret_cast<sherpa_onnx::SpeakerEmbeddingManager *>(ptr);

  jfloat *p = env->GetFloatArrayElements(embedding, nullptr);
  jsize n = env->GetArrayLength(embedding);

  if (n != manager->Dim()) {
    SHERPA_ONNX_LOGE("Expected dim %d, given %d", manager->Dim(),
                     static_cast<int32_t>(n));
    exit(-1);
  }

  const char *p_name = env->GetStringUTFChars(name, nullptr);

  jboolean ok = manager->Verify(p_name, p, threshold);

  env->ReleaseFloatArrayElements(embedding, p, JNI_ABORT);

  env->ReleaseStringUTFChars(name, p_name);

  return ok;
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jboolean JNICALL
Java_com_k2fsa_sherpa_onnx_SpeakerEmbeddingManager_contains(JNIEnv *env,
                                                            jobject /*obj*/,
                                                            jlong ptr,
                                                            jstring name) {
  auto manager = reinterpret_cast<sherpa_onnx::SpeakerEmbeddingManager *>(ptr);

  const char *p_name = env->GetStringUTFChars(name, nullptr);

  jboolean ok = manager->Contains(p_name);

  env->ReleaseStringUTFChars(name, p_name);

  return ok;
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jint JNICALL
Java_com_k2fsa_sherpa_onnx_SpeakerEmbeddingManager_numSpeakers(JNIEnv * /*env*/,
                                                               jobject /*obj*/,
                                                               jlong ptr) {
  auto manager = reinterpret_cast<sherpa_onnx::SpeakerEmbeddingManager *>(ptr);
  return manager->NumSpeakers();
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jobjectArray JNICALL
Java_com_k2fsa_sherpa_onnx_SpeakerEmbeddingManager_allSpeakerNames(
    JNIEnv *env, jobject /*obj*/, jlong ptr) {
  auto manager = reinterpret_cast<sherpa_onnx::SpeakerEmbeddingManager *>(ptr);
  std::vector<std::string> all_speakers = manager->GetAllSpeakers();

  jobjectArray obj_arr = (jobjectArray)env->NewObjectArray(
      all_speakers.size(), env->FindClass("java/lang/String"), nullptr);

  int32_t i = 0;
  for (auto &s : all_speakers) {
    jstring js = env->NewStringUTF(s.c_str());
    env->SetObjectArrayElement(obj_arr, i, js);

    ++i;
  }

  return obj_arr;
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jobjectArray JNICALL
Java_com_k2fsa_sherpa_onnx_SpeakerEmbeddingManager_getBestMatches(
    JNIEnv *env, jobject /*obj*/, jlong ptr, jfloatArray embedding,
    jfloat threshold, jint n) {
  auto manager = reinterpret_cast<sherpa_onnx::SpeakerEmbeddingManager *>(ptr);

  jfloat *p = env->GetFloatArrayElements(embedding, nullptr);
  jsize dim = env->GetArrayLength(embedding);

  if (dim != manager->Dim()) {
    SHERPA_ONNX_LOGE("Expected dim %d, given %d", manager->Dim(),
                     static_cast<int32_t>(dim));
    env->ReleaseFloatArrayElements(embedding, p, JNI_ABORT);
    return nullptr;
  }

  auto matches = manager->GetBestMatches(p, threshold, n);

  env->ReleaseFloatArrayElements(embedding, p, JNI_ABORT);

  if (matches.empty()) {
    return nullptr;
  }

  // Find the SpeakerMatch class
  jclass match_class = env->FindClass("com/k2fsa/sherpa/onnx/SpeakerMatch");
  if (match_class == nullptr) {
    SHERPA_ONNX_LOGE("Failed to find SpeakerMatch class");
    return nullptr;
  }

  // Get the constructor
  jmethodID constructor = env->GetMethodID(match_class, "<init>",
                                          "(Ljava/lang/String;F)V");
  if (constructor == nullptr) {
    SHERPA_ONNX_LOGE("Failed to find SpeakerMatch constructor");
    return nullptr;
  }

  // Create array of SpeakerMatch objects
  jobjectArray obj_arr = (jobjectArray)env->NewObjectArray(
      matches.size(), match_class, nullptr);

  for (size_t i = 0; i < matches.size(); ++i) {
    jstring name = env->NewStringUTF(matches[i].name.c_str());
    jobject match_obj = env->NewObject(match_class, constructor, name,
                                        matches[i].score);
    env->SetObjectArrayElement(obj_arr, i, match_obj);
    env->DeleteLocalRef(name);
    env->DeleteLocalRef(match_obj);
  }

  return obj_arr;
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jfloat JNICALL
Java_com_k2fsa_sherpa_onnx_SpeakerEmbeddingManager_score(JNIEnv *env,
                                                          jobject /*obj*/,
                                                          jlong ptr,
                                                          jstring name,
                                                          jfloatArray embedding) {
  auto manager = reinterpret_cast<sherpa_onnx::SpeakerEmbeddingManager *>(ptr);

  jfloat *p = env->GetFloatArrayElements(embedding, nullptr);
  jsize dim = env->GetArrayLength(embedding);

  if (dim != manager->Dim()) {
    SHERPA_ONNX_LOGE("Expected dim %d, given %d", manager->Dim(),
                     static_cast<int32_t>(dim));
    env->ReleaseFloatArrayElements(embedding, p, JNI_ABORT);
    return -2.0f;
  }

  const char *p_name = env->GetStringUTFChars(name, nullptr);

  float score = manager->Score(p_name, p);

  env->ReleaseStringUTFChars(name, p_name);
  env->ReleaseFloatArrayElements(embedding, p, JNI_ABORT);

  return score;
}