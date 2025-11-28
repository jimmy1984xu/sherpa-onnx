// sherpa-onnx/jni/fast-clustering.cc
//
// Copyright (c)  2024  Xiaomi Corporation

#include <vector>

#include "sherpa-onnx/csrc/fast-clustering.h"

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/jni/common.h"

namespace sherpa_onnx {

static FastClusteringConfig GetFastClusteringConfig(JNIEnv *env, jobject config,
                                                     bool *ok) {
  FastClusteringConfig ans;

  jclass cls = env->GetObjectClass(config);

  SHERPA_ONNX_JNI_READ_INT(ans.num_clusters, numClusters, cls, config);

  SHERPA_ONNX_JNI_READ_FLOAT(ans.threshold, threshold, cls, config);

  *ok = true;
  return ans;
}

}  // namespace sherpa_onnx

SHERPA_ONNX_EXTERN_C
JNIEXPORT jlong JNICALL
Java_com_k2fsa_sherpa_onnx_FastClustering_newFromConfig(JNIEnv *env,
                                                         jobject /*obj*/,
                                                         jobject _config) {
  bool ok = false;
  auto config = sherpa_onnx::GetFastClusteringConfig(env, _config, &ok);

  if (!ok) {
    SHERPA_ONNX_LOGE("Failed to get FastClusteringConfig");
    return 0;
  }

  if (!config.Validate()) {
    SHERPA_ONNX_LOGE("Invalid FastClusteringConfig");
    return 0;
  }

  auto clustering = new sherpa_onnx::FastClustering(config);

  return reinterpret_cast<jlong>(clustering);
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT void JNICALL Java_com_k2fsa_sherpa_onnx_FastClustering_delete(
    JNIEnv *env, jobject /*obj*/, jlong ptr) {
  delete reinterpret_cast<sherpa_onnx::FastClustering *>(ptr);
}

SHERPA_ONNX_EXTERN_C
JNIEXPORT jintArray JNICALL
Java_com_k2fsa_sherpa_onnx_FastClustering_cluster(JNIEnv *env, jobject /*obj*/,
                                                   jlong ptr, jfloatArray embeddings,
                                                   jint numSegments,
                                                   jint embeddingDim) {
  auto clustering = reinterpret_cast<sherpa_onnx::FastClustering *>(ptr);

  if (!clustering) {
    SHERPA_ONNX_LOGE("FastClustering pointer is null");
    return nullptr;
  }

  jsize array_size = env->GetArrayLength(embeddings);
  jsize expected_size = numSegments * embeddingDim;
  if (array_size != expected_size) {
    SHERPA_ONNX_LOGE("Embeddings array size (%d) does not match expected size (%d)",
                     array_size, expected_size);
    return nullptr;
  }

  // Create a copy of embeddings since Cluster will modify them in-place
  std::vector<float> embeddings_copy(array_size);
  env->GetFloatArrayRegion(embeddings, 0, array_size, embeddings_copy.data());

  // Perform clustering
  // Note: Cluster method will modify the embeddings in-place (normalization)
  auto labels = clustering->Cluster(embeddings_copy.data(), numSegments, embeddingDim);

  // Create Java int array for labels
  jintArray result = env->NewIntArray(labels.size());
  if (!result) {
    SHERPA_ONNX_LOGE("Failed to create result array");
    return nullptr;
  }

  env->SetIntArrayRegion(result, 0, labels.size(),
                         reinterpret_cast<const jint *>(labels.data()));

  return result;
}

