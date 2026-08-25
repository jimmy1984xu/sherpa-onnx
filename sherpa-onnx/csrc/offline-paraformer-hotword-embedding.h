#ifndef SHERPA_ONNX_CSRC_OFFLINE_PARAFORMER_HOTWORD_EMBEDDING_H_
#define SHERPA_ONNX_CSRC_OFFLINE_PARAFORMER_HOTWORD_EMBEDDING_H_

#include <cstdint>
#include <memory>
#include <vector>

namespace sherpa_onnx {

// Immutable, row-major hotword embeddings shared by streams of one recognizer.
class OfflineParaformerHotwordEmbedding {
 public:
  OfflineParaformerHotwordEmbedding() = default;

  OfflineParaformerHotwordEmbedding(
      std::shared_ptr<const std::vector<float>> storage,
      int32_t num_embeddings, int32_t embedding_dim)
      : storage_(std::move(storage)),
        num_embeddings_(num_embeddings),
        embedding_dim_(embedding_dim) {}

  bool empty() const {
    return !storage_ || storage_->empty() || num_embeddings_ <= 0 ||
           embedding_dim_ <= 0;
  }

  const float* data() const { return empty() ? nullptr : storage_->data(); }

  size_t size() const { return empty() ? 0 : storage_->size(); }

  int32_t num_embeddings() const { return num_embeddings_; }

  int32_t embedding_dim() const { return embedding_dim_; }

 private:
  std::shared_ptr<const std::vector<float>> storage_;
  int32_t num_embeddings_ = 0;
  int32_t embedding_dim_ = 0;
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_OFFLINE_PARAFORMER_HOTWORD_EMBEDDING_H_
