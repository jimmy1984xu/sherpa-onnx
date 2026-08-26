#include "sherpa-onnx/csrc/offline-paraformer-beam-search-decoder.h"

#include <array>
#include <cmath>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

namespace sherpa_onnx {

TEST(OfflineParaformerBeamSearchDecoder,
     ReturnsAcousticLogProbsForTokensBeforeEos) {
  Ort::MemoryInfo memory_info =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  std::vector<float> logits{
      5.0f, 1.0f, -5.0f,  // token 0
      1.0f, 5.0f, -5.0f,  // token 1
      1.0f, 1.0f, 5.0f,   // EOS
  };
  std::array<int64_t, 3> logits_shape{1, 3, 3};
  auto logits_value = Ort::Value::CreateTensor<float>(
      memory_info, logits.data(), logits.size(), logits_shape.data(),
      logits_shape.size());

  std::array<int64_t, 1> length{3};
  std::array<int64_t, 1> length_shape{1};
  auto length_value = Ort::Value::CreateTensor<int64_t>(
      memory_info, length.data(), length.size(), length_shape.data(),
      length_shape.size());

  OfflineParaformerBeamSearchDecoder decoder(
      /*eos_id=*/2, /*max_active_paths=*/1, /*lm_scale=*/0.0f,
      /*unk_id=*/-1, /*hotwords_score=*/0.0f, /*blank_penalty=*/0.0f);
  auto results = decoder.Decode(std::move(logits_value),
                                std::move(length_value));

  ASSERT_EQ(results.size(), 1);
  ASSERT_EQ(results[0].tokens, (std::vector<int64_t>{0, 1}));
  ASSERT_EQ(results[0].ys_log_probs.size(), results[0].tokens.size());

  const float expected_token_0 =
      5.0f - std::log(std::exp(5.0f) + std::exp(1.0f) + std::exp(-5.0f));
  const float expected_token_1 = expected_token_0;
  EXPECT_NEAR(results[0].ys_log_probs[0], expected_token_0, 1e-5f);
  EXPECT_NEAR(results[0].ys_log_probs[1], expected_token_1, 1e-5f);
}

}  // namespace sherpa_onnx
