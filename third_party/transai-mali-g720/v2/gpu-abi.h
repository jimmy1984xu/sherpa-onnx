#ifndef SHERPA_ONNX_CSRC_MALI_G720_GPU_ABI_H_
#define SHERPA_ONNX_CSRC_MALI_G720_GPU_ABI_H_
#include <stddef.h>
#include <stdint.h>
// Private plugin ABI; no changes to sherpa's public C structures.
struct MaliG720GpuResult {
  const char *utf8;
  size_t utf8_bytes;
  const int32_t *triples;
  int32_t token_count, mel_frames, encoder_frames, processed_samples;
  double elapsed_ms;
};
struct MaliG720GpuResultEx {
  uint32_t abi_version, struct_bytes;
  MaliG720GpuResult base;
  const float *ys_log_probs;
  int32_t score_count;
};
#endif
