#ifndef SHERPA_ONNX_CSRC_MALI_G720_PCM_H_
#define SHERPA_ONNX_CSRC_MALI_G720_PCM_H_
#include <algorithm>
#include <arm_neon.h>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
namespace sherpa_onnx {
// Caller checks finite [-1,1]. FCVTAS implements lround's ties-away-from-zero.
inline int16_t MaliG720Pcm16(float x) {
  const int32_t rounded=vcvtas_s32_f32(x*32768.0f);
  return static_cast<int16_t>(std::min(32767,rounded));
}
inline void MaliG720Pcm16(const float *input,int16_t *output,size_t count) {
  size_t i=0;
  const auto lower=vdupq_n_f32(-1.0f),upper=vdupq_n_f32(1.0f);
  for(;i+4<=count;i+=4) {
    const auto x=vld1q_f32(input+i);
    const auto valid=vandq_u32(vcgeq_f32(x,lower),vcleq_f32(x,upper));
    if(vminvq_u32(valid)!=UINT32_MAX)throw std::invalid_argument("invalid PCM float");
    const auto rounded=vcvtaq_s32_f32(vmulq_n_f32(x,32768.0f));
    vst1_s16(output+i,vqmovn_s32(rounded));
  }
  for(;i<count;++i) {
    const auto x=input[i];
    if(!(x>=-1.0f&&x<=1.0f))throw std::invalid_argument("invalid PCM float");
    output[i]=MaliG720Pcm16(x);
  }
}
}
#endif
