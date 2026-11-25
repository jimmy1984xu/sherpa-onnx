// Built-in ORT API initialization
#include "onnxruntime_cxx_api.h"
#include <arm_neon.h>

#include "sherpa-onnx/csrc/neon-pack-bf16.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <type_traits>
#include "pack/kai_lhs_quant_pack_qai8dxp_f32.h"
#include "pack/kai_rhs_pack_nxk_qsi8cxp_qsi8cx_neon.h"
#include "matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm.h"
#include "pack/kai_lhs_pack_bf16p8x4_f16_neon.h"
#include "pack/kai_rhs_pack_kxn_bf16p12x4biasf32_f16_neon.h"
#include "matmul_clamp_f32_bf16p_bf16p/kai_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla.h"

#include "sherpa-onnx/csrc/neon-ort-kernel.h"

namespace {
using sherpa_onnx::neon::ParallelFor;
template <typename Function>
auto guard(Function fn) noexcept { return sherpa_onnx::neon::Guard<false>(fn); }
template <typename Function>
void parallel(const Ort::KernelContext& ctx, size_t tasks, Function fn) {
  ParallelFor(ctx, tasks, 0, fn);
}
float32x4_t exp4(float32x4_t x){
 x=vminq_f32(vmaxq_f32(x,vdupq_n_f32(-80)),vdupq_n_f32(20));
 auto n=vcvtnq_s32_f32(vmulq_n_f32(x,1.4426950408889634f));auto nf=vcvtq_f32_s32(n);
 auto r=vfmsq_n_f32(x,nf,0.6931471805599453f);
 auto p=vdupq_n_f32(1.f/720);p=vfmaq_f32(vdupq_n_f32(1.f/120),p,r);p=vfmaq_f32(vdupq_n_f32(1.f/24),p,r);p=vfmaq_f32(vdupq_n_f32(1.f/6),p,r);p=vfmaq_f32(vdupq_n_f32(.5f),p,r);p=vfmaq_f32(vdupq_n_f32(1),p,r);p=vfmaq_f32(vdupq_n_f32(1),p,r);
 auto power=vreinterpretq_f32_s32(vshlq_n_s32(vaddq_s32(n,vdupq_n_s32(127)),23));return vmulq_f32(p,power);
}
struct Mish {
 Mish(const OrtKernelInfo*){}
 OrtStatusPtr ComputeV2(OrtKernelContext*raw) noexcept{return guard([&]{Ort::KernelContext ctx(raw);auto x=ctx.GetInput(0);auto shape=x.GetTensorTypeAndShapeInfo().GetShape();auto y=ctx.GetOutput(0,shape);auto n=x.GetTensorTypeAndShapeInfo().GetElementCount();const float*p=x.GetTensorData<float>();float*q=y.GetTensorMutableData<float>();
 auto work=[&](size_t tile){size_t begin=tile*4096,end=std::min(begin+4096,n),i=begin;for(;i+4<=end;i+=4){auto v=vld1q_f32(p+i),e=exp4(v);auto t=vmulq_f32(e,vaddq_f32(e,vdupq_n_f32(2)));auto ratio=vdivq_f32(t,vaddq_f32(t,vdupq_n_f32(2)));vst1q_f32(q+i,vmulq_f32(v,ratio));}for(;i<end;i++){float e=std::exp(std::min(p[i],20.f)),t=e*(e+2);q[i]=p[i]*(t/(t+2));}};
 if(n<8192)for(size_t t=0;t<(n+4095)/4096;t++)work(t);else parallel(ctx,(n+4095)/4096,work);
 });}
};

float32x4_t sin4(float32x4_t x){
 auto turns=vrndnq_f32(vmulq_n_f32(x,0.15915494309189535f));
 auto r=vfmsq_n_f32(x,turns,6.283185307179586f),r2=vmulq_f32(r,r);
 auto p=vdupq_n_f32(1.f/6227020800.f);p=vfmaq_f32(vdupq_n_f32(-1.f/39916800.f),p,r2);p=vfmaq_f32(vdupq_n_f32(1.f/362880.f),p,r2);p=vfmaq_f32(vdupq_n_f32(-1.f/5040.f),p,r2);p=vfmaq_f32(vdupq_n_f32(1.f/120.f),p,r2);p=vfmaq_f32(vdupq_n_f32(-1.f/6.f),p,r2);return vmulq_f32(r,vfmaq_f32(vdupq_n_f32(1),p,r2));
}
void gelu(float*data,int count){
 for(int i=0;i<count;i+=4){auto x=vld1q_f32(data+i);auto z=vmulq_n_f32(vabsq_f32(x),0.7071067811865475f);auto t=vdivq_f32(vdupq_n_f32(1),vfmaq_n_f32(vdupq_n_f32(1),z,0.3275911f));auto p=vfmaq_n_f32(vdupq_n_f32(-1.453152027f),t,1.061405429f);p=vfmaq_f32(vdupq_n_f32(1.421413741f),p,t);p=vfmaq_f32(vdupq_n_f32(-0.284496736f),p,t);p=vfmaq_f32(vdupq_n_f32(0.254829592f),p,t);auto e=exp4(vmulq_n_f32(vmulq_f32(x,x),-.5f));auto erf=vsubq_f32(vdupq_n_f32(1),vmulq_f32(vmulq_f32(p,t),e));erf=vbslq_f32(vcltq_f32(x,vdupq_n_f32(0)),vnegq_f32(erf),erf);vst1q_f32(data+i,vmulq_f32(vmulq_n_f32(x,.5f),vaddq_f32(vdupq_n_f32(1),erf)));}
}
struct SnakeBeta {
 const float*alpha,*coefficient;size_t channels;
 SnakeBeta(const OrtKernelInfo*raw){Ort::ConstKernelInfo info(raw);int c;auto a=info.GetTensorConstantInput(1,&c);if(!c)throw std::runtime_error("Snake alpha");alpha=a.GetTensorData<float>();channels=a.GetTensorTypeAndShapeInfo().GetElementCount();coefficient=info.GetTensorConstantInput(2,&c).GetTensorData<float>();}
 OrtStatusPtr ComputeV2(OrtKernelContext*raw)noexcept{return guard([&]{Ort::KernelContext ctx(raw);auto x=ctx.GetInput(0);auto shape=x.GetTensorTypeAndShapeInfo().GetShape();auto n=x.GetTensorTypeAndShapeInfo().GetElementCount();const float*p=x.GetTensorData<float>();float*q=ctx.GetOutput(0,shape).GetTensorMutableData<float>();
 parallel(ctx,(n+4095)/4096,[&](size_t task){size_t begin=task*4096,end=std::min(n,begin+4096),i=begin;for(;i+4<=end;i+=4){size_t c=i%channels;auto v=vld1q_f32(p+i),s=sin4(vmulq_f32(v,vld1q_f32(alpha+c)));vst1q_f32(q+i,vfmaq_f32(v,vmulq_f32(s,s),vld1q_f32(coefficient+c)));}for(;i<end;i++){float s=std::sin(p[i]*alpha[i%channels]);q[i]=p[i]+s*s*coefficient[i%channels];}});
 });}
};

#include "group_mish.h"
#include "kai_linear.h"
template<class K,int Count>struct Op:Ort::CustomOpBase<Op<K,Count>,K,true>{const char*name;Op(const char*s):name(s){}const char*GetName()const{return name;}size_t GetInputTypeCount()const{return Count;}ONNXTensorElementDataType GetInputType(size_t i)const{return Count==2&&i==1?ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:Count>=4&&i==1?((std::is_same_v<K,F16Linear>)?ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8):ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;}size_t GetOutputTypeCount()const{return 1;}ONNXTensorElementDataType GetOutputType(size_t)const{return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;}OrtStatusPtr CreateKernelV2(const OrtApi&,const OrtKernelInfo*i,void**k)const noexcept{return guard([&]{*k=new K(i);});}};
}
namespace sherpa_onnx {
namespace {
void RegisterMatchaOps(Ort::SessionOptions &options, bool export_graph) {
  static Op<SnakeBeta, 3> snake("SnakeBeta");
  static Op<GroupMish, 3> group("GroupMish");
  static Op<PackedQLinear, 2> packed_q("PackedQLinear");
  static Op<PackedF16Linear, 2> packed_f16("PackedF16Linear");
  static Op<QLinear, 4> q("QLinear");
  static Op<F16Linear, 4> f16("F16Linear");
  static Op<Mish, 1> mish("Mish");
  static Ort::CustomOpDomain runtime = [] {
    Ort::CustomOpDomain domain("transai.matcha");
    domain.Add(&snake);
    domain.Add(&group);
    domain.Add(&packed_q);
    domain.Add(&packed_f16);
    return domain;
  }();
  static Ort::CustomOpDomain exporter = [] {
    Ort::CustomOpDomain domain("transai.matcha");
    domain.Add(&snake);
    domain.Add(&group);
    domain.Add(&packed_q);
    domain.Add(&packed_f16);
    domain.Add(&q);
    domain.Add(&f16);
    domain.Add(&mish);
    return domain;
  }();
  options.Add(export_graph ? exporter : runtime);
}
}  // namespace

void RegisterMatchaNeonOps(Ort::SessionOptions &options) {
  RegisterMatchaOps(options, false);
}

void RegisterMatchaExportOps(Ort::SessionOptions &options) {
  RegisterMatchaOps(options, true);
}
}  // namespace sherpa_onnx
