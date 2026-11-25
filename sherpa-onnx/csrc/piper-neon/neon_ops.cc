#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#include <arm_neon.h>

#include "sherpa-onnx/csrc/neon-pack-bf16.h"
#include "sherpa-onnx/csrc/neon-tts-load-profile.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <type_traits>
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

// Deconvolution is expressed as phase-output channels on an undilated input
// grid. This removes stride-1 zero insertion and redundant zero MACs.
struct Bf16Conv {
 int n,k,channels,kernel,stride,pad,dilation,transpose,phases,offset,activation;
 float slope;
 std::vector<uint8_t>rhs;
 Bf16Conv(const OrtKernelInfo*raw){Ort::ConstKernelInfo i(raw);n=i.GetAttribute<int64_t>("N");k=i.GetAttribute<int64_t>("K");channels=i.GetAttribute<int64_t>("C");kernel=i.GetAttribute<int64_t>("kernel");stride=i.GetAttribute<int64_t>("stride");pad=i.GetAttribute<int64_t>("pad");dilation=i.GetAttribute<int64_t>("dilation");transpose=i.GetAttribute<int64_t>("transpose");phases=i.GetAttribute<int64_t>("phases");offset=i.GetAttribute<int64_t>("offset");activation=i.GetAttribute<int64_t>("activation");slope=i.GetAttribute<float>("slope");int c;auto w=i.GetTensorConstantInput(1,&c);if(!c)throw std::runtime_error("constant weight required");auto bias=i.GetTensorConstantInput(3,&c);
  auto*weight=(const __fp16*)w.GetTensorRawData();std::vector<__fp16>tr(size_t(n)*k);for(int row=0;row<k;row++)for(int col=0;col<n;col++)tr[size_t(row)*n+col]=weight[size_t(col)*k+row];rhs.resize(kai_get_rhs_packed_size_rhs_pack_kxn_bf16p12x4biasf32_f16_neon(n,k));kai_run_rhs_pack_kxn_bf16p12x4biasf32_f16_neon(1,n,k,12,4,1,n*sizeof(__fp16),tr.data(),bias.GetTensorData<float>(),nullptr,rhs.data(),0,nullptr);
 }
 OrtStatusPtr ComputeV2(OrtKernelContext*raw)noexcept{return guard([&]{Ort::KernelContext ctx(raw);auto input=ctx.GetInput(0);auto shape=input.GetTensorTypeAndShapeInfo().GetShape();if(shape.size()!=3||shape[0]!=1||shape[1]!=channels)throw std::runtime_error("Conv expects batch-1 NCT");int tin=shape[2];int tout=transpose?(tin-1)*stride-2*pad+kernel:(tin+2*pad-dilation*(kernel-1)-1)/stride+1;int m=transpose?(tout+pad+phases-1)/phases:tout;
  auto*x=input.GetTensorData<float>();int out_channels=transpose?n/phases:n;auto*y=ctx.GetOutput(0,{1,out_channels,tout}).GetTensorMutableData<float>();
  int taps=k/channels;std::vector<const float*>column(k);std::vector<int>delta(k);for(int j=0;j<k;j++){column[j]=x+size_t(j/taps)*tin;delta[j]=transpose?offset+j%taps:-pad+(j%taps)*dilation;}
  double timings[4][3]{};bool prof=std::getenv("PIPER_PROFILE")!=nullptr;
  parallel(ctx,4,[&](size_t task){
   struct Workspace{std::vector<float>matrix,result;std::vector<uint8_t>lhs;};static thread_local Workspace work;
   const int step=256;int begin=m*task/4,end=m*(task+1)/4;
   for(int start=begin;start<end;start+=step){auto checkpoint=std::chrono::steady_clock::now();auto mark=[&](int phase){if(prof){auto now=std::chrono::steady_clock::now();timings[task][phase]+=std::chrono::duration<double,std::milli>(now-checkpoint).count();checkpoint=now;}};int rows=std::min(step,end-start);work.matrix.resize(size_t(rows)*k);work.result.resize(size_t(rows)*n);auto*matrix=work.matrix.data();auto*result=work.result.data();
    auto load=[&](int col,int row){int t=start+row+delta[col];float32x4_t v;if(t>=0&&t+4<=tin)v=vld1q_f32(column[col]+t);else{float tmp[4];for(int j=0;j<4;j++)tmp[j]=t+j>=0&&t+j<tin?column[col][t+j]:0;v=vld1q_f32(tmp);}if(activation)v=vaddq_f32(vmaxq_f32(v,vdupq_n_f32(0)),vmulq_n_f32(vminq_f32(v,vdupq_n_f32(0)),slope));return v;};
    int row=0;for(;row+4<=rows;row+=4)for(int col=0;col<k;col+=4){auto a=load(col,row),b=load(col+1,row),c=load(col+2,row),d=load(col+3,row);auto ab=vtrnq_f32(a,b),cd=vtrnq_f32(c,d);vst1q_f32(matrix+size_t(row)*k+col,vcombine_f32(vget_low_f32(ab.val[0]),vget_low_f32(cd.val[0])));vst1q_f32(matrix+size_t(row+1)*k+col,vcombine_f32(vget_low_f32(ab.val[1]),vget_low_f32(cd.val[1])));vst1q_f32(matrix+size_t(row+2)*k+col,vcombine_f32(vget_high_f32(ab.val[0]),vget_high_f32(cd.val[0])));vst1q_f32(matrix+size_t(row+3)*k+col,vcombine_f32(vget_high_f32(ab.val[1]),vget_high_f32(cd.val[1])));}
    for(;row<rows;row++)for(int col=0;col<k;col++){int t=start+row+delta[col];float v=t>=0&&t<tin?column[col][t]:0;matrix[size_t(row)*k+col]=activation&&v<0?v*slope:v;}
    mark(0);work.lhs.resize(kai_get_lhs_packed_size_lhs_pack_bf16p8x4_f16_neon(rows,k,8,4,1));sherpa_onnx::PackBf16F32(rows,k,matrix,work.lhs.data());kai_run_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla(rows,n,k,work.lhs.data(),rhs.data(),result,n*sizeof(float),sizeof(float),-INFINITY,INFINITY);
    mark(1);if(transpose){for(int c=0;c<out_channels;c++)for(int row=0;row<rows;row++){int t=(start+row)*phases-pad;auto*src=result+size_t(row)*n+c*phases;if(t>=0&&t+phases<=tout){for(int ph=0;ph<phases;ph+=4)vst1q_f32(y+size_t(c)*tout+t+ph,vld1q_f32(src+ph));}else for(int ph=0;ph<phases;ph++)if(t+ph>=0&&t+ph<tout)y[size_t(c)*tout+t+ph]=src[ph];}}
    else{for(int c=0;c<out_channels;c+=4){int row=0;for(;row+4<=rows;row+=4){auto a=vld1q_f32(result+size_t(row)*n+c),b=vld1q_f32(result+size_t(row+1)*n+c),cc=vld1q_f32(result+size_t(row+2)*n+c),d=vld1q_f32(result+size_t(row+3)*n+c);auto ab=vtrnq_f32(a,b),cd=vtrnq_f32(cc,d);vst1q_f32(y+size_t(c)*tout+start+row,vcombine_f32(vget_low_f32(ab.val[0]),vget_low_f32(cd.val[0])));vst1q_f32(y+size_t(c+1)*tout+start+row,vcombine_f32(vget_low_f32(ab.val[1]),vget_low_f32(cd.val[1])));vst1q_f32(y+size_t(c+2)*tout+start+row,vcombine_f32(vget_high_f32(ab.val[0]),vget_high_f32(cd.val[0])));vst1q_f32(y+size_t(c+3)*tout+start+row,vcombine_f32(vget_high_f32(ab.val[1]),vget_high_f32(cd.val[1])));}for(;row<rows;row++)for(int j=0;j<4;j++)y[size_t(c+j)*tout+start+row]=result[size_t(row)*n+c+j];}}mark(2);
   }
  });
  if(prof&&m>2000){double sums[3]{};for(auto&t:timings)for(int j=0;j<3;j++)sums[j]+=t[j];std::fprintf(stderr,"PHASE half=%d n=%d k=%d m=%d prepare=%g matmul=%g scatter=%g\n",1,n,k,m,sums[0],sums[1],sums[2]);}
 });}
};
#include "i8mm_conv.h"
#include "u8_conv.h"
#include "spline.h"
#include "vector_math.h"
#include "layer_norm.h"
#include "relative_attention.h"
#include "packed_dense_u8.h"
struct Bf16Op : Ort::CustomOpBase<Bf16Op, Bf16Conv, true> {
 const char* GetName() const { return "BF16Conv"; }
 size_t GetInputTypeCount() const { return 4; }
 ONNXTensorElementDataType GetInputType(size_t i) const {
   return i == 1 ? ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 : ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
 }
 size_t GetOutputTypeCount() const { return 1; }
 ONNXTensorElementDataType GetOutputType(size_t) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; }
 OrtStatusPtr CreateKernelV2(const OrtApi &, const OrtKernelInfo *info, void **kernel) const noexcept {
   return guard([&] { *kernel = new Bf16Conv(info); });
 }
};
}
extern "C" __attribute__((visibility("default"))) OrtStatus*ORT_API_CALL PiperRegisterCustomOps(OrtSessionOptions*options,const OrtApiBase*base){auto*api=base->GetApi(ORT_API_VERSION);if(!api)return base->GetApi(1)->CreateStatus(ORT_FAIL,"ORT API mismatch");Ort::InitApi(api);return guard([&]{static Bf16Op b;static U8Op u;static IndexedU8Op indexed;static CarlfmPackedU8Op<false> carlfm;static CarlfmPackedU8Op<true> carlfm_indexed;static PiperSplineOp spline;static PiperNormOp norm;static PiperGateOp gate;static PiperRelativeOp<true> relk;static PiperRelativeOp<false> relv;static AmyPackedU8Op amy;static BryceU8Op bryce;static BrycePackedU8Op bryce_packed;static Ort::CustomOpDomain domain=[](){Ort::CustomOpDomain d("transai.piper");d.Add(&b);d.Add(&u);d.Add(&indexed);d.Add(&carlfm);d.Add(&carlfm_indexed);d.Add(&spline);d.Add(&norm);d.Add(&gate);d.Add(&relk);d.Add(&relv);d.Add(&amy);d.Add(&bryce);d.Add(&bryce_packed);return d;}();Ort::ThrowOnError(api->AddCustomOpDomain(options,domain));});}

namespace sherpa_onnx {void RegisterPiperNeonOps(Ort::SessionOptions& o){Ort::ThrowOnError(PiperRegisterCustomOps(o,OrtGetApiBase()));}}
