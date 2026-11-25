// FP32 exp range reduction and polynomial; callers clamp tails before use.
inline float32x4_t piper_exp(float32x4_t x){
 x=vmaxq_f32(vdupq_n_f32(-87.f),vminq_f32(x,vdupq_n_f32(80.f)));
 auto n=vcvtnq_s32_f32(vmulq_n_f32(x,1.4426950408889634f));auto nf=vcvtq_f32_s32(n);
 auto r=vsubq_f32(vsubq_f32(x,vmulq_n_f32(nf,.693145751953125f)),vmulq_n_f32(nf,1.428606765330187e-6f));
 auto p=vdupq_n_f32(1.f/5040.f);
 for(float c:{1.f/720.f,1.f/120.f,1.f/24.f,1.f/6.f,.5f,1.f,1.f})p=vaddq_f32(vmulq_f32(p,r),vdupq_n_f32(c));
 return vmulq_f32(p,vreinterpretq_f32_s32(vshlq_n_s32(vaddq_s32(n,vdupq_n_s32(127)),23)));
}
inline float32x4_t piper_gelu(float32x4_t x){
 auto z=vmulq_n_f32(x,.7071067811865475f),a=vabsq_f32(z);
 auto t=vdivq_f32(vdupq_n_f32(1),vaddq_f32(vdupq_n_f32(1),vmulq_n_f32(a,.3275911f)));
 auto p=vdupq_n_f32(1.061405429f);
 for(float c:{-1.453152027f,1.421413741f,-.284496736f,.254829592f})p=vaddq_f32(vmulq_f32(p,t),vdupq_n_f32(c));
 auto erf=vsubq_f32(vdupq_n_f32(1),vmulq_f32(vmulq_f32(p,t),piper_exp(vnegq_f32(vmulq_f32(a,a)))));
 erf=vbslq_f32(vcltq_f32(z,vdupq_n_f32(0)),vnegq_f32(erf),erf);
 return vmulq_f32(vmulq_n_f32(x,.5f),vaddq_f32(vdupq_n_f32(1),erf));
}
struct PiperGate {
 explicit PiperGate(const OrtKernelInfo*){}
 OrtStatusPtr ComputeV2(OrtKernelContext*raw)noexcept{return guard([&]{
  Ort::KernelContext ctx(raw);auto a=ctx.GetInput(0),b=ctx.GetInput(1);auto shape=a.GetTensorTypeAndShapeInfo().GetShape();
  if(shape!=b.GetTensorTypeAndShapeInfo().GetShape())throw std::runtime_error("gate shape mismatch");
  size_t count=a.GetTensorTypeAndShapeInfo().GetElementCount();auto*x=a.GetTensorData<float>();auto*z=b.GetTensorData<float>();auto*y=ctx.GetOutput(0,shape).GetTensorMutableData<float>();
  parallel(ctx,4,[&](size_t task){size_t i=count*task/4,end=count*(task+1)/4;
   for(;i+4<=end;i+=4){auto av=vld1q_f32(x+i),bv=vld1q_f32(z+i);auto ae=piper_exp(vmulq_n_f32(vabsq_f32(av),-2));auto be=piper_exp(vnegq_f32(vabsq_f32(bv)));
    auto tanh=vdivq_f32(vsubq_f32(vdupq_n_f32(1),ae),vaddq_f32(vdupq_n_f32(1),ae));tanh=vbslq_f32(vcltq_f32(av,vdupq_n_f32(0)),vnegq_f32(tanh),tanh);
    auto sig=vdivq_f32(vdupq_n_f32(1),vaddq_f32(vdupq_n_f32(1),be));sig=vbslq_f32(vcltq_f32(bv,vdupq_n_f32(0)),vmulq_f32(sig,be),sig);vst1q_f32(y+i,vmulq_f32(tanh,sig));}
   for(;i<end;i++)y[i]=std::tanh(x[i])/(1.f+std::exp(-z[i]));
  });
 });}
};
struct PiperGateOp:Ort::CustomOpBase<PiperGateOp,PiperGate,true>{
 const char*GetName()const{return "PiperTanhSigmoidGate";}size_t GetInputTypeCount()const{return 2;}
 ONNXTensorElementDataType GetInputType(size_t)const{return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;}
 size_t GetOutputTypeCount()const{return 1;}ONNXTensorElementDataType GetOutputType(size_t)const{return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;}
 OrtStatusPtr CreateKernelV2(const OrtApi&,const OrtKernelInfo*i,void**k)const noexcept{return guard([&]{*k=new PiperGate(i);});}
};
