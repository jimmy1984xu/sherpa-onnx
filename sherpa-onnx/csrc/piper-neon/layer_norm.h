// FP32 channel normalization in NCT layout without tensor transposes.
struct PiperNorm {
 int gelu=0;float epsilon;std::vector<float> gamma,beta;
 explicit PiperNorm(const OrtKernelInfo*raw){Ort::ConstKernelInfo i(raw);epsilon=i.GetAttribute<float>("epsilon");
  try{gelu=i.GetAttribute<int64_t>("gelu");}catch(const Ort::Exception&){} int c;auto g=i.GetTensorConstantInput(1,&c);if(!c)throw std::runtime_error("norm constant gamma");
  auto b=i.GetTensorConstantInput(2,&c);if(!c)throw std::runtime_error("norm constant beta");
  size_t n=g.GetTensorTypeAndShapeInfo().GetElementCount();gamma.assign(g.GetTensorData<float>(),g.GetTensorData<float>()+n);beta.assign(b.GetTensorData<float>(),b.GetTensorData<float>()+n);}
 OrtStatusPtr ComputeV2(OrtKernelContext*raw)noexcept{return guard([&]{
  Ort::KernelContext ctx(raw);auto input=ctx.GetInput(0);auto shape=input.GetTensorTypeAndShapeInfo().GetShape();
  if(shape.size()!=3||shape[0]!=1||size_t(shape[1])!=gamma.size())throw std::runtime_error("norm batch/channel shape");
  int c=shape[1],t=shape[2];auto*x=input.GetTensorData<float>();auto*y=ctx.GetOutput(0,shape).GetTensorMutableData<float>();
  parallel(ctx,4,[&](size_t task){int begin=t*task/4,end=t*(task+1)/4,i=begin;
   for(;i+4<=end;i+=4){auto mean=vdupq_n_f32(0);for(int j=0;j<c;j++)mean=vaddq_f32(mean,vld1q_f32(x+size_t(j)*t+i));mean=vmulq_n_f32(mean,1.f/c);
    auto variance=vdupq_n_f32(0);for(int j=0;j<c;j++){auto d=vsubq_f32(vld1q_f32(x+size_t(j)*t+i),mean);variance=vaddq_f32(variance,vmulq_f32(d,d));}
    auto den=vsqrtq_f32(vaddq_f32(vmulq_n_f32(variance,1.f/c),vdupq_n_f32(epsilon)));
    for(int j=0;j<c;j++){auto v=vdivq_f32(vsubq_f32(vld1q_f32(x+size_t(j)*t+i),mean),den);v=vaddq_f32(vmulq_n_f32(v,gamma[j]),vdupq_n_f32(beta[j]));if(gelu)v=piper_gelu(v);vst1q_f32(y+size_t(j)*t+i,v);}}
   for(;i<end;i++){float mean=0,var=0;for(int j=0;j<c;j++)mean+=x[size_t(j)*t+i];mean/=c;for(int j=0;j<c;j++){float d=x[size_t(j)*t+i]-mean;var+=d*d;}float den=std::sqrt(var/c+epsilon);for(int j=0;j<c;j++){float v=((x[size_t(j)*t+i]-mean)/den)*gamma[j]+beta[j];y[size_t(j)*t+i]=gelu?.5f*v*(1.f+std::erf(v*.7071067811865475f)):v;}}
  });
 });}
};
struct PiperNormOp:Ort::CustomOpBase<PiperNormOp,PiperNorm,true>{
 const char*GetName()const{return "PiperLayerNormNCT";}size_t GetInputTypeCount()const{return 3;}
 ONNXTensorElementDataType GetInputType(size_t)const{return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;}
 size_t GetOutputTypeCount()const{return 1;}ONNXTensorElementDataType GetOutputType(size_t)const{return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;}
 OrtStatusPtr CreateKernelV2(const OrtApi&,const OrtKernelInfo*i,void**k)const noexcept{return guard([&]{*k=new PiperNorm(i);});}
};
