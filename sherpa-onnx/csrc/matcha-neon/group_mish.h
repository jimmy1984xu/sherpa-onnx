void group_mish_one(const float*x,float*y,int channels,int frames,int groups,int group,
                    const float*gamma,const float*beta,float epsilon){
 int width=channels/groups,begin=group*width*frames,count=width*frames;
 auto sum=vdupq_n_f32(0);int i=0;
 for(;i+4<=count;i+=4)sum=vaddq_f32(sum,vld1q_f32(x+begin+i));
 float total=vaddvq_f32(sum);for(;i<count;i++)total+=x[begin+i];float mean=total/count;
 auto variance=vdupq_n_f32(0);i=0;
 for(;i+4<=count;i+=4){auto d=vsubq_f32(vld1q_f32(x+begin+i),vdupq_n_f32(mean));variance=vfmaq_f32(variance,d,d);}
 total=vaddvq_f32(variance);for(;i<count;i++){float d=x[begin+i]-mean;total+=d*d;}
 float inv=1/std::sqrt(total/count+epsilon);
 for(int channel=group*width;channel<(group+1)*width;channel++){
  float scale=inv*gamma[channel],offset=beta[channel]-mean*scale;size_t at=size_t(channel)*frames;i=0;
  for(;i+4<=frames;i+=4){auto v=vfmaq_n_f32(vdupq_n_f32(offset),vld1q_f32(x+at+i),scale),e=exp4(v);auto t=vmulq_f32(e,vaddq_f32(e,vdupq_n_f32(2)));vst1q_f32(y+at+i,vmulq_f32(v,vdivq_f32(t,vaddq_f32(t,vdupq_n_f32(2)))));}
  for(;i<frames;i++){float v=x[at+i]*scale+offset,e=std::exp(std::min(v,20.f)),t=e*(e+2);y[at+i]=v*t/(t+2);}
 }
}

struct GroupMish {
 int groups;float epsilon;const float*gamma,*beta;size_t channels;
 GroupMish(const OrtKernelInfo*raw){Ort::ConstKernelInfo info(raw);groups=info.GetAttribute<int64_t>("groups");epsilon=info.GetAttribute<float>("epsilon");int constant;auto g=info.GetTensorConstantInput(1,&constant);if(!constant)throw std::runtime_error("GroupNorm gamma");gamma=g.GetTensorData<float>();channels=g.GetTensorTypeAndShapeInfo().GetElementCount();beta=info.GetTensorConstantInput(2,&constant).GetTensorData<float>();}
 OrtStatusPtr ComputeV2(OrtKernelContext*raw)noexcept{return guard([&]{Ort::KernelContext ctx(raw);auto x=ctx.GetInput(0);auto shape=x.GetTensorTypeAndShapeInfo().GetShape();if(shape.size()!=3||shape[0]!=1||size_t(shape[1])!=channels||channels%groups)throw std::runtime_error("GroupNorm dimensions");const float*p=x.GetTensorData<float>();float*q=ctx.GetOutput(0,shape).GetTensorMutableData<float>();parallel(ctx,groups,[&](size_t g){group_mish_one(p,q,channels,shape[2],groups,g,gamma,beta,epsilon);});});}
};
