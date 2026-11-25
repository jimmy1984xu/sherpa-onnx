// Exact finite-window relative positions, avoiding padded (2*T-1) GEMMs.
template<bool Scores>struct PiperRelative {
 int bins,dim;std::vector<float>weight;
 explicit PiperRelative(const OrtKernelInfo*raw){Ort::ConstKernelInfo i(raw);int constant;auto w=i.GetTensorConstantInput(2,&constant);
  auto s=w.GetTensorTypeAndShapeInfo().GetShape();if(!constant||s.size()!=3||s[0]!=1||s[1]%2!=1)throw std::runtime_error("shared relative embedding shape");
  bins=s[1];dim=s[2];weight.assign(w.GetTensorData<float>(),w.GetTensorData<float>()+bins*dim);}
 OrtStatusPtr ComputeV2(OrtKernelContext*raw)noexcept{return guard([&]{
  Ort::KernelContext ctx(raw);auto input=ctx.GetInput(0),base=ctx.GetInput(1);auto s=input.GetTensorTypeAndShapeInfo().GetShape(),bs=base.GetTensorTypeAndShapeInfo().GetShape();
  if(s.size()!=4||s[0]!=1||s[1]!=2||bs.size()!=4||bs[0]!=1||bs[1]!=2||bs[2]!=s[2])throw std::runtime_error("relative attention input shape");
  int t=s[2],window=bins/2;if((Scores&&(s[3]!=dim||bs[3]!=t))||(!Scores&&(s[3]!=t||bs[3]!=dim)))throw std::runtime_error("relative attention dimensions");
  auto*x=input.GetTensorData<float>();auto*b=base.GetTensorData<float>();auto*y=ctx.GetOutput(0,bs).GetTensorMutableData<float>();
  parallel(ctx,4,[&](size_t task){int begin=2*t*task/4,end=2*t*(task+1)/4;
   for(int row=begin;row<end;row++){int i=row%t;
    if constexpr(Scores){std::copy_n(b+size_t(row)*t,t,y+size_t(row)*t);
     for(int offset=-window;offset<=window;offset++){int j=i+offset;if(j<0||j>=t)continue;auto sum=vdupq_n_f32(0);int c=0;
      for(;c+4<=dim;c+=4)sum=vaddq_f32(sum,vmulq_f32(vld1q_f32(x+size_t(row)*dim+c),vld1q_f32(weight.data()+size_t(offset+window)*dim+c)));
      float value=vaddvq_f32(sum);for(;c<dim;c++)value+=x[size_t(row)*dim+c]*weight[size_t(offset+window)*dim+c];y[size_t(row)*t+j]+=value;
     }
    }else{int c=0;for(;c+4<=dim;c+=4){auto sum=vdupq_n_f32(0);
      for(int offset=-window;offset<=window;offset++){int j=i+offset;if(j>=0&&j<t)sum=vaddq_f32(sum,vmulq_n_f32(vld1q_f32(weight.data()+size_t(offset+window)*dim+c),x[size_t(row)*t+j]));}
      vst1q_f32(y+size_t(row)*dim+c,vaddq_f32(vld1q_f32(b+size_t(row)*dim+c),sum));}
     for(;c<dim;c++){float sum=0;for(int offset=-window;offset<=window;offset++){int j=i+offset;if(j>=0&&j<t)sum+=weight[size_t(offset+window)*dim+c]*x[size_t(row)*t+j];}y[size_t(row)*dim+c]=b[size_t(row)*dim+c]+sum;}
    }
   }
  });
 });}
};
template<bool Scores>struct PiperRelativeOp:Ort::CustomOpBase<PiperRelativeOp<Scores>,PiperRelative<Scores>,true>{
 const char*GetName()const{return Scores?"PiperRelativeScores":"PiperRelativeValues";}size_t GetInputTypeCount()const{return 3;}
 ONNXTensorElementDataType GetInputType(size_t)const{return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;}
 size_t GetOutputTypeCount()const{return 1;}ONNXTensorElementDataType GetOutputType(size_t)const{return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;}
 OrtStatusPtr CreateKernelV2(const OrtApi&,const OrtKernelInfo*i,void**k)const noexcept{return guard([&]{*k=new PiperRelative<Scores>(i);});}
};
