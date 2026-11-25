// Inverse monotonic rational-quadratic spline used by VITS duration flow.
// Algorithm reference: jaywalnut310/vits transforms.py (MIT; see third_party).
// Fuse the export's gather/scatter and shape construction for batch-1 NCT.
struct PiperSpline {
 float width_scale;
 explicit PiperSpline(const OrtKernelInfo* info) {
  width_scale=Ort::ConstKernelInfo(info).GetAttribute<float>("width_scale");
 }
 OrtStatusPtr ComputeV2(OrtKernelContext* raw) noexcept {return guard([&]{
  Ort::KernelContext ctx(raw);auto a=ctx.GetInput(0),b=ctx.GetInput(1),p=ctx.GetInput(2),m=ctx.GetInput(3);
  auto shape=b.GetTensorTypeAndShapeInfo().GetShape();
  if(shape.size()!=3||shape[0]!=1||shape[1]!=1)throw std::runtime_error("spline expects batch-1 N1T");
  const int t=shape[2];
  if(p.GetTensorTypeAndShapeInfo().GetShape()!=std::vector<int64_t>{1,29,t}||
     a.GetTensorTypeAndShapeInfo().GetShape()!=shape||m.GetTensorTypeAndShapeInfo().GetShape()!=shape)
    throw std::runtime_error("spline parameter shapes");
  auto* x0=a.GetTensorData<float>();auto* x1=b.GetTensorData<float>();
  auto* params=p.GetTensorData<float>();auto* mask=m.GetTensorData<float>();
  auto* output=ctx.GetOutput(0,{1,2,t}).GetTensorMutableData<float>();
  parallel(ctx,4,[&](size_t task){for(int i=t*task/4;i<t*(task+1)/4;i++){
   output[i]=x0[i]*mask[i];const float y=x1[i];
   if(mask[i]==0){output[t+i]=0;continue;}
   if(y < -5.f||y > 5.f){output[t+i]=y*mask[i];continue;}
   float ew[10],eh[10],cw[11],ch[11],maxw=-INFINITY,maxh=-INFINITY;
   for(int j=0;j<10;j++){
    ew[j]=params[j*t+i]*mask[i]/width_scale;
    eh[j]=params[(j+10)*t+i]*mask[i]/width_scale;
    maxw=std::max(maxw,ew[j]);maxh=std::max(maxh,eh[j]);
   }
   float sumw=0,sumh=0;
   for(int j=0;j<10;j++){ew[j]=std::exp(ew[j]-maxw);eh[j]=std::exp(eh[j]-maxh);sumw+=ew[j];sumh+=eh[j];}
   cw[0]=ch[0]=-5.f;float sw=0,sh=0;
   for(int j=0;j<10;j++){
    sw+=0.001f+0.99f*(ew[j]/sumw);sh+=0.001f+0.99f*(eh[j]/sumh);
    cw[j+1]=10.f*sw-5.f;ch[j+1]=10.f*sh-5.f;
   }
   cw[10]=ch[10]=5.f;int bin=0;while(bin<9&&y>=ch[bin+1])bin++;
   auto derivative=[&](int j){if(j==0||j==10)return 1.f;float z=params[(19+j)*t+i]*mask[i];
      return .001f+(z>20.f?z:std::log1p(std::exp(z)));};
   const float width=cw[bin+1]-cw[bin],height=ch[bin+1]-ch[bin];
   const float delta=height/width,d0=derivative(bin),d1=derivative(bin+1),offset=y-ch[bin];
   const float curvature=d0+d1-2.f*delta;
   const float qa=offset*curvature+height*(delta-d0),qb=height*d0-offset*curvature,qc=-delta*offset;
   const float root=(2.f*qc)/(-qb-std::sqrt(std::max(0.f,qb*qb-4.f*qa*qc)));
   output[t+i]=(root*width+cw[bin])*mask[i];
  }});
 });}
};
struct PiperSplineOp:Ort::CustomOpBase<PiperSplineOp,PiperSpline,true>{
 const char*GetName()const{return "PiperSplineInverse";}
 size_t GetInputTypeCount()const{return 4;}
 ONNXTensorElementDataType GetInputType(size_t)const{return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;}
 size_t GetOutputTypeCount()const{return 1;}
 ONNXTensorElementDataType GetOutputType(size_t)const{return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;}
 OrtStatusPtr CreateKernelV2(const OrtApi&,const OrtKernelInfo*i,void**k)const noexcept{return guard([&]{*k=new PiperSpline(i);});}
};
