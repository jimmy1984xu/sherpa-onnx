// UINT8 global activation/per-tensor weight quantization compatible with ORT's
// DynamicQuantizeLinear + ConvInteger, accelerated with unsigned I8MM.
template <bool kBryce>
struct PackedDenseU8Conv {
 int n,c,kernel,pad,dilation,stride,group,k,kp,np;uint8_t wz;float ws;
 const uint8_t* packed_rhs=nullptr;std::vector<int32_t>sums;std::vector<float>bias,scales;std::vector<uint8_t>zeros;
 PackedDenseU8Conv(const OrtKernelInfo*raw){
  transai::piper_neon::LoadProfile profile(kBryce ? "Bryce-U8-prepacked" : "Amy-U8-prepacked");Ort::ConstKernelInfo i(raw);
  n=i.GetAttribute<int64_t>("N");c=i.GetAttribute<int64_t>("C");kernel=i.GetAttribute<int64_t>("kernel");pad=i.GetAttribute<int64_t>("pad");dilation=i.GetAttribute<int64_t>("dilation");stride=i.GetAttribute<int64_t>("stride");group=i.GetAttribute<int64_t>("group");
  if(group!=1)throw std::runtime_error("Piper packed U8: packed dense group must be one");k=c*kernel;kp=(k+7)/8*8;np=(n+7)/8*8;
  int constant;auto packed=i.GetTensorConstantInput(1,&constant);if(!constant||packed.GetTensorTypeAndShapeInfo().GetElementCount()!=size_t(np)*kp)throw std::runtime_error("Piper packed U8: packed RHS shape");
  packed_rhs=packed.GetTensorData<uint8_t>(); // Session owns the immutable mapped ORT initializer.
  auto zp=i.GetTensorConstantInput(3,&constant);if(!constant)throw std::runtime_error("Piper packed U8: constant zero point");auto sc=i.GetTensorConstantInput(2,&constant);if(!constant)throw std::runtime_error("Piper packed U8: constant scale");
  const auto scn=sc.GetTensorTypeAndShapeInfo().GetElementCount(),zpn=zp.GetTensorTypeAndShapeInfo().GetElementCount();if((scn!=1&&scn!=size_t(n))||(zpn!=1&&zpn!=size_t(n)))throw std::runtime_error("Piper packed U8: per-channel scale shape");
  wz=*zp.GetTensorData<uint8_t>();ws=*sc.GetTensorData<float>();scales.assign(np,ws);zeros.assign(np,wz);
  for(int col=0;col<n;col++){scales[col]=sc.GetTensorData<float>()[scn==1?0:col];zeros[col]=zp.GetTensorData<uint8_t>()[zpn==1?0:col];}
  auto b=i.GetTensorConstantInput(4,&constant);if(!constant||b.GetTensorTypeAndShapeInfo().GetElementCount()!=size_t(n))throw std::runtime_error("Piper packed U8: bias shape");bias.assign(b.GetTensorData<float>(),b.GetTensorData<float>()+n);
  auto sum=i.GetTensorConstantInput(5,&constant);if(!constant||sum.GetTensorTypeAndShapeInfo().GetElementCount()!=size_t(np))throw std::runtime_error("Piper packed U8: row-sum shape");sums.assign(sum.GetTensorData<int32_t>(),sum.GetTensorData<int32_t>()+np);
 }
 OrtStatusPtr ComputeV2(OrtKernelContext*raw)noexcept{return guard([&]{Ort::KernelContext ctx(raw);auto input=ctx.GetInput(0);auto shape=input.GetTensorTypeAndShapeInfo().GetShape();if(shape.size()!=3||shape[0]!=1||shape[1]!=c)throw std::runtime_error("U8 batch/channels");int tin=shape[2],tout=(tin+2*pad-dilation*(kernel-1)-1)/stride+1;auto*x=input.GetTensorData<float>();auto*y=ctx.GetOutput(0,{1,n,tout}).GetTensorMutableData<float>();size_t count=size_t(c)*tin;
  float minimum=0,maximum=0;
  if constexpr (!kBryce) {
  ScanU8Channels(x, c, tin, minimum, maximum);
  } else {
   // Large decoder tensors benefit from parallel reduction. Ranges include zero
   // and keep the existing asymmetric scale and zero-point calculation.
   if (count < 65536) {
    ScanU8Block(x, 0, count, minimum, maximum);
   } else {
    struct Range { float low, high; } ranges[4]{};
    parallel(ctx, 4, [&](size_t task) {
     ScanU8Block(x, count * task / 4, count * (task + 1) / 4,
          ranges[task].low, ranges[task].high);
    });
    for (const auto &range : ranges) {
     minimum = std::min(minimum, range.low);
     maximum = std::max(maximum, range.high);
    }
   }
  }
  static thread_local std::vector<uint8_t> q;
  if (q.size() < count) q.resize(count);
  auto* qdata = q.data();
  float xs;
  int32_t xz;
  QuantizeU8Activation(ctx, x, count, minimum, maximum, qdata, xs, xz);
  static thread_local std::vector<const uint8_t*>columns;static thread_local std::vector<int>delta;if(columns.size()<size_t(kp)){columns.resize(kp);delta.resize(kp);}auto*column_ptr=columns.data();auto*delta_ptr=delta.data();for(int z=0;z<kp;z++){column_ptr[z]=z<k?qdata+size_t(z/kernel)*tin:nullptr;delta_ptr[z]=-pad+(z%kernel)*dilation;}
  parallel(ctx,4,[&](size_t task){const int step=64;int begin=tout*task/4,end=tout*(task+1)/4;static thread_local std::vector<uint8_t>left;if(left.size()<size_t(step)*kp)left.resize(size_t(step)*kp);int32_t sumx[step];
   for(int start=begin;start<end;start+=step){int rows=std::min(step,end-start),rounded=(rows+7)/8*8;
    auto load=[&](int col,int row){int t=(start+row)*stride+delta_ptr[col];if(column_ptr[col]&&t>=0&&t+8<=tin)return vld1_u8(column_ptr[col]+t);uint8_t temp[8];for(int j=0;j<8;j++)temp[j]=column_ptr[col]&&t+j>=0&&t+j<tin?column_ptr[col][t+j]:xz;return vld1_u8(temp);};
    if constexpr (!kBryce) {
      std::fill_n(sumx, rounded, 0);
      for (int r = 0; r < rounded; r += 8) {
        for (int z = 0; z < kp; z += 8) {
          uint8x8_t packed[8];
          PackU8Tile(load, r, z, left.data() + size_t(r) * kp + size_t(z / 8) * 64, packed);
          for (int lane = 0; lane < 8; ++lane) sumx[r + lane] += vaddlv_u8(packed[lane]);
        }
      }
    } else {
    for (int r = 0; r < rounded; r += 8) {
     uint32x4_t low_sums = vdupq_n_u32(0), high_sums = low_sums;
     for (int z = 0; z < kp; z += 8) {
      uint8x8_t packed[8];
      PackU8Tile(load, r, z, left.data() + size_t(r) * kp + size_t(z / 8) * 64, packed);
      auto reduce_pair = [](uint8x8_t a, uint8x8_t b) {
       return vpaddlq_u16(vpaddlq_u8(vcombine_u8(a, b)));
      };
      low_sums = vaddq_u32(low_sums, vpaddq_u32(
          reduce_pair(packed[0], packed[1]), reduce_pair(packed[2], packed[3])));
      high_sums = vaddq_u32(high_sums, vpaddq_u32(
          reduce_pair(packed[4], packed[5]), reduce_pair(packed[6], packed[7])));
     }
     vst1q_s32(sumx + r, vreinterpretq_s32_u32(low_sums));
     vst1q_s32(sumx + r + 4, vreinterpretq_s32_u32(high_sums));
    }
    }
    for(int r=0;r<rows;r+=8)for(int col=0;col<n;col+=8){uint32x4_t acc[16];auto*a=left.data()+size_t(r)*kp;auto*b=packed_rhs+size_t(col/8)*kp*8;
     AccumulateU8Tile(a, b, kp, acc);
     if constexpr (!kBryce) {
     EmitU8Tile(acc, n, rows, kp, r, col, start, tout, sumx, sums.data(),
                zeros.data(), scales.data(), bias.data(), xz, xs, y);
     } else {
     EmitU8TileInterleaved(acc, n, rows, kp, r, col, start, tout, sumx, sums.data(),
                           zeros.data(), scales.data(), bias.data(), xz, xs, y);
     }

    }
   }
  });
 });}
};
template <bool kBryce>
struct PackedDenseU8Op : Ort::CustomOpBase<PackedDenseU8Op<kBryce>, PackedDenseU8Conv<kBryce>, true> {
 const char*GetName()const{return kBryce ? "BrycePackedU8Conv" : "AmyPackedU8Conv";}size_t GetInputTypeCount()const{return 6;}
 ONNXTensorElementDataType GetInputType(size_t i)const{return i==5?ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:i==1||i==3?ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;}
 size_t GetOutputTypeCount()const{return 1;}ONNXTensorElementDataType GetOutputType(size_t)const{return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;}
 OrtStatusPtr CreateKernelV2(const OrtApi&,const OrtKernelInfo*i,void**k)const noexcept{return guard([&]{*k=new PackedDenseU8Conv<kBryce>(i);});}
};

using AmyPackedU8Op = PackedDenseU8Op<false>;
using BrycePackedU8Op = PackedDenseU8Op<true>;
