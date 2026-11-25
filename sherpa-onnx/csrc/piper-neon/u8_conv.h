// UINT8 global activation/per-tensor weight quantization compatible with ORT's
// DynamicQuantizeLinear + ConvInteger, accelerated with unsigned I8MM.
template <bool kBryce>
struct U8ConvKernel {
 int n,c,kernel,pad,dilation,stride,group,k,kp,np;uint8_t wz;float ws;
 const uint8_t *rhs_data = nullptr;
 std::vector<uint8_t>weights,rhs;std::vector<int32_t>sums;std::vector<float>bias,scales;std::vector<uint8_t>zeros;
 U8ConvKernel(const OrtKernelInfo *raw, int variant = 0,
        int variants = 1, int sums_input = -1) {
  const bool trace_load = !kBryce;
  transai::piper_neon::LoadProfile profile(
      sums_input < 0 ? "U8-pack" : "Carlfm-U8-prepacked", trace_load);
  const Ort::ConstKernelInfo info(raw);
  n = info.GetAttribute<int64_t>("N");
  c = info.GetAttribute<int64_t>("C");
  kernel = info.GetAttribute<int64_t>("kernel");
  pad = info.GetAttribute<int64_t>("pad");
  dilation = info.GetAttribute<int64_t>("dilation");
  stride = info.GetAttribute<int64_t>("stride");
  group = info.GetAttribute<int64_t>("group");
  k = (c / group) * kernel;
  kp = (k + 7) / 8 * 8;
  np = (n + 7) / 8 * 8;
  auto get_constant = [&](size_t index) {
    int constant = 0;
    auto tensor = info.GetTensorConstantInput(index, &constant);
    if (!constant) throw std::runtime_error("U8 parameter must be constant");
    return tensor;
  };
  const Ort::ConstValue weight = get_constant(1);
  const Ort::ConstValue scale = get_constant(2);
  const Ort::ConstValue zero = get_constant(3);
  const Ort::ConstValue offset = get_constant(4);
  const size_t sc_count = scale.GetTensorTypeAndShapeInfo().GetElementCount() / variants;
  const size_t zp_count = zero.GetTensorTypeAndShapeInfo().GetElementCount() / variants;
  if ((sc_count != 1 && sc_count != size_t(n)) ||
      (zp_count != 1 && zp_count != size_t(n)) ||
      scale.GetTensorTypeAndShapeInfo().GetElementCount() != sc_count * variants ||
      zero.GetTensorTypeAndShapeInfo().GetElementCount() != zp_count * variants ||
      offset.GetTensorTypeAndShapeInfo().GetElementCount() != size_t(n) * variants)
    throw std::runtime_error("U8 scale, zero point or bias shape");
  const float *sc_data = scale.GetTensorData<float>() + variant * sc_count;
  const uint8_t *zp_data = zero.GetTensorData<uint8_t>() + variant * zp_count;
  wz = *zp_data;
  ws = *sc_data;
  scales.assign(np, ws);
  zeros.assign(np, wz);
  for (int col = 0; col < n; ++col) {
    scales[col] = sc_data[sc_count == 1 ? 0 : col];
    zeros[col] = zp_data[zp_count == 1 ? 0 : col];
  }
  const float *bias_data = offset.GetTensorData<float>() + variant * n;
  bias.assign(bias_data, bias_data + n);
  if (sums_input >= 0) {
    const Ort::ConstValue packed_sums = get_constant(sums_input);
    const size_t stride = size_t(np) * kp;
    if (group != 1 || weight.GetTensorTypeAndShapeInfo().GetElementCount() != stride * variants ||
        packed_sums.GetTensorTypeAndShapeInfo().GetElementCount() != size_t(np) * variants)
      throw std::runtime_error("Carlfm packed U8 shape");
    // Dense weights are read-only ORT initializers owned by the Session.
    rhs_data = weight.GetTensorData<uint8_t>() + stride * variant;
    const int32_t *sum_data = packed_sums.GetTensorData<int32_t>() + size_t(np) * variant;
    sums.assign(sum_data, sum_data + np);
  } else {
    const size_t stride = size_t(n) * k;
    if (weight.GetTensorTypeAndShapeInfo().GetElementCount() != stride * variants)
      throw std::runtime_error("U8 weight shape");
    const uint8_t *data = weight.GetTensorData<uint8_t>() + size_t(variant) * stride;
    weights.assign(data, data + stride);
    sums.assign(np, 0);
    rhs.resize(size_t(np) * kp);
    for (int col = 0; col < np; ++col) {
      for (int j = 0; j < kp; ++j) {
        const uint8_t value = col < n && j < k ? weights[size_t(col) * k + j] : zeros[col];
        const size_t index = size_t(col / 8) * kp * 8 + size_t(j / 8) * 64 +
                             (col % 8) / 2 * 16 + (col % 2) * 8 + j % 8;
        rhs[index] = value;
        sums[col] += value;
      }
    }
    rhs_data = rhs.data();
  }
 }
 OrtStatusPtr ComputeV2(OrtKernelContext*raw)noexcept{return guard([&]{Ort::KernelContext ctx(raw);auto input=ctx.GetInput(0);auto shape=input.GetTensorTypeAndShapeInfo().GetShape();if(shape.size()!=3||shape[0]!=1||shape[1]!=c)throw std::runtime_error("U8 batch/channels");int tin=shape[2],tout=(tin+2*pad-dilation*(kernel-1)-1)/stride+1;auto*x=input.GetTensorData<float>();auto*y=ctx.GetOutput(0,{1,n,tout}).GetTensorMutableData<float>();size_t count=size_t(c)*tin;
  float minimum=0,maximum=0;
  if constexpr (kBryce) {
    ScanU8Block(x, 0, count, minimum, maximum);
  } else {
    ScanU8Channels(x, c, tin, minimum, maximum);
  }
  static thread_local std::vector<uint8_t> q;
  if (q.size() < count) q.resize(count);
  auto* qdata = q.data();
  float xs;
  int32_t xz;
  QuantizeU8Activation(ctx, x, count, minimum, maximum, qdata, xs, xz);
  if(group!=1){if(group!=c||n!=c)throw std::runtime_error("U8 only dense/depthwise");parallel(ctx,4,[&](size_t task){for(int channel=n*task/4;channel<n*(task+1)/4;channel++)for(int t=0;t<tout;t++){int32_t sum=0;for(int p=0;p<kernel;p++){int time=t*stride-pad+p*dilation;int a=time>=0&&time<tin?qdata[size_t(channel)*tin+time]:xz;sum+=(a-xz)*(int(weights[size_t(channel)*kernel+p])-zeros[channel]);}float value=float(sum)*scales[channel];y[size_t(channel)*tout+t]=value*xs+bias[channel];}});return;}
  static thread_local std::vector<const uint8_t*>columns;static thread_local std::vector<int>delta;if(columns.size()<size_t(kp)){columns.resize(kp);delta.resize(kp);}auto*column_ptr=columns.data();auto*delta_ptr=delta.data();for(int z=0;z<kp;z++){column_ptr[z]=z<k?qdata+size_t(z/kernel)*tin:nullptr;delta_ptr[z]=-pad+(z%kernel)*dilation;}
  parallel(ctx,4,[&](size_t task){const int step=64;int begin=tout*task/4,end=tout*(task+1)/4;static thread_local std::vector<uint8_t>left;if(left.size()<size_t(step)*kp)left.resize(size_t(step)*kp);int32_t sumx[step];
   for(int start=begin;start<end;start+=step){int rows=std::min(step,end-start),rounded=(rows+7)/8*8;std::fill_n(sumx,rounded,0);
    auto load=[&](int col,int row){int t=(start+row)*stride+delta_ptr[col];if(column_ptr[col]&&t>=0&&t+8<=tin)return vld1_u8(column_ptr[col]+t);uint8_t temp[8];for(int j=0;j<8;j++)temp[j]=column_ptr[col]&&t+j>=0&&t+j<tin?column_ptr[col][t+j]:xz;return vld1_u8(temp);};
    for (int r = 0; r < rounded; r += 8) {
      for (int z = 0; z < kp; z += 8) {
        uint8x8_t packed[8];
        PackU8Tile(load, r, z, left.data() + size_t(r) * kp + size_t(z / 8) * 64, packed);
        for (int lane = 0; lane < 8; ++lane) sumx[r + lane] += vaddlv_u8(packed[lane]);
      }
    }
    for(int r=0;r<rows;r+=8)for(int col=0;col<n;col+=8){uint32x4_t acc[16];auto*a=left.data()+size_t(r)*kp;auto*b=rhs_data+size_t(col/8)*kp*8;
     AccumulateU8Tile(a, b, kp, acc);
     if constexpr (kBryce) {
       EmitU8TileInterleaved(acc, n, rows, kp, r, col, start, tout, sumx, sums.data(),
                             zeros.data(), scales.data(), bias.data(), xz, xs, y);
     } else {
       EmitU8Tile(acc, n, rows, kp, r, col, start, tout, sumx, sums.data(),
                  zeros.data(), scales.data(), bias.data(), xz, xs, y);
     }

    }
   }
  });
 });}
};
using U8Conv = U8ConvKernel<false>;

template <bool kBryce>
struct BasicU8Op : Ort::CustomOpBase<BasicU8Op<kBryce>, U8ConvKernel<kBryce>, true> {
  const char* GetName() const { return kBryce ? "BryceU8Conv" : "U8Conv"; }
  size_t GetInputTypeCount() const { return 5; }
  ONNXTensorElementDataType GetInputType(size_t index) const {
    return index == 1 || index == 3 ? ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8
                                    : ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  }
  size_t GetOutputTypeCount() const { return 1; }
  ONNXTensorElementDataType GetOutputType(size_t) const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  }
  OrtStatusPtr CreateKernelV2(const OrtApi&, const OrtKernelInfo* info,
                              void** kernel) const noexcept {
    return guard([&] { *kernel = new U8ConvKernel<kBryce>(info); });
  }
};

using U8Op = BasicU8Op<false>;
using BryceU8Op = BasicU8Op<true>;

struct IndexedU8Conv {
 std::vector<std::unique_ptr<U8Conv>> kernels;
 IndexedU8Conv(const OrtKernelInfo*raw, int sums_input = -1){Ort::ConstKernelInfo i(raw);int64_t variants=i.GetAttribute<int64_t>("variants");if(variants<1||variants>16)throw std::runtime_error("Invalid U8 variants");for(int v=0;v<variants;v++)kernels.emplace_back(new U8Conv(raw,v,variants,sums_input));}
 OrtStatusPtr ComputeV2(OrtKernelContext*raw)noexcept{Ort::KernelContext ctx(raw);const int64_t v=*ctx.GetInput(5).GetTensorData<int64_t>();if(v<0||v>=kernels.size())return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT,"U8 variant index");return kernels[v]->ComputeV2(raw);}
};
struct IndexedU8Op:Ort::CustomOpBase<IndexedU8Op,IndexedU8Conv,true>{
 const char*GetName()const{return "IndexedU8Conv";}size_t GetInputTypeCount()const{return 6;}
 ONNXTensorElementDataType GetInputType(size_t i)const{return i==5?ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:i==1||i==3?ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;}
 size_t GetOutputTypeCount()const{return 1;}ONNXTensorElementDataType GetOutputType(size_t)const{return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;}
 OrtStatusPtr CreateKernelV2(const OrtApi&,const OrtKernelInfo*i,void**k)const noexcept{return guard([&]{*k=new IndexedU8Conv(i);});}
};

template <bool Indexed>
struct CarlfmPackedU8Op
    : Ort::CustomOpBase<CarlfmPackedU8Op<Indexed>,
                        std::conditional_t<Indexed, IndexedU8Conv, U8Conv>, true> {
  const char *GetName() const {
    return Indexed ? "CarlfmIndexedPackedU8Conv" : "CarlfmPackedU8Conv";
  }
  size_t GetInputTypeCount() const { return Indexed ? 7 : 6; }
  ONNXTensorElementDataType GetInputType(size_t i) const {
    if (i == 1 || i == 3) return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
    if (Indexed && i == 5) return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    if (i == (Indexed ? 6u : 5u)) return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  }
  size_t GetOutputTypeCount() const { return 1; }
  ONNXTensorElementDataType GetOutputType(size_t) const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  }
  OrtStatusPtr CreateKernelV2(const OrtApi &, const OrtKernelInfo *info, void **kernel) const noexcept {
    return guard([&] {
      if constexpr (Indexed) {
        *kernel = new IndexedU8Conv(info, 6);
      } else {
        *kernel = new U8Conv(info, 0, 1, 5);
      }
    });
  }
};
