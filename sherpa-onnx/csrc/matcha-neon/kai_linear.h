// Store four independent rows as columns without changing their float bits.
inline void StoreTransposed4(float32x4_t a, float32x4_t b, float32x4_t c,
                             float32x4_t d, float *out, size_t stride) {
  const auto ab = vtrnq_f32(a, b);
  const auto cd = vtrnq_f32(c, d);
  vst1q_f32(out, vcombine_f32(vget_low_f32(ab.val[0]), vget_low_f32(cd.val[0])));
  vst1q_f32(out + stride, vcombine_f32(vget_low_f32(ab.val[1]), vget_low_f32(cd.val[1])));
  vst1q_f32(out + 2 * stride, vcombine_f32(vget_high_f32(ab.val[0]), vget_high_f32(cd.val[0])));
  vst1q_f32(out + 3 * stride, vcombine_f32(vget_high_f32(ab.val[1]), vget_high_f32(cd.val[1])));
}

template<bool Half> struct Linear {
 int n,k,kernel,stride,pad,layout,activation;
 std::vector<uint8_t> rhs;
 const uint8_t* rhs_data=nullptr;
 Linear(const OrtKernelInfo*raw,bool packed=false){
  Ort::ConstKernelInfo info(raw);n=info.GetAttribute<int64_t>("N");k=info.GetAttribute<int64_t>("K");kernel=info.GetAttribute<int64_t>("kernel");stride=info.GetAttribute<int64_t>("stride");pad=info.GetAttribute<int64_t>("pad");layout=info.GetAttribute<int64_t>("layout");int constant;
  activation=info.GetAttribute<int64_t>("activation");auto weight=info.GetTensorConstantInput(1,&constant);if(!constant)throw std::runtime_error("Q weight is not constant");
  if(packed){
   size_t expected=Half?kai_get_rhs_packed_size_rhs_pack_kxn_bf16p12x4biasf32_f16_neon(n,k):kai_get_rhs_packed_size_rhs_pack_nxk_qsi8cxp_qsi8cx_neon(n,k,4,8,1);
   auto offset=info.GetAttribute<int64_t>("rhs_offset");
   if(offset<0||offset>64||weight.GetTensorTypeAndShapeInfo().GetElementCount()!=expected+64)throw std::runtime_error("Packed RHS byte count/alignment offset");
   rhs_data=weight.GetTensorData<uint8_t>()+offset;return;
  }
  auto scale=info.GetTensorConstantInput(2,&constant),bias=info.GetTensorConstantInput(3,&constant);
  if constexpr(Half){
   const auto*w=reinterpret_cast<const __fp16*>(weight.GetTensorRawData());std::vector<__fp16>transposed(size_t(n)*k);for(int row=0;row<k;row++)for(int col=0;col<n;col++)transposed[size_t(row)*n+col]=w[size_t(col)*k+row];
   rhs.resize(kai_get_rhs_packed_size_rhs_pack_kxn_bf16p12x4biasf32_f16_neon(n,k));
   kai_run_rhs_pack_kxn_bf16p12x4biasf32_f16_neon(1,n,k,12,4,1,n*sizeof(__fp16),transposed.data(),bias.GetTensorData<float>(),nullptr,rhs.data(),0,nullptr);
  }else{
   rhs.resize(kai_get_rhs_packed_size_rhs_pack_nxk_qsi8cxp_qsi8cx_neon(n,k,4,8,1));
   const kai_rhs_pack_qsi8cx_params params{1,1.f};
   kai_run_rhs_pack_nxk_qsi8cxp_qsi8cx_neon(1,n,k,4,8,1,weight.GetTensorData<int8_t>(),bias.GetTensorData<float>(),scale.GetTensorData<float>(),rhs.data(),0,&params);
  }
  rhs_data=rhs.data();
  if(const char*directory=std::getenv("MATCHA_DUMP_PACKED_DIR")){
   auto id=info.GetAttribute<int64_t>("pack_id");std::string path=std::string(directory)+"/rhs-"+std::to_string(id)+".bin";
   FILE*file=std::fopen(path.c_str(),"wb");if(!file)throw std::runtime_error("Cannot write offline RHS");
   bool ok=std::fwrite(rhs.data(),1,rhs.size(),file)==rhs.size();ok=std::fclose(file)==0&&ok;if(!ok)throw std::runtime_error("Offline RHS write failed");
  }
 }
 OrtStatusPtr ComputeV2(OrtKernelContext*raw) noexcept{return guard([&]{
  Ort::KernelContext ctx(raw);auto input=ctx.GetInput(0);const float*x=input.GetTensorData<float>();auto shape=input.GetTensorTypeAndShapeInfo().GetShape();int tin=0,channels=0,m;
  if(layout){if(shape.size()!=3||shape[0]!=1)throw std::runtime_error("Conv requires [1,C,T]");channels=shape[1];tin=shape[2];m=layout==2?(tin-1)*stride-2*pad+kernel:(tin+2*pad-kernel)/stride+1;if(channels*kernel!=k)throw std::runtime_error("Conv K");shape={1,n,m};}
  else{if(shape.back()!=k)throw std::runtime_error("Linear K");m=input.GetTensorTypeAndShapeInfo().GetElementCount()/k;shape.back()=n;}
  float*y=ctx.GetOutput(0,shape).GetTensorMutableData<float>();
  const int input_limit = tin;
  struct Work{std::vector<float>matrix,result;std::vector<uint8_t>lhs;};static thread_local Work work;
  if(layout){
   work.matrix.resize(size_t(m)*k);auto*matrix=work.matrix.data();
   parallel(ctx, 4, [&](size_t task) {
     const int begin = m * task / 4, end = m * (task + 1) / 4;
     int row = begin;
     if (layout == 1 && stride == 1) {
       auto load = [&](int col, int first) {
         const float *channel = x + size_t(col / kernel) * tin;
         const int t = first - pad + col % kernel;
         if (t >= 0 && t + 4 <= input_limit) return vld1q_f32(channel + t);
         float lanes[4]{};
         for (int i = 0; i != 4; ++i)
           if (t + i >= 0 && t + i < input_limit) lanes[i] = channel[t + i];
         return vld1q_f32(lanes);
       };
       for (; row + 4 <= end; row += 4) {
         int col = 0;
         for (; col + 4 <= k; col += 4)
           StoreTransposed4(load(col, row), load(col + 1, row),
                            load(col + 2, row), load(col + 3, row),
                            matrix + size_t(row) * k + col, k);
         for (; col < k; ++col) {
           float lanes[4];
           vst1q_f32(lanes, load(col, row));
           for (int i = 0; i != 4; ++i) matrix[size_t(row + i) * k + col] = lanes[i];
         }
       }
     }
     for (; row < end; ++row) {
       for (int c = 0; c < channels; ++c) {
         for (int p = 0; p < kernel; ++p) {
           int t = row * stride - pad + p;
           bool valid = true;
           if (layout == 2) {
             t = row + pad - p;
             valid = t % stride == 0;
             t /= stride;
           }
           float value = 0;
           if (valid && t >= 0 && t < input_limit) {
             value = x[size_t(c) * tin + t];
           }
           const int col = c * kernel + p;
           matrix[size_t(row) * k + col] = value;
         }
       }
     }
   });
   x=matrix;work.result.resize(size_t(m)*n);
  }
  float*result=layout?work.result.data():y;
  if constexpr(Half){
   work.lhs.resize(kai_get_lhs_packed_size_lhs_pack_bf16p8x4_f16_neon(m,k,8,4,1));auto*lhs=work.lhs.data();
   int rowtile=((m+31)/32)*8;
   parallel(ctx,(m+rowtile-1)/rowtile,[&](size_t task){int begin=task*rowtile,rows=std::min(rowtile,m-begin);auto offset=kai_get_lhs_packed_offset_lhs_pack_bf16p8x4_f16_neon(begin,k,8,4,1);sherpa_onnx::PackBf16F32(rows,k,x+size_t(begin)*k,lhs+offset);});
   const int tile=((n+47)/48)*12;
   auto run=[&](int begin,int rows,int col,int columns){auto offset=kai_get_rhs_packed_offset_rhs_pack_kxn_bf16p12x4biasf32_f16_neon(col,k);auto left=kai_get_lhs_packed_offset_lhs_pack_bf16p8x4_f16_neon(begin,k,8,4,1);
    kai_run_matmul_clamp_f32_bf16p8x4_bf16p12x4b_8x12_neon_mmla(rows,columns,k,lhs+left,rhs_data+offset,result+size_t(begin)*n+col,n*sizeof(float),sizeof(float),-INFINITY,INFINITY);
    if(activation==1)for(int row=begin;row<begin+rows;row++)gelu(result+size_t(row)*n+col,columns);
   };
   parallel(ctx,(n+tile-1)/tile,[&](size_t task){int col=task*tile;run(0,m,col,std::min(tile,n-col));});
  }else{
   work.lhs.resize(kai_get_lhs_packed_size_lhs_quant_pack_qai8dxp_f32(m,k,4,8,1));auto*lhs=work.lhs.data();
   int rowtile=((m+15)/16)*4;
   parallel(ctx,(m+rowtile-1)/rowtile,[&](size_t task){int begin=task*rowtile,rows=std::min(rowtile,m-begin);auto offset=kai_get_lhs_packed_offset_lhs_quant_pack_qai8dxp_f32(begin,k,4,8,1);kai_run_lhs_quant_pack_qai8dxp_f32(rows,k,4,8,1,0,x+size_t(begin)*k,k*sizeof(float),lhs+offset);});
   const int tile=((n+15)/16)*4;
   parallel(ctx,(n+tile-1)/tile,[&](size_t task){int col=task*tile,columns=std::min(tile,n-col);auto offset=kai_get_rhs_packed_offset_rhs_pack_nxk_qsi8cxp_qsi8cx_neon(col,k,4,8,1);
    kai_run_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm(m,columns,k,lhs,rhs_data+offset,result+col,n*sizeof(float),sizeof(float),-INFINITY,INFINITY);
    if(activation==1)for(int row=0;row<m;row++)gelu(result+size_t(row)*n+col,columns);
   });
  }
  if (layout) {
    parallel(ctx, 4, [&](size_t task) {
      int col = n * task / 4;
      const int end = n * (task + 1) / 4;
      for (; col + 4 <= end; col += 4) {
        int row = 0;
        for (; row + 4 <= m; row += 4) {
          const float *p = result + size_t(row) * n + col;
          StoreTransposed4(vld1q_f32(p), vld1q_f32(p + n),
                           vld1q_f32(p + 2 * n), vld1q_f32(p + 3 * n),
                           y + size_t(col) * m + row, m);
        }
        for (; row < m; ++row)
          for (int i = 0; i != 4; ++i)
            y[size_t(col + i) * m + row] = result[size_t(row) * n + col + i];
      }
      for (; col < end; ++col)
        for (int row = 0; row < m; ++row)
          y[size_t(col) * m + row] = result[size_t(row) * n + col];
    });
  }
 });}
};

using QLinear=Linear<false>;using F16Linear=Linear<true>;
struct PackedQLinear:QLinear {PackedQLinear(const OrtKernelInfo*info):QLinear(info,true){}};
struct PackedF16Linear:F16Linear {PackedF16Linear(const OrtKernelInfo*info):F16Linear(info,true){}};
