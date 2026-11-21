#define CL_TARGET_OPENCL_VERSION 200
#define CL_ENABLE_BETA_EXTENSIONS
#include <CL/cl.h>
#include <CL/cl_ext.h>
#include <nlohmann/json.hpp>
#include "mali_loader.h"
#include "aot-view.h"
#include "device_target.h"
#include "sherpa-onnx/csrc/mali-g720/init-timer.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <initializer_list>
#include <map>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
using J=nlohmann::json;
using Clock=std::chrono::steady_clock;
#ifdef TRANSAI_SEACO_SCORE_PROBE
static std::vector<char> read(const std::string&p){std::ifstream f(p,std::ios::binary|std::ios::ate);if(!f)throw std::runtime_error("open "+p);auto n=f.tellg();std::vector<char>b(n);f.seekg(0);f.read(b.data(),n);return b;}
static J jsonfile(const std::string&p){std::ifstream f(p);if(!f)throw std::runtime_error("open "+p);J j;f>>j;return j;}
#endif
static void check(cl_int e){if(e)throw std::runtime_error("OpenCL status "+std::to_string(e));}
struct Tensor{std::shared_ptr<cl_mem> b;int r=0,c=0,kind=0;size_t elem=4;cl_mem mem()const{return *b;}};
struct Runtime{
 void*lib=OpenMali();
#define API(x) decltype(&::x) x=reinterpret_cast<decltype(&::x)>(dlsym(lib,#x));
 API(clGetPlatformIDs) API(clGetDeviceIDs) API(clGetDeviceInfo) API(clCreateContext)
 API(clCreateCommandQueue) API(clCreateProgramWithBinary) API(clGetProgramInfo) API(clBuildProgram) API(clGetProgramBuildInfo)
 API(clCreateKernel) API(clSetKernelArg) API(clEnqueueNDRangeKernel) API(clCreateBuffer)
 API(clEnqueueWriteBuffer) API(clEnqueueFillBuffer) API(clFinish)
 decltype(&::clEnqueueReadBuffer) raw_read=reinterpret_cast<decltype(&::clEnqueueReadBuffer)>(dlsym(lib,"clEnqueueReadBuffer"));
 API(clReleaseMemObject) API(clReleaseKernel) API(clReleaseProgram) API(clReleaseCommandQueue) API(clReleaseContext)
 API(clGetExtensionFunctionAddressForPlatform)
 API(clSetKernelExecInfo)
#undef API
 cl_context ctx=nullptr;cl_command_queue queue=nullptr;cl_program program=nullptr;cl_device_id dev=nullptr;
 cl_mem weights=nullptr,dims=nullptr,window=nullptr,melranges=nullptr,tokenbytes=nullptr,tokenoffsets=nullptr;std::shared_ptr<SeacoAot> archive;int max_token_bytes=0;J model,target;std::map<std::string,cl_kernel> kernels;
 std::map<size_t,std::vector<cl_mem>> pool;std::vector<Tensor> seacokv;Tensor embedding;
 bool prepare_only=false;int result_capacity=0;J hotword_key;
 std::string device,driver;int hotcount=0;
#ifdef TRANSAI_SEACO_SCORE_PROBE
 bool capture=false;std::string dumpdir;
#else
 static constexpr bool capture=false;
#endif
 clCreateCommandBufferKHR_fn cbcreate=nullptr;clCommandNDRangeKernelKHR_fn cbkernel=nullptr;
 clFinalizeCommandBufferKHR_fn cbfinal=nullptr;clEnqueueCommandBufferKHR_fn cbenqueue=nullptr;clReleaseCommandBufferKHR_fn cbrelease=nullptr;
 cl_command_buffer_khr recording=nullptr;
 struct Cached{std::vector<cl_command_buffer_khr> commands;Tensor input,result;int R;};
 std::map<std::string,Cached> cache;
 std::vector<cl_command_buffer_khr> recorded;int recorded_count=0;
 static constexpr int gm=1, cbchunk=512, ly=8, gn=4, wbsm=0;
 bool in_infer=false;cl_mem expected_result=nullptr;int result_reads=0,intermediate_reads=0;
 cl_int clEnqueueReadBuffer(cl_command_queue q,cl_mem b,cl_bool blocking,size_t offset,size_t size,void*dst,cl_uint n,const cl_event*wait,cl_event*event){
  if(in_infer){if(b==expected_result)result_reads++;else intermediate_reads++;}
  if(!capture&&(!in_infer||b!=expected_result))throw std::runtime_error("Production policy forbids intermediate GPU-to-CPU reads");
  return raw_read(q,b,blocking,offset,size,dst,n,wait,event);
 }
 Runtime(const std::string&file):archive(std::make_shared<SeacoAot>(file)),model(archive->json("seaco/plan.json")){
  sherpa_onnx::MaliG720InitTimer timer("seaco");
  target=archive->json("seaco/target.json");
  if(model["model_type"]!="seaco_paraformer"||model["encoder"].size()!=50||model["seaco"].size()!=7)throw std::runtime_error("SeACo architecture");
  if (target.value("execution_schema", "") != "seaco_mmul_fused_v1" ||
      target.value("weights_layout", "") != "mmul_lane_tiles") throw std::runtime_error("SeACo requires the current lane-packed execution schema");
  const auto &runtime=target.at("runtime");
  if(gm!=runtime.at("gm")||gn!=runtime.at("gn")||ly!=runtime.at("ly")||cbchunk!=runtime.at("chunk")||wbsm!=runtime.at("wbsm"))throw std::runtime_error("AOT runtime configuration mismatch");
  auto weight_size=archive->view("seaco/weights.bin").size();
  if(weight_size!=target["weights_bytes"].get<size_t>())throw std::runtime_error("AOT weights size");
  for(auto&item:model["assets"].items()){auto&a=item.value();size_t off=a["offset"],bytes=a["bytes"];if(off%64||off>weight_size||bytes>weight_size-off)throw std::runtime_error("AOT tensor bounds");}
  if(model["format"]!=2)throw std::runtime_error("Requires format2 with interleaved FP16 MMUL weights");
  cl_platform_id p;check(clGetPlatformIDs(1,&p,nullptr));check(clGetDeviceIDs(p,CL_DEVICE_TYPE_GPU,1,&dev,nullptr));
  char text[20000];check(clGetDeviceInfo(dev,CL_DEVICE_NAME,sizeof(text),text,nullptr));device=text;
  if(device.find("Mali-G720")==std::string::npos)throw std::runtime_error("Expected Mali-G720: "+device);
  check(clGetDeviceInfo(dev,CL_DRIVER_VERSION,sizeof(text),text,nullptr));driver=text;
  check(clGetDeviceInfo(dev,CL_DEVICE_EXTENSIONS,sizeof(text),text,nullptr));
  if(!strstr(text,"cl_arm_matrix_multiply"))throw std::runtime_error("MMUL extension unavailable");
  if(SeacoDeviceTarget(lib,dev)!=target.at("device"))throw std::runtime_error("AOT GPU/driver identity mismatch");
  cl_int e;ctx=clCreateContext(nullptr,1,&dev,nullptr,nullptr,&e);check(e);queue=clCreateCommandQueue(ctx,dev,0,&e);check(e);
#define EXT(field,type,name) field=reinterpret_cast<type>(clGetExtensionFunctionAddressForPlatform(p,name));if(!field)throw std::runtime_error(name);
  EXT(cbcreate,clCreateCommandBufferKHR_fn,"clCreateCommandBufferKHR")
  EXT(cbkernel,clCommandNDRangeKernelKHR_fn,"clCommandNDRangeKernelKHR")
  EXT(cbfinal,clFinalizeCommandBufferKHR_fn,"clFinalizeCommandBufferKHR")
  EXT(cbenqueue,clEnqueueCommandBufferKHR_fn,"clEnqueueCommandBufferKHR")
  EXT(cbrelease,clReleaseCommandBufferKHR_fn,"clReleaseCommandBufferKHR")
#undef EXT
  timer.Mark("device_and_plan");
  auto executable=archive->view("seaco/program.clbin");const auto*data=reinterpret_cast<const unsigned char*>(executable.data());size_t binary_size=executable.size();cl_int binary_status;
  program=clCreateProgramWithBinary(ctx,1,&dev,&binary_size,&data,&binary_status,&e);check(e);check(binary_status);
  check(clBuildProgram(program,1,&dev,nullptr,nullptr,nullptr));
  cl_program_binary_type binary_type;check(clGetProgramBuildInfo(program,dev,CL_PROGRAM_BINARY_TYPE,sizeof(binary_type),&binary_type,nullptr));
  if(binary_type!=CL_PROGRAM_BINARY_TYPE_EXECUTABLE)throw std::runtime_error("AOT executable required");
  for (const char *name : {"quant_encoder_qkv", "encoder_residual_qnorm"}) {
    cl_int status=CL_SUCCESS;
    auto kernel=clCreateKernel(program,name,&status);
    check(status);
    kernels.emplace(name,kernel);
  }
  timer.Mark("program");
  auto w=archive->view("seaco/weights.bin");
  weights=buffer(w.size(),w.data());dims=buffer(6*sizeof(int));
  auto it=model["assets"]["frontend.window"];window=buffer(1600,w.data()+it["offset"].get<size_t>());
  std::vector<int>ranges;for(int i=0;i<80;i++){ranges.push_back(model.at("mel_ranges").at(i).at(0).get<int>());ranges.push_back(model.at("mel_ranges").at(i).at(1).get<int>());}melranges=buffer(ranges.size()*4,ranges.data());
  std::vector<char>tb;std::vector<int>to{0};for(auto&t:model["tokens"]){auto v=t.get<std::string>();max_token_bytes=std::max(max_token_bytes,int(v.size()));tb.insert(tb.end(),v.begin(),v.end());to.push_back(tb.size());}tokenbytes=buffer(tb.size(),tb.data());tokenoffsets=buffer(to.size()*4,to.data());
  check(clFinish(queue));archive.reset();
  timer.Mark("weights_upload");
  std::cerr<<"GPU "<<device<<" driver "<<driver<<" weights="<<w.size()<<"\n";
 }
 ~Runtime(){for(auto&[n,c]:cache)for(auto cb:c.commands)cbrelease(cb);cache.clear();seacokv.clear();embedding={};for(auto&[n,k]:kernels)clReleaseKernel(k);for(auto&[n,v]:pool)for(auto b:v)clReleaseMemObject(b);
  if(melranges)clReleaseMemObject(melranges);if(tokenbytes)clReleaseMemObject(tokenbytes);if(tokenoffsets)clReleaseMemObject(tokenoffsets);if(window)clReleaseMemObject(window);if(weights)clReleaseMemObject(weights);if(dims)clReleaseMemObject(dims);
  if(program)clReleaseProgram(program);if(queue)clReleaseCommandQueue(queue);if(ctx)clReleaseContext(ctx);if(lib)dlclose(lib);}
 cl_mem buffer(size_t n,const void*p=nullptr,cl_mem_flags flags=CL_MEM_READ_WRITE){cl_int e;auto b=clCreateBuffer(ctx,flags|(p?CL_MEM_COPY_HOST_PTR:0),std::max(n,size_t(4)),const_cast<void*>(p),&e);check(e);return b;}
 Tensor tensor(int r,int c,int kind=0,size_t elem=0){if(!elem)elem=2;size_t needed=size_t(r)*c*elem,n=256;while(n<needed)n<<=1;auto&v=pool[n];cl_mem b;if(v.empty()){b=buffer(n);}else{b=v.back();v.pop_back();}
  return {std::shared_ptr<cl_mem>(new cl_mem(b),[this,n](cl_mem*p){pool[n].push_back(*p);delete p;}),r,c,kind,elem};}
 template<class T>void arg(cl_kernel k,int i,const T&v){check(clSetKernelArg(k,i,sizeof(T),&v));}
 template<class... A>void run(const std::string&name,std::initializer_list<size_t> g,std::initializer_list<size_t> l,const A&...args){
  auto it=kernels.find(name);cl_kernel k;if(it==kernels.end()){cl_int e;k=clCreateKernel(program,name.c_str(),&e);check(e);kernels[name]=k;
   if(name.rfind("gemm8",0)==0||name=="gemmh"||name=="score"||name=="score8")check(clSetKernelExecInfo(k,CL_KERNEL_EXEC_INFO_WORKGROUP_BATCH_SIZE_MODIFIER_ARM,sizeof(wbsm),&wbsm));
  }else k=it->second;
  int i=0;(arg(k,i++,args),...);
  if(recording){if(recorded_count==cbchunk){check(cbfinal(recording));recorded.push_back(recording);cl_int e;recording=cbcreate(1,&queue,nullptr,&e);check(e);recorded_count=0;}
   // The command buffer uses the same in-order queue for every dispatch.
   check(cbkernel(recording,nullptr,nullptr,k,g.size(),nullptr,g.begin(),l.size()?l.begin():nullptr,0,nullptr,nullptr,nullptr));
   recorded_count++;}
  else check(clEnqueueNDRangeKernel(queue,k,g.size(),nullptr,g.begin(),l.size()?l.begin():nullptr,0,nullptr,nullptr));
 }
 cl_ulong off(const std::string&name,int bytes=4){return model["assets"].at(name)["offset"].get<cl_ulong>()/bytes;}
 void zero(const Tensor&t){float z=0;check(clEnqueueFillBuffer(queue,t.mem(),&z,4,0,size_t(t.r)*t.c*t.elem,0,nullptr,nullptr));}
#ifdef TRANSAI_SEACO_SCORE_PROBE
 void dump(const std::string&name,const Tensor&t){if(!capture)return;std::vector<char>x(size_t(t.r)*t.c*t.elem);check(clEnqueueReadBuffer(queue,t.mem(),CL_TRUE,0,x.size(),x.data(),0,nullptr,nullptr));std::ofstream f(dumpdir+"/"+name+(t.elem==2?".f16":".f32"),std::ios::binary);f.write(x.data(),x.size());}
#endif
 Tensor norm(const Tensor&x,const std::string&name){auto y=tensor(x.r,x.c,x.kind);run("norm",{size_t(x.r)*64},{64},x.mem(),weights,off(name+".weight"),off(name+".bias"),y.mem(),x.r,x.c,1.e-12f,dims,x.kind);return y;}
 Tensor linear_s8(const Tensor &q, const Tensor &scale, const std::string &name, int relu, float outscale) {
  const auto &params=model["linears"].at(name);
  const int n=params["n"], k=params["kpad"];
  if(params["precision"]!="s8"||q.c!=k) throw std::runtime_error("Invalid S8 linear: "+name);
  auto y=tensor(q.r,n,q.kind);
  const bool large=(q.kind==0&&q.r>=128)||(q.kind==1&&q.r>=384)||(q.kind==5&&q.r>=512);
  const char *kernel=q.r<=32?"gemm8direct":large?"gemm8_large":"gemm8";
  run(kernel,{size_t((n+4*gn-1)/(4*gn))*16,size_t((q.r+gm*4*ly-1)/(gm*4*ly))*ly},{16,size_t(ly)},
      q.mem(),weights,off(name+".q",1),scale.mem(),weights,off(name+".scale"),off(name+".bias"),y.mem(),
      q.r,k,n,params["npad"].get<int>(),relu,dims,q.kind,outscale);
  return y;
 }
 Tensor linear(const Tensor &x,const std::string &name,int relu=0,const std::string &ln="",float outscale=1.f) {
  const auto &params=model["linears"].at(name);
  const int n=params["n"],k=params["kpad"];
  if(x.c!=params["k"]) throw std::runtime_error("linear dims "+name);
  if(params["precision"]=="s8") {
    auto q=tensor(x.r,k,x.kind,1),scale=tensor(x.r,1,x.kind,4);
    if(ln.empty()) run("quantvec",{size_t(x.r)*64},{64},x.mem(),q.mem(),scale.mem(),x.r,x.c,k,dims,x.kind);
    else run("qnorm_"+std::to_string(x.c),{size_t(x.r)*64},{64},x.mem(),weights,off(ln+".weight"),off(ln+".bias"),q.mem(),scale.mem(),x.r,x.c,k,dims,x.kind);
    return linear_s8(q,scale,name,relu,outscale);
  }
  if(!ln.empty()) return linear(norm(x,ln),name,relu,"",outscale);
  auto y=tensor(x.r,n,x.kind);
  run("gemmh",{size_t((n+15)/16)*16,size_t((x.r+15)/16)},{16,1},x.mem(),weights,off(name+".h",2),weights,off(name+".bias"),y.mem(),x.r,x.c,k,n,relu,dims,x.kind,outscale);
  return y;
 }
 Tensor add(const Tensor&a,const Tensor&b){if(a.r!=b.r||a.c!=b.c)throw std::runtime_error("add dims");auto y=tensor(a.r,a.c,a.kind);run("add",{(size_t(a.r)*a.c+255)/256*256},{256},a.mem(),b.mem(),y.mem(),a.r,a.c,dims,a.kind);return y;}
 Tensor depth(const Tensor&x,const std::string&name,int cols=512,int offset=0){auto a=model["assets"].at(name+".fsmn_block.weight");int ks=a["shape"][2];auto y=tensor(x.r,cols,x.kind);
  run("depth",{size_t(cols/4),size_t(x.r)},{64,1},x.mem(),weights,off(name+".fsmn_block.weight"),y.mem(),x.r,cols,x.c,offset,ks,ks/2,dims,x.kind);return y;}
 Tensor attention(const Tensor &q,const Tensor &kv,int qo,int ko,int vo) {
  auto s=tensor(q.r*4,kv.r,0,4),y=tensor(q.r,512,q.kind);
  const bool fused=q.mem()==kv.mem()&&q.c==1536&&q.kind==0&&qo==0&&ko==512&&vo==1024;
  const int mp=(kv.r+15)/16*16;
  auto vp=tensor(mp*4,128,0,1),vs=tensor(kv.r*4,1,0,4);
  {
    auto qp=tensor(q.r*4,128,0,1),kp=tensor(kv.r*4,128,0,1),qs=tensor(q.r*4,1,0,4),ks=tensor(kv.r*4,1,0,4);
    if(fused) {
      run("quant_encoder_qkv",{size_t(q.r)*12*64},{64},q.mem(),qp.mem(),qs.mem(),kp.mem(),ks.mem(),vp.mem(),vs.mem(),q.r,mp,dims);
    } else {
      run("quant_head",{size_t(q.r)*4*64},{64},q.mem(),qp.mem(),qs.mem(),q.r,q.c,qo,dims,q.kind);
      run("quant_head",{size_t(kv.r)*4*64},{64},kv.mem(),kp.mem(),ks.mem(),kv.r,kv.c,ko,dims,kv.kind);
    }
    run("score8",{size_t((kv.r+15)/16)*16,size_t((q.r+15)/16),4},{16,1,1},qp.mem(),kp.mem(),qs.mem(),ks.mem(),s.mem(),q.r,kv.r,dims,q.kind,kv.kind);
  }
  if(!fused) run("quant_v",{size_t(kv.r)*4*64},{64},kv.mem(),vp.mem(),vs.mem(),kv.r,mp,kv.c,vo,dims,kv.kind);
  auto ap=tensor(q.r*4,mp,0,1),ps=tensor(q.r*4,1,0,4);
  run("prob8",{size_t(q.r)*4*64},{64},s.mem(),vs.mem(),ap.mem(),ps.mem(),q.r,kv.r,mp,dims,q.kind,kv.kind);
  run("context8",{128,size_t((q.r+15)/16),4},{16,1,1},ap.mem(),vp.mem(),ps.mem(),y.mem(),q.r,kv.r,mp,dims,q.kind,kv.kind);
  return y;
 }
 void encoder_residual_qnorm(const Tensor &a,const Tensor &b,const Tensor &residual,const Tensor &out,
                             const Tensor &q,const Tensor &scale,const std::string &norm_name,int operation) {
  run("encoder_residual_qnorm",{size_t(out.r)*64},{64},a.mem(),b.mem(),residual.mem(),out.mem(),q.mem(),scale.mem(),
      weights,off(norm_name+".weight"),off(norm_name+".bias"),dims,operation);
 }
 Tensor encoder(Tensor x) {
  auto pos=tensor(x.r,560);
  run("position",{576,size_t(x.r)},{64,1},x.mem(),pos.mem(),x.r);
  x=pos;
  Tensor next_q,next_scale;
  const auto &layers=model.at("encoder");
  for(size_t index=0;index<layers.size();++index) {
    const auto &layer=layers.at(index);
    const std::string name=layer.at("name");
    const std::string qkv_name=name+".self_attn.linear_q_k_v";
    auto qkv=index==0?linear(x,qkv_name,0,name+".norm1"):linear_s8(next_q,next_scale,qkv_name,0,1.f);
    next_q={};next_scale={};
    auto att=linear(attention(qkv,qkv,0,512,1024),name+".self_attn.linear_out");
    auto memory=depth(qkv,name+".self_attn",512,1024);
    auto residual=tensor(x.r,512),q=tensor(x.r,512,0,1),scale=tensor(x.r,1,0,4);
    encoder_residual_qnorm(att,memory,x,residual,q,scale,name+".norm2",layer.at("in_size")==512?2:1);
    auto hidden=linear_s8(q,scale,name+".feed_forward.w_1",1,1.f);
    auto projected=linear(hidden,name+".feed_forward.w_2",0,"",0.0078125f);
    if(index+1<layers.size()) {
      const auto next_name=layers.at(index+1).at("name").get<std::string>();
      x=tensor(residual.r,512);
      next_q=tensor(residual.r,512,0,1);next_scale=tensor(residual.r,1,0,4);
      encoder_residual_qnorm(projected,residual,residual,x,next_q,next_scale,next_name+".norm1",0);
    } else {
      x=add(projected,residual);
    }
  }
  return norm(x,"encoder.after_norm");
 }
 Tensor decoder(Tensor x,const Tensor&memory,const std::string&branch,const std::vector<Tensor>*selected=nullptr){int i=0;for(auto&l:model[branch]){std::string p=l["name"];
   auto f=linear(x,p+".feed_forward.w_1",1,p+".norm1");f=linear(f,p+".feed_forward.w_2",0,p+".feed_forward.norm");
   if(l["self"]==true)f=add(depth(norm(f,p+".norm2"),p+".self_attn"),x);
   if(l["cross"]==true){auto q=linear(f,p+".src_attn.linear_q",0,p+".norm3");auto kv=branch=="seaco"?(selected?selected->at(i):seacokv.at(i)):linear(memory,p+".src_attn.linear_k_v");f=add(f,linear(attention(q,kv,0,0,512),p+".src_attn.linear_out"));}
   x=f;i++;
  }return norm(x,(branch=="seaco"?"seaco_decoder":"decoder")+std::string(".after_norm"));}
 std::pair<Tensor,std::vector<Tensor>> asf(Tensor x){
  Tensor scores;for(int i=0;i<6;i++){auto&l=model["seaco"][i];std::string p=l["name"];
   auto f=linear(x,p+".feed_forward.w_1",1,p+".norm1");f=linear(f,p+".feed_forward.w_2",0,p+".feed_forward.norm");
   f=add(depth(norm(f,p+".norm2"),p+".self_attn"),x);auto q=linear(f,p+".src_attn.linear_q",0,p+".norm3");auto kv=seacokv[i];
   if(i==5){scores=tensor(q.r*4,kv.r,0,4);run("score",{size_t((kv.r+15)/16)*16,size_t((q.r+15)/16),4},{16,1,1},q.mem(),kv.mem(),scores.mem(),q.r,kv.r,q.c,0,kv.c,0,dims,q.kind,kv.kind);
    run("softmax",{size_t(q.r)*64,4},{64,1},scores.mem(),q.r,kv.r,dims,q.kind,kv.kind);
   }else x=add(f,linear(attention(q,kv,0,0,512),p+".src_attn.linear_out"));
  }
  auto rank=tensor(hotcount,1,0,4),indices=tensor(51,1,0,4);
  run("importance",{size_t((hotcount+63)/64)*64},{64},scores.mem(),rank.mem(),x.r,hotcount,dims);
  run("select_hotwords",{1},{1},rank.mem(),indices.mem(),dims,hotcount);
  auto emb=tensor(51,512,2);run("gather",{512,51},{64,1},embedding.mem(),indices.mem(),emb.mem(),512);
  std::vector<Tensor>kv;for(auto&t:seacokv){if(!t.b){kv.push_back({});continue;}auto dst=tensor(51,t.c,2);run("gather",{size_t(t.c),51},{64,1},t.mem(),indices.mem(),dst.mem(),t.c);kv.push_back(dst);}
  return {emb,kv};
 }
 Tensor frontend(const Tensor&pcm,int R){int frames=R*6;auto power=tensor(frames,257,0,4),mel=tensor(frames,80,0,4),feat=tensor(R,560);
  run("fft",{size_t(frames)*256},{256},pcm.mem(),window,power.mem(),0,frames,dims);
  run("mel",{128,size_t(frames)},{64,1},power.mem(),weights,off("frontend.mel"),mel.mem(),frames,melranges,dims);
  run("lfr",{576,size_t(R)},{64,1},mel.mem(),weights,off("cmvn"),feat.mem(),frames,R,dims);return feat;}
 bool hotwords(const J&h,bool force=false){auto ids=h.at("ids");
  if(!ids.is_array()||ids.empty()||ids.size()>4096)throw std::runtime_error("hotword count");
  for(auto&word:ids){if(!word.is_array()||word.empty()||word.size()>10)throw std::runtime_error("hotword length");
   for(auto&id:word)if(!id.is_number_integer()||id.get<int>()<0||id.get<int>()>=model["vocab_size"].get<int>())throw std::runtime_error("hotword token ID");}
  if(ids.back()!=J::array({1}))throw std::runtime_error("missing SeACo dummy [1]");
  if(!force&&embedding.b&&hotword_key==ids){return true;}
  hotcount=ids.size();
  if(recording)throw std::runtime_error("Cannot update hotwords while recording inference");check(clFinish(queue));
  for(auto&[key,c]:cache)for(auto cb:c.commands)cbrelease(cb);cache.clear();recorded.clear();
  std::vector<int>input(hotcount*10),lengths(hotcount);for(int b=0;b<hotcount;b++){lengths[b]=ids[b].size();if(lengths[b]<1||lengths[b]>10)throw std::runtime_error("hotword length");for(int t=0;t<lengths[b];t++)input[b*10+t]=ids[b][t];}
  auto ii=tensor(hotcount,10,0,4),ll=tensor(hotcount,1,0,4);check(clEnqueueWriteBuffer(queue,ii.mem(),CL_TRUE,0,input.size()*4,input.data(),0,nullptr,nullptr));check(clEnqueueWriteBuffer(queue,ll.mem(),CL_TRUE,0,lengths.size()*4,lengths.data(),0,nullptr,nullptr));
  Tensor h0=tensor(hotcount,512,-1),c0=tensor(hotcount,512,-1,4),h1=tensor(hotcount,512,-1),c1=tensor(hotcount,512,-1,4);zero(h0);zero(c0);zero(h1);zero(c1);
  for(int t=0;t<10;t++)for(int layer=0;layer<2;layer++){auto x=tensor(hotcount,1024,-1);run("lstmin",{1024,size_t(hotcount)},{64,1},weights,off("decoder.embed.0.weight"),ii.mem(),ll.mem(),h0.mem(),layer?h1.mem():h0.mem(),x.mem(),hotcount,t,layer);
   auto gates=linear(x,"hotword.lstm"+std::to_string(layer));run("lstmstep",{512,size_t(hotcount)},{64,1},gates.mem(),layer?h1.mem():h0.mem(),layer?c1.mem():c0.mem(),ll.mem(),hotcount,t);}
  embedding=h1;embedding.kind=2;seacokv.clear();int ds[3]={0,0,hotcount};check(clEnqueueWriteBuffer(queue,dims,CL_TRUE,0,sizeof(ds),ds,0,nullptr,nullptr));
  for(auto&l:model["seaco"]){if(l["cross"]==true)seacokv.push_back(linear(embedding,l["name"].get<std::string>()+".src_attn.linear_k_v"));else seacokv.push_back({});}
  check(clFinish(queue));hotword_key=ids;return false;
 }
 std::vector<int> infer(const std::vector<char>& bytes,bool use_hw,double& ms) {
  return infer(bytes.data(), bytes.size(), use_hw, ms);
 }
 std::vector<int> infer(const void *data,size_t bytes,bool use_hw,double&ms){
  in_infer=true;expected_result=nullptr;result_reads=0;intermediate_reads=0;
  int samples=bytes/2;int R=(((samples-400)/160+1)+5)/6;
  if(bytes%2||samples<400||samples>480000)throw std::runtime_error("PCM must contain 400..480000 int16 samples");
  if(R<1||R>500)throw std::runtime_error("input must be 16k mono PCM <=30 seconds");
  int actualR=R;R=plan_capacity(R);result_capacity=R;
  int score_offset=R+2+(R*(max_token_bytes+2)+4)/4;int result_words=score_offset+R;
  std::string key=std::to_string(R)+":"+std::to_string(use_hw);
  auto found=cache.find(key);
  if(!capture&&found!=cache.end()){
   auto&c=found->second;expected_result=c.result.mem();int ds[6]={actualR,0,hotcount,samples,(samples-400)/160+1,0};auto start=Clock::now();
   check(clEnqueueWriteBuffer(queue,dims,CL_FALSE,0,sizeof(ds),ds,0,nullptr,nullptr));
   check(clEnqueueWriteBuffer(queue,c.input.mem(),CL_FALSE,0,bytes,data,0,nullptr,nullptr));
   for(auto cb:c.commands)check(cbenqueue(0,nullptr,cb,0,nullptr,nullptr));std::vector<int>res(result_words);
   check(clEnqueueReadBuffer(queue,c.result.mem(),CL_TRUE,0,res.size()*4,res.data(),0,nullptr,nullptr));
   ms=std::chrono::duration<double,std::milli>(Clock::now()-start).count();in_infer=false;return res;
  }
  auto input=tensor(R*960+400,1,0,2);auto result=tensor(result_words,1,0,4);
  expected_result=result.mem();
  auto start=Clock::now();int ds[6]={actualR,0,hotcount,samples,(samples-400)/160+1,0};
  if(!prepare_only){check(clEnqueueWriteBuffer(queue,dims,CL_FALSE,0,sizeof(ds),ds,0,nullptr,nullptr));check(clEnqueueWriteBuffer(queue,input.mem(),CL_FALSE,0,bytes,data,0,nullptr,nullptr));}
  if(!capture){cl_int e;recording=cbcreate(1,&queue,nullptr,&e);check(e);recorded.clear();recorded_count=0;}
  auto feats=frontend(input,R);auto enc=encoder(feats);
  auto unfolded=tensor(R,1536);run("unfold",{1536,size_t(R)},{64,1},enc.mem(),unfolded.mem(),R,dims);auto conv=tensor(R,512);
  run("gemmh",{512,size_t((R+15)/16)},{16,1},unfolded.mem(),weights,off("predictor.conv.h",2),weights,off("predictor.conv.bias"),conv.mem(),R,1536,1536,512,1,dims,0,1.f);
  auto alpha=tensor(R,1,0,4);
  run("alpha_project",{size_t(R)*64},{64},conv.mem(),weights,off("predictor.cif_output.h",2),off("predictor.cif_output.bias"),alpha.mem(),R);
  auto cif=tensor(R,512,1);
  run("cif",{512},{64},enc.mem(),alpha.mem(),cif.mem(),dims,R,model["predictor_tail"].get<float>());
  auto hidden=decoder(cif,enc,"decoder");auto logits=linear(hidden,"decoder.output_layer");Tensor hw=logits;
  if(use_hw){Tensor mem=embedding;std::vector<Tensor>selected;if(hotcount>50){auto filtered=asf(hidden);mem=filtered.first;selected=std::move(filtered.second);}
   auto stacked=tensor(R*2,512,5);
   run("stack_branches",{size_t(R)*1024},{256},cif.mem(),hidden.mem(),stacked.mem(),R,dims);
   auto decoded=decoder(stacked,mem,"seaco",selected.empty()?nullptr:&selected),sum=tensor(R,512,1);
   run("merge_branches",{size_t(R)*512},{256},decoded.mem(),sum.mem(),R,dims);
   hw=linear(sum,"hotword_output_layer");}
  run("choose",{size_t(R)*64},{64},logits.mem(),hw.mem(),result.mem(),R,model["vocab_size"].get<int>(),int(use_hw),model["no_bias"].get<int>(),dims,score_offset);
  run("text_result",{1},{1},result.mem(),tokenbytes,tokenoffsets,R);
  if(recording){check(cbfinal(recording));recorded.push_back(recording);cache[key]={recorded,input,result,R};if(!prepare_only)for(auto cb:recorded)check(cbenqueue(0,nullptr,cb,0,nullptr,nullptr));recording=nullptr;}
  if(prepare_only){in_infer=false;ms=0;return {};}std::vector<int>res(result_words);check(clEnqueueReadBuffer(queue,result.mem(),CL_TRUE,0,res.size()*4,res.data(),0,nullptr,nullptr));ms=std::chrono::duration<double,std::milli>(Clock::now()-start).count();in_infer=false;return res;
 }
 // Keep short inputs on tightly sized plans and share plans for longer inputs.
 static int plan_capacity(int frames) {
  const int step=frames<=64?16:frames<=256?32:64;
  return (frames+step-1)/step*step;
 }
 void prepare_plans(bool hw){
  if(prepare_only||capture)throw std::runtime_error("invalid plan preparation mode");
  prepare_only=true;
  for(int cap=16;cap<=512;cap=plan_capacity(cap+1)){double ms;infer(nullptr,size_t(std::min(cap*960,480000))*2,hw,ms);}
  prepare_only=false;
 }
};
