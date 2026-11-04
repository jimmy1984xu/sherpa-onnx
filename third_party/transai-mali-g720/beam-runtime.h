#pragma once
// Precompiled GPU beam program is linked into the architecture library.
extern const unsigned char transai_beam_binary[];
extern const size_t transai_beam_binary_bytes;
extern const char transai_beam_target[];
struct BeamGpu {
  Gpu &gpu;
  cl_program program{};
  cl_kernel kernel{},initialize{},step{},expand{},select{},logits{},cell0{},cell1{},joint{},rec{},x1{},pred{};
  std::vector<std::unique_ptr<transai::opencl::CommandBuffer>> batches;
  std::vector<cl_mem> buffers;
  int beam_size=0;
  std::vector<int32_t> graph_cache;
  explicit BeamGpu(Gpu &g) : gpu(g) {
    using namespace transai::aot;
    Require(Json::parse(transai_beam_target)==Target(g.cl.lib,g.device),"GPU beam target mismatch");
    auto create=Entry<decltype(&clCreateProgramWithBinary)>(g.cl.lib,"clCreateProgramWithBinary");
    auto build=Entry<decltype(&clBuildProgram)>(g.cl.lib,"clBuildProgram");
    auto info=Entry<decltype(&clGetProgramBuildInfo)>(g.cl.lib,"clGetProgramBuildInfo");
    const unsigned char *data=transai_beam_binary;
    cl_int status=0,binary_status=0;
    program=create(g.ctx,1,&g.device,&transai_beam_binary_bytes,&data,&binary_status,&status);
    try {
      Check(status,"beam executable");Check(binary_status,"beam binary");
      Check(build(program,1,&g.device,nullptr,nullptr,nullptr),"beam finalize");
      cl_program_binary_type type=0;
      Check(info(program,g.device,CL_PROGRAM_BINARY_TYPE,sizeof(type),&type,nullptr),"beam binary type");
      Require(type==CL_PROGRAM_BINARY_TYPE_EXECUTABLE,"GPU beam executable required");
      kernel=g.cl.clCreateKernel(program,"beam_fast",&status);Check(status,"GPU beam kernel");
      initialize=g.cl.clCreateKernel(program,"beam_init",&status);Check(status,"GPU beam init");
      step=g.cl.clCreateKernel(program,"beam_prepare",&status);Check(status,"GPU parallel model step");
      expand=g.cl.clCreateKernel(program,"beam_expand",&status);Check(status,"GPU per-path expansion");
      select=g.cl.clCreateKernel(program,"beam_select",&status);Check(status,"GPU beam selection");
      logits=g.cl.clCreateKernel(program,"beam_affine",&status);Check(status,"GPU MMUL joiner");
      cell0=g.cl.clCreateKernel(program,"beam_cell",&status);Check(status,"GPU cell0");
      cell1=g.cl.clCreateKernel(program,"beam_cell",&status);Check(status,"GPU cell1");
      joint=g.cl.clCreateKernel(program,"beam_joint",&status);Check(status,"GPU joint");
      rec=g.cl.clCreateKernel(program,"beam_affine",&status);Check(status,"GPU recurrent MMUL");
      x1=g.cl.clCreateKernel(program,"beam_affine",&status);Check(status,"GPU input MMUL");
      pred=g.cl.clCreateKernel(program,"beam_affine",&status);Check(status,"GPU predictor MMUL");
    } catch (...) { for(auto k:{kernel,initialize,step,expand,select,logits,cell0,cell1,joint,rec,x1,pred})if(k)g.cl.clReleaseKernel(k);if(program)g.cl.clReleaseProgram(program);throw; }
    std::cerr<<"AOT_LOAD role=beam embedded=1 source_compiles=0\n";
  }
  void ReleaseBuffers() {
    batches.clear();
    for(auto b:buffers)gpu.cl.clReleaseMemObject(b);
    buffers.clear();beam_size=0;graph_cache.clear();
  }
  ~BeamGpu() { ReleaseBuffers();for(auto k:{kernel,initialize,step,expand,select,logits,cell0,cell1,joint,rec,x1,pred})if(k)gpu.cl.clReleaseKernel(k);if(program)gpu.cl.clReleaseProgram(program); }
  void Prepare(int beam,const int32_t *graph,size_t words) {
    if(beam<1||beam>64||!graph||words<4+MALI_G720_VOCAB_SIZE||words>16*1024*1024 ||
       graph[0]<1||graph[1]<0||graph[1]>MALI_G720_VOCAB_SIZE||graph[2]!=MALI_G720_VOCAB_SIZE)
      throw std::runtime_error("invalid GPU beam graph/config");
    size_t table=size_t(graph[0])*(graph[1]+1);
    if(words!=4+MALI_G720_VOCAB_SIZE+2*table+graph[0])throw std::runtime_error("GPU beam graph size");
    if(beam_size!=beam || graph_cache.size()!=words) {
      ReleaseBuffers();
      size_t sizes[]={size_t(3*beam)*5*640*4,size_t(beam)*48,size_t(beam)*(beam+1)*48,
                      size_t(beam)*4128*20,size_t(beam)*(2*MALI_G720_VOCAB_SIZE+5)*4,words*4,32,size_t(3*beam)*640,size_t(3*beam)*4,size_t(3*beam)*4,size_t(2*beam)*2560*4};
      for(size_t n:sizes) {
        cl_int status;auto b=gpu.cl.clCreateBuffer(gpu.ctx,CL_MEM_READ_WRITE,n,nullptr,&status);
        Check(status,"GPU beam buffer");buffers.push_back(b);
      }
      beam_size=beam;
    }
    if(graph_cache.size()!=words||!std::equal(graph_cache.begin(),graph_cache.end(),graph)) {
      Check(gpu.cl.clEnqueueWriteBuffer(gpu.queue,buffers[5],CL_TRUE,0,words*4,graph,0,nullptr,nullptr),"GPU hotword graph upload");
      graph_cache.assign(graph,graph+words);
    }
  }
  void Decode(int beam,float blank_penalty,int frame_bound) {
    auto bind=[&](cl_kernel k,std::initializer_list<cl_mem> args) {
      int i=0;for(auto value:args)Check(gpu.cl.clSetKernelArg(k,i++,sizeof(cl_mem),&value),"GPU beam binding");
    };
    int trace=getenv("MALI_G720_SCORE_TRACE")?1:0;
    bind(initialize,{gpu.packet,gpu.length,buffers[0],buffers[1],buffers[6],gpu.enc});
    Check(gpu.cl.clSetKernelArg(initialize,6,4,&beam),"GPU beam init paths");
    bind(step,{buffers[0],buffers[1],buffers[6],buffers[7],buffers[8],buffers[9]});
    Check(gpu.cl.clSetKernelArg(step,6,4,&beam),"GPU prepare paths");
    for(int bank=0;bank<2;++bank){auto k=bank?cell1:cell0;
      bind(k,{buffers[0],buffers[1],buffers[6],buffers[10],buffers[7],buffers[8],buffers[9]});
      Check(gpu.cl.clSetKernelArg(k,7,4,&beam),"GPU cell paths");
      Check(gpu.cl.clSetKernelArg(k,8,4,&bank),"GPU cell bank");}
    bind(joint,{gpu.enc,buffers[0],buffers[1],buffers[6],buffers[7],buffers[8],buffers[9]});
    Check(gpu.cl.clSetKernelArg(joint,7,4,&beam),"GPU joint paths");
    cl_kernel affine[]={rec,x1,pred,logits};
    for(int op=0;op<4;++op){auto k=affine[op];
      bind(k,{gpu.weights,buffers[7],buffers[8],buffers[9],buffers[1],buffers[6],buffers[0],buffers[10],buffers[4],gpu.packet});
      Check(gpu.cl.clSetKernelArg(k,10,4,&beam),"GPU MMUL paths");
      Check(gpu.cl.clSetKernelArg(k,11,4,&op),"GPU MMUL operation");
      Check(gpu.cl.clSetKernelArg(k,12,4,&trace),"GPU MMUL trace");}
    bind(expand,{buffers[1],buffers[2],buffers[4],buffers[5],buffers[6]});
    Check(gpu.cl.clSetKernelArg(expand,5,4,&beam),"GPU expansion paths");
    Check(gpu.cl.clSetKernelArg(expand,6,4,&blank_penalty),"GPU expansion penalty");
    bind(select,{gpu.packet,gpu.length,buffers[0],buffers[1],buffers[2],buffers[3],buffers[6]});
    Check(gpu.cl.clSetKernelArg(select,7,4,&beam),"GPU selection paths");
    bind(kernel,{gpu.enc,gpu.weights,gpu.packet,gpu.length,buffers[0],buffers[1],buffers[2],buffers[3],buffers[4],buffers[5],buffers[6],buffers[10]});
    Check(gpu.cl.clSetKernelArg(kernel,12,4,&beam),"GPU tail paths");
    Check(gpu.cl.clSetKernelArg(kernel,13,4,&blank_penalty),"GPU tail penalty");
    Check(gpu.cl.clSetKernelArg(kernel,14,4,&trace),"GPU tail trace");
    if(batches.empty()) {
      // Pair original eight-round blocks, with a separate eight-round remainder.
      for(int b=0;b<14;++b) {
        auto batch=std::make_unique<transai::opencl::CommandBuffer>(gpu.queue);
        auto group=[&](cl_kernel k){transai::opencl::CommandDispatch d;d.kernel=k;d.global[0]=128*beam;d.local[0]=128;batch->Add(d);};
        auto mmul=[&](cl_kernel k,int columns,int pairs){transai::opencl::CommandDispatch d;d.kernel=k;d.dimensions=3;d.global[0]=((columns+63)/64)*64;d.global[1]=(beam+3)/4;d.global[2]=pairs;d.local[0]=64;d.local[1]=d.local[2]=1;batch->Add(d);};
        for(int i=0;i<(b==13?8:16);++i){
          group(step);mmul(rec,2560,2);group(cell0);mmul(x1,2560,1);group(cell1);
          mmul(pred,640,1);group(joint);mmul(logits,MALI_G720_VOCAB_SIZE+5,1);
          transai::opencl::CommandDispatch d;d.kernel=expand;d.global[0]=512*beam;d.local[0]=512;batch->Add(d);
          d.kernel=select;d.global[0]=128;d.local[0]=128;batch->Add(d);
        }
        batch->Finalize();batches.push_back(std::move(batch));
      }
    }
    size_t global=256,init_group=128;
    Check(gpu.cl.clEnqueueNDRangeKernel(gpu.queue,initialize,1,nullptr,&init_group,&init_group,0,nullptr,nullptr),"GPU beam initialize");
    int blocks=std::clamp((frame_bound/2+16+7)/8,1,26);
    for(int i=0;i<blocks/2;++i)batches[i]->Enqueue(false);
    if(blocks&1)batches[13]->Enqueue(false);
    Check(gpu.cl.clEnqueueNDRangeKernel(gpu.queue,kernel,1,nullptr,&global,&global,0,nullptr,nullptr),"GPU beam tail");
    size_t one=1;
    Check(gpu.cl.clEnqueueNDRangeKernel(gpu.queue,gpu.text_kernel,1,nullptr,&one,&one,0,nullptr,nullptr),"GPU beam text");
    if(gpu.profiling)Check(gpu.cl.clEnqueueMarkerWithWaitList(gpu.queue,0,nullptr,&gpu.tdt_end),"beam end marker");
    Check(gpu.cl.clEnqueueReadBuffer(gpu.queue,gpu.packet,CL_TRUE,0,gpu.result.size()*4,gpu.result.data(),0,nullptr,nullptr),"GPU beam final packet");
    if(gpu.profiling)gpu.FinishProfile();
    if(gpu.result[0]||gpu.result[1]<0||gpu.result[1]>3750)throw std::runtime_error("GPU beam status "+std::to_string(gpu.result[0]));
    std::cerr<<"GPU_BEAM cpu_search=0 cpu_context_graph=0 cpu_model=0 encoder_d2h=0 final_packet_reads=1 paths="<<beam
             <<" max_active="<<(uint32_t(gpu.result[3])>>24)<<" evaluations="<<(gpu.result[3]&0xffffff)
             <<" rounds="<<gpu.result[2]<<" all_tdt_affine_mmul=1 parallel_groups="<<beam<<" command_buffer_blocks="<<blocks<<"\n";
  }
};
