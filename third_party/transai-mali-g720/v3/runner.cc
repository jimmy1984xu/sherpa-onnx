#include "mali_loader.h"
#include "program_binary.h"

#include "dispatch/opencl_runtime.h"

#include <CL/cl.h>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>
inline bool MultiEnabled() { const char* v=getenv("TRANSAI_TDT_MULTI"); return v && std::strcmp(v,"0")!=0; }
using Clock = std::chrono::steady_clock;
using transai::teacher::dispatch::BytecodeReader;
using transai::teacher::dispatch::OpenClRuntime;
double Ms(Clock::time_point t) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}
template <class T> std::vector<T> Read(const std::string &p) {
  std::ifstream f(p, std::ios::binary | std::ios::ate);
  if (!f)
    throw std::runtime_error("missing " + p);
  size_t n = f.tellg();
  if (n % sizeof(T))
    throw std::runtime_error("size " + p);
  std::vector<T> v(n / sizeof(T));
  f.seekg(0);
  f.read((char *)v.data(), n);
  if (!f)
    throw std::runtime_error("read " + p);
  return v;
}
void Check(cl_int s, const char *p) {
  if (s)
    throw std::runtime_error(std::string(p) + " " + std::to_string(s));
}
struct Cl {
  void *lib = OpenMali();
#define E(n)                                                                   \
  decltype(&::n) n = reinterpret_cast<decltype(&::n)>(dlsym(lib, #n));
  E(clGetEventProfilingInfo)
  E(clEnqueueMarkerWithWaitList) E(clReleaseEvent) E(clGetPlatformIDs)
      E(clGetDeviceIDs) E(clGetDeviceInfo) E(clCreateContext)
          E(clCreateCommandQueue) E(clCreateBuffer) E(clEnqueueWriteBuffer)
              E(clEnqueueReadBuffer) E(clFinish) E(clCreateKernel)
                      E(clSetKernelArg) E(clEnqueueNDRangeKernel)
                          E(clGetKernelWorkGroupInfo) E(clReleaseMemObject)
                              E(clReleaseKernel) E(clReleaseProgram)
                                  E(clReleaseCommandQueue) E(clReleaseContext)
#undef E
                                      ~Cl() {
    dlclose(lib);
  }
};
struct Gpu {
  Cl cl;
  cl_context ctx{};
  cl_command_queue queue{};
  cl_device_id device{};
  cl_program program{};
  cl_kernel tdt{};
  cl_mem pcm_buffer{}, front_params{}, power_buffer{}, log_buffer{};
  std::vector<cl_kernel> front_kernels;
  cl_kernel text_kernel{};
  cl_mem state{};
  cl_kernel init{};
  std::vector<cl_kernel> phases;
  std::vector<std::unique_ptr<transai::opencl::CommandBuffer>> batches;
  cl_mem mel{}, length{}, enc{}, weights{}, packet{}, debug{};
  std::vector<uint8_t> plan_bytes;
  BytecodeReader plan;
  std::unique_ptr<OpenClRuntime> core;
  std::vector<int> result = std::vector<int>((45032 + 131072 + 3750*4) / 4);
  int block_steps =
      getenv("TRANSAI_TDT_BLOCK") ? atoi(getenv("TRANSAI_TDT_BLOCK")) : 8;
  int wg = 256;
  bool profiling = getenv("TRANSAI_TDT_PROFILE") != nullptr;
  cl_event enc_begin{}, enc_end{}, tdt_end{};
  double gpu_encoder_ms = 0, gpu_tdt_ms = 0, submit_ms = 0, wait_ms = 0;
  int valid_storage = 0;
  bool batch_created = false;
  Gpu(const std::string &base, const std::string &dir) {
    if (getenv("MALI_G720_SCORE_TRACE")) result.resize((191104+8193*4)/4);
    auto cleanup = [](Gpu *p) { p->Cleanup(); };
    std::unique_ptr<Gpu, decltype(cleanup)> rollback(this, cleanup);
    cl_uint np = 0;
    Check(cl.clGetPlatformIDs(0, nullptr, &np), "platforms");
    std::vector<cl_platform_id> platforms(np);
    Check(cl.clGetPlatformIDs(np, platforms.data(), nullptr), "platforms");
    for (auto p : platforms)
      if (!cl.clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 1, &device, nullptr))
        break;
    if (!device)
      throw std::runtime_error("no gpu");
    auto info = [&](cl_device_info k) {
      size_t n = 0;
      Check(cl.clGetDeviceInfo(device, k, 0, nullptr, &n), "info");
      std::string s(n, '\0');
      Check(cl.clGetDeviceInfo(device, k, n, s.data(), nullptr), "info");
      return s;
    };
    auto name = info(CL_DEVICE_NAME), ext = info(CL_DEVICE_EXTENSIONS);
    std::cerr << "DEVICE " << name << " DRIVER " << info(CL_DRIVER_VERSION)
              << " EXT " << ext << std::endl;
    if (name.find("G720") == std::string::npos ||
        ext.find("cl_arm_integer_dot_product_int8") == std::string::npos ||
        ext.find("cl_arm_matrix_multiply") == std::string::npos)
      throw std::runtime_error("unsupported gpu");
    cl_int status = 0;
    ctx = cl.clCreateContext(nullptr, 1, &device, nullptr, nullptr, &status);
    Check(status, "context");
    queue = cl.clCreateCommandQueue(
        ctx, device, profiling ? CL_QUEUE_PROFILING_ENABLE : 0, &status);
    Check(status, "queue");
    auto alloc = [&](size_t n) {
      auto b = cl.clCreateBuffer(ctx, CL_MEM_READ_WRITE, n, nullptr, &status);
      Check(status, "buffer");
      return b;
    };
    state = alloc(131072);
    mel = alloc(3000 * 128 * 2);
    length = alloc(16);
    pcm_buffer = alloc(480000 * 2);
    power_buffer = alloc(3000 * 257 * 4);
    log_buffer = alloc(3000 * 128 * 4);
    auto front = Read<uint8_t>(dir + "/frontend.bin");
    if (front.size() != 212779)
      throw std::runtime_error("frontend parameters mismatch");
    front_params =
        cl.clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY,
                          front.size(), nullptr, &status);
    Check(status, "front params");
    Check(cl.clEnqueueWriteBuffer(queue, front_params, CL_TRUE, 0, front.size(),
                                  front.data(), 0, nullptr, nullptr),
          "front params upload");
    enc = alloc(375 * 640 * 2);
    packet = alloc(result.size() * 4);
    debug = alloc((5 * 640 + 8198) * 4);
    auto w = Read<uint8_t>(dir + "/tdt_weights.bin");
    if (w.size() != 94667424)
      throw std::runtime_error("TDT weight size mismatch");
    weights = cl.clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY,
                                w.size(), nullptr, &status);
    Check(status, "readonly weights");
    Check(cl.clEnqueueWriteBuffer(queue, weights, CL_TRUE, 0, w.size(),
                                  w.data(), 0, nullptr, nullptr),
          "weights");
    if (getenv("TRANSAI_TDT_WG"))
      wg = atoi(getenv("TRANSAI_TDT_WG"));
    if (wg < 64 || wg > 1024 || (wg & (wg - 1)))
      throw std::runtime_error("invalid workgroup");
    std::string options = "-cl-std=CL2.0 -DWG=" + std::to_string(wg) +
                          (getenv("TRANSAI_TDT_NATIVE") ? " -DNATIVE_MATH=1"
                                                        : " -DNATIVE_MATH=0");
    options += getenv("MALI_G720_SCORE_TRACE") ? " -DMALI_SCORE_TRACE=1" : " -DMALI_SCORE_TRACE=0";
    program = transai::aot::Load(cl.lib, ctx, queue, dir, options);
    tdt = cl.clCreateKernel(
        program, MultiEnabled() ? "tdt_tail" : "tdt_persistent",
        &status);
    Check(status, "kernel");
    cl_ulong local = 0;
    size_t max = 0;
    cl.clGetKernelWorkGroupInfo(tdt, device, CL_KERNEL_LOCAL_MEM_SIZE,
                                sizeof(local), &local, nullptr);
    cl.clGetKernelWorkGroupInfo(tdt, device, CL_KERNEL_WORK_GROUP_SIZE,
                                sizeof(max), &max, nullptr);
    std::cerr << "TDT_LOCAL " << local << " MAX_WG " << max << " WG " << wg
              << std::endl;

    for (auto name : {"front_trim", "front_fft", "front_mel", "front_norm"}) {
      auto k = cl.clCreateKernel(program, name, &status);
      Check(status, "frontend kernel");
      cl_mem args[] = {pcm_buffer,   front_params, length,
                       power_buffer, log_buffer,   mel};
      Check(cl.clSetKernelArg(k, 0, sizeof(cl_mem), &args[0]), "front pcm");
      for (int i = 1; i < 6; ++i)
        Check(cl.clSetKernelArg(k, i + 1, sizeof(cl_mem), &args[i]),
              "front args");
      front_kernels.push_back(k);
    }
    text_kernel = cl.clCreateKernel(program, "final_text", &status);
    Check(status, "text kernel");
    cl_mem text_args[] = {packet, front_params, length};
    for (int i = 0; i < 3; ++i)
      Check(cl.clSetKernelArg(text_kernel, i, sizeof(cl_mem), &text_args[i]),
            "text args");
    int gp = 1;
    Check(cl.clSetKernelArg(text_kernel, 3, 4, &gp), "text gpu flag");
    if (MultiEnabled()) {
      init = cl.clCreateKernel(program, "tdt_init", &status);
      Check(status, "init");
      auto bind = [&](cl_kernel k) {
        cl_mem args[] = {state, enc, weights, packet};
        for (int i = 0; i < 4; ++i)
          Check(cl.clSetKernelArg(k, i, sizeof(cl_mem), &args[i]),
                "phase args");
      };
      bind(init);
      Check(cl.clSetKernelArg(init, 5, sizeof(cl_mem), &length), "init meta");
      std::vector<const char *> phase_names =
          getenv("TRANSAI_TDT_FUSED")
              ? std::vector<const char *>{"tdt_rec", "tdt_fused_step"}
              : std::vector<const char *>{"tdt_rec", "tdt_l0",    "tdt_x1",
                                          "tdt_l1",  "tdt_pred",  "tdt_joint",
                                          "tdt_out", "tdt_select"};
      for (auto name : phase_names) {
        auto k = cl.clCreateKernel(program, name, &status);
        Check(status, "phase");
        bind(k);
        phases.push_back(k);
      }
      if (block_steps < 1 || block_steps > 256 || 256 % block_steps)
        throw std::runtime_error("invalid block");
      for (int b = 0; b < 256 / block_steps; ++b) {
        auto batch = std::make_unique<transai::opencl::CommandBuffer>(queue);
        const size_t global[] = {5120, (size_t)wg, 2560, (size_t)wg,
                                 640,  (size_t)wg, 8256, (size_t)wg};
        for (int step = 0; step < block_steps; ++step)
          for (int j = 0; j < phases.size(); ++j) {
            transai::opencl::CommandDispatch d;
            d.kernel = phases[j];
            d.global[0] = global[j];
            d.local[0] =
                (j == 0 || (phases.size() > 2 && (j == 2 || j == 4 || j == 6)))
                    ? 64
                    : wg;
            batch->Add(d);
          }
        batch->Finalize();
        batches.push_back(std::move(batch));
      }
    }
    plan_bytes = Read<uint8_t>(base + "/encoder.full.nemo.plan.taig720");
    std::string error;
    if (!BytecodeReader::Parse(plan_bytes.data(), plan_bytes.size(), &plan,
                               &error))
      throw std::runtime_error(error);
    core = OpenClRuntime::Create(
        ctx, queue, plan, base + "/encoder.weights.bin", false, true, &error);
    if (!core)
      throw std::runtime_error(error);
    core->SetDeferredWait(true);
    std::cerr << "GPU_CONTRACT cpu_frontend=0 encoder_d2h=0 cpu_encoder=0 "
                 "cpu_decoder=0 cpu_token_to_text=0 initial_pcm_uploads=1 "
                 "final_packet_reads=1"
              << std::endl;
    rollback.release();
  }
  void Decode(int frames, int trace) {
    if (frames < 1 || frames > 375)
      throw std::runtime_error("frames outside1..375");
    auto begin = Clock::now();
    if (profiling && !enc_end) {
      Check(cl.clEnqueueMarkerWithWaitList(queue, 0, nullptr, &enc_end),
            "encoder marker");
    }
    batch_created = false;
    int resume = init ? 1 : 0;
    if (init) {
      int init_frames = -1;
      Check(cl.clSetKernelArg(init, 4, 4, &init_frames), "init frames");
      const std::vector<int> shape_plans{
          24,  32,  40,  48,  64,  80,  96,  112, 128, 144, 160, 176, 192,
          208, 224, 240, 256, 272, 288, 304, 320, 336, 352, 368, 375};
      int capacity =
          *std::lower_bound(shape_plans.begin(), shape_plans.end(), frames);
      int pct = getenv("TRANSAI_TDT_BUDGET")
                    ? atoi(getenv("TRANSAI_TDT_BUDGET"))
                    : 70;
      int steps = std::min(256, ((capacity * pct / 100 + 23) / 8) * 8);
      size_t g = wg;
      Check(cl.clEnqueueNDRangeKernel(queue, init, 1, nullptr, &g, &g, 0,
                                      nullptr, nullptr),
            "init enqueue");
      for (int i = 0; i < (steps + block_steps - 1) / block_steps; ++i) {
        auto e = batches[i]->Enqueue();
        if (e)
          cl.clReleaseEvent(e);
      }
    }
    Check(cl.clSetKernelArg(tdt, 6, sizeof(cl_mem), &state), "state arg");
    Check(cl.clSetKernelArg(tdt, 7, 4, &resume), "resume arg");
    Check(cl.clSetKernelArg(tdt, 8, sizeof(cl_mem), &length), "meta");
    int gp = 1;
    Check(cl.clSetKernelArg(tdt, 9, 4, &gp), "gpu front");
    if (frames < 1 || frames > 375)
      throw std::runtime_error("frames");
    cl_mem args[] = {enc, weights, packet, debug};
    Check(cl.clSetKernelArg(tdt, 0, sizeof(cl_mem), &args[0]), "arg0");
    Check(cl.clSetKernelArg(tdt, 1, 4, &frames), "frames");
    for (int i = 1; i < 4; ++i)
      Check(cl.clSetKernelArg(tdt, i + 1, sizeof(cl_mem), &args[i]), "arg");
    Check(cl.clSetKernelArg(tdt, 5, 4, &trace), "trace");
    size_t n = wg;
    Check(cl.clEnqueueNDRangeKernel(queue, tdt, 1, nullptr, &n, &n, 0, nullptr,
                                    nullptr),
          "tdt enqueue");
    size_t one = 1;
    Check(cl.clEnqueueNDRangeKernel(queue, text_kernel, 1, nullptr, &one, &one,
                                    0, nullptr, nullptr),
          "final text");
    if (profiling)
      Check(cl.clEnqueueMarkerWithWaitList(queue, 0, nullptr, &tdt_end),
            "tdt end");
    submit_ms = Ms(begin);
    auto wait_begin = Clock::now();
    Check(cl.clEnqueueReadBuffer(queue, packet, CL_TRUE, 0, result.size() * 4,
                                 result.data(), 0, nullptr, nullptr),
          "final read");
    wait_ms = Ms(wait_begin);
    if (profiling) {
      auto end = [&](cl_event e) {
        cl_ulong v = 0;
        Check(cl.clGetEventProfilingInfo(e, CL_PROFILING_COMMAND_END, sizeof(v),
                                         &v, nullptr),
              "event time");
        return v;
      };
      gpu_tdt_ms = (end(tdt_end) - end(enc_end)) / 1e6;
      if (enc_begin)
        gpu_encoder_ms = (end(enc_end) - end(enc_begin)) / 1e6;
      for (auto e : {enc_begin, enc_end, tdt_end})
        if (e)
          cl.clReleaseEvent(e);
      enc_begin = enc_end = tdt_end = nullptr;
    }
    if (result[1] < 0 || result[1] > 3750)
      throw std::runtime_error("token capacity");
    if (result[0])
      throw std::runtime_error("GPU TDT status " + std::to_string(result[0]));
  }

  void EncodePcm(const std::vector<int16_t> &pcm) {
    if (pcm.empty() || pcm.size() > 480000)
      throw std::runtime_error("PCM outside1..480000");
    int count = pcm.size();
    Check(cl.clEnqueueWriteBuffer(queue, pcm_buffer, CL_FALSE, 0,
                                  pcm.size() * 2, pcm.data(), 0, nullptr,
                                  nullptr),
          "initial PCM upload");
    int valid = std::max(1, count / 160);
    size_t global[] = {256, (size_t)valid * 256, (size_t)valid * 128,
                       128 * 128},
           local[] = {256, 256, 128, 128};
    for (int i = 0; i < 4; ++i) {
      Check(cl.clSetKernelArg(front_kernels[i], 1, 4, &count), "sample count");
      Check(cl.clEnqueueNDRangeKernel(queue, front_kernels[i], 1, nullptr,
                                      &global[i], &local[i], 0, nullptr,
                                      nullptr),
            "front enqueue");
    }
    if (profiling)
      Check(cl.clEnqueueMarkerWithWaitList(queue, 0, nullptr, &enc_begin),
            "enc begin");
    const transai::teacher::dispatch::PlanRecord *selected = nullptr;
    for (auto &p : plan.plans())
      if (p.encoder_frames >= (valid + 7) / 8) {
        selected = &p;
        break;
      }
    if (!selected)
      throw std::runtime_error("PCM shape");
    std::string error;
    if (!core->Invoke(mel, length, enc, valid, *selected, &error))
      throw std::runtime_error(error);
    if (profiling)
      Check(cl.clEnqueueMarkerWithWaitList(queue, 0, nullptr, &enc_end),
            "enc end");
  }
  void Cleanup() {
    if (queue)
      cl.clFinish(queue);
    core.reset();
    batches.clear();
    for (auto k : phases)
      cl.clReleaseKernel(k);
    if (init)
      cl.clReleaseKernel(init);
    for (auto k : front_kernels)
      cl.clReleaseKernel(k);
    if (text_kernel)
      cl.clReleaseKernel(text_kernel);
    for (auto b : {pcm_buffer, front_params, power_buffer, log_buffer})
      if (b)
        cl.clReleaseMemObject(b);
    if (state)
      cl.clReleaseMemObject(state);
    for (auto b : {mel, length, enc, weights, packet, debug})
      if (b)
        cl.clReleaseMemObject(b);
    if (tdt)
      cl.clReleaseKernel(tdt);
    if (program)
      cl.clReleaseProgram(program);
    if (queue)
      cl.clReleaseCommandQueue(queue);
    if (ctx)
      cl.clReleaseContext(ctx);
  }
  ~Gpu() { Cleanup(); }
};
