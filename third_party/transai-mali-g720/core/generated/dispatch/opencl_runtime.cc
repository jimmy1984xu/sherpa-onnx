// Copyright 2026 TransAI
// SPDX-License-Identifier: Apache-2.0

#include "dispatch/opencl_runtime.h"

#include <dlfcn.h>
#include <iostream>
#include "mali_loader.h"
#include <cstdlib>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <fstream>
#include <limits>
#include <mutex>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace transai::teacher::dispatch {
namespace {



constexpr size_t kHalfBytes = 2;
constexpr size_t kMaxFrames = 375;
constexpr size_t kModelChannels = 1024;

constexpr uint32_t kFp16Raw = 1;
constexpr uint32_t kFp16MmulRhs = 3;
constexpr uint32_t kFp16Vector = 5;
constexpr uint32_t kS8MmulRhs = 8;
constexpr uint64_t WeightKey(uint32_t semantic, int32_t layer,
                             uint32_t storage) {
  return (static_cast<uint64_t>(semantic) << 40) |
         (static_cast<uint64_t>(static_cast<uint32_t>(layer + 1)) << 8) |
         storage;
}

size_t RoundUp(size_t value, size_t alignment) {
  return (value + alignment - 1) / alignment * alignment;
}

std::string ReadTextFile(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) return {};
  std::ostringstream output;
  output << input.rdbuf();
  return output.str();
}

template <typename Value>
bool SetArg(decltype(&clSetKernelArg) set_arg, cl_kernel kernel, cl_uint index,
            const Value& value) {
  return set_arg(kernel, index, sizeof(Value), &value) == CL_SUCCESS;
}

}  // namespace

struct OpenClRuntime::Api {
  void* library = nullptr;
  decltype(&clCreateProgramWithSource) create_program = nullptr;
  decltype(&clBuildProgram) build_program = nullptr;
  decltype(&clGetProgramInfo) get_program_info = nullptr;
  decltype(&clGetProgramBuildInfo) get_build_info = nullptr;
  decltype(&clCreateKernel) create_kernel = nullptr;
  decltype(&clCreateBuffer) create_buffer = nullptr;
  decltype(&clEnqueueWriteBuffer) enqueue_write = nullptr;
  decltype(&clEnqueueReadBuffer) enqueue_read = nullptr;
  decltype(&clSetKernelArg) set_arg = nullptr;
  decltype(&clSetKernelExecInfo) set_kernel_exec_info = nullptr;
  decltype(&clEnqueueNDRangeKernel) enqueue_kernel = nullptr;
  decltype(&clWaitForEvents) wait_events = nullptr;
  decltype(&clGetEventProfilingInfo) get_event_profiling_info = nullptr;
  decltype(&clReleaseEvent) release_event = nullptr;
  decltype(&clReleaseMemObject) release_mem = nullptr;
  decltype(&clReleaseKernel) release_kernel = nullptr;
  decltype(&clReleaseProgram) release_program = nullptr;

  Api() {
    library = OpenMali();
    if (library == nullptr) return;
#define LOAD(name, field) \
  field = reinterpret_cast<decltype(field)>(dlsym(library, name))
    LOAD("clCreateProgramWithSource", create_program);
    LOAD("clBuildProgram", build_program);
    LOAD("clGetProgramInfo", get_program_info);
    LOAD("clGetProgramBuildInfo", get_build_info);
    LOAD("clCreateKernel", create_kernel);
    LOAD("clCreateBuffer", create_buffer);
    LOAD("clEnqueueWriteBuffer", enqueue_write);
    LOAD("clEnqueueReadBuffer", enqueue_read);
    LOAD("clSetKernelArg", set_arg);
    LOAD("clSetKernelExecInfo", set_kernel_exec_info);
    LOAD("clEnqueueNDRangeKernel", enqueue_kernel);
    LOAD("clWaitForEvents", wait_events);
    LOAD("clGetEventProfilingInfo", get_event_profiling_info);
    LOAD("clReleaseEvent", release_event);
    LOAD("clReleaseMemObject", release_mem);
    LOAD("clReleaseKernel", release_kernel);
    LOAD("clReleaseProgram", release_program);
#undef LOAD
  }

  ~Api() {
    if (library != nullptr) dlclose(library);
  }

  bool valid() const {
    return create_program != nullptr && build_program != nullptr &&
           get_program_info != nullptr &&
           get_build_info != nullptr && create_kernel != nullptr &&
           create_buffer != nullptr && enqueue_write != nullptr &&
            enqueue_read != nullptr &&
           set_arg != nullptr && set_kernel_exec_info != nullptr &&
           enqueue_kernel != nullptr &&
           wait_events != nullptr && get_event_profiling_info != nullptr &&
           release_event != nullptr && release_mem != nullptr &&
           release_kernel != nullptr &&
           release_program != nullptr;
  }
};

std::unique_ptr<OpenClRuntime> OpenClRuntime::Create(
    cl_context context, cl_command_queue queue, const BytecodeReader& bytecode,
    const std::string& weights_path, bool diagnostic, bool nemo_padding, std::string* error) {
  if (context == nullptr || queue == nullptr || error == nullptr) return nullptr;
  auto runtime = std::unique_ptr<OpenClRuntime>(new OpenClRuntime());
  runtime->api_ = std::make_unique<Api>();
  runtime->context_ = context;
  runtime->queue_ = queue;
  if (!runtime->api_->valid()) {
    *error = "required OpenCL entry point unavailable";
    return nullptr;
  }
  runtime->diagnostic_ = diagnostic;
  if (const char* value = std::getenv("TRANSAI_G720_FFN_DIRECT_LIMIT")) {
    const std::string limit(value);
    if (limit != "32" && limit != "40" && limit != "65") { *error="unsupported immutable FFN dispatch policy";return nullptr; }
    runtime->ffn_direct_limit_ = std::stoi(limit);
  }


  auto choice = [&](const char* name, int fallback, std::initializer_list<int> allowed) {
    const char* value=std::getenv(name); if(!value)return fallback;
    int parsed=std::stoi(value);
    if(std::find(allowed.begin(),allowed.end(),parsed)==allowed.end())throw std::runtime_error(std::string("unsupported tuning: ")+name);
    return parsed;
  };
  runtime->dense_lx_=choice("TRANSAI_G720_DENSE_LX",32,{16,32,64,128});
  runtime->dense_ly_=choice("TRANSAI_G720_DENSE_LY",2,{1,2,4});
  runtime->broadcast_y_=choice("TRANSAI_G720_BROADCAST_Y",4,{0,1,2,3,4,5,6,7,8,16});
  runtime->small_m_=choice("TRANSAI_G720_SMALL_M",4,{2,4,8});
  runtime->small_n_=choice("TRANSAI_G720_SMALL_N",4,{2,4,8});
  runtime->dense_tile_max_m_=choice("TRANSAI_G720_DENSE_TILE_MAX_M",32,{32,64,128,192,375});
  runtime->bcast_k_=choice("TRANSAI_G720_BCAST_K",64,{32,64,128,256,512});
  runtime->projection_direct_limit_=choice("TRANSAI_G720_PROJECTION_DIRECT_LIMIT",40,{32,40});
  runtime->wbsm_=choice("TRANSAI_G720_WBSM",-1,{-8,-4,-1,0,1,4,8,16});
  runtime->wbsm_dense_=std::getenv("TRANSAI_G720_WBSM")!=nullptr;
  runtime->small_tile_=std::getenv("TRANSAI_G720_SMALL_M")!=nullptr;
  runtime->ffn_broadcast_tile_=std::getenv("TRANSAI_G720_FFN_BROADCAST_TILE")!=nullptr;
  runtime->projection_tile_=std::getenv("TRANSAI_G720_PROJECTION_TILE")!=nullptr;
  runtime->projection_y_=choice("TRANSAI_G720_PROJECTION_Y",4,{0,1,2,3,4,5,6,7,8,16});
  std::cerr<<"G720_TUNING dense="<<runtime->dense_lx_<<"x"<<runtime->dense_ly_<<" broadcast_y="<<runtime->broadcast_y_<<std::endl;
  for (const WeightRecord& weight : bytecode.weights()) {
    if ((weight.offset % kHalfBytes) != 0) {
      *error = "TAIENC6 weight offset is not FP16 aligned";
      return nullptr;
    }
    if (weight.semantic != 0) {
      runtime->weight_offsets_.emplace(
          WeightKey(weight.semantic, weight.layer, weight.storage),
          weight.storage == kS8MmulRhs ? weight.offset
                                      : weight.offset / kHalfBytes);
      if (weight.storage == kFp16MmulRhs) {
        runtime->weight_bias_offsets_.emplace(
            WeightKey(weight.semantic, weight.layer, weight.storage),
            weight.bias_offset / kHalfBytes);
      } else if (weight.storage == kS8MmulRhs) {
        runtime->weight_scale_byte_offsets_.emplace(
            WeightKey(weight.semantic, weight.layer, weight.storage),
            weight.scale_offset);
        runtime->weight_bias_byte_offsets_.emplace(
            WeightKey(weight.semantic, weight.layer, weight.storage),
            weight.bias_offset);
      }
    }
  }
  constexpr std::array<uint32_t, 4> kFfnSemantics = {7, 8, 14, 15};
  for (int32_t layer = 0; layer < 24; ++layer) {
    const size_t descriptor_count =
        std::count_if(kFfnSemantics.begin(), kFfnSemantics.end(),
                      [&](uint32_t semantic) {
                        return bytecode.FindWeight(
                                   semantic, layer, kS8MmulRhs) != nullptr;
                      });
    if (descriptor_count != 0 && descriptor_count != kFfnSemantics.size()) {
      *error = "TAIENC6 S8 FFN layer descriptor set is incomplete";
      return nullptr;
    }
    runtime->w8_ffn_layers_[layer] =
        descriptor_count == kFfnSemantics.size();
    runtime->has_any_w8_ffn_ =
        runtime->has_any_w8_ffn_ || runtime->w8_ffn_layers_[layer];
  }
  const auto projection_is_complete = [&](uint32_t semantic,
                                          bool* enabled) {
    const size_t descriptor_count = static_cast<size_t>(std::count_if(
        bytecode.weights().begin(), bytecode.weights().end(),
        [&](const WeightRecord& weight) {
          return weight.semantic == semantic &&
                 weight.storage == kS8MmulRhs &&
                 weight.layer >= 0 && weight.layer < 24;
        }));
    if (descriptor_count != 0 && descriptor_count != 24) {
      *error = "TAIENC6 S8 projection descriptor set is incomplete";
      return false;
    }
    *enabled = descriptor_count == 24;
    return true;
  };
  if (!projection_is_complete(9, &runtime->w8_qkv_) ||
      !projection_is_complete(10, &runtime->w8_attention_out_) ||
      !projection_is_complete(11, &runtime->w8_conv_pw1_) ||
      !projection_is_complete(13, &runtime->w8_conv_pw2_)) {
    return nullptr;
  }
  runtime->has_any_w8_projection_ =
      runtime->w8_qkv_ || runtime->w8_attention_out_ ||
      runtime->w8_conv_pw1_ || runtime->w8_conv_pw2_;
  runtime->has_fp16_quantized_projection_ =
      runtime->w8_attention_out_ || runtime->w8_conv_pw2_;
  constexpr std::array<std::pair<uint32_t, uint32_t>, 16> kRequiredWeights = {{
      {1, kFp16Raw}, {2, kFp16Raw}, {3, kFp16MmulRhs},
      {6, kFp16MmulRhs}, {7, kS8MmulRhs}, {9, kS8MmulRhs},
      {10, kS8MmulRhs}, {11, kS8MmulRhs}, {13, kS8MmulRhs},
      {12, kFp16Raw}, {16, kFp16Raw}, {17, kFp16Raw},
      {18, kFp16Raw}, {19, kFp16Raw}, {20, kFp16Raw},
      {21, kFp16MmulRhs}}};
  for (const auto [semantic, required_storage] : kRequiredWeights) {
    const uint32_t storage = ((semantic == 7 && !runtime->w8_ffn_layers_[0]) || (semantic == 9 && !runtime->w8_qkv_) || (semantic == 10 && !runtime->w8_attention_out_) || (semantic == 11 && !runtime->w8_conv_pw1_) || (semantic == 13 && !runtime->w8_conv_pw2_)) ? kFp16MmulRhs : required_storage;
    const int32_t layer = semantic >= 7 && semantic <= 20 ? 0 : -1;
    if (bytecode.FindWeight(semantic, layer, storage) == nullptr) {
      *error = "TAIENC6 production weight contract is incomplete";
      return nullptr;
    }
  }
  const uint64_t weight_bytes = bytecode.has_external_payload()
                                    ? bytecode.external_payload_size()
                                    : bytecode.payload_size();
  if (weight_bytes == 0 || weight_bytes > std::numeric_limits<size_t>::max()) {
    *error = "TAIENC6 weight payload exceeds OpenCL address space";
    return nullptr;
  }
  if (bytecode.has_external_payload()) {
    runtime->weights_fd_ = open(weights_path.c_str(), O_RDONLY | O_CLOEXEC);
    struct stat file_status {};
    if (runtime->weights_fd_ < 0 ||
        fstat(runtime->weights_fd_, &file_status) != 0 ||
        file_status.st_size != static_cast<off_t>(weight_bytes)) {
      *error = "TAIENC6 external weight file contract mismatch";
      return nullptr;
    }
  }
  cl_int status = CL_SUCCESS;
  runtime->weights_ = runtime->api_->create_buffer(
      context, CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY,
      static_cast<size_t>(weight_bytes), nullptr, &status);
  if (status != CL_SUCCESS || runtime->weights_ == nullptr) {
    *error = "TAIENC6 persistent weight allocation failed: " +
             std::to_string(status);
    return nullptr;
  }
  if (bytecode.has_external_payload()) {
    constexpr size_t kUploadChunkBytes = 16U * 1024U * 1024U;
    std::vector<uint8_t> staging(kUploadChunkBytes);
    size_t offset = 0;
    while (offset < static_cast<size_t>(weight_bytes)) {
      const size_t bytes = std::min(
          staging.size(), static_cast<size_t>(weight_bytes) - offset);
      const ssize_t read_bytes = pread(
          runtime->weights_fd_, staging.data(), bytes,
          static_cast<off_t>(offset));
      if (read_bytes != static_cast<ssize_t>(bytes) ||
          runtime->api_->enqueue_write(
              queue, runtime->weights_, CL_TRUE, offset, bytes, staging.data(),
              0, nullptr, nullptr) != CL_SUCCESS) {
        *error = "TAIENC6 external weight upload failed at byte " +
                 std::to_string(offset);
        return nullptr;
      }
      offset += bytes;
    }
    close(runtime->weights_fd_);
    runtime->weights_fd_ = -1;
  } else if (runtime->api_->enqueue_write(
                 queue, runtime->weights_, CL_TRUE, 0,
                 static_cast<size_t>(weight_bytes), bytecode.payload(), 0,
                 nullptr, nullptr) != CL_SUCCESS) {
    *error = "TAIENC6 embedded weight upload failed";
    return nullptr;
  }

  const size_t separator = weights_path.find_last_of('/');
  const std::string kernel_path =
      (separator == std::string::npos ? std::string() :
                                       weights_path.substr(0, separator + 1)) +
      "teacher_kernels.cl";
  const std::string source = ReadTextFile(kernel_path);
  if (source.empty()) {
    *error = "TAIENC6 OpenCL kernel artifact is missing";
    return nullptr;
  }
  const char* source_address = source.data();
  const size_t source_size = source.size();
  runtime->program_ = runtime->api_->create_program(
      context, 1, &source_address, &source_size, &status);
  if (status != CL_SUCCESS || runtime->program_ == nullptr) {
    *error = "TAIENC6 OpenCL program creation failed";
    return nullptr;
  }
  std::string build_options="-cl-std=CL2.0 -DTAI_DENSE_LX="+std::to_string(runtime->dense_lx_)+" -DTAI_DENSE_LY="+std::to_string(runtime->dense_ly_)+" -DTAI_SM="+std::to_string(runtime->small_m_)+" -DTAI_SN="+std::to_string(runtime->small_n_)+" -DTAI_BCAST_K="+std::to_string(runtime->bcast_k_)+(nemo_padding?" -DTAI_NEMO_ZERO_HALO=1":"");

  const int registers=choice("TRANSAI_G720_MMUL_REGISTERS",0,{0,32,64});
  build_options+=" -DTAI_DENSE_UNROLL="+std::to_string(choice("TRANSAI_G720_DENSE_UNROLL",1,{1,2,4,8}));
  if(registers){
    build_options+=" -fregister-allocation=";
    bool first=true;
    for(const char* name:{"dense_fp16_g720","dense_fp16_g720_small",
      "gemm_w8a8_taienc6_1024x4096","gemm_w8a8_taienc6_4096x1024_residual_half",
      "w8a8_taienc6_expand_direct","w8a8_taienc6_contract_direct",
      "w8a8_taienc6_expand_small","w8a8_taienc6_contract_small",
      "w8a8_taienc6_expand_tile","w8a8_taienc6_contract_tile"}){
      if(!first)build_options+=",";first=false;
      build_options+=std::string(name)+":"+std::to_string(registers);
    }
  }
  std::cerr<<"G720_BUILD_OPTIONS "<<build_options<<std::endl;
  status = runtime->api_->build_program(runtime->program_, 0, nullptr,
                                        build_options.c_str(), nullptr, nullptr);
  if (status != CL_SUCCESS) {
    size_t device_bytes = 0;
    runtime->api_->get_program_info(runtime->program_, CL_PROGRAM_DEVICES, 0,
                                    nullptr, &device_bytes);
    std::vector<cl_device_id> devices(device_bytes / sizeof(cl_device_id));
    if (!devices.empty()) {
      runtime->api_->get_program_info(runtime->program_, CL_PROGRAM_DEVICES,
                                      device_bytes, devices.data(), nullptr);
    }
    size_t log_size = 0;
    const cl_device_id device = devices.empty() ? nullptr : devices.front();
    runtime->api_->get_build_info(runtime->program_, device,
                                  CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
    std::string log(log_size, '\0');
    if (log_size != 0) {
      runtime->api_->get_build_info(runtime->program_, device,
                                    CL_PROGRAM_BUILD_LOG, log.size(),
                                    log.data(), nullptr);
    }
    *error = "TAIENC6 OpenCL program build failed (" +
             std::to_string(status) + "): " + log;
    return nullptr;
  }
  if (!runtime->CreateKernels(error) || !runtime->AllocateWorkspace(error)) {
    return nullptr;
  }
  runtime->shape_tile_limit_ = choice("TRANSAI_G720_SHAPE_TILE_LIMIT",0,{0,32,64,96,112,128,192});
  runtime->alternate_m_ = choice("TRANSAI_G720_ALT_M",8,{4,8});
  runtime->alternate_n_ = choice("TRANSAI_G720_ALT_N",4,{4,8});
  runtime->adaptive_tile_n_ = std::getenv("TRANSAI_G720_ADAPTIVE_TILE_N") != nullptr;
  runtime->row_reuse_pad_ = std::getenv("TRANSAI_G720_ROW_REUSE_PAD") != nullptr;
  runtime->tiny_limit_ = choice("TRANSAI_G720_TINY_LIMIT",0,{0,32,64});
  if (runtime->adaptive_tile_n_) {
    if (runtime->shape_tile_limit_ == 0) { *error="adaptive tiles require a short-plan limit";return nullptr; }
    runtime->alternate_m_=4;runtime->alternate_n_=8;
  }
  if (runtime->shape_tile_limit_ != 0) {
    if (runtime->small_m_ != 4 || runtime->small_n_ != 4) {
      *error="shape tile policy requires primary 4x4"; return nullptr;
    }
    runtime->alternate_program_=runtime->api_->create_program(context,1,&source_address,&source_size,&status);
    if(status!=CL_SUCCESS || !runtime->alternate_program_) {
      *error="alternate MMUL program creation failed"; return nullptr;
    }
    std::swap(runtime->program_,runtime->alternate_program_);
    runtime->kernels_.swap(runtime->alternate_kernels_);
    auto alternate_options=build_options;
    const auto marker=alternate_options.find("-DTAI_SM=4 ");
    if(marker==std::string::npos) { *error="missing immutable primary tile option";return nullptr; }
    alternate_options.replace(marker,11,"-DTAI_SM="+std::to_string(runtime->alternate_m_)+" ");
    const auto n_marker=alternate_options.find("-DTAI_SN=4 ");
    if(n_marker==std::string::npos) { *error="missing primary N tile option";return nullptr; }
    alternate_options.replace(n_marker,11,"-DTAI_SN="+std::to_string(runtime->alternate_n_)+" ");
    const int alternate_k=choice("TRANSAI_G720_ALT_BCAST_K",runtime->bcast_k_,{32,64,128,256,512});
    const auto original_k="-DTAI_BCAST_K="+std::to_string(runtime->bcast_k_)+" ";
    const auto k_marker=alternate_options.find(original_k);
    if(k_marker==std::string::npos) { *error="missing primary RHS chunk option";return nullptr; }
    alternate_options.replace(k_marker,original_k.size(),"-DTAI_BCAST_K="+std::to_string(alternate_k)+" ");
    std::cerr<<"G720_ALTERNATE_BUILD_OPTIONS "<<alternate_options<<std::endl;
    status=runtime->api_->build_program(runtime->program_,0,nullptr,alternate_options.c_str(),nullptr,nullptr);
    if(status!=CL_SUCCESS) {
      *error="alternate MMUL program build failed: "+std::to_string(status);return nullptr;
    }
    if(!runtime->CreateKernels(error))return nullptr;
    std::swap(runtime->program_,runtime->alternate_program_);
    runtime->kernels_.swap(runtime->alternate_kernels_);
    std::cerr<<"G720_SHAPE_TILE limit="<<runtime->shape_tile_limit_<<" primary=4x4 alternate="
             <<runtime->alternate_m_<<"x"<<runtime->alternate_n_<<" shared_weights_workspace=1"<<std::endl;
    if (runtime->adaptive_tile_n_ && !runtime->row_reuse_pad_) {
      runtime->fallback_program_=runtime->api_->create_program(context,1,&source_address,&source_size,&status);
      if(status!=CL_SUCCESS || !runtime->fallback_program_) { *error="fallback MMUL program creation failed";return nullptr; }
      std::swap(runtime->program_,runtime->fallback_program_);
      runtime->kernels_.swap(runtime->fallback_kernels_);
      auto fallback_options=build_options;
      const auto fm=fallback_options.find("-DTAI_SM=4 ");
      if(fm==std::string::npos) { *error="missing fallback tile option";return nullptr; }
      fallback_options.replace(fm,11,"-DTAI_SM=8 ");
      std::cerr<<"G720_FALLBACK_BUILD_OPTIONS "<<fallback_options<<std::endl;
      status=runtime->api_->build_program(runtime->program_,0,nullptr,fallback_options.c_str(),nullptr,nullptr);
      if(status!=CL_SUCCESS) { *error="fallback MMUL build failed: "+std::to_string(status);return nullptr; }
      if(!runtime->CreateKernels(error))return nullptr;
      std::swap(runtime->program_,runtime->fallback_program_);
      runtime->kernels_.swap(runtime->fallback_kernels_);
      std::cerr<<"G720_ADAPTIVE_N long=4x8 degenerate_y=8x4 shared_weights_workspace=1"<<std::endl;
    }
  }
  if(runtime->tiny_limit_!=0) {
    if(runtime->small_m_!=4 || runtime->small_n_!=4 || !runtime->shape_tile_limit_) { *error="tiny tile requires primary4x4 and shape policy";return nullptr; }
    runtime->tiny_program_=runtime->api_->create_program(context,1,&source_address,&source_size,&status);
    if(status!=CL_SUCCESS || !runtime->tiny_program_) { *error="tiny MMUL program creation failed";return nullptr; }
    std::swap(runtime->program_,runtime->tiny_program_);runtime->kernels_.swap(runtime->tiny_kernels_);
    auto options=build_options;const auto marker=options.find("-DTAI_SM=4 ");
    if(marker==std::string::npos) { *error="tiny primary option missing";return nullptr; }
    options.replace(marker,11,"-DTAI_SM=2 ");
    std::cerr<<"G720_TINY_BUILD_OPTIONS "<<options<<std::endl;
    status=runtime->api_->build_program(runtime->program_,0,nullptr,options.c_str(),nullptr,nullptr);
    if(status!=CL_SUCCESS) { *error="tiny MMUL build failed: "+std::to_string(status);return nullptr; }
    if(!runtime->CreateKernels(error))return nullptr;
    std::swap(runtime->program_,runtime->tiny_program_);runtime->kernels_.swap(runtime->tiny_kernels_);
  }
  return runtime;
}

bool OpenClRuntime::CreateKernels(std::string* error) {
  constexpr std::array<const char*, 13> kNames = {
      "pre_conv0", "pre_depthwise3", "dense_fp16_g720", "dense_fp16_g720_small",
      "pre_reorder",        "layernorm_fp16",    "attention_score",
      "attention_relative_fuse", "attention_softmax", "attention_value",
      "conv_glu", "conv_depthwise9_silu", "output_fp16"};
  constexpr std::array<const char*, 10> kW8FfnNames = {
      "layernorm_s8_dynamic_row_no_affine_1024",
      "gemm_w8a8_taienc6_1024x4096",
      "ffn_silu_quantize_epilogue",
      "gemm_w8a8_taienc6_4096x1024_residual_half",
      "w8a8_taienc6_expand_direct",
      "w8a8_taienc6_contract_direct", "w8a8_taienc6_expand_small", "w8a8_taienc6_contract_small", "w8a8_taienc6_expand_tile", "w8a8_taienc6_contract_tile"};
  constexpr std::array<const char*, 4> kW8ProjectionNames = {
      "w8a8_taienc6_projection",
      "w8a8_taienc6_projection_direct", "w8a8_taienc6_projection_small", "w8a8_taienc6_projection_tile"};
  const auto create = [&](const char* name) {
    cl_int status = CL_SUCCESS;
    cl_kernel kernel = api_->create_kernel(program_, name, &status);
    if (status != CL_SUCCESS || kernel == nullptr) {
      *error = std::string("TAIENC6 kernel creation failed: ") + name;
      return false;
    }
    const bool w8_mmul =
        std::strcmp(name, "gemm_w8a8_taienc6_1024x4096") == 0 ||
        std::strcmp(name, "gemm_w8a8_taienc6_4096x1024_residual_half") == 0 ||
        std::strcmp(name, "w8a8_taienc6_expand_small") == 0 || std::strcmp(name, "w8a8_taienc6_contract_small") == 0 || std::strcmp(name, "w8a8_taienc6_expand_tile") == 0 || std::strcmp(name, "w8a8_taienc6_contract_tile") == 0 ||
        std::strcmp(name, "w8a8_taienc6_expand_direct") == 0 ||
        std::strcmp(name, "w8a8_taienc6_contract_direct") == 0 ||
        std::strcmp(name, "w8a8_taienc6_projection_small") == 0 || std::strcmp(name, "w8a8_taienc6_projection_tile") == 0 || std::strcmp(name, "w8a8_taienc6_projection") == 0 ||
        std::strcmp(name, "w8a8_taienc6_projection_direct") == 0;
    const cl_int kWorkgroupBatchSizeModifier = wbsm_;
    if ((w8_mmul || (wbsm_dense_ && (std::strcmp(name,"dense_fp16_g720")==0 || std::strcmp(name,"dense_fp16_g720_small")==0))) &&
        api_->set_kernel_exec_info(
            kernel, CL_KERNEL_EXEC_INFO_WORKGROUP_BATCH_SIZE_MODIFIER_ARM,
            sizeof(kWorkgroupBatchSizeModifier),
            &kWorkgroupBatchSizeModifier) != CL_SUCCESS) {
      api_->release_kernel(kernel);
      *error = std::string("TAIENC6 scheduling control failed: ") + name;
      return false;
    }
    kernels_.emplace(name, kernel);
    return true;
  };
  for (const char* name : kNames) {
    if (!create(name)) return false;
  }
  if (has_any_w8_ffn_) {
    for (const char* name : kW8FfnNames) {
      if (!create(name)) return false;
    }
  }
  if (has_any_w8_projection_) {
    for (const char* name : kW8ProjectionNames) {
      if (!create(name)) return false;
    }
  }
  if (has_fp16_quantized_projection_ &&
      !create("quantize_fp16_s8_dynamic_row_1024")) {
    return false;
  }
  return true;
}

bool OpenClRuntime::AllocateWorkspace(std::string* error) {
  const auto allocate = [&](size_t bytes, cl_mem* output) -> bool {
    cl_int status = CL_SUCCESS;
    *output = api_->create_buffer(context_, CL_MEM_READ_WRITE, bytes, nullptr,
                                  &status);
    if (status != CL_SUCCESS || *output == nullptr) {
      *error = "TAIENC6 workspace allocation failed: " +
               std::to_string(status);
      return false;
    }
    return true;
  };
  return allocate(1500ULL * 64 * 256 * kHalfBytes, &pre0_) &&
         allocate(750ULL * 32 * 256 * kHalfBytes, &pre1a_) &&
         allocate(750ULL * 32 * 256 * kHalfBytes, &pre1b_) &&
         allocate(375ULL * 16 * 256 * kHalfBytes, &pre2a_) &&
         allocate(375ULL * 16 * 256 * kHalfBytes, &pre2b_) &&
         allocate(kMaxFrames * 4096 * kHalfBytes, &pre_flat_) &&
         allocate(kMaxFrames * kModelChannels * kHalfBytes, &activation_a_) &&
         allocate(kMaxFrames * kModelChannels * kHalfBytes, &activation_b_) &&
         allocate(kMaxFrames * 4096 * kHalfBytes, &ffn_) &&
         allocate(kMaxFrames * 4096, &ffn_q8_) &&
         allocate(kMaxFrames * sizeof(float), &ffn_scale_) &&
         allocate(kMaxFrames * 3072 * kHalfBytes, &qkv_) &&
         allocate(8ULL * kMaxFrames * kMaxFrames * kHalfBytes, &scores_) &&
         allocate(kMaxFrames * kModelChannels * kHalfBytes, &attention_) &&
         allocate(kMaxFrames * 2048 * kHalfBytes, &conv_) &&
         allocate(kMaxFrames * 640 * kHalfBytes, &joint_);
}

OpenClRuntime::~OpenClRuntime() {
  if (api_ == nullptr) return;
  if (pending_profile_event_) api_->release_event(pending_profile_event_);
  profiler_.Release(api_->release_event);
  cached_plans_.clear(); // Release recorded commands before their kernels and buffers.
  const std::array<cl_mem, 17> buffers = {
      joint_,       conv_,        attention_,   scores_,      qkv_,
      ffn_scale_, ffn_q8_, ffn_, activation_b_, activation_a_,
      pre_flat_, pre2b_, pre2a_, pre1b_, pre1a_, pre0_, weights_};
  for (cl_mem buffer : buffers) {
    if (buffer != nullptr) api_->release_mem(buffer);
  }
  if (weights_fd_ >= 0) close(weights_fd_);
  for (const auto& [name, kernel] : kernels_) {
    (void)name;
    api_->release_kernel(kernel);
  }
  for (const auto& [name,kernel] : alternate_kernels_) {
    (void)name; api_->release_kernel(kernel);
  }
  if (alternate_program_ != nullptr) api_->release_program(alternate_program_);
  for (const auto& [name,kernel] : fallback_kernels_) {
    (void)name; api_->release_kernel(kernel);
  }
  if (fallback_program_ != nullptr) api_->release_program(fallback_program_);
  for(const auto& [name,kernel]:tiny_kernels_) { (void)name;api_->release_kernel(kernel); }
  if(tiny_program_)api_->release_program(tiny_program_);
  if (program_ != nullptr) api_->release_program(program_);
}

uint64_t OpenClRuntime::WeightOffset(uint32_t semantic, int32_t layer,
                                     uint32_t storage) const {
  const auto found = weight_offsets_.find(WeightKey(semantic, layer, storage));
  return found == weight_offsets_.end() ? std::numeric_limits<uint64_t>::max()
                                        : found->second;
}

uint64_t OpenClRuntime::WeightBiasOffset(uint32_t semantic,
                                         int32_t layer) const {
  const auto found = weight_bias_offsets_.find(
      WeightKey(semantic, layer, kFp16MmulRhs));
  return found == weight_bias_offsets_.end()
             ? std::numeric_limits<uint64_t>::max()
             : found->second;
}

uint64_t OpenClRuntime::WeightScaleByteOffset(uint32_t semantic,
                                              int32_t layer) const {
  const auto found = weight_scale_byte_offsets_.find(
      WeightKey(semantic, layer, kS8MmulRhs));
  return found == weight_scale_byte_offsets_.end()
             ? std::numeric_limits<uint64_t>::max()
             : found->second;
}

uint64_t OpenClRuntime::WeightBiasByteOffset(uint32_t semantic,
                                             int32_t layer) const {
  const auto found = weight_bias_byte_offsets_.find(
      WeightKey(semantic, layer, kS8MmulRhs));
  return found == weight_bias_byte_offsets_.end()
             ? std::numeric_limits<uint64_t>::max()
             : found->second;
}

bool OpenClRuntime::UseW8FfnLayer(int layer) const {
  return !capture_force_fp16_ && layer >= 0 &&
         layer < static_cast<int>(w8_ffn_layers_.size()) &&
         w8_ffn_layers_[static_cast<size_t>(layer)];
}

bool OpenClRuntime::UseW8Projection(uint32_t semantic) const {
  return (semantic == 9 && w8_qkv_) ||
         (semantic == 10 && w8_attention_out_) ||
         (semantic == 11 && w8_conv_pw1_) ||
         (semantic == 13 && w8_conv_pw2_);
}

cl_kernel OpenClRuntime::Kernel(const char* name) const {
  const auto& active=use_tiny_tile_?tiny_kernels_:(use_fallback_tile_?fallback_kernels_:(use_alternate_tile_?alternate_kernels_:kernels_));
  const auto found = active.find(name);
  return found == active.end() ? nullptr : found->second;
}

bool OpenClRuntime::LoadCaptureRequest(std::string* error) {
  capture_stage_.clear();
  capture_path_.clear();
  capture_layer_ = -1;
  capture_written_ = false;
  capture_force_fp16_ = false;
  if (!diagnostic_) return true;
  const char* stage = std::getenv("TRANSAI_TEACHER_CAPTURE_STAGE");
  const char* path = std::getenv("TRANSAI_TEACHER_CAPTURE_PATH");
  if (stage == nullptr && path == nullptr) return true;
  if (stage == nullptr || stage[0] == '\0' || path == nullptr || path[0] == '\0') {
    *error = "teacher stage capture requires nonempty stage and path";
    return false;
  }
  capture_stage_ = stage;
  if (capture_stage_.compare(0, 5, "fp16:") == 0) {
    capture_stage_.erase(0, 5);
    capture_force_fp16_ = true;
  }
  capture_path_ = path;
  if (const char* layer = std::getenv("TRANSAI_TEACHER_CAPTURE_LAYER")) {
    char* end = nullptr;
    const long parsed = std::strtol(layer, &end, 10);
    if (end == layer || *end != '\0' || parsed < -1 || parsed >= 24) {
      *error = "teacher stage capture layer is invalid";
      return false;
    }
    capture_layer_ = static_cast<int>(parsed);
  }
  return true;
}

bool OpenClRuntime::MaybeCapture(const char* stage, int layer, cl_mem buffer,
                                 std::vector<uint32_t> shape,
                                 std::string* error) {
  return MaybeCaptureTyped(stage, layer, buffer, std::move(shape),
                           sizeof(uint16_t), "fp16", error);
}

bool OpenClRuntime::MaybeCaptureTyped(const char* stage, int layer,
                                      cl_mem buffer,
                                      std::vector<uint32_t> shape,
                                      size_t element_bytes,
                                      const char* dtype,
                                      std::string* error) {
  if (capture_stage_.empty() || capture_written_ || capture_stage_ != stage ||
      capture_layer_ != layer) {
    return true;
  }
  uint64_t elements = 1;
  for (uint32_t dimension : shape) {
    if (dimension == 0 || elements > std::numeric_limits<size_t>::max() / dimension) {
      *error = "teacher stage capture shape is invalid";
      return false;
    }
    elements *= dimension;
  }
  if (element_bytes == 0 ||
      elements > std::numeric_limits<size_t>::max() / element_bytes) {
    *error = "teacher stage capture byte size is invalid";
    return false;
  }
  const size_t bytes = static_cast<size_t>(elements) * element_bytes;
  std::vector<uint8_t> values(bytes);
  if (api_->enqueue_read(queue_, buffer, CL_TRUE, 0, bytes, values.data(),
                         0, nullptr, nullptr) != CL_SUCCESS) {
    *error = "teacher stage capture readback failed";
    return false;
  }
  std::ofstream output(capture_path_, std::ios::binary);
  output.write(reinterpret_cast<const char*>(values.data()),
               static_cast<std::streamsize>(bytes));
  if (!output) {
    *error = "teacher stage capture output write failed";
    return false;
  }
  std::ofstream metadata(capture_path_ + ".json", std::ios::binary);
  metadata << "{\"format_version\":1,\"stage\":\"" << capture_stage_
           << "\",\"layer\":" << capture_layer_
           << ",\"valid_mel_frames\":" << capture_valid_mel_frames_
           << ",\"valid_encoder_frames\":"
           << ((capture_valid_mel_frames_ + 7) / 8)
           << ",\"selected_plan\":" << capture_plan_
           << ",\"dtype\":\"" << dtype << "\",\"shape\":[";
  for (size_t index = 0; index < shape.size(); ++index) {
    if (index != 0) metadata << ',';
    metadata << shape[index];
  }
  metadata << "],\"elements\":" << elements << "}";
  if (!metadata) {
    *error = "teacher stage capture metadata write failed";
    return false;
  }
  capture_written_ = true;
  return true;
}

bool OpenClRuntime::Enqueue(const char* name, const char* category, int layer,
                            cl_kernel kernel, cl_uint dimensions,
                            const size_t* global, const size_t* local,
                            bool final, cl_event* event, std::string* error) {
  cl_event next_event = nullptr;
  if (!diagnostic_) {
    if (recording_plan_ == nullptr || recording_plan_->commands == nullptr) {
      *error = "TAIENC6 production command buffer is not recording";
      return false;
    }
    try {
      transai::opencl::CommandDispatch dispatch;
      dispatch.kernel = kernel;
      dispatch.dimensions = dimensions;
      std::copy_n(global, dimensions, dispatch.global);
      dispatch.explicit_local = local != nullptr;
      if (local != nullptr) std::copy_n(local, dimensions, dispatch.local);
      recording_plan_->commands->Add(dispatch);
      ++recording_plan_->command_count;
      if (final) {
        recording_plan_->commands->Finalize();
        const auto submit_start = EventProfiler::Clock::now();
        next_event = recording_plan_->commands->Enqueue();
        command_submit_ms_ = std::chrono::duration<double, std::milli>(
                                 EventProfiler::Clock::now() - submit_start)
                                 .count();
      }
    } catch (const std::exception& exception) {
      *error = std::string("TAIENC6 command buffer record failed: ") +
               exception.what();
      return false;
    }
  } else {
    const cl_uint wait_count = previous_diagnostic_event_ == nullptr ? 0U : 1U;
    const cl_int status = api_->enqueue_kernel(
        queue_, kernel, dimensions, nullptr, global, local, wait_count,
        wait_count == 0U ? nullptr : &previous_diagnostic_event_, &next_event);
    if (status != CL_SUCCESS) {
      *error = "teacher kernel enqueue failed for " + std::string(name) +
               ": " + std::to_string(status);
      return false;
    }
    profiler_.Add(name, category, layer, next_event);
    previous_diagnostic_event_ = next_event;
  }
  if (event != nullptr) {
    *event = next_event;
  }
  return true;
}

bool OpenClRuntime::Invoke(cl_mem mel, cl_mem valid_mel_frames, cl_mem output,
                           uint32_t valid_mel_frames_value,
                           const PlanRecord& plan, std::string* error) {

  previous_diagnostic_event_ = nullptr;
  const bool profile_command = !diagnostic_ && std::getenv("TRANSAI_G720_PROFILE_COMMAND_BUFFER");
  if (profile_command) profiler_.Begin(valid_mel_frames_value, plan.encoder_frames);
  if (mel == nullptr || valid_mel_frames == nullptr || output == nullptr ||
      error == nullptr) {
    return false;
  }
  capture_valid_mel_frames_ = valid_mel_frames_value;
  capture_plan_ = plan.encoder_frames;
  if (!LoadCaptureRequest(error)) return false;
  const int rows = static_cast<int>(plan.encoder_frames);
  if (rows <= 0 || rows > static_cast<int>(kMaxFrames)) {
    *error = "TAIENC6 command plan frame count is invalid";
    return false;
  }
  if (shape_tile_limit_ != 0) {
    use_alternate_tile_=rows>shape_tile_limit_;
    small_m_=use_alternate_tile_?alternate_m_:4;
    small_n_=use_alternate_tile_?alternate_n_:4;
    use_fallback_tile_=false;
    if (adaptive_tile_n_ && use_alternate_tile_ && !row_reuse_pad_) {
      const size_t sections=(rows+15)/16;
      size_t reuse=std::min<size_t>(8,sections);while(sections%reuse)--reuse;
      if (reuse==1) { use_fallback_tile_=true;small_m_=8;small_n_=4; }
    }
    use_tiny_tile_=tiny_limit_!=0 && rows<=tiny_limit_;
    if(use_tiny_tile_) { small_m_=2;small_n_=4; }
  }
  if (!diagnostic_) {
    const auto cached = std::find_if(
        cached_plans_.begin(), cached_plans_.end(),
        [&](const CachedPlan& value) {
          return value.encoder_frames == plan.encoder_frames &&
                 value.mel == mel && value.length == valid_mel_frames &&
                 value.output == output;
        });
    if (cached != cached_plans_.end()) {
      cl_event event = nullptr;
      try {
        const auto submit_start = EventProfiler::Clock::now();
        event = cached->commands->Enqueue();
        command_submit_ms_ = std::chrono::duration<double, std::milli>(
                                 EventProfiler::Clock::now() - submit_start)
                                 .count();
      } catch (const std::exception& exception) {
        *error = std::string("TAIENC6 command buffer enqueue failed: ") +
                 exception.what();
        return false;
      }
      const cl_int status = defer_wait_ && !diagnostic_ ? CL_SUCCESS : api_->wait_events(1, &event);
      if (status != CL_SUCCESS) {
        api_->release_event(event);
        *error = "TAIENC6 command buffer wait failed";
        return false;
      }
      if (profile_command) pending_profile_event_ = event;
      else api_->release_event(event);
      last_profile_json_ = "{}";
      return true;
    }
    try {
      cached_plans_.push_back(CachedPlan{
          plan.encoder_frames, mel, valid_mel_frames, output, 0,
          std::make_unique<transai::opencl::CommandBuffer>(queue_)});
    } catch (const std::exception& exception) {
      *error = std::string("TAIENC6 command buffer creation failed: ") +
               exception.what();
      return false;
    }
    recording_plan_ = &cached_plans_.back();
  }
  if (diagnostic_) profiler_.Begin(valid_mel_frames_value, plan.encoder_frames);
  const int pre0_frames = static_cast<int>(plan.pre0_frames);
  const int pre1_frames = static_cast<int>(plan.pre1_frames);
  const int pre2_frames = static_cast<int>(plan.pre2_frames);
  if (plan.mel_capacity != plan.encoder_frames * 8 ||
      pre0_frames != rows * 4 || pre1_frames != rows * 2 ||
      pre2_frames != rows || plan.output_frames != plan.encoder_frames) {
    *error = "TAIENC6 plan-specific frame contract mismatch";
    return false;
  }
  const uint64_t no_offset = 0;
  cl_mem residual_placeholder = activation_a_;

  cl_kernel kernel = Kernel("pre_conv0");
  uint64_t weight = WeightOffset(1, -1, kFp16Raw);
  uint64_t bias = WeightOffset(1, -1, kFp16Vector);
  if (!SetArg(api_->set_arg, kernel, 0, mel) ||
      !SetArg(api_->set_arg, kernel, 1, valid_mel_frames) ||
      !SetArg(api_->set_arg, kernel, 2, weights_) ||
      !SetArg(api_->set_arg, kernel, 3, weight) ||
      !SetArg(api_->set_arg, kernel, 4, bias) ||
      !SetArg(api_->set_arg, kernel, 5, pre0_)) {
    *error = "TAIENC6 preencoder conv0 arguments failed";
    return false;
  }
  const size_t conv0_global[] = {
      256, 64, static_cast<size_t>(pre0_frames)};
  if (!Enqueue("pre_conv0", "preencoder", -1, kernel, 3, conv0_global,
               nullptr, false, nullptr, error) ||
      !MaybeCapture(
          "pre_conv0", -1, pre0_,
          {1, static_cast<uint32_t>(pre0_frames), 64U * 256U}, error)) {
    return false;
  }

  const auto depthwise3 = [&](cl_mem input, cl_mem destination,
                              uint32_t semantic,
                              int input_frames, int input_frequencies,
                              int output_frames, int output_frequencies,
                              int subsampling) -> bool {
    cl_kernel current = Kernel("pre_depthwise3");
    const uint64_t weight_value = WeightOffset(semantic, -1, kFp16Raw);
    const uint64_t bias_value = WeightOffset(semantic, -1, kFp16Vector);
    if (!SetArg(api_->set_arg, current, 0, input) ||
        !SetArg(api_->set_arg, current, 1, valid_mel_frames) ||
        !SetArg(api_->set_arg, current, 2, weights_) ||
        !SetArg(api_->set_arg, current, 3, weight_value) ||
        !SetArg(api_->set_arg, current, 4, bias_value) ||
        !SetArg(api_->set_arg, current, 5, destination) ||
        !SetArg(api_->set_arg, current, 6, input_frames) ||
        !SetArg(api_->set_arg, current, 7, input_frequencies) ||
        !SetArg(api_->set_arg, current, 8, output_frames) ||
        !SetArg(api_->set_arg, current, 9, output_frequencies) ||
        !SetArg(api_->set_arg, current, 10, subsampling)) {
      *error = "TAIENC6 preencoder depthwise arguments failed";
      return false;
    }
    const size_t global[] = {32, static_cast<size_t>(output_frequencies),
                             static_cast<size_t>(output_frames)};
    return Enqueue("pre_depthwise3", "preencoder", -1, current, 3, global,
                   nullptr, false, nullptr, error);
  };

  const char* dense_category = "preencoder";
  int dense_layer = -1;
  const auto dense = [&](cl_mem input, cl_mem destination,
                         uint32_t semantic, int semantic_layer,
                         cl_mem residual, int dense_rows, int input_channels,
                         int output_channels, int spatial_width,
                         int subsampling, int mode) -> bool {
    const bool small=small_tile_ && dense_rows<=dense_tile_max_m_;
    const char* dense_name=small?"dense_fp16_g720_small":"dense_fp16_g720";
    cl_kernel current = Kernel(dense_name);
    const uint64_t weight_value =
        WeightOffset(semantic, semantic_layer, kFp16MmulRhs);
    const uint64_t bias_value =
        (mode & 1) != 0 ? WeightBiasOffset(semantic, semantic_layer) : no_offset;
    if (weight_value == std::numeric_limits<uint64_t>::max() ||
        ((mode & 1) != 0 &&
         bias_value == std::numeric_limits<uint64_t>::max())) {
      *error = "TAIENC6 dense weight lookup failed";
      return false;
    }
    if (!SetArg(api_->set_arg, current, 0, input) ||
        !SetArg(api_->set_arg, current, 1, weights_) ||
        !SetArg(api_->set_arg, current, 2, weight_value) ||
        !SetArg(api_->set_arg, current, 3, bias_value) ||
        !SetArg(api_->set_arg, current, 4, residual) ||
        !SetArg(api_->set_arg, current, 5, valid_mel_frames) ||
        !SetArg(api_->set_arg, current, 6, destination) ||
        !SetArg(api_->set_arg, current, 7, dense_rows) ||
        !SetArg(api_->set_arg, current, 8, input_channels) ||
        !SetArg(api_->set_arg, current, 9, output_channels) ||
        !SetArg(api_->set_arg, current, 10, spatial_width) ||
        !SetArg(api_->set_arg, current, 11, subsampling) ||
        !SetArg(api_->set_arg, current, 12, mode)) {
      *error = "TAIENC6 dense arguments failed";
      return false;
    }
    const size_t section_rows =
        (static_cast<size_t>(dense_rows) + (small?4*small_m_:16)-1) / (small?4*small_m_:16);
    const size_t global[] = {static_cast<size_t>(output_channels)*(small?4:small_n_)/small_n_,
                             RoundUp(section_rows, small?1:dense_ly_)};
    const size_t local[] = {static_cast<size_t>(dense_lx_), static_cast<size_t>(small?1:dense_ly_)};
    return Enqueue(dense_name, dense_category, dense_layer, current, 2,
                   global, local, false, nullptr, error);
  };

  if (!depthwise3(pre0_, pre1a_, 2, pre0_frames, 64,
                  pre1_frames, 32, 4) ||
      !MaybeCapture(
          "pre_dw1", -1, pre1a_,
          {1, static_cast<uint32_t>(pre1_frames), 32U * 256U}, error) ||
      !dense(pre1a_, pre1b_, 3, -1, residual_placeholder,
             pre1_frames * 32, 256, 256, 32, 4, 1 | 8) ||
      !MaybeCapture(
          "pre_pw1", -1, pre1b_,
          {1, static_cast<uint32_t>(pre1_frames), 32U * 256U}, error) ||
      !depthwise3(pre1b_, pre2a_, 4, pre1_frames, 32,
                  pre2_frames, 16, 8) ||
      !MaybeCapture(
          "pre_dw2", -1, pre2a_,
          {1, static_cast<uint32_t>(pre2_frames), 16U * 256U}, error) ||
      !dense(pre2a_, pre2b_, 5, -1, residual_placeholder,
             pre2_frames * 16, 256, 256, 16, 8, 1 | 8) ||
      !MaybeCapture(
          "pre_pw2", -1, pre2b_,
          {1, static_cast<uint32_t>(pre2_frames), 16U * 256U}, error)) {
    return false;
  }

  kernel = Kernel("pre_reorder");
  if (!SetArg(api_->set_arg, kernel, 0, pre2b_) ||
      !SetArg(api_->set_arg, kernel, 1, valid_mel_frames) ||
      !SetArg(api_->set_arg, kernel, 2, pre_flat_)) {
    *error = "TAIENC6 preencoder reorder arguments failed";
    return false;
  }
  const size_t reorder_global[] = {4096, static_cast<size_t>(rows)};
  if (!Enqueue("pre_reorder", "preencoder", -1, kernel, 2,
               reorder_global, nullptr, false, nullptr, error) ||
      !dense(pre_flat_, activation_a_, 6, -1, residual_placeholder, rows,
              4096, 1024, 1, 8, 1)) {
    return false;
  }
  if (!MaybeCapture("pre_out", -1, activation_a_,
                    {1, static_cast<uint32_t>(rows), 1024}, error)) {
    return false;
  }

  const auto layernorm = [&](cl_mem input, cl_mem destination, int layer,
                             bool affine) -> bool {
    cl_kernel current = Kernel("layernorm_fp16");
    const uint64_t gamma =
        affine ? WeightOffset(16, layer, kFp16Raw) : no_offset;
    const uint64_t beta =
        affine ? WeightOffset(17, layer, kFp16Raw) : no_offset;
    const int affine_value = affine ? 1 : 0;
    if (!SetArg(api_->set_arg, current, 0, input) ||
        !SetArg(api_->set_arg, current, 1, weights_) ||
        !SetArg(api_->set_arg, current, 2, gamma) ||
        !SetArg(api_->set_arg, current, 3, beta) ||
        !SetArg(api_->set_arg, current, 4, valid_mel_frames) ||
        !SetArg(api_->set_arg, current, 5, destination) ||
        !SetArg(api_->set_arg, current, 6, rows) ||
        !SetArg(api_->set_arg, current, 7, affine_value)) {
      *error = "TAIENC6 layernorm arguments failed";
      return false;
    }
    const size_t global[] = {static_cast<size_t>(rows) * 8};
    constexpr size_t local[] = {8};
    return Enqueue("layernorm_fp16", "layernorm", layer, current, 1,
                   global, local, false, nullptr, error);
  };
  const auto layernorm_s8 = [&](cl_mem input, int layer,
                                const char* stage_prefix) -> bool {
    cl_kernel current =
        Kernel("layernorm_s8_dynamic_row_no_affine_1024");
    constexpr float kLayerNormEpsilon = 1.0e-5F;
    if (!SetArg(api_->set_arg, current, 0, input) ||
        !SetArg(api_->set_arg, current, 1, valid_mel_frames) ||
        !SetArg(api_->set_arg, current, 2, ffn_q8_) ||
        !SetArg(api_->set_arg, current, 3, ffn_scale_) ||
        !SetArg(api_->set_arg, current, 4, kLayerNormEpsilon)) {
      *error = "TAIENC6 quantizing LayerNorm arguments failed";
      return false;
    }
    const size_t global[] = {static_cast<size_t>(rows) * 8};
    constexpr size_t local[] = {8};
    return Enqueue("layernorm_s8_dynamic_row_no_affine_1024", "layernorm",
                   layer, current, 1, global, local, false, nullptr, error) &&
           MaybeCaptureTyped(
               std::string(stage_prefix).append("_norm_q8").c_str(), layer,
               ffn_q8_, {1, static_cast<uint32_t>(rows), 1024},
               sizeof(int8_t), "int8", error) &&
           MaybeCaptureTyped(
               std::string(stage_prefix).append("_norm_scale").c_str(),
               layer, ffn_scale_, {1, static_cast<uint32_t>(rows)},
               sizeof(float), "fp32", error);
  };
  const auto w8_projection = [&](cl_mem input, cl_mem destination,
                                 uint32_t semantic, int layer,
                                 cl_mem residual_value,
                                 int output_channels, int mode,
                                 const char* stage_prefix,
                                 bool quantize_fp16) -> bool {
    if (quantize_fp16) {
      cl_kernel quantize = Kernel("quantize_fp16_s8_dynamic_row_1024");
      if (!SetArg(api_->set_arg, quantize, 0, input) ||
          !SetArg(api_->set_arg, quantize, 1, valid_mel_frames) ||
          !SetArg(api_->set_arg, quantize, 2, ffn_q8_) ||
          !SetArg(api_->set_arg, quantize, 3, ffn_scale_)) {
        *error = "TAIENC6 FP16 dynamic-row quantization arguments failed";
        return false;
      }
      const size_t quantize_global[] = {static_cast<size_t>(rows) * 64};
      constexpr size_t quantize_local[] = {64};
      if (!Enqueue("quantize_fp16_s8_dynamic_row_1024", dense_category,
                   layer, quantize, 1, quantize_global, quantize_local,
                   false, nullptr, error)) {
        return false;
      }
    } else if (!layernorm_s8(input, layer, stage_prefix)) {
      return false;
    }
    const uint64_t weight =
        WeightOffset(semantic, layer, kS8MmulRhs);
    const uint64_t scale =
        WeightScaleByteOffset(semantic, layer);
    const uint64_t bias =
        WeightBiasByteOffset(semantic, layer);
    if (weight == std::numeric_limits<uint64_t>::max() ||
        scale == std::numeric_limits<uint64_t>::max() ||
        ((mode & 1) != 0 &&
         bias == std::numeric_limits<uint64_t>::max())) {
      *error = "TAIENC6 W8 projection descriptor lookup failed";
      return false;
    }
    const bool direct_rhs = rows < projection_direct_limit_;
    const bool small=small_tile_ && direct_rhs && rows<=32;
    const bool tile=small_tile_ && !direct_rhs && projection_tile_;
    const char* name = direct_rhs
                           ? (small?"w8a8_taienc6_projection_small":"w8a8_taienc6_projection_direct")
                           : (tile?"w8a8_taienc6_projection_tile":"w8a8_taienc6_projection");
    cl_kernel current = Kernel(name);
    constexpr int input_channels = 1024;
    if (!SetArg(api_->set_arg, current, 0, ffn_q8_) ||
        !SetArg(api_->set_arg, current, 1, weights_) ||
        !SetArg(api_->set_arg, current, 2, weight) ||
        !SetArg(api_->set_arg, current, 3, ffn_scale_) ||
        !SetArg(api_->set_arg, current, 4, scale) ||
        !SetArg(api_->set_arg, current, 5, bias) ||
        !SetArg(api_->set_arg, current, 6, residual_value) ||
        !SetArg(api_->set_arg, current, 7, destination) ||
        !SetArg(api_->set_arg, current, 8, valid_mel_frames) ||
        !SetArg(api_->set_arg, current, 9, rows) ||
        !SetArg(api_->set_arg, current, 10, input_channels) ||
        !SetArg(api_->set_arg, current, 11, output_channels) ||
        !SetArg(api_->set_arg, current, 12, mode)) {
      *error = "TAIENC6 W8 projection arguments failed";
      return false;
    }
    const size_t section_rows =
        (static_cast<size_t>(rows)+(small||tile?4*small_m_:16)-1)/(small||tile?4*small_m_:16);
    const size_t local_x = direct_rhs ? 32 : 16;
    const size_t adaptive_y=(section_rows%4==0?4:(section_rows%2==0?2:1));
    const size_t exact_y=RowReuseY(section_rows);
    const size_t local_y = small?1:(direct_rhs?2:(projection_y_==16?exact_y:(projection_y_?projection_y_:adaptive_y)));
    const size_t global[] = {
        static_cast<size_t>(output_channels)*((small||tile)?4:small_n_)/small_n_,
        RoundUp(section_rows, local_y)};
    const size_t local[] = {local_x, local_y};
    return Enqueue(name, dense_category, layer, current, 2, global, local,
                   false, nullptr, error);
  };
  const auto ffn_w8 = [&](cl_mem input, cl_mem destination,
                          uint32_t input_semantic,
                          uint32_t output_semantic, int layer,
                          const char* stage_prefix) -> bool {
    if (!layernorm_s8(input, layer, stage_prefix)) {
      return false;
    }

    const uint64_t weight1 =
        WeightOffset(input_semantic, layer, kS8MmulRhs);
    const uint64_t scale1 =
        WeightScaleByteOffset(input_semantic, layer);
    const uint64_t bias1 =
        WeightBiasByteOffset(input_semantic, layer);
    const uint64_t weight2 =
        WeightOffset(output_semantic, layer, kS8MmulRhs);
    const uint64_t scale2 =
        WeightScaleByteOffset(output_semantic, layer);
    const uint64_t bias2 =
        WeightBiasByteOffset(output_semantic, layer);
    if (weight1 == std::numeric_limits<uint64_t>::max() ||
        scale1 == std::numeric_limits<uint64_t>::max() ||
        bias1 == std::numeric_limits<uint64_t>::max() ||
        weight2 == std::numeric_limits<uint64_t>::max() ||
        scale2 == std::numeric_limits<uint64_t>::max() ||
        bias2 == std::numeric_limits<uint64_t>::max()) {
      *error = "TAIENC6 W8 FFN descriptor lookup failed";
      return false;
    }
    const size_t section_rows =
        (static_cast<size_t>(rows) + 15) / 16;
    const bool direct_rhs = rows < ffn_direct_limit_;
    const bool tile=small_tile_ && !direct_rhs && ffn_broadcast_tile_;
    const bool small=small_tile_ && direct_rhs && rows<=32;
    const size_t gemm_local_x = direct_rhs ? 32 : 16;
    const size_t tile_sections=tile?(rows+4*small_m_-1)/(4*small_m_):section_rows;
    const size_t adaptive_y=(tile_sections%4==0?4:(tile_sections%2==0?2:1));
    const size_t exact_y=RowReuseY(tile_sections);
    const size_t gemm_local_y = small?1:(direct_rhs?2:(broadcast_y_==16?exact_y:(broadcast_y_?broadcast_y_:adaptive_y)));
    const size_t gemm_global_y = RoundUp((small||tile)?(rows+4*small_m_-1)/(4*small_m_):section_rows,gemm_local_y);
    const size_t gemm_local[] = {gemm_local_x, gemm_local_y};

    const char* expand_name = direct_rhs
                                  ? (small?"w8a8_taienc6_expand_small":"w8a8_taienc6_expand_direct")
                                  : (tile?"w8a8_taienc6_expand_tile":"gemm_w8a8_taienc6_1024x4096");
    cl_kernel expand = Kernel(expand_name);
    if (!SetArg(api_->set_arg, expand, 0, ffn_q8_) ||
        !SetArg(api_->set_arg, expand, 1, weights_) ||
        !SetArg(api_->set_arg, expand, 2, weight1) ||
        !SetArg(api_->set_arg, expand, 3, ffn_scale_) ||
        !SetArg(api_->set_arg, expand, 4, scale1) ||
        !SetArg(api_->set_arg, expand, 5, bias1) ||
        !SetArg(api_->set_arg, expand, 6, ffn_) ||
        !SetArg(api_->set_arg, expand, 7, rows)) {
      *error = "TAIENC6 W8 FFN expand arguments failed";
      return false;
    }
    const size_t expand_global[] = {static_cast<size_t>(4096*((small||tile)?4:small_n_)/small_n_),gemm_global_y};
    if (!Enqueue(expand_name, dense_category, layer,
                 expand, 2, expand_global, gemm_local, false, nullptr, error)) {
      return false;
    }

    cl_kernel silu = Kernel("ffn_silu_quantize_epilogue");
    constexpr int ffn_channels = 4096;
    if (!SetArg(api_->set_arg, silu, 0, ffn_) ||
        !SetArg(api_->set_arg, silu, 1, ffn_q8_) ||
        !SetArg(api_->set_arg, silu, 2, ffn_scale_) ||
        !SetArg(api_->set_arg, silu, 3, rows) ||
        !SetArg(api_->set_arg, silu, 4, ffn_channels)) {
      *error = "TAIENC6 W8 FFN SiLU arguments failed";
      return false;
    }
    const size_t silu_global[] = {static_cast<size_t>(rows) * 64};
    constexpr size_t silu_local[] = {64};
    if (!Enqueue("ffn_silu_quantize_epilogue", dense_category, layer, silu,
                 1, silu_global, silu_local, false, nullptr, error) ||
        !MaybeCaptureTyped(
            std::string(stage_prefix).append("_post_silu_q8").c_str(), layer,
            ffn_q8_, {1, static_cast<uint32_t>(rows), 4096}, sizeof(int8_t),
            "int8", error) ||
        !MaybeCaptureTyped(
            std::string(stage_prefix).append("_post_silu_scale").c_str(),
            layer, ffn_scale_, {1, static_cast<uint32_t>(rows)},
            sizeof(float), "fp32", error) ||
        !MaybeCapture(
            std::string(stage_prefix).append("_post_silu").c_str(), layer,
            ffn_, {1, static_cast<uint32_t>(rows), 4096}, error)) {
      return false;
    }

    const char* contract_name =
        direct_rhs ? (small?"w8a8_taienc6_contract_small":"w8a8_taienc6_contract_direct")
                   : (tile?"w8a8_taienc6_contract_tile":"gemm_w8a8_taienc6_4096x1024_residual_half");
    cl_kernel contract = Kernel(contract_name);
    if (!SetArg(api_->set_arg, contract, 0, ffn_q8_) ||
        !SetArg(api_->set_arg, contract, 1, weights_) ||
        !SetArg(api_->set_arg, contract, 2, weight2) ||
        !SetArg(api_->set_arg, contract, 3, ffn_scale_) ||
        !SetArg(api_->set_arg, contract, 4, scale2) ||
        !SetArg(api_->set_arg, contract, 5, bias2) ||
        !SetArg(api_->set_arg, contract, 6, input) ||
        !SetArg(api_->set_arg, contract, 7, destination) ||
        !SetArg(api_->set_arg, contract, 8, valid_mel_frames) ||
        !SetArg(api_->set_arg, contract, 9, rows)) {
      *error = "TAIENC6 W8 FFN contract arguments failed";
      return false;
    }
    const size_t contract_global[] = {static_cast<size_t>(1024*((small||tile)?4:small_n_)/small_n_),gemm_global_y};
    return Enqueue(contract_name, dense_category, layer, contract, 2,
                   contract_global,
                   gemm_local, false, nullptr, error);
  };

  cl_mem residual = activation_a_;
  cl_mem workspace = activation_b_;
  for (int layer = 0; layer < 24; ++layer) {
    dense_layer = layer;
    dense_category = "ffn1";
    if (UseW8FfnLayer(layer)) {
      if (!ffn_w8(residual, workspace, 7, 8, layer, "ff1")) return false;
    } else {
      if (!layernorm(residual, workspace, layer, false) ||
          !MaybeCapture("norm_ff1", layer, workspace,
                        {1, static_cast<uint32_t>(rows), 1024}, error) ||
          !dense(workspace, ffn_, 7, layer, residual_placeholder, rows,
                 1024, 4096, 1, 8, 1 | 2) ||
          !MaybeCapture("ff1_post_silu", layer, ffn_,
                        {1, static_cast<uint32_t>(rows), 4096}, error) ||
          !dense(ffn_, workspace, 8, layer, residual, rows, 4096, 1024,
                 1, 8, 4)) {
        return false;
      }
    }
    if (!MaybeCapture("ff1_residual", layer, workspace,
                      {1, static_cast<uint32_t>(rows), 1024}, error)) return false;
    std::swap(residual, workspace);

    dense_category = "attention";
    if (UseW8Projection(9)) {
      if (!w8_projection(residual, qkv_, 9, layer, residual_placeholder,
                         3072, 1 | 16 | 32, "att", false)) {
        return false;
      }
    } else {
      if (!layernorm(residual, workspace, layer, false) ||
          !MaybeCapture("norm_att", layer, workspace,
                        {1, static_cast<uint32_t>(rows), 1024}, error) ||
          !dense(workspace, qkv_, 9, layer, residual_placeholder,
                 rows, 1024, 3072, 1, 8, 1 | 16)) {
        return false;
      }
    }
    if (!MaybeCapture("qkv", layer, qkv_,
                      {3, 8, static_cast<uint32_t>(rows), 128}, error)) return false;
    kernel = Kernel("attention_score");
    const uint64_t position = WeightOffset(18, layer, kFp16Raw);
    const uint64_t bias_u = WeightOffset(19, layer, kFp16Raw);
    const uint64_t bias_v = WeightOffset(20, layer, kFp16Raw);
    if (!SetArg(api_->set_arg, kernel, 0, qkv_) ||
        !SetArg(api_->set_arg, kernel, 1, weights_) ||
        !SetArg(api_->set_arg, kernel, 2, bias_u) ||
        !SetArg(api_->set_arg, kernel, 3, scores_) ||
        !SetArg(api_->set_arg, kernel, 4, rows)) {
      *error = "TAIENC6 attention score arguments failed";
      return false;
    }
    const size_t score_global[] = {RoundUp(static_cast<size_t>(rows), 16),
                                   (static_cast<size_t>(rows) + 15) / 16, 8};
    constexpr size_t score_tile[] = {16, 1, 1};
    if (!Enqueue("attention_score", "attention", layer, kernel, 3,
                  score_global, score_tile, false, nullptr, error)) {
      return false;
    }
    if (!MaybeCapture("content_score", layer, scores_,
                      {8, static_cast<uint32_t>(rows), static_cast<uint32_t>(rows)},
                      error)) return false;

    kernel = Kernel("attention_relative_fuse");
    if (!SetArg(api_->set_arg, kernel, 0, qkv_) ||
        !SetArg(api_->set_arg, kernel, 1, weights_) ||
        !SetArg(api_->set_arg, kernel, 2, position) ||
        !SetArg(api_->set_arg, kernel, 3, bias_v) ||
        !SetArg(api_->set_arg, kernel, 4, valid_mel_frames) ||
        !SetArg(api_->set_arg, kernel, 5, scores_) ||
        !SetArg(api_->set_arg, kernel, 6, rows)) {
      *error = "TAIENC6 relative attention arguments failed";
      return false;
    }
    const size_t relative_global[] = {
        RoundUp(static_cast<size_t>(rows), 8),
        RoundUp(static_cast<size_t>(rows), 8), 8};
    constexpr size_t relative_local[] = {8, 8, 1};
    if (!Enqueue("attention_relative_fuse", "attention", layer, kernel, 3,
                  relative_global, relative_local, false, nullptr, error)) {
      return false;
    }
    if (!MaybeCapture("combined_score", layer, scores_,
                      {8, static_cast<uint32_t>(rows), static_cast<uint32_t>(rows)},
                      error)) return false;

    kernel = Kernel("attention_softmax");
    if (!SetArg(api_->set_arg, kernel, 0, scores_) ||
        !SetArg(api_->set_arg, kernel, 1, valid_mel_frames) ||
        !SetArg(api_->set_arg, kernel, 2, rows)) {
      *error = "TAIENC6 attention softmax arguments failed";
      return false;
    }
    const size_t softmax_global[] = {64, static_cast<size_t>(rows), 8};
    constexpr size_t softmax_local[] = {64, 1, 1};
    if (!Enqueue("attention_softmax", "attention", layer, kernel, 3,
                  softmax_global, softmax_local, false, nullptr, error)) {
      return false;
    }
    if (!MaybeCapture("softmax", layer, scores_,
                      {8, static_cast<uint32_t>(rows), static_cast<uint32_t>(rows)},
                      error)) return false;

    kernel = Kernel("attention_value");
    if (!SetArg(api_->set_arg, kernel, 0, scores_) ||
        !SetArg(api_->set_arg, kernel, 1, qkv_) ||
        !SetArg(api_->set_arg, kernel, 2, valid_mel_frames) ||
        !SetArg(api_->set_arg, kernel, 3, attention_) ||
        !SetArg(api_->set_arg, kernel, 4, rows)) {
      *error = "TAIENC6 attention value arguments failed";
      return false;
    }
    const size_t value_global[] = {
        128, (static_cast<size_t>(rows) + 15) / 16, 8};
    constexpr size_t value_tile[] = {16, 1, 1};
    if (!Enqueue("attention_value", "attention", layer, kernel, 3,
                 value_global, value_tile, false, nullptr, error) ||
        !MaybeCapture("attention_value", layer, attention_,
                      {1, static_cast<uint32_t>(rows), 1024}, error)) {
      return false;
    }
    if (UseW8Projection(10)) {
      if (!w8_projection(attention_, workspace, 10, layer, residual,
                         1024, 4 | 32, "att_out", true)) {
        return false;
      }
    } else if (!dense(attention_, workspace, 10, layer, residual, rows,
                      1024, 1024, 1, 8, 4)) {
      return false;
    }
    if (!MaybeCapture("attention_residual", layer, workspace,
                      {1, static_cast<uint32_t>(rows), 1024}, error)) return false;
    std::swap(residual, workspace);

    dense_category = "conv";
    if (UseW8Projection(11)) {
      if (!w8_projection(residual, conv_, 11, layer, residual_placeholder,
                         2048, 1 | 32, "conv", false)) {
        return false;
      }
    } else {
      if (!layernorm(residual, workspace, layer, false) ||
          !MaybeCapture("norm_conv", layer, workspace,
                        {1, static_cast<uint32_t>(rows), 1024}, error) ||
          !dense(workspace, conv_, 11, layer,
                 residual_placeholder, rows, 1024, 2048, 1, 8, 1)) {
        return false;
      }
    }
    if (!MaybeCapture("conv_pw1", layer, conv_,
                      {1, static_cast<uint32_t>(rows), 2048}, error)) return false;
    kernel = Kernel("conv_glu");
    if (!SetArg(api_->set_arg, kernel, 0, conv_) ||
        !SetArg(api_->set_arg, kernel, 1, valid_mel_frames) ||
        !SetArg(api_->set_arg, kernel, 2, attention_) ||
        !SetArg(api_->set_arg, kernel, 3, rows)) {
      *error = "TAIENC6 conv GLU arguments failed";
      return false;
    }
    const size_t model_global[] = {128, static_cast<size_t>(rows)};
    if (!Enqueue("conv_glu", "conv", layer, kernel, 2, model_global,
                  nullptr, false, nullptr, error)) {
      return false;
    }
    if (!MaybeCapture("conv_glu", layer, attention_,
                      {1, static_cast<uint32_t>(rows), 1024}, error)) return false;

    kernel = Kernel("conv_depthwise9_silu");
    weight = WeightOffset(12, layer, kFp16Raw);
    bias = WeightOffset(12, layer, kFp16Vector);
    const uint64_t pointwise_bias_bytes =
        UseW8Projection(11) ? WeightBiasByteOffset(11, layer) : WeightBiasOffset(11, layer);
    if (pointwise_bias_bytes == std::numeric_limits<uint64_t>::max() ||
        (pointwise_bias_bytes % kHalfBytes) != 0) {
      *error = "TAIENC6 conv pointwise bias lookup failed";
      return false;
    }
    const uint64_t pointwise_bias = UseW8Projection(11) ? pointwise_bias_bytes / kHalfBytes : pointwise_bias_bytes;
    if (!SetArg(api_->set_arg, kernel, 0, attention_) ||
        !SetArg(api_->set_arg, kernel, 1, weights_) ||
        !SetArg(api_->set_arg, kernel, 2, weight) ||
        !SetArg(api_->set_arg, kernel, 3, bias) ||
        !SetArg(api_->set_arg, kernel, 4, pointwise_bias) ||
        !SetArg(api_->set_arg, kernel, 5, valid_mel_frames) ||
        !SetArg(api_->set_arg, kernel, 6, qkv_) ||
        !SetArg(api_->set_arg, kernel, 7, rows)) {
      *error = "TAIENC6 conv depthwise arguments failed";
      return false;
    }
    if (!Enqueue("conv_depthwise9_silu", "conv", layer, kernel, 2,
                 model_global, nullptr, false, nullptr, error) ||
        !MaybeCapture("conv_depthwise_silu", layer, qkv_,
                      {1, static_cast<uint32_t>(rows), 1024}, error)) {
      return false;
    }
    if (UseW8Projection(13)) {
      if (!w8_projection(qkv_, workspace, 13, layer, residual,
                         1024, 4 | 32, "conv_out", true)) {
        return false;
      }
    } else if (!dense(qkv_, workspace, 13, layer, residual, rows,
                      1024, 1024, 1, 8, 4)) {
      return false;
    }
    if (!MaybeCapture("conv_residual", layer, workspace,
                      {1, static_cast<uint32_t>(rows), 1024}, error)) return false;
    std::swap(residual, workspace);

    dense_category = "ffn2";
    if (UseW8FfnLayer(layer)) {
      if (!ffn_w8(residual, workspace, 14, 15, layer, "ff2")) return false;
    } else {
      if (!layernorm(residual, workspace, layer, false) ||
          !MaybeCapture("norm_ff2", layer, workspace,
                        {1, static_cast<uint32_t>(rows), 1024}, error) ||
          !dense(workspace, ffn_, 14, layer, residual_placeholder,
                 rows, 1024, 4096, 1, 8, 1 | 2) ||
          !MaybeCapture("ff2_post_silu", layer, ffn_,
                        {1, static_cast<uint32_t>(rows), 4096}, error) ||
          !dense(ffn_, workspace, 15, layer, residual, rows, 4096, 1024,
                 1, 8, 4)) {
        return false;
      }
    }
    if (!MaybeCapture("ff2_residual", layer, workspace,
                      {1, static_cast<uint32_t>(rows), 1024}, error)) return false;
    std::swap(residual, workspace);
    if (!layernorm(residual, workspace, layer, layer < 23)) return false;
    std::swap(residual, workspace);
    if (!MaybeCapture("norm_out", layer, residual,
                      {1, static_cast<uint32_t>(rows), 1024}, error)) return false;
  }

  dense_category = "joint";
  dense_layer = -1;
  if (!dense(residual, joint_, 21, -1, residual_placeholder, rows, 1024,
             640, 1, 8, 1)) {
    return false;
  }
  if (!MaybeCapture("joint", -1, joint_,
                    {1, static_cast<uint32_t>(rows), 640}, error)) return false;
  kernel = Kernel("output_fp16");
  if (!SetArg(api_->set_arg, kernel, 0, joint_) ||
      !SetArg(api_->set_arg, kernel, 1, valid_mel_frames) ||
      !SetArg(api_->set_arg, kernel, 2, output) ||
      !SetArg(api_->set_arg, kernel, 3, rows)) {
    *error = "TAIENC6 output arguments failed";
    return false;
  }
  const size_t output_global[] = {640, static_cast<size_t>(rows)};
  cl_event final_event = nullptr;
  if (!Enqueue("output_fp16", "joint", -1, kernel, 2, output_global,
               nullptr, true, &final_event, error) ||
      final_event == nullptr) {
    return false;
  }
  const cl_int wait_status = defer_wait_ && !diagnostic_ ? CL_SUCCESS : api_->wait_events(1, &final_event);
  if (wait_status != CL_SUCCESS) {
    if (!diagnostic_) api_->release_event(final_event);
    *error = "TAIENC6 final event wait failed";
    return false;
  }
  if (diagnostic_ &&
      !profiler_.Finish(api_->get_event_profiling_info, api_->release_event,
                        &last_profile_json_, error)) {
    return false;
  }
  if (!diagnostic_) {
    if (recording_plan_ == nullptr ||
        recording_plan_->command_count != plan.command_count) {
      api_->release_event(final_event);
      recording_plan_ = nullptr;
      *error = "TAIENC6 recorded command count does not match plan";
      return false;
    }
    if (profile_command) pending_profile_event_ = final_event;
    else api_->release_event(final_event);
    last_profile_json_ = "{}";
    recording_plan_ = nullptr;
  }
  if (!capture_stage_.empty() && !capture_written_) {
    *error = "requested teacher stage capture was not reached";
    return false;
  }
  return true;
}

bool OpenClRuntime::FinishCommandProfile(std::string* error) {
  if (!pending_profile_event_) return true;
  cl_event event = pending_profile_event_;
  pending_profile_event_ = nullptr;
  return profiler_.FinishCommandBuffer(event, command_submit_ms_, api_->get_event_profiling_info,
                                      api_->release_event, &last_profile_json_, error);
}

size_t OpenClRuntime::RowReuseY(size_t sections) const {
  size_t best=std::min<size_t>(8,sections);while(sections%best)--best;
  if (!row_reuse_pad_ || sections<=8) return best;
  size_t best_groups=sections/best,best_padded=sections;
  for(size_t y=2;y<=8;++y) {
    const size_t padded=((sections+y-1)/y)*y;
    if(padded*8>sections*9)continue; // At most 12.5% padded MMUL row sections.
    const size_t groups=padded/y;
    if(groups<best_groups || (groups==best_groups && padded<best_padded)) {
      best=y;best_groups=groups;best_padded=padded;
    }
  }
  return best;
}

}  // namespace transai::teacher::dispatch
