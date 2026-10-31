// Copyright 2026 TransAI
// SPDX-License-Identifier: Apache-2.0

#ifndef TRANSAI_TEACHER_OPENCL_RUNTIME_H_
#define TRANSAI_TEACHER_OPENCL_RUNTIME_H_

#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>

#include <array>
#include <chrono>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "dispatch/bytecode_reader.h"
#include "dispatch/event_profiler.h"
#include "src/native/runtime/opencl/command_buffer.h"

namespace transai::teacher::dispatch {

class OpenClRuntime {
 public:
  static std::unique_ptr<OpenClRuntime> Create(cl_context context,
                                               cl_command_queue queue,
                                               const BytecodeReader& bytecode,
                                               const std::string& weights_path, bool diagnostic, bool nemo_padding,
                                               std::string* error);
  ~OpenClRuntime();

  OpenClRuntime(const OpenClRuntime&) = delete;
  OpenClRuntime& operator=(const OpenClRuntime&) = delete;

  bool Invoke(cl_mem mel, cl_mem valid_mel_frames, cl_mem output,
              uint32_t valid_mel_frames_value, const PlanRecord& plan,
              std::string* error);
  void SetDeferredWait(bool value) { defer_wait_ = value; }
  bool FinishCommandProfile(std::string* error);
  int active_tile_m() const { return small_m_; }
  int active_tile_n() const { return small_n_; }
  const std::string& last_profile_json() const { return last_profile_json_; }

 private:
  OpenClRuntime() = default;

  struct Api;
  struct CachedPlan {
    uint32_t encoder_frames = 0;
    cl_mem mel = nullptr;
    cl_mem length = nullptr;
    cl_mem output = nullptr;
    size_t command_count = 0;
    std::unique_ptr<transai::opencl::CommandBuffer> commands;
  };
  bool CreateKernels(std::string* error);
  bool AllocateWorkspace(std::string* error);
  bool Enqueue(const char* name, const char* category, int layer,
               cl_kernel kernel, cl_uint dimensions, const size_t* global,
               const size_t* local, bool final, cl_event* event,
               std::string* error);
  bool LoadCaptureRequest(std::string* error);
  bool MaybeCapture(const char* stage, int layer, cl_mem buffer,
                    std::vector<uint32_t> shape, std::string* error);
  bool MaybeCaptureTyped(const char* stage, int layer, cl_mem buffer,
                         std::vector<uint32_t> shape, size_t element_bytes,
                         const char* dtype, std::string* error);
  uint64_t WeightOffset(uint32_t semantic, int32_t layer,
                        uint32_t storage) const;
  uint64_t WeightBiasOffset(uint32_t semantic, int32_t layer) const;
  uint64_t WeightScaleByteOffset(uint32_t semantic, int32_t layer) const;
  uint64_t WeightBiasByteOffset(uint32_t semantic, int32_t layer) const;
  bool UseW8FfnLayer(int layer) const;
  bool UseW8Projection(uint32_t semantic) const;
  cl_kernel Kernel(const char* name) const;

  std::unique_ptr<Api> api_;
  cl_context context_ = nullptr;
  cl_command_queue queue_ = nullptr;
  cl_program program_ = nullptr;
  cl_program alternate_program_ = nullptr;
  std::unordered_map<std::string, cl_kernel> alternate_kernels_;
  int shape_tile_limit_ = 0;
  bool use_alternate_tile_ = false;
  int alternate_m_ = 8, alternate_n_ = 4;
  cl_program fallback_program_ = nullptr;
  std::unordered_map<std::string, cl_kernel> fallback_kernels_;
  bool adaptive_tile_n_ = false, use_fallback_tile_ = false;
  bool row_reuse_pad_ = false;
  size_t RowReuseY(size_t sections) const;
  cl_program tiny_program_ = nullptr;
  std::unordered_map<std::string, cl_kernel> tiny_kernels_;
  int tiny_limit_ = 0;
  bool use_tiny_tile_ = false;
  bool diagnostic_ = false;
  int ffn_direct_limit_ = 40;
  int projection_direct_limit_ = 40;
  bool defer_wait_ = false;
  int dense_lx_=32, dense_ly_=2, broadcast_y_=4;
  int small_m_=4,small_n_=4,dense_tile_max_m_=32,bcast_k_=64,wbsm_=-1,projection_y_=4;
  bool small_tile_=false,ffn_broadcast_tile_=false,wbsm_dense_=false,projection_tile_=false;
  EventProfiler profiler_;
  std::string last_profile_json_ = "{}";
  std::unordered_map<std::string, cl_kernel> kernels_;
  std::unordered_map<uint64_t, uint64_t> weight_offsets_;
  std::unordered_map<uint64_t, uint64_t> weight_bias_offsets_;
  std::unordered_map<uint64_t, uint64_t> weight_scale_byte_offsets_;
  std::unordered_map<uint64_t, uint64_t> weight_bias_byte_offsets_;
  std::vector<CachedPlan> cached_plans_;
  CachedPlan* recording_plan_ = nullptr;
  double command_submit_ms_ = 0.0;
  cl_event pending_profile_event_ = nullptr;
  std::string capture_stage_;
  std::string capture_path_;
  int capture_layer_ = -1;
  uint32_t capture_valid_mel_frames_ = 0;
  uint32_t capture_plan_ = 0;
  bool capture_written_ = false;
  std::array<bool, 24> w8_ffn_layers_{};
  bool has_any_w8_ffn_ = false;
  bool w8_qkv_ = false;
  bool w8_conv_pw1_ = false;
  bool w8_attention_out_ = false;
  bool w8_conv_pw2_ = false;
  bool has_any_w8_projection_ = false;
  bool has_fp16_quantized_projection_ = false;
  bool capture_force_fp16_ = false;
  cl_event previous_diagnostic_event_ = nullptr;
  cl_mem weights_ = nullptr;
  cl_mem pre0_ = nullptr;
  cl_mem pre1a_ = nullptr;
  cl_mem pre1b_ = nullptr;
  cl_mem pre2a_ = nullptr;
  cl_mem pre2b_ = nullptr;
  cl_mem pre_flat_ = nullptr;
  cl_mem activation_a_ = nullptr;
  cl_mem activation_b_ = nullptr;
  cl_mem ffn_ = nullptr;
  cl_mem ffn_q8_ = nullptr;
  cl_mem ffn_scale_ = nullptr;
  cl_mem qkv_ = nullptr;
  cl_mem scores_ = nullptr;
  cl_mem attention_ = nullptr;
  cl_mem conv_ = nullptr;
  cl_mem joint_ = nullptr;
  int weights_fd_ = -1;
};

}  // namespace transai::teacher::dispatch

#endif  // TRANSAI_TEACHER_OPENCL_RUNTIME_H_
