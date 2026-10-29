#pragma once

#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <CL/cl_ext.h>

#include <cstddef>

namespace transai::opencl {

struct CommandDispatch {
  cl_kernel kernel = nullptr;
  cl_uint dimensions = 1;
  size_t global[3] = {1, 1, 1};
  size_t local[3] = {1, 1, 1};
  bool explicit_local = true;
};

class CommandBuffer {
 public:
  CommandBuffer(cl_platform_id platform, cl_command_queue queue);
  explicit CommandBuffer(cl_command_queue queue);
  ~CommandBuffer();

  CommandBuffer(const CommandBuffer&) = delete;
  CommandBuffer& operator=(const CommandBuffer&) = delete;

  void Add(const CommandDispatch& dispatch);
  void Finalize();
  cl_event Enqueue();

 private:
  void* library_ = nullptr;
  cl_command_queue queue_ = nullptr;
  cl_command_buffer_khr command_buffer_ = nullptr;
  clCommandNDRangeKernelKHR_fn command_ndrange_ = nullptr;
  clFinalizeCommandBufferKHR_fn finalize_ = nullptr;
  clEnqueueCommandBufferKHR_fn enqueue_ = nullptr;
  clReleaseCommandBufferKHR_fn release_ = nullptr;
  cl_sync_point_khr last_sync_point_ = 0;
  bool has_sync_point_ = false;
};

}  // namespace transai::opencl
