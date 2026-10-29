#include "src/native/runtime/opencl/command_buffer.h"

#include <dlfcn.h>
#include "mali_loader.h"

#include <stdexcept>
#include <string>

namespace transai::opencl {
namespace {

cl_platform_id FirstPlatform(void* library) {
  auto get_platform_ids = reinterpret_cast<decltype(&clGetPlatformIDs)>(
      dlsym(library, "clGetPlatformIDs"));
  cl_platform_id platform = nullptr;
  cl_uint count = 0;
  if (get_platform_ids == nullptr ||
      get_platform_ids(1, &platform, &count) != CL_SUCCESS || count == 0 ||
      platform == nullptr) {
    throw std::runtime_error("OpenCL platform unavailable for command buffer");
  }
  return platform;
}

template <typename Function, typename GetExtension>
Function LoadExtension(
    cl_platform_id platform, const char* name, GetExtension get_extension) {
  auto function = reinterpret_cast<Function>(
      get_extension(platform, name));
  if (function == nullptr) {
    throw std::runtime_error(std::string("OpenCL extension entry point missing: ") + name);
  }
  return function;
}

void Check(cl_int status, const char* operation) {
  if (status != CL_SUCCESS) {
    throw std::runtime_error(
        std::string(operation) + " failed: " + std::to_string(status));
  }
}

}  // namespace

CommandBuffer::CommandBuffer(cl_platform_id platform, cl_command_queue queue)
    : queue_(queue) {
  library_ = OpenMali();
  if (library_ == nullptr) {
    throw std::runtime_error("libOpenCL.so unavailable for command buffer");
  }
  auto get_extension = reinterpret_cast<decltype(&clGetExtensionFunctionAddressForPlatform)>(
      dlsym(library_, "clGetExtensionFunctionAddressForPlatform"));
  if (get_extension == nullptr) {
    throw std::runtime_error("clGetExtensionFunctionAddressForPlatform unavailable");
  }
  auto create = LoadExtension<clCreateCommandBufferKHR_fn>(
      platform, "clCreateCommandBufferKHR", get_extension);
  command_ndrange_ = LoadExtension<clCommandNDRangeKernelKHR_fn>(
      platform, "clCommandNDRangeKernelKHR", get_extension);
  finalize_ = LoadExtension<clFinalizeCommandBufferKHR_fn>(
      platform, "clFinalizeCommandBufferKHR", get_extension);
  enqueue_ = LoadExtension<clEnqueueCommandBufferKHR_fn>(
      platform, "clEnqueueCommandBufferKHR", get_extension);
  release_ = LoadExtension<clReleaseCommandBufferKHR_fn>(
      platform, "clReleaseCommandBufferKHR", get_extension);
  cl_int status = CL_SUCCESS;
  command_buffer_ = create(1, &queue_, nullptr, &status);
  Check(status, "clCreateCommandBufferKHR");
}

CommandBuffer::CommandBuffer(cl_command_queue queue) : queue_(queue) {
  library_ = OpenMali();
  if (library_ == nullptr) {
    throw std::runtime_error("libOpenCL.so unavailable for command buffer");
  }
  const cl_platform_id platform = FirstPlatform(library_);
  auto get_extension =
      reinterpret_cast<decltype(&clGetExtensionFunctionAddressForPlatform)>(
          dlsym(library_, "clGetExtensionFunctionAddressForPlatform"));
  if (get_extension == nullptr) {
    throw std::runtime_error(
        "clGetExtensionFunctionAddressForPlatform unavailable");
  }
  auto create = LoadExtension<clCreateCommandBufferKHR_fn>(
      platform, "clCreateCommandBufferKHR", get_extension);
  command_ndrange_ = LoadExtension<clCommandNDRangeKernelKHR_fn>(
      platform, "clCommandNDRangeKernelKHR", get_extension);
  finalize_ = LoadExtension<clFinalizeCommandBufferKHR_fn>(
      platform, "clFinalizeCommandBufferKHR", get_extension);
  enqueue_ = LoadExtension<clEnqueueCommandBufferKHR_fn>(
      platform, "clEnqueueCommandBufferKHR", get_extension);
  release_ = LoadExtension<clReleaseCommandBufferKHR_fn>(
      platform, "clReleaseCommandBufferKHR", get_extension);
  cl_int status = CL_SUCCESS;
  command_buffer_ = create(1, &queue_, nullptr, &status);
  Check(status, "clCreateCommandBufferKHR");
}

CommandBuffer::~CommandBuffer() {
  if (command_buffer_ != nullptr && release_ != nullptr) {
    release_(command_buffer_);
  }
  if (library_ != nullptr) dlclose(library_);
}

void CommandBuffer::Add(const CommandDispatch& dispatch) {
  cl_sync_point_khr next_sync_point = 0;
  Check(
      command_ndrange_(
          command_buffer_, nullptr, nullptr, dispatch.kernel,
          dispatch.dimensions, nullptr, dispatch.global,
          dispatch.explicit_local ? dispatch.local : nullptr,
          has_sync_point_ ? 1U : 0U,
          has_sync_point_ ? &last_sync_point_ : nullptr,
          &next_sync_point, nullptr),
      "clCommandNDRangeKernelKHR");
  last_sync_point_ = next_sync_point;
  has_sync_point_ = true;
}

void CommandBuffer::Finalize() {
  Check(finalize_(command_buffer_), "clFinalizeCommandBufferKHR");
}

cl_event CommandBuffer::Enqueue() {
  cl_event event = nullptr;
  Check(enqueue_(0, nullptr, command_buffer_, 0, nullptr, &event),
        "clEnqueueCommandBufferKHR");
  return event;
}

}  // namespace transai::opencl
