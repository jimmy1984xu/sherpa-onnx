#include "src/native/runtime/opencl/event_profiler.h"

#include <stdexcept>

namespace transai::opencl {
namespace {

uint64_t ReadTimestamp(
    cl_event event,
    cl_profiling_info field,
    GetEventProfilingInfoFn get_event_profiling_info) {
  cl_ulong value = 0;
  const cl_int status = get_event_profiling_info(
      event, field, sizeof(value), &value, nullptr);
  if (status != CL_SUCCESS) {
    throw std::runtime_error(
        "clGetEventProfilingInfo failed: " + std::to_string(status));
  }
  return static_cast<uint64_t>(value);
}

double Milliseconds(uint64_t begin, uint64_t end) {
  return end >= begin ? static_cast<double>(end - begin) / 1.0e6 : 0.0;
}

}  // namespace

double EventProfile::device_ms() const { return Milliseconds(start_ns, end_ns); }

double EventProfile::submit_delay_ms() const {
  return Milliseconds(queued_ns, submit_ns);
}

double EventProfile::queue_delay_ms() const {
  return Milliseconds(submit_ns, start_ns);
}

EventProfile ReadEventProfile(
    const std::string& name,
    cl_event event,
    GetEventProfilingInfoFn get_event_profiling_info) {
  if (event == nullptr || get_event_profiling_info == nullptr) {
    throw std::runtime_error("OpenCL profiling event is unavailable");
  }
  return EventProfile{
      name,
      ReadTimestamp(event, CL_PROFILING_COMMAND_QUEUED, get_event_profiling_info),
      ReadTimestamp(event, CL_PROFILING_COMMAND_SUBMIT, get_event_profiling_info),
      ReadTimestamp(event, CL_PROFILING_COMMAND_START, get_event_profiling_info),
      ReadTimestamp(event, CL_PROFILING_COMMAND_END, get_event_profiling_info),
  };
}

}  // namespace transai::opencl
