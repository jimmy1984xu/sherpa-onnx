#pragma once

#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>

#include <cstdint>
#include <string>

namespace transai::opencl {

struct EventProfile {
  std::string name;
  uint64_t queued_ns = 0;
  uint64_t submit_ns = 0;
  uint64_t start_ns = 0;
  uint64_t end_ns = 0;

  double device_ms() const;
  double submit_delay_ms() const;
  double queue_delay_ms() const;
};

using GetEventProfilingInfoFn = decltype(&clGetEventProfilingInfo);

EventProfile ReadEventProfile(
    const std::string& name,
    cl_event event,
    GetEventProfilingInfoFn get_event_profiling_info);

}  // namespace transai::opencl
