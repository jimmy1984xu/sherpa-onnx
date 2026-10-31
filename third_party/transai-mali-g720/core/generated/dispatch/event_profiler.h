// Copyright 2026 TransAI
// SPDX-License-Identifier: Apache-2.0

#ifndef TRANSAI_TEACHER_DISPATCH_EVENT_PROFILER_H_
#define TRANSAI_TEACHER_DISPATCH_EVENT_PROFILER_H_

#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>

#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

#include "src/native/runtime/opencl/event_profiler.h"

namespace transai::teacher::dispatch {

struct KernelTrace {
  std::string name;
  std::string category;
  int layer = -1;
  cl_event event = nullptr;
};

class EventProfiler {
 public:
  using Clock = std::chrono::steady_clock;

  EventProfiler() = default;
  EventProfiler(const EventProfiler&) = delete;
  EventProfiler& operator=(const EventProfiler&) = delete;

  void Begin(uint32_t valid_mel_frames, uint32_t plan);
  void Add(std::string name, std::string category, int layer, cl_event event);
  bool Finish(decltype(&clGetEventProfilingInfo) get_event_profiling_info,
              decltype(&clReleaseEvent) release_event,
              std::string* output_json, std::string* error);
  bool FinishCommandBuffer(
      cl_event event, double host_submit_ms,
      decltype(&clGetEventProfilingInfo) get_event_profiling_info,
      decltype(&clReleaseEvent) release_event, std::string* output_json,
      std::string* error);
  void Release(decltype(&clReleaseEvent) release_event);

 private:
  uint32_t valid_mel_frames_ = 0;
  uint32_t plan_ = 0;
  Clock::time_point begin_{};
  std::vector<KernelTrace> traces_;
};

}  // namespace transai::teacher::dispatch

#endif  // TRANSAI_TEACHER_DISPATCH_EVENT_PROFILER_H_
