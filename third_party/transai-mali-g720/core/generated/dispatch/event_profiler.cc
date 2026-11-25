// Copyright 2026 TransAI
// SPDX-License-Identifier: Apache-2.0

#include "dispatch/event_profiler.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <iomanip>
#include <map>
#include <sstream>
#include <utility>

namespace transai::teacher::dispatch {
namespace {

double Milliseconds(EventProfiler::Clock::time_point begin,
                    EventProfiler::Clock::time_point end) {
  return std::chrono::duration<double, std::milli>(end - begin).count();
}

std::string EscapeJson(const std::string& value) {
  std::string output;
  output.reserve(value.size());
  for (char character : value) {
    if (character == '"' || character == '\\') output.push_back('\\');
    output.push_back(character);
  }
  return output;
}

}  // namespace

void EventProfiler::Begin(uint32_t valid_mel_frames, uint32_t plan) {
  valid_mel_frames_ = valid_mel_frames;
  plan_ = plan;
  begin_ = Clock::now();
  traces_.clear();
}

void EventProfiler::Add(std::string name, std::string category, int layer,
                        cl_event event) {
  traces_.push_back(
      {std::move(name), std::move(category), layer, event});
}

bool EventProfiler::FinishCommandBuffer(
    cl_event event, double host_submit_ms,
    decltype(&clGetEventProfilingInfo) get_event_profiling_info,
    decltype(&clReleaseEvent) release_event, std::string* output_json,
    std::string* error) {
  if (event == nullptr || get_event_profiling_info == nullptr ||
      release_event == nullptr || output_json == nullptr || error == nullptr) {
    if (error != nullptr) *error = "OpenCL command-buffer profile is incomplete";
    return false;
  }
  try {
    const auto profile = transai::opencl::ReadEventProfile(
        "teacher_command_buffer", event, get_event_profiling_info);
    const double host_invoke_ms = Milliseconds(begin_, Clock::now());
    std::ostringstream output;
    output << std::fixed << std::setprecision(6)
           << "{\"valid_mel_frames\":" << valid_mel_frames_
           << ",\"required_encoder_frames\":"
           << ((valid_mel_frames_ + 7) / 8) << ",\"selected_plan\":" << plan_
           << ",\"command_buffer\":true,\"command_enqueue_count\":1"
           << ",\"host_invoke_ms\":" << host_invoke_ms
           << ",\"device_span_ms\":" << profile.device_ms()
           << ",\"host_submit_ms\":" << host_submit_ms
           << ",\"host_non_gpu_ms\":"
           << std::max(0.0, host_invoke_ms - profile.device_ms())
           << ",\"accounting_error_ratio\":0.0,\"categories\":{}"
           << ",\"layers\":[],\"kernels\":[{\"name\":\"teacher_command_buffer\""
           << ",\"category\":\"encoder\",\"layer\":-1"
           << ",\"queued_ns\":" << profile.queued_ns
           << ",\"submit_ns\":" << profile.submit_ns
           << ",\"start_ns\":" << profile.start_ns
           << ",\"end_ns\":" << profile.end_ns
           << ",\"device_ms\":" << profile.device_ms()
           << ",\"submit_delay_ms\":" << profile.submit_delay_ms()
           << ",\"queue_delay_ms\":" << profile.queue_delay_ms() << "}]}";
    *output_json = output.str();
  } catch (const std::exception& exception) {
    *error = exception.what();
    release_event(event);
    return false;
  }
  release_event(event);
  return true;
}

bool EventProfiler::Finish(
    decltype(&clGetEventProfilingInfo) get_event_profiling_info,
    decltype(&clReleaseEvent) release_event, std::string* output_json,
    std::string* error) {
  if (get_event_profiling_info == nullptr || release_event == nullptr ||
      output_json == nullptr || error == nullptr || traces_.empty()) {
    if (error != nullptr) *error = "OpenCL diagnostic profile is incomplete";
    Release(release_event);
    return false;
  }
  std::vector<transai::opencl::EventProfile> profiles;
  profiles.reserve(traces_.size());
  try {
    for (const auto& trace : traces_) {
      profiles.push_back(transai::opencl::ReadEventProfile(
          trace.name, trace.event, get_event_profiling_info));
    }
  } catch (const std::exception& exception) {
    *error = exception.what();
    Release(release_event);
    return false;
  }
  const auto finish = Clock::now();
  const uint64_t first_start = profiles.front().start_ns;
  const uint64_t last_end = profiles.back().end_ns;
  const double device_span_ms =
      last_end >= first_start
          ? static_cast<double>(last_end - first_start) / 1.0e6
          : 0.0;
  const double host_invoke_ms = Milliseconds(begin_, finish);
  std::map<std::string, double> categories;
  std::map<int, double> layers;
  for (size_t index = 0; index < profiles.size(); ++index) {
    categories[traces_[index].category] += profiles[index].device_ms();
    if (traces_[index].layer >= 0) {
      layers[traces_[index].layer] += profiles[index].device_ms();
    }
  }

  std::ostringstream output;
  output << std::fixed << std::setprecision(6)
         << "{\"valid_mel_frames\":" << valid_mel_frames_
         << ",\"required_encoder_frames\":"
         << ((valid_mel_frames_ + 7) / 8) << ",\"selected_plan\":" << plan_
         << ",\"host_invoke_ms\":" << host_invoke_ms
         << ",\"device_span_ms\":" << device_span_ms
         << ",\"host_submit_ms\":"
         << (profiles.back().submit_ns >= profiles.front().queued_ns
                 ? static_cast<double>(profiles.back().submit_ns -
                                       profiles.front().queued_ns) /
                       1.0e6
                 : 0.0)
         << ",\"host_non_gpu_ms\":"
         << std::max(0.0, host_invoke_ms - device_span_ms)
         << ",\"accounting_error_ratio\":"
         << (host_invoke_ms > 0.0
                 ? std::abs(host_invoke_ms - device_span_ms -
                            std::max(0.0, host_invoke_ms - device_span_ms)) /
                       host_invoke_ms
                 : 0.0)
         << ",\"categories\":{";
  bool first = true;
  for (const auto& [name, duration] : categories) {
    if (!first) output << ',';
    first = false;
    output << '"' << EscapeJson(name) << "\":" << duration;
  }
  output << "},\"layers\":[";
  first = true;
  for (const auto& [layer, duration] : layers) {
    if (!first) output << ',';
    first = false;
    output << "{\"layer\":" << layer << ",\"device_ms\":" << duration
           << '}';
  }
  output << "],\"kernels\":[";
  for (size_t index = 0; index < profiles.size(); ++index) {
    if (index != 0) output << ',';
    const auto& profile = profiles[index];
    output << "{\"name\":\"" << EscapeJson(traces_[index].name)
           << "\",\"category\":\"" << EscapeJson(traces_[index].category)
           << "\",\"layer\":" << traces_[index].layer
           << ",\"queued_ns\":" << profile.queued_ns
           << ",\"submit_ns\":" << profile.submit_ns
           << ",\"start_ns\":" << profile.start_ns
           << ",\"end_ns\":" << profile.end_ns
           << ",\"device_ms\":" << profile.device_ms()
           << ",\"submit_delay_ms\":" << profile.submit_delay_ms()
           << ",\"queue_delay_ms\":" << profile.queue_delay_ms() << '}';
  }
  output << "]}";
  *output_json = output.str();
  Release(release_event);
  return true;
}

void EventProfiler::Release(decltype(&clReleaseEvent) release_event) {
  if (release_event != nullptr) {
    for (auto& trace : traces_) {
      if (trace.event != nullptr) release_event(trace.event);
      trace.event = nullptr;
    }
  }
  traces_.clear();
}

}  // namespace transai::teacher::dispatch
