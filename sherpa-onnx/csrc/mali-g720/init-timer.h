#pragma once
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>
namespace sherpa_onnx {
class MaliG720InitTimer {
 public:
  explicit MaliG720InitTimer(std::string name) : name_(std::move(name)), enabled_(std::getenv("TRANSAI_INIT_PROFILE") != nullptr) { if (enabled_) start_ = Clock::now(); }
  void Mark(const std::string& stage) {
    if (!enabled_) return;
    const auto now=Clock::now();
    std::cerr << "INIT_STAGE name=" << name_ << "." << stage << " ms=" << std::chrono::duration<double,std::milli>(now-start_).count() << std::endl;
    start_=Clock::now();
  }
 private:
  using Clock=std::chrono::steady_clock;
  std::string name_; bool enabled_; Clock::time_point start_;
};
}
