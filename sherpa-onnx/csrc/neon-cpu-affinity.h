#ifndef SHERPA_ONNX_CSRC_NEON_CPU_AFFINITY_H_
#define SHERPA_ONNX_CSRC_NEON_CPU_AFFINITY_H_

#include <array>
#include <cstdint>

#include "sherpa-onnx/csrc/neon-threads.h"

#if defined(__ANDROID__) && defined(__aarch64__)
#include <pthread.h>
#include <sched.h>

#include <memory>
#include <new>
#include <vector>

#include "onnxruntime_c_api.h"
#endif

namespace sherpa_onnx::neon {

struct CpuAffinity {
  std::array<int32_t, 8> cores{};
  int32_t count = 0;

  static CpuAffinity Resolve(const int32_t *requested, int32_t count,
                             int32_t model_threads) {
    CpuAffinity result;
#if defined(__ANDROID__) && defined(__aarch64__)
    if (!requested || !ValidNumThreads(count) || count == 0 ||
        count != ResolveNumThreads(model_threads)) {
      return result;
    }
    cpu_set_t available, selected;
    if (sched_getaffinity(0, sizeof(available), &available) != 0) return result;
    CPU_ZERO(&selected);
    for (int32_t i = 0; i < count; ++i) {
      const int32_t core = requested[i];
      if (core < 0 || core >= CPU_SETSIZE || !CPU_ISSET(core, &available) ||
          CPU_ISSET(core, &selected)) {
        return {};
      }
      CPU_SET(core, &selected);
      result.cores[i] = core;
    }
    result.count = count;
#else
    (void)requested;
    (void)count;
    (void)model_threads;
#endif
    return result;
  }
};

class ScopedCpuAffinityRequest {
 public:
  explicit ScopedCpuAffinityRequest(CpuAffinity &affinity)
      : affinity_(affinity), previous_(current_) {
    current_ = this;
  }
  ~ScopedCpuAffinityRequest() { current_ = previous_; }
  ScopedCpuAffinityRequest(const ScopedCpuAffinityRequest &) = delete;
  ScopedCpuAffinityRequest &operator=(const ScopedCpuAffinityRequest &) = delete;

  template <typename SessionOptions>
  static void ConfigureWorkers(SessionOptions &options) {
#if defined(__ANDROID__) && defined(__aarch64__)
    if (!current_ || current_->affinity_.count < 2) return;
    current_->next_core_ = 1;
    // Android ORT does not apply session.intra_op_thread_affinities to its
    // workers. Capture the actual threads through its session callbacks.
    options.SetCustomCreateThreadFn(CreateWorker)
        .SetCustomJoinThreadFn(JoinWorker);
#else
    (void)options;
#endif
  }

  void ApplyWorkers() {
#if defined(__ANDROID__) && defined(__aarch64__)
    if (!affinity_.count) return;
    if (affinity_.count > 1 && (workers_.empty() || next_core_ != 1)) {
      affinity_.count = 0;
      return;
    }
    cpu_set_t caller;
    if (sched_getaffinity(0, sizeof(caller), &caller) != 0 ||
        !CPU_ISSET(affinity_.cores[0], &caller)) {
      affinity_.count = 0;
      return;
    }
    for (auto &worker : workers_) {
      if (sched_getaffinity(worker.tid, sizeof(worker.original),
                            &worker.original) != 0 ||
          !CPU_ISSET(worker.core, &worker.original)) {
        affinity_.count = 0;
        return;
      }
    }
    for (auto &worker : workers_) {
      cpu_set_t selected;
      CPU_ZERO(&selected);
      CPU_SET(worker.core, &selected);
      if (sched_setaffinity(worker.tid, sizeof(selected), &selected) != 0) {
        for (const auto &applied : workers_) {
          if (&applied == &worker) break;
          sched_setaffinity(applied.tid, sizeof(applied.original),
                            &applied.original);
        }
        affinity_.count = 0;
        return;
      }
    }
#endif
  }

 private:
  CpuAffinity &affinity_;
  ScopedCpuAffinityRequest *previous_;
  inline static thread_local ScopedCpuAffinityRequest *current_ = nullptr;

#if defined(__ANDROID__) && defined(__aarch64__)
  struct Worker {
    OrtThreadWorkerFn run;
    void *arg;
    pthread_t thread{};
  };

  struct WorkerAffinity {
    pid_t tid;
    int32_t core;
    cpu_set_t original{};
  };

  static void *RunWorker(void *arg) {
    auto *worker = static_cast<Worker *>(arg);
    worker->run(worker->arg);
    return nullptr;
  }

  static OrtCustomThreadHandle CreateWorker(void *, OrtThreadWorkerFn run,
                                            void *arg) {
    std::unique_ptr<Worker> worker(new (std::nothrow) Worker{run, arg});
    if (!worker ||
        pthread_create(&worker->thread, nullptr, RunWorker, worker.get()) != 0) {
      return nullptr;
    }
    if (current_) current_->RememberWorker(pthread_gettid_np(worker->thread));
    return reinterpret_cast<OrtCustomThreadHandle>(worker.release());
  }

  static void JoinWorker(OrtCustomThreadHandle handle) {
    auto *worker = reinterpret_cast<Worker *>(
        const_cast<OrtCustomHandleType *>(handle));
    pthread_join(worker->thread, nullptr);
    delete worker;
  }

  void RememberWorker(pid_t tid) {
    if (!affinity_.count) return;
    if (tid <= 0) {
      affinity_.count = 0;
      return;
    }
    try {
      workers_.push_back({tid, affinity_.cores[next_core_]});
      // A pair of ORT sessions can reuse the same SessionOptions.
      if (++next_core_ == affinity_.count) next_core_ = 1;
    } catch (const std::bad_alloc &) {
      affinity_.count = 0;
    }
  }

  std::vector<WorkerAffinity> workers_;
  int32_t next_core_ = 1;
#endif
};

class ScopedInferenceCore {
 public:
  explicit ScopedInferenceCore(const CpuAffinity &affinity) {
#if defined(__ANDROID__) && defined(__aarch64__)
    if (affinity.count == 0 ||
        sched_getaffinity(0, sizeof(previous_), &previous_) != 0 ||
        !CPU_ISSET(affinity.cores[0], &previous_)) {
      return;
    }
    cpu_set_t selected;
    CPU_ZERO(&selected);
    CPU_SET(affinity.cores[0], &selected);
    restore_ = !CPU_EQUAL(&selected, &previous_) &&
               sched_setaffinity(0, sizeof(selected), &selected) == 0;
#else
    (void)affinity;
#endif
  }
  ~ScopedInferenceCore() {
#if defined(__ANDROID__) && defined(__aarch64__)
    if (restore_) sched_setaffinity(0, sizeof(previous_), &previous_);
#endif
  }
  ScopedInferenceCore(const ScopedInferenceCore &) = delete;
  ScopedInferenceCore &operator=(const ScopedInferenceCore &) = delete;

 private:
#if defined(__ANDROID__) && defined(__aarch64__)
  cpu_set_t previous_{};
  bool restore_ = false;
#endif
};

}  // namespace sherpa_onnx::neon

#endif  // SHERPA_ONNX_CSRC_NEON_CPU_AFFINITY_H_
