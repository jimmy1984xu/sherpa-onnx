#include "gpu-abi.h"
#include "runner.cc"
#include <mutex>

namespace {
void Error(char* target,size_t capacity,const char* message) {
  if(target&&capacity){std::snprintf(target,capacity,"%s",message);}
}
void Defaults() {
  // Device tuning only. No governor/frequency/thermal writes or app affinity changes.
  static std::once_flag once;
  std::call_once(once,[]{
    const std::pair<const char*,const char*> values[]={
      {"TRANSAI_G720_SMALL_M","4"},{"TRANSAI_G720_SMALL_N","4"},
      {"TRANSAI_G720_DENSE_TILE_MAX_M","32"},{"TRANSAI_G720_DENSE_LY","2"},
      {"TRANSAI_G720_BROADCAST_Y","16"},{"TRANSAI_G720_FFN_BROADCAST_TILE","1"},
      {"TRANSAI_G720_PROJECTION_TILE","1"},{"TRANSAI_G720_PROJECTION_Y","16"},
      {"TRANSAI_G720_SHAPE_TILE_LIMIT","112"},{"TRANSAI_G720_FFN_DIRECT_LIMIT","32"},
      {"TRANSAI_G720_ADAPTIVE_TILE_N","1"},{"TRANSAI_G720_ROW_REUSE_PAD","1"},
      {"TRANSAI_G720_PROJECTION_DIRECT_LIMIT","32"},{"TRANSAI_G720_TINY_LIMIT","64"},
      {"TRANSAI_G720_BCAST_K","256"},{"TRANSAI_G720_ALT_BCAST_K","64"},
      {"TRANSAI_TDT_WG","512"},{"TRANSAI_TDT_MULTI","1"},
      {"TRANSAI_TDT_BLOCK","8"},{"TRANSAI_TDT_BUDGET","50"}
    };
    for(auto [name,value]:values)if(!getenv(name))setenv(name,value,0);
  });
}
struct Session {
  Gpu gpu;
  std::mutex mutex;
  std::vector<int16_t> input;
  Session(const char* encoder,const char* tdt):gpu(encoder,tdt){input.reserve(480000);}
};
}
#define API extern "C" __attribute__((visibility("default")))
API int transai_parakeet_gpu_create(const char* encoder,const char* tdt,
                                  void** output,char* error,size_t capacity) {
  if(output)*output=nullptr;
  try {
    if(!encoder||!tdt||!output)throw std::runtime_error("null create argument");
    Defaults();*output=new Session(encoder,tdt);Error(error,capacity,"");return 0;
  }catch(const std::exception& e){Error(error,capacity,e.what());return -1;}
}
#define MALI_G720_TRACE_LOGITS_BYTES 4100
#include "../scored-sdk.inc"
