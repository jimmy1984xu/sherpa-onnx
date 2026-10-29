#pragma once
#include <dlfcn.h>
#include <stdexcept>
#include <string>
// Shell retains the original SP-HAL path. Normal apps use the OEM public OpenCL
// library declared through <uses-native-library android:name="libOpenCL.so"/>.
// Neither path permits a CPU ASR fallback or copies the vendor driver.
inline void* OpenMali() {
  void* support=dlopen("libvndksupport.so",RTLD_NOW|RTLD_LOCAL);
  using Load=void*(*)(const char*,int);
  auto load=support?reinterpret_cast<Load>(dlsym(support,"android_load_sphal_library")):nullptr;
  const int flags=RTLD_NOW|RTLD_LOCAL|RTLD_NODELETE;
  void* driver=load?load("/vendor/lib64/egl/mt6989/libGLES_mali.so",flags):nullptr;
  if(support)dlclose(support);
  if(!driver)driver=dlopen("libOpenCL.so",flags);
  if(!driver){const char* error=dlerror();throw std::runtime_error(std::string("Mali SP-HAL/public OpenCL unavailable: ")+(error?error:"loader error"));}
  return driver;
}
