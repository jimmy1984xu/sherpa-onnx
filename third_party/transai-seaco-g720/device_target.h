#pragma once
#include <CL/cl.h>
#include <CL/cl_ext.h>
#include <nlohmann/json.hpp>
#include <dlfcn.h>
#include <string>
inline nlohmann::json SeacoDeviceTarget(void*lib,cl_device_id dev){
 auto info=reinterpret_cast<decltype(&::clGetDeviceInfo)>(dlsym(lib,"clGetDeviceInfo"));
 auto text=[&](cl_device_info key){size_t n=0;if(info(dev,key,0,nullptr,&n)!=CL_SUCCESS)throw std::runtime_error("GPU identity");std::string s(n,'\0');if(info(dev,key,n,s.data(),nullptr)!=CL_SUCCESS)throw std::runtime_error("GPU identity");while(!s.empty()&&!s.back())s.pop_back();return s;};
 auto uuid=[&](cl_device_info key){unsigned char x[16];if(info(dev,key,sizeof(x),x,nullptr)!=CL_SUCCESS)return std::string();std::string s;const char*h="0123456789abcdef";for(auto b:x){s+=h[b>>4];s+=h[b&15];}return s;};
 return {{"name",text(CL_DEVICE_NAME)},{"vendor",text(CL_DEVICE_VENDOR)},{"driver",text(CL_DRIVER_VERSION)},
  {"device_version",text(CL_DEVICE_VERSION)},{"opencl_c",text(CL_DEVICE_OPENCL_C_VERSION)},
  {"device_uuid",uuid(CL_DEVICE_UUID_KHR)},{"driver_uuid",uuid(CL_DRIVER_UUID_KHR)}};
}
