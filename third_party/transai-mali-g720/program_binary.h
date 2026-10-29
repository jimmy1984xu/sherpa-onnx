#pragma once
#include <CL/cl.h>
#include <CL/cl_ext.h>
#include <dlfcn.h>
#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include "nlohmann/json.hpp"
#include "sherpa-onnx/csrc/mali-g720/sha256.h"

namespace transai::aot {
using Json=nlohmann::json;
inline void Require(bool value,const std::string& message) {
  if(!value)throw std::runtime_error("MaliG720 AOT: "+message);
}
template<class T> T Entry(void* library,const char* name) {
  auto value=reinterpret_cast<T>(dlsym(library,name));Require(value!=nullptr,std::string("missing ")+name);return value;
}
inline Json Target(void* library,cl_device_id device) {
  auto info=Entry<decltype(&clGetDeviceInfo)>(library,"clGetDeviceInfo");
  auto text=[&](cl_device_info field) {
    size_t size=0;Require(info(device,field,0,nullptr,&size)==CL_SUCCESS&&size>0&&size<65536,"device identity size");
    std::string value(size,'\0');Require(info(device,field,size,value.data(),nullptr)==CL_SUCCESS,"device identity");
    while(!value.empty()&&value.back()=='\0')value.pop_back();return value;
  };
  cl_uint vendor=0;Require(info(device,CL_DEVICE_VENDOR_ID,sizeof(vendor),&vendor,nullptr)==CL_SUCCESS,"vendor identity");
  std::array<unsigned char,CL_UUID_SIZE_KHR> uuid{};
  Require(info(device,CL_DRIVER_UUID_KHR,uuid.size(),uuid.data(),nullptr)==CL_SUCCESS,"driver UUID");
  std::ostringstream hex;hex<<std::hex<<std::setfill('0');for(auto c:uuid)hex<<std::setw(2)<<int(c);
  return Json{{"device",text(CL_DEVICE_NAME)},{"vendor",text(CL_DEVICE_VENDOR)},{"vendor_id",vendor},
              {"driver",text(CL_DRIVER_VERSION)},{"device_version",text(CL_DEVICE_VERSION)},
              {"opencl_c_version",text(CL_DEVICE_OPENCL_C_VERSION)},{"driver_uuid",hex.str()}};
}
inline cl_program Load(void* library,cl_context context,cl_command_queue queue,
                       const std::string& directory,const std::string& options) {
  const auto start=std::chrono::steady_clock::now();
  const auto slash=directory.find_last_of('/');Require(slash!=std::string::npos,"invalid model directory");
  const std::string role=directory.substr(slash+1),root=directory.substr(0,slash);
  Require(role=="encoder"||role=="gpu","invalid program role");
  std::ifstream input(root+"/model.json");Require(bool(input),"missing compiled model manifest");
  const Json model=Json::parse(input);Require(model.at("format")==2,"compiled model format 2 required");
  const auto& aot=model.at("aot");Require(aot.at("format")==1,"unsupported executable format");
  auto queue_info=Entry<decltype(&clGetCommandQueueInfo)>(library,"clGetCommandQueueInfo");
  cl_device_id device=nullptr;Require(queue_info(queue,CL_QUEUE_DEVICE,sizeof(device),&device,nullptr)==CL_SUCCESS,"queue device");
  Require(aot.at("target")==Target(library,device),"GPU/driver mismatch; compile for the target device before deployment");
  const Json* selected=nullptr;
  for(const auto& program:aot.at("programs"))if(program.at("role")==role&&program.at("options")==options) {
    Require(selected==nullptr,"duplicate executable options");selected=&program;
  }
  Require(selected!=nullptr,"no executable for requested options: "+options);
  const std::string name=selected->at("name");
  Require(!name.empty()&&name.find('/')==std::string::npos&&name.find('\\')==std::string::npos&&
          name.size()>6&&name.substr(name.size()-6)==".clbin","invalid executable name");
  const Json* asset=nullptr;
  for(const auto& item:model.at("assets"))if(item.at("role")==role&&item.at("name")==name) {
    Require(asset==nullptr,"duplicate executable asset");asset=&item;
  }
  Require(asset!=nullptr,"executable is not a model asset");
  const std::string path=directory+"/"+name;
  std::ifstream binary(path,std::ios::binary|std::ios::ate);
  Require(bool(binary)&&binary.tellg()>0,"missing or empty executable: "+path);
  const auto length=static_cast<size_t>(binary.tellg());
  Require(length<=64U*1024U*1024U&&length==asset->at("bytes").get<size_t>(),"executable size mismatch");
  Require(sherpa_onnx::MaliG720Sha256(path)==asset->at("sha256").get<std::string>(),"executable SHA mismatch");
  std::vector<unsigned char> bytes(length);binary.seekg(0);binary.read(reinterpret_cast<char*>(bytes.data()),length);
  Require(bool(binary),"executable read failed");const unsigned char* data=bytes.data();cl_int status=0,binary_status=0;
  auto create=Entry<decltype(&clCreateProgramWithBinary)>(library,"clCreateProgramWithBinary");
  auto build=Entry<decltype(&clBuildProgram)>(library,"clBuildProgram");
  auto get_build=Entry<decltype(&clGetProgramBuildInfo)>(library,"clGetProgramBuildInfo");
  auto release=Entry<decltype(&clReleaseProgram)>(library,"clReleaseProgram");
  cl_program program=create(context,1,&device,&length,&data,&binary_status,&status);
  if(status!=CL_SUCCESS||binary_status!=CL_SUCCESS||!program) {
    if(program)release(program);throw std::runtime_error("MaliG720 AOT executable rejected: "+std::to_string(status)+"/"+std::to_string(binary_status));
  }
  // Finalize an already executable binary; this path never submits OpenCL source.
  status=build(program,1,&device,nullptr,nullptr,nullptr);
  cl_program_binary_type type=CL_PROGRAM_BINARY_TYPE_NONE;
  if(status!=CL_SUCCESS||get_build(program,device,CL_PROGRAM_BINARY_TYPE,sizeof(type),&type,nullptr)!=CL_SUCCESS||type!=CL_PROGRAM_BINARY_TYPE_EXECUTABLE) {
    release(program);throw std::runtime_error("MaliG720 AOT executable finalization failed: "+std::to_string(status));
  }
  const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
  std::cerr<<"AOT_LOAD role="<<role<<" file="<<name<<" ms="<<ms<<" source_compiles=0"<<std::endl;
  return program;
}
}
