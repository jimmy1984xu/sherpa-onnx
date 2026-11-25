#pragma once
#include "sherpa-onnx/csrc/mali-g720/aot-file.h"
// Map the unified container only during initialization. No extraction/hash scan.
class SeacoAot {
 public:
  struct View { const char *p;size_t n;const char*data()const{return p;}size_t size()const{return n;} };
  explicit SeacoAot(const std::string &path):file_(transai::aot::File::Open(path)) {
    const auto &m=file_->manifest();
    if(m.at("architecture").at("family")!="seaco-paraformer" || m.at("architecture").at("version")!=1)
      throw std::runtime_error("SeACo AOT architecture mismatch");
    struct stat st{};if(fstat(file_->fd(),&st))throw std::runtime_error("SeACo AOT stat");
    bytes_=st.st_size;mapping_=mmap(nullptr,bytes_,PROT_READ,MAP_PRIVATE,file_->fd(),0);
    if(mapping_==MAP_FAILED){mapping_=nullptr;throw std::runtime_error("SeACo AOT mmap");}
  }
  ~SeacoAot(){if(mapping_)munmap(mapping_,bytes_);}
  View view(const std::string &name)const {auto r=file_->Asset(name);return {static_cast<const char*>(mapping_)+r.offset,static_cast<size_t>(r.bytes)};}
  nlohmann::json json(const std::string &name)const {auto v=view(name);return nlohmann::json::parse(v.data(),v.data()+v.size());}
 private:
  std::shared_ptr<transai::aot::File> file_;void *mapping_=nullptr;size_t bytes_=0;
};
