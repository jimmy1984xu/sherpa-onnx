#ifndef SHERPA_ONNX_CSRC_MALI_G720_AOT_FILE_H_
#define SHERPA_ONNX_CSRC_MALI_G720_AOT_FILE_H_

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>
#include "nlohmann/json.hpp"

namespace transai::aot {
// Uncompressed, little-endian, 64 KiB aligned assets. No extraction or runtime
// content hashing: offline delivery verification owns the SHA checks.
class File {
 public:
  struct Region { uint64_t offset, bytes; };
  static std::shared_ptr<File> Open(const std::string &path) {
    static std::mutex mutex;
    static std::map<std::string, std::weak_ptr<File>> cache;
    std::lock_guard<std::mutex> lock(mutex);
    for (auto it=cache.begin();it!=cache.end();)
      if(it->second.expired()) it=cache.erase(it); else ++it;
    if (auto file=cache[path].lock()) return file;
    auto file=std::shared_ptr<File>(new File(path));
    cache[path]=file;
    return file;
  }
  ~File() { if(fd_>=0) close(fd_); }
  File(const File&)=delete;
  File& operator=(const File&)=delete;
  int fd() const { return fd_; }
  const nlohmann::json& manifest() const { return manifest_; }
  Region Asset(const std::string &name) const {
    auto it=entries_.find(name);
    if(it==entries_.end()) throw std::runtime_error("MaliG720 AOT missing asset: "+name);
    return it->second;
  }
  void Read(const Region &region, void *output) const {
    auto p=static_cast<unsigned char*>(output);
    uint64_t done=0;
    while(done<region.bytes) {
      const size_t count=static_cast<size_t>(std::min<uint64_t>(region.bytes-done,16U*1024U*1024U));
      const auto n=pread(fd_,p+done,count,static_cast<off_t>(region.offset+done));
      if(n<0 && errno==EINTR) continue;
      if(n<=0) throw std::runtime_error("MaliG720 AOT asset read failed");
      done+=n;
    }
  }
  std::string Text(const std::string &name) const {
    auto r=Asset(name);
    if(r.bytes>1024U*1024U) throw std::runtime_error("MaliG720 AOT text too large");
    std::string text(static_cast<size_t>(r.bytes),'\0');Read(r,text.data());return text;
  }
 private:
  explicit File(const std::string &path) {
    if(path.size()<5 || path.substr(path.size()-4)!=".aot")
      throw std::runtime_error("MaliG720 requires one .aot model file");
    fd_=open(path.c_str(),O_RDONLY|O_CLOEXEC);
    try {
      struct stat status{};
      if(fd_<0 || fstat(fd_,&status) || status.st_size<64)
        throw std::runtime_error("MaliG720 AOT cannot open: "+path);
      unsigned char header[64];Read({0,64},header);
      auto u32=[&](int n) { uint32_t v=0;for(int i=0;i<4;++i)v|=uint32_t(header[n+i])<<(8*i);return v; };
      auto u64=[&](int n) { uint64_t v=0;for(int i=0;i<8;++i)v|=uint64_t(header[n+i])<<(8*i);return v; };
      const uint64_t index_bytes=u64(16),total=u64(24);
      if(std::memcmp(header,"TAIAOT01",8) || u32(8)!=1 || u32(12)!=64 ||
         index_bytes==0 || index_bytes>1024U*1024U || total!=uint64_t(status.st_size) ||
         index_bytes>total-64)
        throw std::runtime_error("MaliG720 AOT invalid header/length");
      for(int i=32;i<64;++i) if(header[i]) throw std::runtime_error("MaliG720 AOT reserved header");
      std::string index(static_cast<size_t>(index_bytes),'\0');Read({64,index_bytes},index.data());
      const auto data=nlohmann::json::parse(index);
      if(data.at("format")!=1 || data.at("entries").size()!=11)
        throw std::runtime_error("MaliG720 AOT invalid index");
      uint64_t end=64+index_bytes;
      for(const auto &item:data.at("entries")) {
        const std::string name=item.at("name");
        Region r{item.at("offset").get<uint64_t>(),item.at("bytes").get<uint64_t>()};
        if(name.empty() || name.front()=='/' || name.find("..")!=std::string::npos ||
           name.find('\\')!=std::string::npos || r.offset%65536 || r.offset<end ||
           r.bytes==0 || r.offset>total || r.bytes>total-r.offset ||
           !entries_.emplace(name,r).second)
          throw std::runtime_error("MaliG720 AOT invalid asset bounds/order: "+name);
        end=r.offset+r.bytes;
      }
      if(end!=total) throw std::runtime_error("MaliG720 AOT trailing/missing payload");
      manifest_=nlohmann::json::parse(Text("model.json"));
      if(manifest_.at("format")!=2 || manifest_.at("assets").size()!=10)
        throw std::runtime_error("MaliG720 AOT compiled manifest required");
      std::map<std::string,bool> seen;
      for(const auto &asset:manifest_.at("assets")) {
        const std::string role=asset.at("role"),name=asset.at("name");
        if((role!="tokens"&&role!="encoder"&&role!="gpu") || name.empty() ||
           name.find_first_of("/\\")!=std::string::npos || name=="." || name=="..")
          throw std::runtime_error("MaliG720 AOT invalid asset name");
        const std::string key=(role=="tokens"?"":role+"/")+name;
        if(!seen.emplace(key,true).second || Asset(key).bytes!=asset.at("bytes").get<uint64_t>())
          throw std::runtime_error("MaliG720 AOT manifest/index size mismatch: "+key);
      }
    } catch(...) { if(fd_>=0)close(fd_);fd_=-1;throw; }
  }
  int fd_=-1;
  std::map<std::string,Region> entries_;
  nlohmann::json manifest_;
};

// Virtual asset paths stay private to the runtime; they never address files on
// disk. Holding the File keeps one descriptor/index alive for session creation.
struct Slice {
  std::shared_ptr<File> file;
  File::Region region;
  explicit Slice(const std::string &path) {
    const auto slash=path.rfind(".aot/");
    if(slash==std::string::npos) throw std::runtime_error("MaliG720 AOT asset path required");
    file=File::Open(path.substr(0,slash+4));
    region=file->Asset(path.substr(slash+5));
  }
};
class Mapping {
 public:
  explicit Mapping(const std::string &path):slice_(path) {
    const long page_size=sysconf(_SC_PAGESIZE);
    if(page_size<=0) throw std::runtime_error("MaliG720 AOT invalid page size");
    const uint64_t base=slice_.region.offset/uint64_t(page_size)*uint64_t(page_size);
    delta_=static_cast<size_t>(slice_.region.offset-base);
    bytes_=static_cast<size_t>(slice_.region.bytes)+delta_;
    address_=mmap(nullptr,bytes_,PROT_READ,MAP_PRIVATE,slice_.file->fd(),static_cast<off_t>(base));
    if(address_==MAP_FAILED) throw std::runtime_error("MaliG720 AOT asset mapping failed");
  }
  ~Mapping() { if(address_!=MAP_FAILED) munmap(address_,bytes_); }
  Mapping(const Mapping&)=delete;
  Mapping& operator=(const Mapping&)=delete;
  size_t size() const { return static_cast<size_t>(slice_.region.bytes); }
  const unsigned char* data() const { return static_cast<const unsigned char*>(address_)+delta_; }
 private:
  Slice slice_;
  void *address_=MAP_FAILED;
  size_t bytes_=0,delta_=0;
};
template<class T> inline std::vector<T> Read(const std::string &path) {
  Slice slice(path);
  if(slice.region.bytes%sizeof(T)) throw std::runtime_error("MaliG720 AOT element size");
  std::vector<T> data(static_cast<size_t>(slice.region.bytes/sizeof(T)));
  slice.file->Read(slice.region,data.data());return data;
}
}  // namespace transai::aot
#endif
