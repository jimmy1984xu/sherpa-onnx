#ifndef SHERPA_ONNX_NEON_TTS_FRONTEND_H_
#define SHERPA_ONNX_NEON_TTS_FRONTEND_H_
#ifdef SHERPA_ONNX_ENABLE_MATCHA_NEON
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "sherpa-onnx/csrc/mali-g720/aot-file.h"
namespace transai::tts_neon {
// espeak/fst APIs consume paths. Materialize only these small resources once;
// acoustic/vocoder tensors always remain mmap-backed in the original AOT.
inline std::string Frontend(const std::string& path,bool materialize=true) {
  namespace fs=std::filesystem;
  const auto file=aot::File::Open(path);const auto& m=file->manifest();
  if(m.at("frontend").at("format")!=1 || m.at("frontend").at("asset")!="tts/frontend.bin")
    throw std::runtime_error("Unsupported bundled NEON TTS frontend");
  std::string key;
  for(const auto& a:m.at("assets"))if(a.at("name")=="frontend.bin")key=a.at("sha256");
  if(key.size()!=64 || key.find_first_not_of("0123456789abcdef")!=std::string::npos)
    throw std::runtime_error("Invalid frontend cache identity");
  const auto family = m.at("architecture").at("family").get<std::string>();
  if (family != "matcha-tts" && family != "piper-en-amy-tts" &&
      family != "piper-en-bryce-tts" && family != "piper-es-carlfm-tts")
    throw std::runtime_error("Unsupported NEON TTS frontend family");
  const auto parent = fs::path(path).parent_path() / ("." + family + "-cache");
  const auto target=parent/key;const auto marker=target/".ready";
  if(!materialize)return target.string();
  fs::create_directories(parent);
  const int lock=open((parent/(key+".lock")).c_str(),O_CREAT|O_RDWR|O_CLOEXEC,0600);
  if(lock<0)throw std::runtime_error("Cannot open frontend cache lock");
  struct Guard{int fd;~Guard(){flock(fd,LOCK_UN);close(fd);}} guard{lock};
  if(flock(lock,LOCK_EX))throw std::runtime_error("Cannot lock frontend cache");
  if(fs::is_regular_file(marker))return target.string();
  const auto asset=file->Asset("tts/frontend.bin");unsigned char header[16];
  if(asset.bytes<16)throw std::runtime_error("Short frontend bundle");
  file->Read({asset.offset,16},header);uint64_t count=0;
  for(int i=0;i<8;++i)count|=uint64_t(header[8+i])<<(8*i);
  if(std::memcmp(header,"TAIFNT01",8) || count==0 || count>1024*1024 || count>asset.bytes-16)
    throw std::runtime_error("Invalid frontend header");
  std::string index(count,'\0');file->Read({asset.offset+16,count},index.data());
  const auto data=nlohmann::json::parse(index);const auto& entries=data.at("entries");
  if(data.at("format")!=1 || entries.size()!=m.at("frontend").at("files") || entries.size()>4096)
    throw std::runtime_error("Invalid frontend index");
  fs::create_directories(target);uint64_t end=16+count;
  std::map<std::string,bool> seen;
  for(const auto& e:entries) {
    const std::string name=e.at("name");const uint64_t offset=e.at("offset"),bytes=e.at("bytes");
    const fs::path relative(name);
    if(name.empty()||relative.is_absolute()||name.find('\\')!=std::string::npos||name.find("..")!=std::string::npos||
       !seen.emplace(name,true).second||offset<end||offset>asset.bytes||bytes>asset.bytes-offset)
      throw std::runtime_error("Invalid frontend entry");
    end=offset+bytes;const auto destination=target/relative;fs::create_directories(destination.parent_path());
    // No symlink traversal in writable caches.
    for(auto p=destination;p!=parent;p=p.parent_path())if(fs::is_symlink(p))throw std::runtime_error("Symlink in frontend cache");
    std::ofstream out(destination,std::ios::binary|std::ios::trunc);
    if(!out)throw std::runtime_error("Cannot create cached frontend file");
    std::vector<char> buffer(65536);uint64_t done=0;
    while(done<bytes){const auto n=std::min<uint64_t>(buffer.size(),bytes-done);file->Read({asset.offset+offset+done,n},buffer.data());out.write(buffer.data(),n);done+=n;}
    out.close();if(!out)throw std::runtime_error("Cannot write frontend cache");
  }
  if(end!=asset.bytes)throw std::runtime_error("Trailing frontend bytes");
  std::ofstream ready(marker);ready<<key<<'\n';ready.close();
  if(!ready)throw std::runtime_error("Cannot commit frontend cache");
  return target.string();
}
}
#endif
#endif
