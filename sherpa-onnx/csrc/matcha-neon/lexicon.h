#ifndef SHERPA_ONNX_MATCHA_NEON_LEXICON_H_
#define SHERPA_ONNX_MATCHA_NEON_LEXICON_H_
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <stdexcept>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <unistd.h>
namespace transai::matcha_neon {
// Immutable open-addressed lexicon. Offline compilation replaces per-load
// string parsing and tens of thousands of map/set allocations.
class Lexicon {
 const unsigned char* data_=nullptr;size_t bytes_=0;
 uint32_t count_=0,capacity_=0,slots_=0,entries_=0;
 uint32_t U32(size_t at)const{if(at>bytes_||bytes_-at<4)throw std::runtime_error("Lexicon bounds");uint32_t v;std::memcpy(&v,data_+at,4);return v;}
 public:
 explicit Lexicon(const std::string& path) {
  const int fd=open(path.c_str(),O_RDONLY|O_CLOEXEC);struct stat st{};
  if(fd<0)throw std::runtime_error("Cannot open compiled lexicon");
  if(fstat(fd,&st)||st.st_size<32){close(fd);throw std::runtime_error("Invalid compiled lexicon size");}
  bytes_=st.st_size;void* p=mmap(nullptr,bytes_,PROT_READ,MAP_PRIVATE,fd,0);close(fd);
  if(p==MAP_FAILED)throw std::runtime_error("Cannot map compiled lexicon");data_=static_cast<unsigned char*>(p);
  try {
   if(std::memcmp(data_,"TAILEX01",8))throw std::runtime_error("Compiled lexicon magic");
   count_=U32(8);capacity_=U32(12);slots_=U32(16);entries_=U32(20);
   if(!count_||capacity_<count_*2||capacity_>(1U<<24)||(capacity_&(capacity_-1))||slots_!=32||
      entries_!=32+capacity_*4||uint64_t(entries_)+uint64_t(count_)*16>bytes_)
    throw std::runtime_error("Compiled lexicon header");
  }catch(...){munmap(const_cast<unsigned char*>(data_),bytes_);data_=nullptr;throw;}
 }
 ~Lexicon(){if(data_)munmap(const_cast<unsigned char*>(data_),bytes_);}
 Lexicon(const Lexicon&)=delete;
 bool Lookup(const std::string& word,std::vector<int32_t>* ids=nullptr)const {
  uint32_t hash=2166136261u;for(unsigned char c:word){hash^=c;hash*=16777619u;}
  uint32_t slot=hash&(capacity_-1);
  for(uint32_t probes=0;probes<capacity_;++probes,slot=(slot+1)&(capacity_-1)) {
   const uint32_t index=U32(slots_+slot*4);if(!index)return false;
   if(index>count_)throw std::runtime_error("Compiled lexicon slot");
   const size_t entry=entries_+size_t(index-1)*16;
   const uint32_t offset=U32(entry),length=U32(entry+4);
   if(offset>bytes_||length>bytes_-offset)throw std::runtime_error("Compiled lexicon word bounds");
   if(length!=word.size()||std::memcmp(data_+offset,word.data(),length))continue;
   if(ids){const uint32_t pos=U32(entry+8),n=U32(entry+12);
    if(pos%4||pos>bytes_||uint64_t(n)*4>bytes_-pos)throw std::runtime_error("Compiled lexicon ids bounds");
    ids->resize(n);std::memcpy(ids->data(),data_+pos,size_t(n)*4);}
   return true;
  }
  throw std::runtime_error("Compiled lexicon probe overflow");
 }
};
}
#endif
