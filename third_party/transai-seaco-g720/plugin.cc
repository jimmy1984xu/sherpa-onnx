#include "runtime.h"
#include "sherpa-onnx/csrc/mali-g720/seaco-abi.h"
struct Session {
 Runtime gpu;
 std::vector<int> packet;
 std::vector<float> scores;
 std::string text, applied_hotwords;

 explicit Session(const char *file, const char *words) : gpu(file) {
  sherpa_onnx::MaliG720InitTimer timer("seaco");
  const bool hot = words && *words;
  if (hot) {
   gpu.hotwords(J::parse(words));
   applied_hotwords = words;
   timer.Mark("hotwords");
  }
  gpu.prepare_plans(hot);
  timer.Mark("command_plans");
 }
};
static void error(char *out,size_t n,const std::exception &e){if(n){std::strncpy(out,e.what(),n-1);out[n-1]=0;}}
#define EXPORT extern "C" __attribute__((visibility("default")))
EXPORT int transai_seaco_create(const char *file,const char *words,void **handle,char *err,size_t n) {
 try {if(!file||!handle)throw std::invalid_argument("SeACo create arguments");*handle=new Session(file,words);return 0;}
 catch(const std::exception &e){error(err,n,e);return 1;}
}
EXPORT void transai_seaco_destroy(void *handle){delete static_cast<Session*>(handle);}
EXPORT int transai_seaco_recognize(void *handle,const int16_t *pcm,size_t samples,const char *words,SeacoGpuResult *out,char *err,size_t n) {
 try {
  if(!handle||!pcm||!out||samples<400||samples>480000)throw std::invalid_argument("SeACo inference arguments");
  auto &s=*static_cast<Session*>(handle);bool hot=words&&*words;
  if (hot && s.applied_hotwords != words) {
    s.gpu.hotwords(J::parse(words));
    s.applied_hotwords = words;
  }
  double ms;
  s.packet=s.gpu.infer(pcm,samples*sizeof(*pcm),hot,ms);int R=s.gpu.result_capacity,count=s.packet.at(0),len=s.packet.at(R+1);
  int score_offset=R+2+(R*(s.gpu.max_token_bytes+2)+4)/4;
  if(count<0||count>R||len<0||len>=int((score_offset-R-2)*4)||s.packet.size()!=size_t(score_offset+R))throw std::runtime_error("SeACo GPU packet bounds");
  s.text.assign(reinterpret_cast<const char*>(s.packet.data()+R+2),len);s.scores.resize(count);
  for(int i=0;i<count;++i)std::memcpy(&s.scores[i],&s.packet[score_offset+i],4);
  *out={s.packet.data()+1,s.scores.data(),s.text.data(),size_t(count),s.text.size(),s.gpu.result_reads,s.gpu.intermediate_reads};
  if(out->result_reads!=1||out->intermediate_reads)throw std::runtime_error("SeACo GPU transport policy");
  std::cerr<<"SEACO_GPU cpu_model=0 cpu_hotword_lstm=0 intermediate_reads=0 result_reads=1 score_logsoftmax=1 hotwords="<<hot<<"\n";
  return 0;
 } catch(const std::exception &e){error(err,n,e);return 1;}
}
