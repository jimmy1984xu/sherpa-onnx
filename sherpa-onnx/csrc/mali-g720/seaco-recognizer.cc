#include "sherpa-onnx/csrc/mali-g720/seaco-recognizer.h"
#include "sherpa-onnx/csrc/mali-g720/aot-file.h"
#include "sherpa-onnx/csrc/mali-g720/config.h"
#include "sherpa-onnx/csrc/mali-g720/pcm.h"
#include "sherpa-onnx/csrc/mali-g720/seaco-abi.h"
#include <dlfcn.h>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>
namespace sherpa_onnx { namespace {
using Json=nlohmann::json;
class SeacoRecognizer final : public OfflineRecognizerImpl {
 public:
  explicit SeacoRecognizer(const OfflineRecognizerConfig &c):OfflineRecognizerImpl(c) {
    if(!ValidateMaliG720(c.model_config)||c.feat_config.sampling_rate!=16000||
       (c.decoding_method!="greedy_search"&&c.decoding_method!="modified_beam_search")||
       !c.lm_config.model.empty()||!c.ctc_fst_decoder_config.graph.empty()||c.blank_penalty!=0||
       !c.model_config.bpe_vocab.empty()||!std::isfinite(c.hotwords_score))
      throw std::invalid_argument("SeACo MaliG720 requires 16k PCM and native greedy/hotwords; no fallback");
    auto file=transai::aot::File::Open(c.model_config.transducer.encoder_filename);
    const auto &a=file->manifest().at("architecture");
    if(a.at("family")!="seaco-paraformer"||a.at("version")!=1)throw std::invalid_argument("SeACo architecture mismatch");
    auto tokens=Json::parse(Read(*file,"tokenizer/tokens.json"));
    tokens_=tokens.get<std::vector<std::string>>();
    if(tokens_.size()!=a.at("vocab_size").get<size_t>()||tokens_.size()<4)throw std::invalid_argument("SeACo vocabulary size");
    for(size_t i=0;i<tokens_.size();++i)if(!ids_.emplace(tokens_[i],i).second)throw std::invalid_argument("duplicate SeACo token");
    tokenizer_file_ = std::move(file);
    if(!c.hotwords_file.empty()){std::ifstream in(c.hotwords_file);if(!in)throw std::invalid_argument("cannot open SeACo hotwords_file");
      global_=Lines(in);}
    global_encoded_ = Words("");
    library_=dlopen("libtransai_seaco_paraformer_gpu.so",RTLD_NOW|RTLD_LOCAL);
    if(!library_)throw std::runtime_error(std::string("SeACo GPU dlopen: ")+dlerror());
    try {
      auto create=Load<int(*)(const char*,const char*,void**,char*,size_t)>("transai_seaco_create");
      destroy_=Load<void(*)(void*)>("transai_seaco_destroy");recognize_=Load<Recognize>("transai_seaco_recognize");
      char err[2048]{};if(create(c.model_config.transducer.encoder_filename.c_str(),global_encoded_.c_str(),&handle_,err,sizeof(err)))throw std::runtime_error(err);
    }catch(...){dlclose(library_);library_=nullptr;throw;}
  }
  ~SeacoRecognizer()override{if(handle_)destroy_(handle_);if(library_)dlclose(library_);}
  std::unique_ptr<OfflineStream> CreateStream()const override{return Stream(global_encoded_);}
  std::unique_ptr<OfflineStream> CreateStream(const std::string &words)const override{
    if (words.empty()) return CreateStream();
    std::lock_guard<std::mutex> lock(hotword_mutex_);
    if (words != cached_words_) {
      auto encoded = Words(words);
      cached_words_ = words;
      cached_encoded_ = std::move(encoded);
    }
    return Stream(cached_encoded_);
  }
  void DecodeStreams(OfflineStream **streams,int32_t count)const override {
    if(count<0||(count&&!streams))throw std::invalid_argument("SeACo stream array");
    std::lock_guard<std::mutex> lock(mutex_);
    for(int i=0;i<count;++i){auto *s=streams[i];if(!s||s->GetOption("mali_g720.seaco")!="1")throw std::invalid_argument("foreign SeACo stream");
      s->SetResult({});if(!s->GetOption("mali_g720.error").empty())throw std::runtime_error(s->GetOption("mali_g720.error"));
      auto input=s->GetFrames();if(input.size()<400||input.size()>480000)throw std::invalid_argument("SeACo requires 400..480000 PCM samples");
      std::vector<int16_t> pcm(input.size());MaliG720Pcm16(input.data(),pcm.data(),input.size());
      SeacoGpuResult gpu{};char err[2048]{};const auto words=s->GetOption("mali_g720.seaco.words");
      if(recognize_(handle_,pcm.data(),pcm.size(),words.c_str(),&gpu,err,sizeof(err)))throw std::runtime_error(err);
      if(gpu.count>512||gpu.text_bytes>131072||!gpu.text||(gpu.count&&(!gpu.ids||!gpu.scores))||gpu.result_reads!=1||gpu.intermediate_reads)
        throw std::runtime_error("invalid SeACo GPU result");
      OfflineRecognitionResult r;r.text.assign(gpu.text,gpu.text_bytes);
      for(size_t j=0;j<gpu.count;++j){int id=gpu.ids[j];float score=gpu.scores[j];
        if(id<0||id>=int(tokens_.size())||!std::isfinite(score)||score>2e-5f)throw std::runtime_error("invalid SeACo token/score");
        if(id<=2)continue; // Same control tokens excluded by GPU UTF-8 generation.
        r.tokens.push_back(tokens_[id]);r.ys_log_probs.push_back(score);
      }
      // No timestamp model is claimed by the source GPU release.
      r.text=ApplyInverseTextNormalization(std::move(r.text));r.text=ApplyHomophoneReplacer(std::move(r.text));s->SetResult(r);
    }
  }
  OfflineRecognizerConfig GetConfig()const override{return config_;}
  void SetConfig(const OfflineRecognizerConfig&)override{throw std::invalid_argument("SeACo model/config is immutable");}
 private:
  static std::string Read(const transai::aot::File &f,const std::string &name){auto a=f.Asset(name);if(a.bytes>32*1024*1024)throw std::invalid_argument("SeACo tokenizer too large");std::string s(a.bytes,'\0');f.Read(a,s.data());return s;}
  template<class T>T Load(const char *name){auto p=dlsym(library_,name);if(!p)throw std::runtime_error(std::string("missing SeACo GPU symbol ")+name);return reinterpret_cast<T>(p);}
  static std::vector<std::string> Lines(std::istream &in){std::vector<std::string> result;std::string line;while(std::getline(in,line)){
    if(line.size()>=3&&line.compare(0,3,"\xef\xbb\xbf")==0)line.erase(0,3);
    auto first=line.find_first_not_of(" \t\r"),last=line.find_last_not_of(" \t\r");if(first!=std::string::npos)result.push_back(line.substr(first,last-first+1));}return result;}
  int Piece(const std::string &s,bool continuation)const{auto i=ids_.find(s+(continuation?"@@":""));if(i!=ids_.end())return i->second;i=ids_.find(s);return i==ids_.end()?-1:i->second;}
  std::vector<int> English(const std::string &s)const{
    std::string lower=s;bool alpha=false,upper=true;for(char &c:lower){unsigned char u=c;alpha|=std::isalpha(u);if(std::islower(u))upper=false;c=std::tolower(u);}
    if(alpha&&upper){std::vector<int> seq;for(size_t i=0;i<lower.size();++i){int id=Piece(lower.substr(i,1),i+1<lower.size());if(id<0){seq.clear();break;}seq.push_back(id);}if(!seq.empty())return seq;}
    std::call_once(dictionary_once_, [this] {
      std::map<std::string, std::vector<int>> parsed;
    std::istringstream dict(Read(*tokenizer_file_,"tokenizer/seg_dict"));std::string line;
    while(std::getline(dict,line)){std::istringstream in(line);std::string word,piece;in>>word;std::vector<int> seq;bool ok=true;
      while(in>>piece){auto i=ids_.find(piece);if(i==ids_.end()){ok=false;break;}seq.push_back(i->second);}
      if(ok&&!word.empty()&&!seq.empty())parsed[word]=std::move(seq);
    }
      segments_ = std::move(parsed);
      tokenizer_file_.reset();
    });
    auto seg=segments_.find(lower);if(seg!=segments_.end())return seg->second;
    for(const auto &word:std::vector<std::string>{lower,s}){std::vector<std::vector<int>> best(word.size()+1);std::vector<bool> valid(word.size()+1);valid[0]=true;
      for(size_t begin=0;begin<word.size();++begin)if(valid[begin])for(size_t end=begin+1;end<=word.size();++end){int id=Piece(word.substr(begin,end-begin),end<word.size());
        if(id>=0&&(!valid[end]||best[begin].size()+1<best[end].size())){best[end]=best[begin];best[end].push_back(id);valid[end]=true;}}
      if(valid.back())return best.back();}
    throw std::invalid_argument("cannot encode SeACo English hotword: "+s);
  }
  std::vector<int> Encode(const std::string &line)const{
    std::istringstream input(line);std::vector<std::string> fields;std::string field;while(input>>field)fields.push_back(field);
    if(fields.size()>1){char *end=nullptr;std::strtof(fields.back().c_str(),&end);if(end&&!*end)fields.pop_back();}
    std::vector<int> out;
    for(const auto &word:fields)for(size_t i=0;i<word.size();){unsigned char c=word[i];
      if(c<128){size_t end=i;while(end<word.size()&&static_cast<unsigned char>(word[end])<128)++end;auto seq=English(word.substr(i,end-i));out.insert(out.end(),seq.begin(),seq.end());i=end;}
      else {size_t n=(c&0xe0)==0xc0?2:(c&0xf0)==0xe0?3:(c&0xf8)==0xf0?4:0;
        if(!n||i+n>word.size())throw std::invalid_argument("invalid UTF-8 SeACo hotword");auto token=word.substr(i,n);auto id=ids_.find(token);
        if(id==ids_.end())throw std::invalid_argument("unknown SeACo hotword token");out.push_back(id->second);i+=n;}}
    if(out.empty()||out.size()>10||std::any_of(out.begin(),out.end(),[](int id){return id<=2;}))throw std::invalid_argument("SeACo hotword must have 1..10 non-control tokens");return out;
  }
  std::string Words(std::string stream)const{
    std::replace(stream.begin(),stream.end(),'/','\n');std::istringstream in(stream);auto words=Lines(in);words.insert(words.end(),global_.begin(),global_.end());
    if(words.empty())return {};if(words.size()>4095)throw std::invalid_argument("SeACo supports at most 4095 hotwords");
    Json ids=Json::array();for(const auto &word:words)ids.push_back(Encode(word));ids.push_back({1});return Json{{"ids",ids}}.dump();
  }
  std::unique_ptr<OfflineStream> Stream(const std::string &words)const{auto s=std::make_unique<OfflineStream>(MaliG720Tag{},nullptr);s->SetOption("mali_g720.seaco","1");s->SetOption("mali_g720.seaco.words",words);return s;}
  using Recognize=int(*)(void*,const int16_t*,size_t,const char*,SeacoGpuResult*,char*,size_t);
  std::vector<std::string> tokens_,global_;std::map<std::string,int> ids_;
  mutable std::map<std::string,std::vector<int>> segments_;
  mutable std::shared_ptr<transai::aot::File> tokenizer_file_;
  mutable std::once_flag dictionary_once_;
  std::string global_encoded_;
  mutable std::mutex hotword_mutex_;
  mutable std::string cached_words_, cached_encoded_;
  void *library_=nullptr,*handle_=nullptr;void(*destroy_)(void*)=nullptr;Recognize recognize_=nullptr;mutable std::mutex mutex_;
};
}
std::unique_ptr<OfflineRecognizerImpl> CreateSeacoMaliG720Recognizer(const OfflineRecognizerConfig &config){return std::make_unique<SeacoRecognizer>(config);}
}
