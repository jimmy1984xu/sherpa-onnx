#ifndef SHERPA_ONNX_CSRC_MALI_G720_HOTWORD_GRAPH_H_
#define SHERPA_ONNX_CSRC_MALI_G720_HOTWORD_GRAPH_H_
#include <algorithm>
#include <cstring>
#include <map>
#include <set>
#include <stdexcept>
#include <vector>
#include "sherpa-onnx/csrc/context-graph.h"
namespace sherpa_onnx {
// Serialize sherpa's exact non-strict ContextGraph transitions once, before
// inference. GPU search consumes the table without host callbacks or logits.
inline std::vector<int32_t> MaliHotwordGraph(const ContextGraphPtr &graph, int vocab) {
  std::vector<const ContextState *> nodes{graph ? graph->Root() : nullptr};
  std::map<const ContextState *, int32_t> ids{{nodes[0], 0}};
  std::set<int32_t> alphabet;
  if (graph) for (size_t i=0; i<nodes.size(); ++i) {
    std::map<int32_t,const ContextState *> ordered;
    for (const auto &edge:nodes[i]->next) ordered[edge.first]=edge.second.get();
    for (const auto &edge:ordered) {
      alphabet.insert(edge.first); ids[edge.second]=nodes.size(); nodes.push_back(edge.second);
    }
  }
  size_t stride=alphabet.size()+1, count=nodes.size()*stride;
  size_t words=4+vocab+2*count+nodes.size();
  if (words>16*1024*1024) throw std::invalid_argument("MaliG720 hotword transition table exceeds 64MiB");
  std::vector<int32_t> result(words, 0), symbols(alphabet.begin(),alphabet.end());
  symbols.push_back(-1);  // any token outside the hotword alphabet
  result[0]=nodes.size();result[1]=alphabet.size();result[2]=vocab;
  std::fill(result.begin()+4,result.begin()+4+vocab,alphabet.size());
  for (size_t i=0;i<alphabet.size();++i) {
    if (symbols[i]<0 || symbols[i]>=vocab-1) throw std::invalid_argument("invalid hotword token");
    result[4+symbols[i]]=i;
  }
  auto put_float=[&](size_t index,float value) {std::memcpy(&result[index],&value,4);};
  for (size_t i=0;i<nodes.size();++i) {
    for (size_t j=0;j<stride;++j) {
      int next=0;float score=0;
      if (graph) {
        auto t=graph->ForwardOneStep(nodes[i],symbols[j],false);
        next=ids.at(std::get<1>(t));score=std::get<0>(t);
      }
      result[4+vocab+i*stride+j]=next;
      put_float(4+vocab+count+i*stride+j,score);
    }
    put_float(4+vocab+2*count+i,graph?graph->Finalize(nodes[i]).first:0);
  }
  return result;
}
}  // namespace sherpa_onnx
#endif
