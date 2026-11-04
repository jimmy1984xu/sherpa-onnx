API int transai_parakeet_gpu_recognize_beam(void *opaque,const int16_t *pcm,
    size_t samples,int32_t beam,float blank_penalty,const int32_t *graph,size_t words,
    MaliG720GpuResultEx *output,char *error,size_t capacity) {
  if(!output||output->abi_version!=1||output->struct_bytes!=sizeof(*output)) {
    Error(error,capacity,"beam result ABI mismatch");return -1;
  }
  output->base={};output->ys_log_probs=nullptr;output->score_count=0;
  try {
    if(!opaque||!pcm||!samples||samples>480000)throw std::runtime_error("invalid beam input");
    auto &s=*static_cast<Session*>(opaque);std::lock_guard lock(s.mutex);
    auto begin=Clock::now();
    if(!s.beam)s.beam=std::make_unique<BeamGpu>(s.gpu);
    s.beam->Prepare(beam,graph,words);
    s.input.assign(pcm,pcm+samples);s.gpu.EncodePcm(s.input);
    s.beam->Decode(beam,blank_penalty,int((samples/160+7)/8));
    const auto &packet=s.gpu.result;
    if(packet[7]<0||packet[7]>131072)throw std::runtime_error("invalid beam UTF8 size");
    output->base={reinterpret_cast<const char*>(packet.data())+45032,size_t(packet[7]),
                  packet.data()+8,packet[1],packet[5],packet[4],packet[6],Ms(begin)};
    output->ys_log_probs=reinterpret_cast<const float*>(packet.data())+176104/4;
    output->score_count=packet[1];
    if(const char *trace=getenv("MALI_G720_SCORE_TRACE")) {
      if(packet[1] && packet[9]==0) {
        std::ofstream file(trace,std::ios::binary);
        file.write(reinterpret_cast<const char*>(packet.data())+191104,MALI_G720_TRACE_LOGITS_BYTES);
        file.write(reinterpret_cast<const char*>(packet.data())+176104,4);
        file.write(reinterpret_cast<const char*>(packet.data()+8),4);
        if(!file)throw std::runtime_error("cannot save beam score trace");
      } else throw std::runtime_error("beam score trace requires first token at frame zero");
    }
    Error(error,capacity,"");return 0;
  } catch(const std::exception &e) {Error(error,capacity,e.what());return -1;}
}
