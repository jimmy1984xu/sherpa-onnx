// The plugin exposes only the scored private ABI consumed by sherpa-onnx.
API int transai_parakeet_gpu_recognize_ex(void* opaque, const int16_t* pcm,
    size_t samples, MaliG720GpuResultEx* output, char* error, size_t capacity) {
  if (!output || output->abi_version != 1 || output->struct_bytes != sizeof(*output)) {
    Error(error, capacity, "score result ABI mismatch");
    return -1;
  }
  output->base = {};
  output->ys_log_probs = nullptr;
  output->score_count = 0;
  try {
    if (!opaque || !pcm || !samples || samples > 480000)
      throw std::runtime_error("expected PCM16 mono16000Hz, 1..480000 samples");
    auto& session = *static_cast<Session*>(opaque);
    std::lock_guard lock(session.mutex);
    auto begin = Clock::now();
    session.input.assign(pcm, pcm + samples);
    session.gpu.EncodePcm(session.input);
    session.gpu.Decode((std::max<size_t>(1, samples / 160) + 7) / 8, 0);
    const auto& packet = session.gpu.result;
    if (packet[7] < 0 || packet[7] > 131072)
      throw std::runtime_error("invalid GPU UTF8 size");
    output->base = {reinterpret_cast<const char*>(packet.data()) + 45032,
                   static_cast<size_t>(packet[7]), packet.data() + 8, packet[1],
                   packet[5], packet[4], packet[6], Ms(begin)};
    output->ys_log_probs = reinterpret_cast<const float*>(packet.data()) + 176104 / 4;
    output->score_count = packet[1];
    if (const char* trace = getenv("MALI_G720_SCORE_TRACE")) {
      if (packet[1]) {
        std::ofstream file(trace, std::ios::binary);
        file.write(reinterpret_cast<const char*>(packet.data()) + 191104,
                   MALI_G720_TRACE_LOGITS_BYTES);
        file.write(reinterpret_cast<const char*>(packet.data()) + 176104, 4);
        file.write(reinterpret_cast<const char*>(packet.data() + 8), 4);
        if (!file) throw std::runtime_error("cannot save score trace");
      }
    }
    Error(error, capacity, "");
    return 0;
  } catch (const std::exception& e) {
    Error(error, capacity, e.what());
    return -1;
  }
}

API void transai_parakeet_gpu_destroy(void* handle) {
  delete static_cast<Session*>(handle);
}
