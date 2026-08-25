// sherpa-onnx/csrc/sherpa-onnx-offline.cc
//
// Copyright (c)  2022-2023  Xiaomi Corporation

#include <stdio.h>

#include <chrono>  // NOLINT
#include <string>
#include <vector>
#include <fstream>
#include <unordered_map>

#include "sherpa-onnx/csrc/offline-recognizer.h"
#include "sherpa-onnx/csrc/parse-options.h"
#include "sherpa-onnx/csrc/wave-reader.h"

int main(int32_t argc, char *argv[]) {
  const char *kUsageMessage = R"usage(
Speech recognition using non-streaming models with sherpa-onnx.

Usage:

(1) Transducer from icefall

See https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-transducer/index.html

  ./bin/sherpa-onnx-offline \
    --tokens=/path/to/tokens.txt \
    --encoder=/path/to/encoder.onnx \
    --decoder=/path/to/decoder.onnx \
    --joiner=/path/to/joiner.onnx \
    --num-threads=1 \
    --decoding-method=greedy_search \
    /path/to/foo.wav [bar.wav foobar.wav ...]


(2) Paraformer from FunASR

See https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-paraformer/index.html

  ./bin/sherpa-onnx-offline \
    --tokens=/path/to/tokens.txt \
    --paraformer=/path/to/model.onnx \
    --num-threads=1 \
    --decoding-method=greedy_search \
    /path/to/foo.wav [bar.wav foobar.wav ...]

(3) Moonshine models

See https://k2-fsa.github.io/sherpa/onnx/moonshine/index.html

  ./bin/sherpa-onnx-offline \
    --moonshine-preprocessor=/Users/fangjun/open-source/sherpa-onnx/scripts/moonshine/preprocess.onnx \
    --moonshine-encoder=/Users/fangjun/open-source/sherpa-onnx/scripts/moonshine/encode.int8.onnx \
    --moonshine-uncached-decoder=/Users/fangjun/open-source/sherpa-onnx/scripts/moonshine/uncached_decode.int8.onnx \
    --moonshine-cached-decoder=/Users/fangjun/open-source/sherpa-onnx/scripts/moonshine/cached_decode.int8.onnx \
    --tokens=/Users/fangjun/open-source/sherpa-onnx/scripts/moonshine/tokens.txt \
    --num-threads=1 \
    /path/to/foo.wav [bar.wav foobar.wav ...]

(4) Whisper models

See https://k2-fsa.github.io/sherpa/onnx/pretrained_models/whisper/tiny.en.html

  ./bin/sherpa-onnx-offline \
    --whisper-encoder=./sherpa-onnx-whisper-base.en/base.en-encoder.int8.onnx \
    --whisper-decoder=./sherpa-onnx-whisper-base.en/base.en-decoder.int8.onnx \
    --tokens=./sherpa-onnx-whisper-base.en/base.en-tokens.txt \
    --num-threads=1 \
    /path/to/foo.wav [bar.wav foobar.wav ...]

(5) NeMo CTC models

See https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-ctc/index.html

  ./bin/sherpa-onnx-offline \
    --tokens=./sherpa-onnx-nemo-ctc-en-conformer-medium/tokens.txt \
    --nemo-ctc-model=./sherpa-onnx-nemo-ctc-en-conformer-medium/model.onnx \
    --num-threads=2 \
    --decoding-method=greedy_search \
    --debug=false \
    ./sherpa-onnx-nemo-ctc-en-conformer-medium/test_wavs/0.wav \
    ./sherpa-onnx-nemo-ctc-en-conformer-medium/test_wavs/1.wav \
    ./sherpa-onnx-nemo-ctc-en-conformer-medium/test_wavs/8k.wav

(6) TDNN CTC model for the yesno recipe from icefall

See https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-ctc/yesno/index.html
      //
  ./build/bin/sherpa-onnx-offline \
    --sample-rate=8000 \
    --feat-dim=23 \
    --tokens=./sherpa-onnx-tdnn-yesno/tokens.txt \
    --tdnn-model=./sherpa-onnx-tdnn-yesno/model-epoch-14-avg-2.onnx \
    ./sherpa-onnx-tdnn-yesno/test_wavs/0_0_0_1_0_0_0_1.wav \
    ./sherpa-onnx-tdnn-yesno/test_wavs/0_0_1_0_0_0_1_0.wav

(7) Using wav.scp file and saving results to file

  ./bin/sherpa-onnx-offline \
    --tokens=/path/to/tokens.txt \
    --encoder=/path/to/encoder.onnx \
    --decoder=/path/to/decoder.onnx \
    --joiner=/path/to/joiner.onnx \
    --wav-scp=wav.scp \
    --result-file=result.txt \
    --batch-size=32

Note: It supports decoding multiple files in batches

foo.wav should be of single channel, 16-bit PCM encoded wave file; its
sampling rate can be arbitrary and does not need to be 16kHz.

Please refer to
https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html
for a list of pre-trained models to download.
)usage";

  sherpa_onnx::ParseOptions po(kUsageMessage);
  sherpa_onnx::OfflineRecognizerConfig config;
  config.Register(&po);

  // Add new options
  std::string wav_scp;
  std::string result_file;
  int32_t batch_size = 1;

  po.Register("wav-scp", &wav_scp,
              "Path to wav.scp file containing id and wav path pairs");
  po.Register("result-file", &result_file,
              "Path to save recognition results in 'id text' format");
  po.Register("batch-size", &batch_size,
              "Batch size for decoding (default: 1)");

  po.Read(argc, argv);

  bool use_wav_scp = !wav_scp.empty();
  bool save_result = !result_file.empty();

  if (!use_wav_scp && po.NumArgs() < 1) {
    fprintf(stderr, "Error: Please provide at least 1 wave file or use --wav-scp.\n\n");
    po.PrintUsage();
    exit(EXIT_FAILURE);
  }

  if (use_wav_scp && po.NumArgs() > 0) {
    fprintf(stderr, "Warning: Both wav.scp and command line wave files provided. "
                    "Using wav.scp and ignoring command line files.\n");
  }

  fprintf(stderr, "%s\n", config.ToString().c_str());

  if (!config.Validate()) {
    fprintf(stderr, "Errors in config!\n");
    return -1;
  }

  fprintf(stderr, "Creating recognizer ...\n");
  sherpa_onnx::OfflineRecognizer recognizer(config);

  // Prepare file list
  std::vector<std::pair<std::string, std::string>> files; // id, filename pairs
  if (use_wav_scp) {
    fprintf(stderr, "Reading wav.scp from %s\n", wav_scp.c_str());
    std::ifstream infile(wav_scp);
    if (!infile.is_open()) {
      fprintf(stderr, "Failed to open wav.scp: %s\n", wav_scp.c_str());
      return -1;
    }
    std::string line;
    while (std::getline(infile, line)) {
      size_t pos = line.find(' ');
      if (pos != std::string::npos) {
        std::string id = line.substr(0, pos);
        std::string filename = line.substr(pos + 1);
        files.emplace_back(id, filename);
      }
    }
    infile.close();
  } else {
    for (int32_t i = 1; i <= po.NumArgs(); ++i) {
      std::string filename = po.GetArg(i);
      // Use filename as id
      size_t slash_pos = filename.find_last_of("/\\");
      size_t dot_pos = filename.find_last_of('.');
      std::string id = (slash_pos != std::string::npos) ?
                       filename.substr(slash_pos + 1, dot_pos - slash_pos - 1) :
                       filename.substr(0, dot_pos);
      files.emplace_back(id, filename);
    }
  }

  fprintf(stderr, "Found %zu files to process\n", files.size());
  fprintf(stderr, "Using batch size: %d\n", batch_size);

  std::ofstream result_stream;
  if (save_result) {
    result_stream.open(result_file);
    if (!result_stream.is_open()) {
      fprintf(stderr, "Failed to open result file: %s\n", result_file.c_str());
      return -1;
    }
    fprintf(stderr, "Saving results to: %s\n", result_file.c_str());
  }

  fprintf(stderr, "Started\n");
  const auto begin = std::chrono::steady_clock::now();

  float total_duration = 0;
  int32_t processed_files = 0;
  std::vector<std::string> all_results;

  // Process files in batches
  for (size_t start_idx = 0; start_idx < files.size(); start_idx += batch_size) {
    size_t end_idx = std::min(start_idx + batch_size, files.size());
    size_t current_batch_size = end_idx - start_idx;

    fprintf(stderr, "Processing batch [%zu-%zu] of %zu files\n",
            start_idx + 1, end_idx, files.size());

    std::vector<std::unique_ptr<sherpa_onnx::OfflineStream>> ss;
    std::vector<sherpa_onnx::OfflineStream *> ss_pointers;
    float batch_duration = 0;

    // Load waveforms for current batch
    for (size_t i = start_idx; i < end_idx; ++i) {
      const auto& [id, filename] = files[i];

      int32_t sampling_rate = -1;
      bool is_ok = false;
      std::vector<float> samples = sherpa_onnx::ReadWave(filename, &sampling_rate, &is_ok);

      if (!is_ok) {
        fprintf(stderr, "Failed to read '%s'\n", filename.c_str());
        // Create empty stream for failed files
        auto s = recognizer.CreateStream();
        ss.push_back(std::move(s));
        continue;
      }

      float duration = samples.size() / static_cast<float>(sampling_rate);
      batch_duration += duration;
      total_duration += duration;

      auto s = recognizer.CreateStream();
      s->AcceptWaveform(sampling_rate, samples.data(), samples.size());
      ss.push_back(std::move(s));
    }

    // Prepare pointers for decoding
    for (auto& s : ss) {
      ss_pointers.push_back(s.get());
    }

    // Decode current batch
    recognizer.DecodeStreams(ss_pointers.data(), ss_pointers.size());

    // Collect results for current batch
    for (size_t i = 0; i < current_batch_size; ++i) {
      size_t file_idx = start_idx + i;
      const auto& [id, filename] = files[file_idx];

      std::string result_text = ss[i]->GetResult().text;
      std::string result_line = id + " " + result_text;

      all_results.push_back(result_line);

      if (save_result) {
        result_stream << result_line << std::endl;
      }

      // Also print to stderr for immediate feedback
      fprintf(stderr, "%s: %s\n", id.c_str(), result_text.c_str());
    }

    processed_files += current_batch_size;
    fprintf(stderr, "Processed %d/%zu files\n", processed_files, files.size());
  }

  const auto end = std::chrono::steady_clock::now();

  fprintf(stderr, "Done!\n\n");

  // Print all results to stderr if not saving to file
  if (!save_result) {
    for (const auto& result : all_results) {
      fprintf(stderr, "%s\n", result.c_str());
    }
    fprintf(stderr, "----\n");
  }

  if (save_result) {
    result_stream.close();
    fprintf(stderr, "Results saved to: %s\n", result_file.c_str());
  }

  float elapsed_seconds =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - begin)
          .count() /
      1000.;

  fprintf(stderr, "num threads: %d\n", config.model_config.num_threads);
  fprintf(stderr, "decoding method: %s\n", config.decoding_method.c_str());
  if (config.decoding_method == "modified_beam_search") {
    fprintf(stderr, "max active paths: %d\n", config.max_active_paths);
  }

  fprintf(stderr, "Total audio duration: %.3f s\n", total_duration);
  fprintf(stderr, "Elapsed seconds: %.3f s\n", elapsed_seconds);
  float rtf = elapsed_seconds / total_duration;
  fprintf(stderr, "Real time factor (RTF): %.3f / %.3f = %.3f\n",
          elapsed_seconds, total_duration, rtf);

  return 0;
}