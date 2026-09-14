// c-api-examples/speaker-segmentation-c-api.c
//
// Copyright (c) 2026

#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "sherpa-onnx/c-api/c-api.h"

static void Usage(const char *program) {
  fprintf(stderr, "Usage: %s --model MODEL --audio WAV [options]\n", program);
  fprintf(stderr, "Options:\n");
  fprintf(stderr, "  --num-threads N\n");
  fprintf(stderr, "  --min-duration-on SECONDS\n");
  fprintf(stderr, "  --min-duration-off SECONDS\n");
  fprintf(stderr, "  --change-vote-threshold VALUE\n");
}

int main(int argc, char *argv[]) {
  const char *model = NULL;
  const char *audio_filename = NULL;
  int32_t num_threads = 1;
  float min_duration_on = 0.30F;
  float min_duration_off = 0.50F;
  float change_vote_threshold = 0.50F;

  for (int32_t i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--help") == 0) {
      Usage(argv[0]);
      return 0;
    }
    if (i + 1 >= argc) {
      Usage(argv[0]);
      return 1;
    }
    if (strcmp(argv[i], "--model") == 0) {
      model = argv[++i];
    } else if (strcmp(argv[i], "--audio") == 0) {
      audio_filename = argv[++i];
    } else if (strcmp(argv[i], "--num-threads") == 0) {
      sscanf(argv[++i], "%d", &num_threads);
    } else if (strcmp(argv[i], "--min-duration-on") == 0) {
      sscanf(argv[++i], "%f", &min_duration_on);
    } else if (strcmp(argv[i], "--min-duration-off") == 0) {
      sscanf(argv[++i], "%f", &min_duration_off);
    } else if (strcmp(argv[i], "--change-vote-threshold") == 0) {
      sscanf(argv[++i], "%f", &change_vote_threshold);
    } else {
      Usage(argv[0]);
      return 1;
    }
  }

  if (!model || !audio_filename) {
    Usage(argv[0]);
    return 1;
  }

  const SherpaOnnxWave *wave = SherpaOnnxReadWave(audio_filename);
  if (!wave) {
    fprintf(stderr, "Failed to read %s\n", audio_filename);
    return 1;
  }

  SherpaOnnxSpeakerSegmentationConfig config;
  memset(&config, 0, sizeof(config));
  config.model.pyannote.model = model;
  config.model.num_threads = num_threads;
  config.model.provider = "cpu";
  config.min_duration_on = min_duration_on;
  config.min_duration_off = min_duration_off;
  config.change_vote_threshold = change_vote_threshold;

  const SherpaOnnxSpeakerSegmentation *segmenter =
      SherpaOnnxCreateSpeakerSegmentation(&config);
  if (!segmenter) {
    fprintf(stderr, "Failed to create speaker segmentation runner\n");
    SherpaOnnxFreeWave(wave);
    return 1;
  }

  if (SherpaOnnxSpeakerSegmentationGetSampleRate(segmenter) !=
      wave->sample_rate) {
    fprintf(stderr, "Expected sample rate %d, got %d\n",
            SherpaOnnxSpeakerSegmentationGetSampleRate(segmenter),
            wave->sample_rate);
    SherpaOnnxDestroySpeakerSegmentation(segmenter);
    SherpaOnnxFreeWave(wave);
    return 1;
  }

  for (int32_t start = 0; start < wave->num_samples; start += 512) {
    int32_t n = wave->num_samples - start;
    if (n > 512) {
      n = 512;
    }
    SherpaOnnxSpeakerSegmentationAcceptWaveform(segmenter,
                                                 wave->samples + start, n);
    while (!SherpaOnnxSpeakerSegmentationEmpty(segmenter)) {
      const SherpaOnnxSpeakerSegmentationSpan *span =
          SherpaOnnxSpeakerSegmentationFront(segmenter);
      assert(span->start < span->end);
      assert(span->speaker_count >= 0 && span->speaker_count <= 2);
      assert((span->flag &
              ~(SherpaOnnxSpeakerSegmentationFlagSpeakerCountChanged |
                SherpaOnnxSpeakerSegmentationFlagSingleSpeakerChanged |
                SherpaOnnxSpeakerSegmentationFlagInputFinished)) == 0);
      printf("%.3f -- %.3f count=%d flag=%d\n", span->start, span->end,
             span->speaker_count, span->flag);
      SherpaOnnxSpeakerSegmentationPop(segmenter);
    }
  }

  SherpaOnnxSpeakerSegmentationInputFinished(segmenter);
  while (!SherpaOnnxSpeakerSegmentationEmpty(segmenter)) {
    const SherpaOnnxSpeakerSegmentationSpan *span =
        SherpaOnnxSpeakerSegmentationFront(segmenter);
    assert(span->start < span->end);
    assert(span->speaker_count >= 0 && span->speaker_count <= 2);
    assert((span->flag &
            ~(SherpaOnnxSpeakerSegmentationFlagSpeakerCountChanged |
              SherpaOnnxSpeakerSegmentationFlagSingleSpeakerChanged |
              SherpaOnnxSpeakerSegmentationFlagInputFinished)) == 0);
    printf("%.3f -- %.3f count=%d flag=%d\n", span->start, span->end,
           span->speaker_count, span->flag);
    SherpaOnnxSpeakerSegmentationPop(segmenter);
  }

  SherpaOnnxDestroySpeakerSegmentation(segmenter);
  SherpaOnnxFreeWave(wave);
  return 0;
}