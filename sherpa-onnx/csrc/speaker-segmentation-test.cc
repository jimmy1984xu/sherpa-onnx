// sherpa-onnx/csrc/speaker-segmentation-test.cc
//
// Copyright (c) 2026

#include "sherpa-onnx/csrc/speaker-segmentation-fusion.h"
#include "sherpa-onnx/csrc/speaker-segmentation.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <vector>

#include "gtest/gtest.h"

namespace sherpa_onnx {
namespace {

SpeakerSegmentationFusionConfig MakeFusionConfig(
    float min_duration_on = 0.0F, float min_duration_off = 0.0F,
    float change_vote_threshold = 0.5F) {
  SpeakerSegmentationFusionConfig config;
  config.frame_shift_samples = 160;
  config.window_shift_samples = 16000;
  config.sample_rate = 16000;
  config.min_duration_on = min_duration_on;
  config.min_duration_off = min_duration_off;
  config.change_vote_threshold = change_vote_threshold;
  return config;
}

TEST(SpeakerSegmentationFusion, CountsUseCoverageAverageAndRound) {
  SpeakerSegmentationFusion fuser(MakeFusionConfig());
  fuser.AddWindow(/*start_frame=*/0, {0b001, 0b011});
  fuser.AddWindow(/*start_frame=*/0, {0b001, 0b001});

  auto frames = fuser.FinalizeBefore(/*frame_exclusive=*/2);

  ASSERT_EQ(frames.size(), 2);
  EXPECT_EQ(frames[0].frame_index, 0);
  EXPECT_EQ(frames[0].speaker_count, 1);
  EXPECT_FALSE(frames[0].single_speaker_changed_before);
  EXPECT_EQ(frames[1].frame_index, 1);
  EXPECT_EQ(frames[1].speaker_count, 2);  // round(1.5) == 2
}

TEST(SpeakerSegmentationFusion, ReversedLocalTracksVoteForOneChange) {
  SpeakerSegmentationFusion fuser(MakeFusionConfig());
  fuser.AddWindow(/*start_frame=*/0, {0b001, 0b001, 0b010, 0b010});
  fuser.AddWindow(/*start_frame=*/0, {0b010, 0b010, 0b001, 0b001});

  auto frames = fuser.FinalizeBefore(/*frame_exclusive=*/4);

  ASSERT_EQ(frames.size(), 4);
  EXPECT_FALSE(frames[0].single_speaker_changed_before);
  EXPECT_FALSE(frames[1].single_speaker_changed_before);
  EXPECT_TRUE(frames[2].single_speaker_changed_before);
  EXPECT_FALSE(frames[3].single_speaker_changed_before);
  for (const auto &frame : frames) {
    EXPECT_EQ(frame.speaker_count, 1);
  }
}

TEST(SpeakerSegmentationFusion, MinDurationOnDoesNotRewriteCounts) {
  SpeakerSegmentationFusion fuser(
      MakeFusionConfig(/*min_duration_on=*/0.03F));  // 3 frames at 10 ms
  fuser.AddWindow(/*start_frame=*/0, {0b001, 0b001, 0b010, 0b001, 0b001});

  auto frames = fuser.FinalizeBefore(/*frame_exclusive=*/5);

  ASSERT_EQ(frames.size(), 5);
  for (const auto &frame : frames) {
    EXPECT_EQ(frame.speaker_count, 1);
    EXPECT_FALSE(frame.single_speaker_changed_before);
  }
}

TEST(SpeakerSegmentationFusion, MinDurationOffDoesNotRewriteCounts) {
  SpeakerSegmentationFusion fuser(
      MakeFusionConfig(/*min_duration_on=*/0.0F, /*min_duration_off=*/0.03F));
  fuser.AddWindow(/*start_frame=*/0,
                  {0b001, 0b001, 0b000, 0b001, 0b001, 0b010, 0b010});

  auto frames = fuser.FinalizeBefore(/*frame_exclusive=*/7);

  ASSERT_EQ(frames.size(), 7);
  EXPECT_EQ(frames[2].speaker_count, 0);  // Raw count preserves the gap.
  EXPECT_TRUE(frames[5].single_speaker_changed_before);
}


OfflineSpeakerSegmentationPyannoteModelMetaData MakeTestMetaData() {
  OfflineSpeakerSegmentationPyannoteModelMetaData meta_data;
  meta_data.sample_rate = 16000;
  meta_data.window_size = 160000;
  meta_data.window_shift = 16000;
  meta_data.receptive_field_size = 1600;
  meta_data.receptive_field_shift = 1600;
  meta_data.num_speakers = 3;
  meta_data.powerset_max_classes = 2;
  meta_data.num_classes = 7;
  return meta_data;
}

std::unique_ptr<SpeakerSegmentation> CreateTestSegmenter() {
  SpeakerSegmentationConfig config;
  return SpeakerSegmentation::CreateForTesting(
      config, MakeTestMetaData(),
      [](const std::vector<float> &window) {
        EXPECT_EQ(window.size(), 160000);
        return std::vector<uint8_t>(100, 0b001);
      });
}

std::vector<SpeakerSegmentationSpan> Drain(SpeakerSegmentation *segmenter) {
  std::vector<SpeakerSegmentationSpan> spans;
  while (!segmenter->Empty()) {
    spans.push_back(segmenter->Front());
    segmenter->Pop();
  }
  return spans;
}

TEST(SpeakerSegmentation, ChunkedAndOneShotHaveSameFinalSpans) {
  std::vector<float> audio(196800, 0.1F);  // 12.3 seconds at 16 kHz

  auto one_shot = CreateTestSegmenter();
  one_shot->AcceptWaveform(audio.data(), static_cast<int32_t>(audio.size()));
  one_shot->InputFinished();
  const auto one_shot_spans = Drain(one_shot.get());

  auto chunked = CreateTestSegmenter();
  for (size_t start = 0; start < audio.size(); start += 512) {
    const int32_t n = static_cast<int32_t>(
        std::min<size_t>(512, audio.size() - start));
    chunked->AcceptWaveform(audio.data() + start, n);
  }
  chunked->InputFinished();
  const auto chunked_spans = Drain(chunked.get());

  ASSERT_EQ(one_shot_spans.size(), chunked_spans.size());
  for (size_t i = 0; i != one_shot_spans.size(); ++i) {
    EXPECT_FLOAT_EQ(one_shot_spans[i].start, chunked_spans[i].start);
    EXPECT_FLOAT_EQ(one_shot_spans[i].end, chunked_spans[i].end);
    EXPECT_EQ(one_shot_spans[i].speaker_count,
              chunked_spans[i].speaker_count);
    EXPECT_EQ(one_shot_spans[i].flag, chunked_spans[i].flag);
  }
}

TEST(SpeakerSegmentation, TenSecondFirstWindowPublishesFirstOneSecond) {
  auto segmenter = CreateTestSegmenter();
  std::vector<float> audio(160000, 0.1F);
  segmenter->AcceptWaveform(audio.data(), static_cast<int32_t>(audio.size()));

  ASSERT_FALSE(segmenter->Empty());
  const auto &span = segmenter->Front();
  EXPECT_FLOAT_EQ(span.start, 0.0F);
  EXPECT_FLOAT_EQ(span.end, 1.0F);
  EXPECT_EQ(span.speaker_count, 1);
  EXPECT_EQ(span.flag, kSpeakerSegmentationContinue);
}

TEST(SpeakerSegmentation, InputFinishedPadsTailButClipsToAudioEnd) {
  auto segmenter = CreateTestSegmenter();
  std::vector<float> audio(196800, 0.1F);  // 12.3 seconds
  segmenter->AcceptWaveform(audio.data(), static_cast<int32_t>(audio.size()));
  segmenter->InputFinished();
  const auto spans = Drain(segmenter.get());

  ASSERT_FALSE(spans.empty());
  EXPECT_FLOAT_EQ(spans.back().end, 12.3F);
  EXPECT_NE(spans.back().flag & kSpeakerSegmentationInputFinished, 0);
  for (const auto &span : spans) {
    EXPECT_LE(span.end, 12.3F);
  }
}

TEST(SpeakerSegmentation, ResetAndTwoObjectsDoNotShareState) {
  auto first = CreateTestSegmenter();
  auto second = CreateTestSegmenter();
  std::vector<float> audio(160000, 0.1F);

  first->AcceptWaveform(audio.data(), static_cast<int32_t>(audio.size()));
  EXPECT_FALSE(first->Empty());
  EXPECT_TRUE(second->Empty());

  first->Reset();
  EXPECT_TRUE(first->Empty());
  first->AcceptWaveform(audio.data(), static_cast<int32_t>(audio.size()));
  EXPECT_FALSE(first->Empty());
  EXPECT_TRUE(second->Empty());
}

TEST(SpeakerSegmentation, SpanFlagsDescribeRightBoundary) {
  auto create_with_masks = [](std::vector<uint8_t> masks) {
    SpeakerSegmentationConfig config;
    return SpeakerSegmentation::CreateForTesting(
        config, MakeTestMetaData(),
        [masks = std::move(masks)](const std::vector<float> &window) {
          EXPECT_EQ(window.size(), 160000);
          return masks;
        });
  };

  std::vector<float> audio(240000, 0.1F);  // enough to publish through 6 s
  std::vector<uint8_t> single_speaker_change(100, 0b001);
  std::fill(single_speaker_change.begin() + 50,
            single_speaker_change.end(), 0b010);
  auto single = create_with_masks(std::move(single_speaker_change));
  single->AcceptWaveform(audio.data(), static_cast<int32_t>(audio.size()));
  single->InputFinished();
  const auto single_spans = Drain(single.get());

  ASSERT_FALSE(single_spans.empty());
  EXPECT_NE(std::find_if(single_spans.begin(), single_spans.end(),
                         [](const SpeakerSegmentationSpan &span) {
                           return span.flag &
                                  kSpeakerSegmentationSingleSpeakerChanged;
                         }),
            single_spans.end());

  std::vector<uint8_t> count_change(100, 0b001);
  std::fill(count_change.begin() + 50, count_change.end(), 0b011);
  auto count = create_with_masks(std::move(count_change));
  count->AcceptWaveform(audio.data(), static_cast<int32_t>(audio.size()));
  count->InputFinished();
  const auto count_spans = Drain(count.get());

  ASSERT_FALSE(count_spans.empty());
  EXPECT_NE(std::find_if(count_spans.begin(), count_spans.end(),
                         [](const SpeakerSegmentationSpan &span) {
                           return span.flag &
                                  kSpeakerSegmentationSpeakerCountChanged;
                         }),
            count_spans.end());
  EXPECT_NE(count_spans.back().flag & kSpeakerSegmentationInputFinished, 0);
}
TEST(SpeakerSegmentation, RejectsAcceptAfterInputFinishedUntilReset) {
  auto segmenter = CreateTestSegmenter();
  std::vector<float> audio(160000, 0.1F);
  segmenter->AcceptWaveform(audio.data(), static_cast<int32_t>(audio.size()));
  segmenter->InputFinished();
  Drain(segmenter.get());

  segmenter->AcceptWaveform(audio.data(), 16000);
  EXPECT_TRUE(segmenter->Empty());

  segmenter->Reset();
  segmenter->AcceptWaveform(audio.data(), static_cast<int32_t>(audio.size()));
  EXPECT_FALSE(segmenter->Empty());
}
}  // namespace
}  // namespace sherpa_onnx
