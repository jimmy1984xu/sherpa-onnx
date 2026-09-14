// sherpa-onnx/csrc/speaker-segmentation-test.cc
//
// Copyright (c) 2026

#include "sherpa-onnx/csrc/speaker-segmentation-fusion.h"

#include <cstdint>
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

}  // namespace
}  // namespace sherpa_onnx
