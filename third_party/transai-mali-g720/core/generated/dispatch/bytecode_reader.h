// Copyright 2026 TransAI
// SPDX-License-Identifier: Apache-2.0

#ifndef TRANSAI_TEACHER_BYTECODE_READER_H_
#define TRANSAI_TEACHER_BYTECODE_READER_H_

#include <cstddef>
#include <cstdint>
#include <array>
#include <string>
#include <vector>

namespace transai::teacher::dispatch {

struct ScratchRecord {
  uint32_t id = 0;
  uint32_t first_command = 0;
  uint32_t last_command = 0;
  uint64_t offset = 0;
  uint64_t size = 0;
};

struct PlanRecord {
  uint32_t encoder_frames = 0;
  uint32_t mel_capacity = 0;
  uint32_t pre0_frames = 0;
  uint32_t pre1_frames = 0;
  uint32_t pre2_frames = 0;
  uint32_t output_frames = 0;
  uint32_t command_begin = 0;
  uint32_t command_count = 0;
  uint64_t scratch_bytes = 0;
};

struct CommandRecord {
  uint16_t kernel_id = 0;
  uint16_t dimensions = 0;
  uint32_t flags = 0;
  uint32_t global[3]{};
  uint32_t local[3]{};
  uint32_t arg_begin = 0;
  uint32_t arg_count = 0;
  uint32_t semantic = 0;
  int32_t layer = -1;
  uint32_t mode = 0;
  uint32_t rows = 0;
  uint32_t input_channels = 0;
  uint32_t output_channels = 0;
};

struct ArgumentRecord {
  uint32_t kind = 0;
  uint32_t slot = 0;
  uint64_t value = 0;
};

struct WeightRecord {
  uint32_t tensor_index = 0;
  uint32_t element_type = 0;
  uint32_t storage = 0;
  std::vector<uint32_t> dimensions;
  uint64_t offset = 0;
  uint64_t size = 0;
  uint32_t semantic = 0;
  int32_t layer = -1;
  uint32_t input_channels = 0;
  uint32_t output_channels = 0;
  uint32_t pack_format = 0;
  uint64_t scale_offset = 0;
  uint64_t bias_offset = 0;
  std::array<uint8_t, 32> source_hash{};
};

class BytecodeReader {
 public:
  static bool Parse(const uint8_t* data, size_t size, BytecodeReader* output,
                    std::string* error);

  const uint8_t* payload() const { return payload_; }
  uint64_t payload_size() const { return payload_size_; }
  bool has_external_payload() const { return external_payload_size_ != 0; }
  uint64_t external_payload_size() const { return external_payload_size_; }
  const std::vector<ScratchRecord>& scratch() const { return scratch_; }
  const std::vector<PlanRecord>& plans() const { return plans_; }
  const std::vector<WeightRecord>& weights() const { return weights_; }
  const std::vector<CommandRecord>& commands() const { return commands_; }
  const std::vector<ArgumentRecord>& arguments() const { return arguments_; }
  const WeightRecord* FindWeight(uint32_t tensor_index) const;
  const WeightRecord* FindWeight(uint32_t semantic, int32_t layer,
                                 uint32_t storage) const;

 private:
  const uint8_t* payload_ = nullptr;
  uint64_t payload_size_ = 0;
  uint64_t external_payload_size_ = 0;
  std::vector<ScratchRecord> scratch_;
  std::vector<PlanRecord> plans_;
  std::vector<WeightRecord> weights_;
  std::vector<CommandRecord> commands_;
  std::vector<ArgumentRecord> arguments_;
};

}  // namespace transai::teacher::dispatch

#endif  // TRANSAI_TEACHER_BYTECODE_READER_H_
