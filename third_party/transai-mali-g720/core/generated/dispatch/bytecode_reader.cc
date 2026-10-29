// Copyright 2026 TransAI
// SPDX-License-Identifier: Apache-2.0

#include "dispatch/bytecode_reader.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <set>
#include <string_view>

namespace transai::teacher::dispatch {
namespace {

constexpr uint64_t kHeaderSize = 320;
constexpr uint64_t kKernelRecordSize = 80;
constexpr uint64_t kWeightRecordSize = 144;
constexpr uint64_t kScratchRecordSize = 40;
constexpr uint64_t kPlanRecordSize = 48;
constexpr uint64_t kCommandRecordSize = 64;
constexpr uint64_t kArgumentRecordSize = 24;
constexpr uint64_t kSourceHashRecordSize = 40;
constexpr std::array<uint32_t, 11> kLegacyPlans = {24,32,40,48,64,96,128,192,256,288,375};
constexpr std::array<uint32_t,25> kFinePlans = {24,32,40,48,64,80,96,112,128,144,160,176,192,208,224,240,256,272,288,304,320,336,352,368,375};
uint32_t ReadU32(const uint8_t* data) {
  uint32_t value = 0;
  for (uint32_t index = 0; index < 4; ++index) {
    value |= static_cast<uint32_t>(data[index]) << (index * 8);
  }
  return value;
}

uint64_t ReadU64(const uint8_t* data) {
  uint64_t value = 0;
  for (uint32_t index = 0; index < 8; ++index) {
    value |= static_cast<uint64_t>(data[index]) << (index * 8);
  }
  return value;
}

bool RangeValid(uint64_t offset, uint64_t length, uint64_t limit) {
  return offset <= limit && length <= limit - offset;
}

}  // namespace

bool BytecodeReader::Parse(const uint8_t* data, size_t size,
                           BytecodeReader* output, std::string* error) {
  if (data == nullptr || output == nullptr || error == nullptr) return false;
  *output = {};
  error->clear();
  if (size < kHeaderSize || std::memcmp(data, "TAIG720", 7) != 0 ||
      ReadU32(data + 8) != 7 || ReadU32(data + 12) != kHeaderSize) {
    *error = "invalid TAIG720 header";
    return false;
  }
  const uint32_t op_count = ReadU32(data + 28);
  if (ReadU32(data + 16) != 3000 || ReadU32(data + 20) != 375 ||
      ReadU32(data + 24) != 24 ||
      op_count == 0) {
    *error = "TAIG720 teacher identity mismatch";
    return false;
  }
  const uint32_t kernel_count = ReadU32(data + 32);
  const uint32_t weight_count = ReadU32(data + 36);
  const uint32_t scratch_count = ReadU32(data + 40);
  const uint32_t plan_count = ReadU32(data + 44);
  std::vector<uint32_t> expected_plans(kFinePlans.begin(),kFinePlans.end());
  if(plan_count==kLegacyPlans.size())expected_plans.assign(kLegacyPlans.begin(),kLegacyPlans.end());
  const uint64_t kernel_offset = ReadU64(data + 48);
  const uint64_t weight_offset = ReadU64(data + 56);
  const uint64_t scratch_offset = ReadU64(data + 64);
  const uint64_t plan_offset = ReadU64(data + 72);
  const uint64_t payload_offset = ReadU64(data + 80);
  const uint64_t payload_size = ReadU64(data + 88);
  const uint64_t total_size = ReadU64(data + 96);
  const bool external_payload = ReadU32(data + 232) == 1;
  const uint64_t external_payload_size = ReadU64(data + 240);
  const uint32_t command_count = ReadU32(data + 248);
  const uint32_t argument_count = ReadU32(data + 252);
  const uint64_t command_offset = ReadU64(data + 256);
  const uint64_t argument_offset = ReadU64(data + 264);
  const uint32_t source_hash_count = ReadU32(data + 272);
  const uint32_t source_hash_record_size = ReadU32(data + 276);
  const uint64_t source_hash_offset = ReadU64(data + 280);
  const uint32_t quant_policy_size = ReadU32(data + 288);
  const uint64_t quant_policy_offset = ReadU64(data + 296);
  if (kernel_count != 21 ||
      weight_count < 468 ||
      scratch_count != 7 ||
      plan_count != expected_plans.size() || total_size != size ||
      !RangeValid(kernel_offset, kernel_count * kKernelRecordSize,
                  payload_offset) ||
      !RangeValid(weight_offset, weight_count * kWeightRecordSize,
                  payload_offset) ||
      !RangeValid(scratch_offset, scratch_count * kScratchRecordSize,
                  payload_offset) ||
      !RangeValid(plan_offset, plan_count * kPlanRecordSize, payload_offset) ||
      !RangeValid(command_offset, command_count * kCommandRecordSize,
                  payload_offset) ||
      !RangeValid(argument_offset, argument_count * kArgumentRecordSize,
                  payload_offset) ||
      (source_hash_count != 0 &&
       (source_hash_count != weight_count ||
        source_hash_record_size != kSourceHashRecordSize ||
        !RangeValid(source_hash_offset,
                    source_hash_count * kSourceHashRecordSize,
                    payload_offset))) ||
      (quant_policy_size != 0 &&
       !RangeValid(quant_policy_offset, quant_policy_size, payload_offset)) ||
      (kernel_count >= 15 && quant_policy_size == 0) ||
      !RangeValid(payload_offset, payload_size, size) ||
      payload_offset + payload_size != size ||
      (external_payload != (payload_size == 0)) ||
      (external_payload && external_payload_size == 0)) {
    *error = "TAIG720 table or payload bounds mismatch";
    return false;
  }
  const std::string_view policy(
      reinterpret_cast<const char*>(data + quant_policy_offset),
      quant_policy_size);
  const bool teacher_v17 =
      policy.find("\"format_version\": 17") != std::string_view::npos &&
      policy.find(
          "\"policy_id\": \"teacher-v17/full-w8a8-major-linear\"") !=
          std::string_view::npos &&
      policy.find("\"ffn_w8_layers\": \"0-23\"") !=
          std::string_view::npos &&
      policy.find(
          "\"w8_projection_families\": [\"ffn\", \"qkv\", \"conv_pw1\", "
          "\"attention_out\", \"conv_pw2\"]") !=
          std::string_view::npos &&
      policy.find("\"scale_type\": \"fp32\"") !=
          std::string_view::npos;
  const bool middle = policy.find("g720/ffn-middle-sensitive-fp16") != std::string_view::npos;
  const bool disabled = policy.find("g720/ffn-disabled-sensitive-fp16") != std::string_view::npos;
  const bool ffn_only = (middle || disabled || policy.find("g720/ffn-w8a8-sensitive-fp16") != std::string_view::npos) && policy.find("\"w8_projection_families\": [\"ffn\"]") != std::string_view::npos;
  const uint32_t expected_projection_mask = ffn_only ? 0 : 0x0fU;
  const uint32_t expected_ffn_count = disabled ? 0 : (middle ? 16 : 24);
  if (!teacher_v17 && !ffn_only) {
    *error = "TAIG720 Teacher V17 major-linear W8A8 policy mismatch";
    return false;
  }
  uint32_t previous_tensor = 0;
  uint32_t previous_storage = 0;
  std::set<uint32_t> source_tensors;
  std::array<uint32_t, 24> s8_ffn_layer_masks{};
  std::array<uint32_t, 24> s8_projection_layer_masks{};
  output->weights_.reserve(weight_count);
  for (uint32_t index = 0; index < weight_count; ++index) {
    const uint8_t* record = data + weight_offset + index * kWeightRecordSize;
    const uint32_t tensor_index = ReadU32(record);
    const uint32_t rank = ReadU32(record + 12);
    const uint64_t relative_offset = ReadU64(record + 48);
    const uint64_t bytes = ReadU64(record + 56);
    const uint32_t storage = ReadU32(record + 8);
    if ((index != 0 &&
         (tensor_index < previous_tensor ||
          (tensor_index == previous_tensor && storage <= previous_storage))) ||
        rank > 8 ||
        !RangeValid(relative_offset, bytes,
                    external_payload ? external_payload_size : payload_size)) {
      *error = "TAIG720 weight descriptor mismatch";
      return false;
    }
    previous_tensor = tensor_index;
    previous_storage = storage;
    source_tensors.insert(tensor_index);
    WeightRecord value;
    value.tensor_index = tensor_index;
    value.element_type = ReadU32(record + 4);
    value.storage = storage;
    value.dimensions.reserve(rank);
    for (uint32_t dimension = 0; dimension < rank; ++dimension) {
      value.dimensions.push_back(ReadU32(record + 16 + dimension * 4));
    }
    value.offset = relative_offset;
    value.size = bytes;
    value.semantic = ReadU32(record + 96);
    value.layer = static_cast<int32_t>(ReadU32(record + 100));
    value.input_channels = ReadU32(record + 104);
    value.output_channels = ReadU32(record + 108);
    value.pack_format = ReadU32(record + 112);
    value.scale_offset = ReadU64(record + 120);
    value.bias_offset = ReadU64(record + 128);
    if (source_hash_count != 0) {
      const uint8_t* hash_record =
          data + source_hash_offset + index * kSourceHashRecordSize;
      if (ReadU32(hash_record) != value.tensor_index ||
          ReadU32(hash_record + 4) != value.storage) {
        *error = "TAIG720 source hash descriptor mismatch";
        return false;
      }
      std::memcpy(value.source_hash.data(), hash_record + 8,
                  value.source_hash.size());
    }
    const uint64_t scale_size =
        value.storage == 8 ? static_cast<uint64_t>(value.output_channels) * 4
                           : 0;
    const uint64_t bias_size =
        value.storage == 8 ? static_cast<uint64_t>(value.output_channels) * 2
                           : 0;
    const bool s8_with_bias =
        value.semantic == 7 || value.semantic == 9 ||
        value.semantic == 11 || value.semantic == 14;
    const bool s8_without_bias =
        value.semantic == 8 || value.semantic == 10 ||
        value.semantic == 13 || value.semantic == 15;
    if (value.storage > 8 || value.semantic > 21 ||
        (value.storage == 3 &&
         (value.pack_format != 2 ||
          value.input_channels == 0 ||
          value.output_channels == 0)) ||
        (value.storage == 8 &&
         (value.pack_format != 4 ||
          (!s8_with_bias && !s8_without_bias) ||
          value.layer < 0 || value.layer >= 24 ||
          value.input_channels == 0 ||
          value.output_channels == 0 ||
          !RangeValid(value.scale_offset, scale_size,
                      external_payload ? external_payload_size
                                       : payload_size) ||
          (s8_with_bias &&
           (value.bias_offset == 0 ||
            !RangeValid(value.bias_offset, bias_size,
                        external_payload ? external_payload_size
                                         : payload_size))) ||
          (s8_without_bias && value.bias_offset != 0)))) {
      *error = "TAIG720 weight packing metadata mismatch";
      return false;
    }
    if (value.storage == 8) {
      uint32_t* layer_mask = nullptr;
      uint32_t semantic_bit = 0;
      if (value.semantic == 7) {
        layer_mask = &s8_ffn_layer_masks[value.layer];
        semantic_bit = 1U;
      } else if (value.semantic == 8) {
        layer_mask = &s8_ffn_layer_masks[value.layer];
        semantic_bit = 2U;
      } else if (value.semantic == 14) {
        layer_mask = &s8_ffn_layer_masks[value.layer];
        semantic_bit = 4U;
      } else if (value.semantic == 15) {
        layer_mask = &s8_ffn_layer_masks[value.layer];
        semantic_bit = 8U;
      } else if (value.semantic == 9) {
        layer_mask = &s8_projection_layer_masks[value.layer];
        semantic_bit = 1U;
      } else if (value.semantic == 11) {
        layer_mask = &s8_projection_layer_masks[value.layer];
        semantic_bit = 2U;
      } else if (value.semantic == 10) {
        layer_mask = &s8_projection_layer_masks[value.layer];
        semantic_bit = 4U;
      } else {
        layer_mask = &s8_projection_layer_masks[value.layer];
        semantic_bit = 8U;
      }
      if ((*layer_mask & semantic_bit) != 0) {
        *error = "TAIG720 S8 layer descriptor is duplicated";
        return false;
      }
      *layer_mask |= semantic_bit;
    }
    output->weights_.push_back(std::move(value));
  }
  if (source_tensors.size() != 468) {
    *error = "TAIG720 canonical source tensor count mismatch";
    return false;
  }
  const uint32_t w8_layer_count = static_cast<uint32_t>(std::count_if(
      s8_ffn_layer_masks.begin(), s8_ffn_layer_masks.end(),
      [](uint32_t mask) { return mask == 0x0fU; }));
  if (std::any_of(s8_ffn_layer_masks.begin(), s8_ffn_layer_masks.end(),
                  [](uint32_t mask) {
                    return mask != 0U && mask != 0x0fU;
                  })) {
    *error = "TAIG720 S8 FFN layer descriptor set is incomplete";
    return false;
  }
  const uint32_t w8_projection_layer_count =
      static_cast<uint32_t>(std::count(
          s8_projection_layer_masks.begin(),
          s8_projection_layer_masks.end(), expected_projection_mask));
  if (std::any_of(
          s8_projection_layer_masks.begin(),
          s8_projection_layer_masks.end(),
          [&](uint32_t mask) { return mask != expected_projection_mask; })) {
    *error = "TAIG720 S8 projection layer descriptor set is incomplete";
    return false;
  }
  for (int i=0;i<24;++i) { const bool expected = !disabled && (!middle || (i>=4 && i<20)); if(s8_ffn_layer_masks[i] != (expected ? 15U : 0U)) { *error="FFN precision mask mismatch";return false; } }
  if (weight_count != 468 || w8_layer_count != expected_ffn_count ||
      w8_projection_layer_count != 24) {
    *error =
        "TAIG720 Teacher V17 canonical major-linear descriptor contract mismatch";
    return false;
  }
  output->scratch_.reserve(scratch_count);
  for (uint32_t index = 0; index < scratch_count; ++index) {
    const uint8_t* record = data + scratch_offset + index * kScratchRecordSize;
    ScratchRecord value{ReadU32(record), ReadU32(record + 4),
                        ReadU32(record + 8), ReadU64(record + 16),
                        ReadU64(record + 24)};
    if (!RangeValid(value.offset, value.size,
                    std::numeric_limits<uint64_t>::max())) {
      *error = "TAIG720 scratch descriptor overflow";
      return false;
    }
    output->scratch_.push_back(value);
  }
  output->plans_.reserve(plan_count);
  for (uint32_t index = 0; index < plan_count; ++index) {
    const uint8_t* record = data + plan_offset + index * kPlanRecordSize;
    PlanRecord value{
        ReadU32(record), ReadU32(record + 4), ReadU32(record + 8),
        ReadU32(record + 12), ReadU32(record + 16), ReadU32(record + 20),
        ReadU32(record + 24), ReadU32(record + 28), ReadU64(record + 32)};
    if (value.encoder_frames != expected_plans[index] ||
        value.mel_capacity != value.encoder_frames * 8 ||
        value.pre0_frames != value.encoder_frames * 4 ||
        value.pre1_frames != value.encoder_frames * 2 ||
        value.pre2_frames != value.encoder_frames ||
        value.output_frames != value.encoder_frames ||
        value.command_count != 513U + 2U * w8_layer_count - (ffn_only ? 48U : 0U) ||
        !RangeValid(value.command_begin, value.command_count, command_count)) {
      *error = "TAIG720 command plan mismatch";
      return false;
    }
    output->plans_.push_back(value);
  }
  output->commands_.reserve(command_count);
  for (uint32_t index = 0; index < command_count; ++index) {
    const uint8_t* record = data + command_offset + index * kCommandRecordSize;
    const uint32_t identity = ReadU32(record);
    CommandRecord command;
    command.kernel_id = static_cast<uint16_t>(identity & 0xffffU);
    command.dimensions = static_cast<uint16_t>(identity >> 16);
    command.flags = ReadU32(record + 4);
    for (size_t axis = 0; axis < 3; ++axis) {
      command.global[axis] = ReadU32(record + 8 + axis * 4);
      command.local[axis] = ReadU32(record + 20 + axis * 4);
    }
    command.arg_begin = ReadU32(record + 32);
    command.arg_count = ReadU32(record + 36);
    command.semantic = ReadU32(record + 40);
    command.layer = static_cast<int32_t>(ReadU32(record + 44));
    command.mode = ReadU32(record + 48);
    command.rows = ReadU32(record + 52);
    command.input_channels = ReadU32(record + 56);
    command.output_channels = ReadU32(record + 60);
    if (command.kernel_id >= kernel_count || command.dimensions == 0 ||
        command.dimensions > 3 ||
        command.semantic > 21 ||
        !RangeValid(command.arg_begin, command.arg_count, argument_count)) {
      *error = "TAIG720 command descriptor mismatch";
      return false;
    }
    output->commands_.push_back(command);
  }
  output->arguments_.reserve(argument_count);
  for (uint32_t index = 0; index < argument_count; ++index) {
    const uint8_t* record = data + argument_offset + index * kArgumentRecordSize;
    ArgumentRecord argument{ReadU32(record), ReadU32(record + 4),
                            ReadU64(record + 8)};
    if (argument.kind > 7) {
      *error = "TAIG720 argument descriptor mismatch";
      return false;
    }
    output->arguments_.push_back(argument);
  }
  output->payload_ = data + payload_offset;
  output->payload_size_ = payload_size;
  output->external_payload_size_ = external_payload_size;
  return true;
}

const WeightRecord* BytecodeReader::FindWeight(uint32_t tensor_index) const {
  const auto found = std::lower_bound(
      weights_.begin(), weights_.end(), tensor_index,
      [](const WeightRecord& value, uint32_t index) {
        return value.tensor_index < index;
      });
  return found == weights_.end() || found->tensor_index != tensor_index
             ? nullptr
             : &*found;
}

const WeightRecord* BytecodeReader::FindWeight(uint32_t semantic,
                                               int32_t layer,
                                               uint32_t storage) const {
  const auto found = std::find_if(
      weights_.begin(), weights_.end(), [&](const WeightRecord& value) {
        return value.semantic == semantic && value.layer == layer &&
               value.storage == storage;
      });
  return found == weights_.end() ? nullptr : &*found;
}

}  // namespace transai::teacher::dispatch
