#pragma once
#include <stddef.h>
#include <stdint.h>
// Private plugin ABI. Public sherpa C structs/functions remain untouched.
struct SeacoGpuResult {const int32_t *ids;const float *scores;const char *text;size_t count,text_bytes;int result_reads,intermediate_reads;};
