# 与交付包专用构建保持相同的源码、优化选项和插件 ABI。
if(NOT ANDROID OR NOT ANDROID_ABI STREQUAL "arm64-v8a")
  message(FATAL_ERROR "TransAI Mali-G720 plugins require Android arm64-v8a")
endif()

add_library(transai-mali-g720-core STATIC
  core/generated/dispatch/bytecode_reader.cc
  core/generated/dispatch/opencl_runtime.cc
  core/generated/dispatch/event_profiler.cc
  core/generated/command_buffer.cc
  core/vendor/src/native/runtime/opencl/event_profiler.cc
)
set_target_properties(transai-mali-g720-core PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
)
target_include_directories(transai-mali-g720-core PUBLIC
  "${CMAKE_SOURCE_DIR}"
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${CMAKE_CURRENT_SOURCE_DIR}/core"
  "${CMAKE_CURRENT_SOURCE_DIR}/core/generated"
  "${CMAKE_CURRENT_SOURCE_DIR}/core/vendor"
  "${CMAKE_CURRENT_SOURCE_DIR}/core/vendor/src/native/include"
  "${CMAKE_CURRENT_SOURCE_DIR}/core/vendor/third_party/OpenCL-Headers"
)
target_compile_definitions(transai-mali-g720-core PUBLIC
  CL_ENABLE_BETA_EXTENSIONS NDEBUG
)
target_compile_options(transai-mali-g720-core PUBLIC -O3 -ffp-contract=off)

# 将随交付包提供的 Mali-G720 beam AOT 程序嵌入对应插件。
# 这与 app/android/build.py 的生成规则一致，避免运行时依赖外部 clbin 文件。
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/beam-programs.json" _transai_beam_receipt)
string(FIND "${_transai_beam_receipt}" "\"target\"" _transai_beam_target_key)
if(_transai_beam_target_key EQUAL -1)
  message(FATAL_ERROR "Unable to find transai beam target key")
endif()
string(SUBSTRING "${_transai_beam_receipt}" ${_transai_beam_target_key} -1 _transai_beam_target_tail)
string(FIND "${_transai_beam_target_tail}" "{" _transai_beam_target_open)
string(FIND "${_transai_beam_target_tail}" "}" _transai_beam_target_close)
if(_transai_beam_target_open EQUAL -1 OR _transai_beam_target_close EQUAL -1 OR _transai_beam_target_close LESS _transai_beam_target_open)
  message(FATAL_ERROR "Unable to extract transai beam target object")
endif()
math(EXPR _transai_beam_target_length "${_transai_beam_target_close} - ${_transai_beam_target_open} + 1")
string(SUBSTRING "${_transai_beam_target_tail}" ${_transai_beam_target_open} ${_transai_beam_target_length} _transai_beam_target_json)

function(transai_generate_beam_source version)
  set(binary "${CMAKE_CURRENT_SOURCE_DIR}/v${version}/beam.clbin")
  if(NOT EXISTS "${binary}")
    message(FATAL_ERROR "Missing Mali-G720 beam binary: ${binary}")
  endif()

  file(READ "${binary}" binary_hex HEX)
  string(REGEX REPLACE "([0-9A-Fa-f][0-9A-Fa-f])" "0x\\1," binary_values "${binary_hex}")
  set(source "#include <stddef.h>\n")
  string(APPEND source "extern const unsigned char transai_beam_binary[] = {")
  string(APPEND source "${binary_values}")
  string(APPEND source "};\n")
  string(APPEND source "extern const size_t transai_beam_binary_bytes = sizeof(transai_beam_binary);\n")
  string(APPEND source "extern const char transai_beam_target[] = R\"json(${_transai_beam_target_json})json\";\n")

  set(generated "${CMAKE_CURRENT_BINARY_DIR}/beam_binary_v${version}.cc")
  file(GENERATE OUTPUT "${generated}" CONTENT "${source}")
  set(TRANS_AI_BEAM_SOURCE_${version} "${generated}" PARENT_SCOPE)
endfunction()

foreach(version IN ITEMS 2 3)
  transai_generate_beam_source(${version})
  set(plugin transai_parakeet_gpu_v${version})
  set(exports "${CMAKE_CURRENT_SOURCE_DIR}/v${version}/sdk.exports")
  add_library(${plugin} SHARED
    v${version}/sdk.cc
    "${TRANS_AI_BEAM_SOURCE_${version}}"
  )
  set_target_properties(${plugin} PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_VISIBILITY_PRESET hidden
    LINK_DEPENDS "${exports}"
  )
  target_link_libraries(${plugin} PRIVATE transai-mali-g720-core ${CMAKE_DL_LIBS} log)
  # 插件通过纯 C ABI 交互，独立静态链接 C++ 运行库，与原交付方式一致。
  target_link_options(${plugin} PRIVATE
    -static-libstdc++ "-Wl,--version-script=${exports}"
  )
  install(TARGETS ${plugin} LIBRARY DESTINATION lib)
endforeach()
