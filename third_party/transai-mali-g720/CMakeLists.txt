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

foreach(version IN ITEMS 2 3)
  set(plugin transai_parakeet_gpu_v${version})
  set(exports "${CMAKE_CURRENT_SOURCE_DIR}/v${version}/sdk.exports")
  add_library(${plugin} SHARED v${version}/sdk.cc)
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
