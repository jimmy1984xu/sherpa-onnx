# 对齐输入包 app/android/build.py 的独立 SeACo 插件构建。
if(NOT ANDROID OR NOT ANDROID_ABI STREQUAL "arm64-v8a")
  message(FATAL_ERROR "TransAI SeACo Mali-G720 plugin requires Android arm64-v8a")
endif()

set(seaco_exports "${CMAKE_CURRENT_SOURCE_DIR}/plugin.exports")
add_library(transai_seaco_paraformer_gpu SHARED plugin.cc)
set_target_properties(transai_seaco_paraformer_gpu PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
  CXX_VISIBILITY_PRESET hidden
  LINK_DEPENDS "${seaco_exports}"
)
target_include_directories(transai_seaco_paraformer_gpu PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/../.."
  "${CMAKE_CURRENT_SOURCE_DIR}/../transai-mali-g720/core/vendor/third_party/OpenCL-Headers"
)
target_compile_options(transai_seaco_paraformer_gpu PRIVATE -O3 -ffp-contract=off)
target_link_libraries(transai_seaco_paraformer_gpu PRIVATE
  nlohmann_json::nlohmann_json ${CMAKE_DL_LIBS} log
)
target_link_options(transai_seaco_paraformer_gpu PRIVATE
  -static-libstdc++ "-Wl,--version-script=${seaco_exports}"
)
install(TARGETS transai_seaco_paraformer_gpu LIBRARY DESTINATION lib)
