set(vox_kai_root "${PROJECT_SOURCE_DIR}/third_party/titanet-kleidiai")
set(vox_matmul "${vox_kai_root}/kai/ukernels/matmul")
set(vox_i8mm "${vox_matmul}/matmul_clamp_f32_qai8dxp_qsi8cxp/kai_matmul_clamp_f32_qai8dxp4x8_qsi8cxp4x8_16x4_neon_i8mm")

if(NOT TARGET onnxruntime)
  message(FATAL_ERROR "voxlingua107_neon_ops requires the configured onnxruntime target")
endif()
get_target_property(vox_onnxruntime_include_dirs onnxruntime INTERFACE_INCLUDE_DIRECTORIES)
if(NOT vox_onnxruntime_include_dirs)
  message(FATAL_ERROR "onnxruntime target does not expose its include directories")
endif()

add_library(voxlingua107_neon_ops SHARED
  neon-ops.cc
  "${vox_matmul}/pack/kai_lhs_quant_pack_qai8dxp_f32.c"
  "${vox_matmul}/pack/kai_rhs_pack_nxk_qsi8cxp_qsi8cx_neon.c"
  "${vox_i8mm}.c"
  "${vox_i8mm}_asm.S"
)
target_include_directories(voxlingua107_neon_ops PRIVATE
  "${vox_kai_root}"
  "${vox_matmul}"
  ${vox_onnxruntime_include_dirs}
)
target_compile_options(voxlingua107_neon_ops PRIVATE
  -O3
  -march=armv8.6-a+dotprod+i8mm
  -mtune=cortex-a520
  -fvisibility=hidden
)
target_link_options(voxlingua107_neon_ops PRIVATE -Wl,-z,max-page-size=16384)

install(TARGETS voxlingua107_neon_ops
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib
  ARCHIVE DESTINATION lib
)
