// Immutable SHA-256 identities for the bundled release assets.
#ifndef SHERPA_ONNX_CSRC_MALI_ASSET_HASHES_H_
#define SHERPA_ONNX_CSRC_MALI_ASSET_HASHES_H_
namespace sherpa_onnx {
struct MaliG720Asset { int version; const char *role,*name,*sha256; };
inline constexpr MaliG720Asset kMaliG720Assets[] = {
  {2,"encoder","encoder.full.nemo.plan.taig720","b57475d6675dc250dd4c6d49d0e40bd946d044d75090dbd9d856d21c3b611bb3"},
  {2,"encoder","encoder.weights.bin","8d5b24aef8164d572af0a0a92539b59c6d5acebea80b9f4ed977e656015bad62"},
  {2,"encoder","teacher_kernels.cl","42b8b668bc3599bc72bb84ec2b16f4aa2220589db0fea1041c9d59545ac321e9"},
  {2,"gpu","frontend.bin","64028cd11841f1d0722893d224e4497f2bb0fd9ceac73a8e81c42aa246843f47"},
  {2,"gpu","frontend.cl","a6ec2d85f6ec28cc5d9e348f468eeb1096d770a78731fb3a147e497ea6de5381"},
  {2,"gpu","frontend_layout.clh","4b1f5618cc944bc2d63af014ecb35d1068d3e6054f5c17e45952103bfaf4d495"},
  {2,"gpu","tdt_weights.bin","5240e79ff8e74935354d4439b1caaa3d74251676b35733e08c1bfe2eb27ff7db"},
  {2,"gpu","tdt_layout.clh","010ed941685e6de8fc1452736aaa5c41f11e52b1db2b4e10652721105d77cca8"},
  {2,"gpu","tdt.cl","777d7731e27192a943ae862f8c7d6aca1c4a221a39fedca29e3933bed396a99d"},
  {2,"gpu","multi.cl","a6abe0b4ebc1f2b636c83f0f3ec82151a765914fd27eb96478f893c2bdf85b0b"},
  {2,"tokens","tokens.txt","d5b4ca312b07f10cc89b6ff1f500201b5bd826dc3f9500c115caeb36c0b10831"},
  {3,"encoder","encoder.full.nemo.plan.taig720","8f6b7ab09b6f1d803f2de4d5d6211532d0329e3fbd771914829f80fed5c7845c"},
  {3,"encoder","encoder.weights.bin","dd220c20ce0bd0d4ace64cc1ebfcf5dd61fc041454adee963f66c84884b34436"},
  {3,"encoder","teacher_kernels.cl","42b8b668bc3599bc72bb84ec2b16f4aa2220589db0fea1041c9d59545ac321e9"},
  {3,"gpu","frontend.bin","d505e3b00cad8713126f4a1da9cee7fc8fc73add69b28bb055d8ac978ab373b7"},
  {3,"gpu","frontend.cl","f21fe4f715788777702b702637353bdb1d5506cd42f691046c668bd4703db2a2"},
  {3,"gpu","frontend_layout.clh","0aa12b5a11b07d0b246cc7cd38f47f769990c9a2a4b69a52d7cd02d63d83b614"},
  {3,"gpu","tdt_weights.bin","5fce9a464ae30f3fef76e134d6ce196864d7075ce929f00d137730625eadea60"},
  {3,"gpu","tdt_layout.clh","4cc3f87580c28d03a4b1b4fec27de71ba024563c05c6c793264d0c1e6818d95b"},
  {3,"gpu","tdt.cl","e4651c5cc027d1301c4acc34a9a8b9d28873b3ddb4d8f7ea9c00e2ed590bff31"},
  {3,"gpu","multi.cl","821f4143a397c67bf91abeacc2b9faaeeb2111f4e07e0ace07e81c9c68137b64"},
  {3,"tokens","tokens.txt","ba8e4007c65f4bb4358ffe2ecc13d9ccc7a10351151065242b5c3a943e685742"},
};
}
#endif
