#!/usr/bin/env python3

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


def load_module():
    script_path = Path(__file__).resolve().parent.parent / "utils-tts_model_benchmark.py"
    spec = importlib.util.spec_from_file_location("utils_tts_model_benchmark", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakeKokoroConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class FakeModelConfig:
    def __init__(self, num_threads, provider):
        self.num_threads = num_threads
        self.provider = provider
        self.kokoro = None


class FakeTtsConfig:
    def __init__(self, model, rule_fsts="", max_num_sentences=1):
        self.model = model
        self.rule_fsts = rule_fsts
        self.max_num_sentences = max_num_sentences


class FakeGenerationConfig:
    def __init__(self):
        self.sid = 0
        self.speed = 1.0


class FakeSherpaOnnx:
    OfflineTtsKokoroModelConfig = FakeKokoroConfig
    OfflineTtsModelConfig = FakeModelConfig
    OfflineTtsConfig = FakeTtsConfig
    GenerationConfig = FakeGenerationConfig


class TestUtilsTtsModelBenchmark(unittest.TestCase):
    def setUp(self):
        self.module = load_module()
        self.fake_sherpa_onnx = FakeSherpaOnnx()
        self.args = SimpleNamespace(num_threads=2, provider="cpu", sid=3, speed=1.2)

    def test_build_kokoro_config_uses_multilang_zh_resources(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "model.int8.onnx").write_bytes(b"")
            (root / "voices.bin").write_bytes(b"")
            (root / "tokens.txt").write_text("a", encoding="utf-8")
            (root / "espeak-ng-data").mkdir()
            (root / "lexicon-us-en.txt").write_text("hello h e l o", encoding="utf-8")
            (root / "lexicon-zh.txt").write_text("开始 k a i", encoding="utf-8")
            (root / "phone-zh.fst").write_bytes(b"")
            (root / "date-zh.fst").write_bytes(b"")
            (root / "number-zh.fst").write_bytes(b"")

            config, gen_config = self.module.build_kokoro_config(
                self.fake_sherpa_onnx,
                root,
                self.args,
                "zh",
            )

        self.assertEqual(
            config.model.kokoro.lexicon,
            f"{root / 'lexicon-us-en.txt'},{root / 'lexicon-zh.txt'}",
        )
        self.assertEqual(config.model.kokoro.lang, "")
        self.assertEqual(
            config.rule_fsts,
            ",".join(
                str(root / name)
                for name in ("phone-zh.fst", "date-zh.fst", "number-zh.fst")
            ),
        )
        self.assertEqual(gen_config.sid, 3)
        self.assertEqual(gen_config.speed, 1.2)

    def test_resolve_effective_lang_prefers_user_language_for_multi_model(self):
        candidate = self.module.CandidateModel(
            name="kokoro-int8-multi-lang-v1_1",
            lang="multi",
            asset_name="kokoro-int8-multi-lang-v1_1.tar.bz2",
        )
        args = SimpleNamespace(languages="zh", models="kokoro-int8-multi-lang-v1_1")

        effective_lang = self.module.resolve_effective_lang(candidate, args)

        self.assertEqual(effective_lang, "zh")


if __name__ == "__main__":
    unittest.main()
