// sherpa-onnx/python/csrc/speaker-segmentation.cc
//
// Copyright (c) 2026

#include "sherpa-onnx/python/csrc/speaker-segmentation.h"

#include <vector>

#include "sherpa-onnx/csrc/speaker-segmentation.h"

namespace sherpa_onnx {

void PybindSpeakerSegmentation(py::module *m) {
  using PyConfig = SpeakerSegmentationConfig;
  py::class_<PyConfig>(*m, "SpeakerSegmentationConfig")
      .def(py::init<>())
      .def(py::init<const OfflineSpeakerSegmentationModelConfig &, float,
                    float, float>(),
           py::arg("model"), py::arg("min_duration_on") = 0.30F,
           py::arg("min_duration_off") = 0.50F,
           py::arg("change_vote_threshold") = 0.50F)
      .def_readwrite("model", &PyConfig::model)
      .def_readwrite("min_duration_on", &PyConfig::min_duration_on)
      .def_readwrite("min_duration_off", &PyConfig::min_duration_off)
      .def_readwrite("change_vote_threshold",
                     &PyConfig::change_vote_threshold)
      .def("__str__", &PyConfig::ToString)
      .def("validate", &PyConfig::Validate);

  using PySpan = SpeakerSegmentationSpan;
  py::class_<PySpan>(*m, "SpeakerSegmentationSpan")
      .def_property_readonly("start", [](const PySpan &self) {
        return self.start;
      })
      .def_property_readonly("end", [](const PySpan &self) {
        return self.end;
      })
      .def_property_readonly("speaker_count", [](const PySpan &self) {
        return self.speaker_count;
      })
      .def_property_readonly("flag", [](const PySpan &self) {
        return self.flag;
      })
      .def_property_readonly("local_speaker_mask", [](const PySpan &self) {
        return self.local_speaker_mask;
      })
      .def_property_readonly("local_speaker_mask_confidence",
                             [](const PySpan &self) {
                               return self.local_speaker_mask_confidence;
                             });

  using PyClass = SpeakerSegmentation;
  py::class_<PyClass>(*m, "SpeakerSegmentation",
                      R"(
An object-level streaming pyannote speaker-segmentation runner.

It returns finalized, non-overlapping spans with speaker_count, a fused
session-local three-bit speaker mask, its fused confidence, and an
identity-free right-boundary flag. The mask is not a global speaker ID.
Calling front when empty() is True is an error. The returned span is a value copy.
                      )")
      .def(py::init<const PyConfig &>(), py::arg("config"),
           py::call_guard<py::gil_scoped_release>())
      .def_property_readonly("sample_rate", &PyClass::SampleRate)
      .def(
          "accept_waveform",
          [](PyClass &self, const std::vector<float> &samples) {
            self.AcceptWaveform(samples.data(), samples.size());
          },
          py::arg("samples"), py::call_guard<py::gil_scoped_release>())
      .def("input_finished", &PyClass::InputFinished,
           py::call_guard<py::gil_scoped_release>())
      .def("empty", &PyClass::Empty)
      .def_property_readonly("front", [](const PyClass &self) {
        return self.Front();
      })
      .def("pop", &PyClass::Pop)
      .def("reset", &PyClass::Reset,
           py::call_guard<py::gil_scoped_release>());

  m->attr("CONTINUE") =
      py::int_(static_cast<int32_t>(kSpeakerSegmentationContinue));
  m->attr("SPEAKER_COUNT_CHANGED") =
      py::int_(static_cast<int32_t>(
          kSpeakerSegmentationSpeakerCountChanged));
  m->attr("SINGLE_SPEAKER_CHANGED") =
      py::int_(static_cast<int32_t>(
          kSpeakerSegmentationSingleSpeakerChanged));
  m->attr("INPUT_FINISHED") =
      py::int_(static_cast<int32_t>(kSpeakerSegmentationInputFinished));
}

}  // namespace sherpa_onnx
