// sherpa-onnx/csrc/offline-paraformer-hotwords.cc
//
// Copyright (c)  2026  Xiaomi Corporation

#include "sherpa-onnx/csrc/offline-paraformer-hotwords.h"

#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>

#include "sherpa-onnx/csrc/macros.h"
#include "sherpa-onnx/csrc/text-utils.h"
#include "sherpa-onnx/csrc/utils.h"

namespace sherpa_onnx {
std::string GetParaformerSegDictPath(const std::string &tokens) {
  const auto pos = tokens.find_last_of("/\\");
  if (pos == std::string::npos) {
    return "seg_dict";
  }
  return tokens.substr(0, pos + 1) + "seg_dict";
}

namespace {

using SegDict = std::unordered_map<std::string, std::string>;

bool IsAsciiLetter(const std::string &s) {
  if (s.empty()) {
    return false;
  }

  for (char c : s) {
    if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'))) {
      return false;
    }
  }
  return true;
}

std::string ToLowerAscii(std::string s) {
  for (auto &c : s) {
    if (c >= 'A' && c <= 'Z') {
      c = static_cast<char>(c - 'A' + 'a');
    }
  }
  return s;
}

void LoadSegDict(std::istream &is, SegDict *seg_dict) {
  std::string line;
  while (std::getline(is, line)) {
    const auto pos = line.find('\t');
    if (pos == std::string::npos || pos == 0 || pos + 1 == line.size()) {
      continue;
    }
    seg_dict->emplace(line.substr(0, pos), line.substr(pos + 1));
  }
}

void LoadSegDict(const std::string &path, SegDict *seg_dict) {
  std::ifstream is(path);
  if (!is) {
    SHERPA_ONNX_LOGE(
        "Warning: Cannot open Paraformer seg_dict: '%s'. English hotwords "
        "will use character fallback.",
        path.c_str());
    return;
  }

  LoadSegDict(is, seg_dict);
}

void EncodeEnglishWord(const std::string &word, const SegDict &seg_dict,
                       const SymbolTable &symbol_table,
                       std::ostringstream *oss) {
  const std::string lower_word = ToLowerAscii(word);
  const int32_t n = static_cast<int32_t>(lower_word.size());
  const int32_t kInfinity = std::numeric_limits<int32_t>::max();

  struct Segmentation {
    int32_t cost = std::numeric_limits<int32_t>::max();
    int32_t first_piece_length = 0;
    std::vector<std::string> tokens;
  };

  std::vector<Segmentation> best(n + 1);
  best[n].cost = 0;

  for (int32_t i = n - 1; i >= 0; --i) {
    for (int32_t j = i + 1; j <= n; ++j) {
      std::vector<std::string> tokens;
      const std::string piece = lower_word.substr(i, j - i);
      const auto iter = seg_dict.find(piece);
      if (iter != seg_dict.end()) {
        std::istringstream is(iter->second);
        std::string token;
        while (is >> token) {
          tokens.push_back(std::move(token));
        }
      } else if (j == i + 1) {
        tokens.push_back(piece);
      } else {
        continue;
      }

      if (j < n) {
        std::string &last_token = tokens.back();
        if (last_token.size() < 2 ||
            last_token.compare(last_token.size() - 2, 2, "@@") != 0) {
          last_token += "@@";
        }
      }

      if (best[j].cost == kInfinity) {
        continue;
      }

      bool valid = true;
      for (const auto &token : tokens) {
        if (!symbol_table.Contains(token)) {
          valid = false;
          break;
        }
      }
      if (!valid) {
        continue;
      }

      const int32_t cost =
          static_cast<int32_t>(tokens.size()) + best[j].cost;
      const int32_t piece_length = j - i;
      if (cost > best[i].cost ||
          (cost == best[i].cost &&
           piece_length <= best[i].first_piece_length)) {
        continue;
      }

      tokens.insert(tokens.end(), best[j].tokens.begin(),
                    best[j].tokens.end());
      best[i].cost = cost;
      best[i].first_piece_length = piece_length;
      best[i].tokens = std::move(tokens);
    }
  }

  if (symbol_table.Contains(lower_word) && best[0].cost > 1) {
    best[0].cost = 1;
    best[0].tokens = {lower_word};
  }

  if (best[0].cost != kInfinity) {
    for (const auto &token : best[0].tokens) {
      *oss << " " << token;
    }
    return;
  }

  for (size_t i = 0; i != lower_word.size(); ++i) {
    *oss << " " << lower_word[i];
    if (i + 1 != lower_word.size()) {
      *oss << "@@";
    }
  }
}

void EncodeWord(const std::string &word, const SymbolTable &symbol_table,
                const SegDict &seg_dict, std::ostringstream *oss) {
  const auto chars = SplitUtf8(word);
  bool has_ascii_letter = false;
  for (const auto &c : chars) {
    if (IsAsciiLetter(c)) {
      has_ascii_letter = true;
      break;
    }
  }

  // English words in Paraformer use seg_dict rather than direct token lookup.
  if (!has_ascii_letter && symbol_table.Contains(word)) {
    *oss << " " << word;
    return;
  }

  for (size_t i = 0; i < chars.size();) {
    if (!IsAsciiLetter(chars[i])) {
      *oss << " " << chars[i++];
      continue;
    }

    std::string english_word;
    do {
      english_word += chars[i++];
    } while (i < chars.size() && IsAsciiLetter(chars[i]));
    EncodeEnglishWord(english_word, seg_dict, symbol_table, oss);
  }
}

bool EncodeParaformerHotwordsImpl(
    std::istream &is, const SymbolTable &symbol_table, const SegDict &seg_dict,
    std::vector<std::vector<int32_t>> *hotwords,
    std::vector<float> *boost_scores) {
  std::ostringstream tokenized_hotwords;
  std::string line;
  std::string word;

  while (std::getline(is, line)) {
    std::string score;
    std::ostringstream phrase;
    std::istringstream line_stream(line);
    while (line_stream >> word) {
      if (word[0] == ':') {
        score = word;
        continue;
      }
      if (!score.empty()) {
        SHERPA_ONNX_LOGE(
            "Boosting score should be put after the words/phrase, given %s.",
            line.c_str());
        return false;
      }
      phrase << " " << word;
    }

    const std::string phrase_text = phrase.str();
    if (phrase_text.empty()) {
      continue;
    }

    std::istringstream phrase_stream(phrase_text.substr(1));
    while (phrase_stream >> word) {
      EncodeWord(word, symbol_table, seg_dict, &tokenized_hotwords);
    }
    if (!score.empty()) {
      tokenized_hotwords << " " << score;
    }
    tokenized_hotwords << "\n";
  }

  std::string tokenized = tokenized_hotwords.str();
  SHERPA_ONNX_LOGE("[debug][v3] Paraformer hotwords tokenized: %s",
                   tokenized.c_str());
  std::istringstream token_stream(tokenized);
  return EncodeKeywords(token_stream, symbol_table, hotwords, nullptr,
                        boost_scores, nullptr);
}

}  // namespace

bool EncodeParaformerHotwords(std::istream &is, const std::string &tokens,
                              const SymbolTable &symbol_table,
                              std::vector<std::vector<int32_t>> *hotwords,
                              std::vector<float> *boost_scores) {
  SegDict seg_dict;
  LoadSegDict(GetParaformerSegDictPath(tokens), &seg_dict);
  return EncodeParaformerHotwordsImpl(is, symbol_table, seg_dict, hotwords,
                                      boost_scores);
}

bool EncodeParaformerHotwords(std::istream &is,
                              const SymbolTable &symbol_table,
                              std::vector<std::vector<int32_t>> *hotwords,
                              std::vector<float> *boost_scores) {
  const SegDict seg_dict;
  return EncodeParaformerHotwordsImpl(is, symbol_table, seg_dict, hotwords,
                                      boost_scores);
}

bool EncodeParaformerHotwords(std::istream &is, std::istream &seg_dict_stream,
                              const SymbolTable &symbol_table,
                              std::vector<std::vector<int32_t>> *hotwords,
                              std::vector<float> *boost_scores) {
  SegDict seg_dict;
  LoadSegDict(seg_dict_stream, &seg_dict);
  return EncodeParaformerHotwordsImpl(is, symbol_table, seg_dict, hotwords,
                                      boost_scores);
}

}  // namespace sherpa_onnx
