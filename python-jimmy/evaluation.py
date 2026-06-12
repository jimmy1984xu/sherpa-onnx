#!/usr/bin/env python3
#
# Copyright 2021-2023 Xiaomi Corporation (Author: Fangjun Kuang,
#                                                 Zengwei Yao)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import argparse
import logging
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TextIO
import string
try:
    import kaldialign  # type: ignore
    _HAS_KALDIALIGN = True
except Exception:
    # 在部分环境中 kaldialign 的二进制扩展（_kaldialign）会因 DLL 依赖缺失而无法加载。
    # 为了保证 WER 计算可用，这里提供纯 Python 的兜底对齐实现。
    kaldialign = None  # type: ignore
    _HAS_KALDIALIGN = False


def _pure_python_align(ref: List[str], hyp: List[str], err_token: str, sclite_mode: bool = False):
    """
    纯 Python 版对齐：最小化插入/删除/替换的编辑距离，并输出类似 kaldialign 的 (ref, hyp) 对序列。

    注：sclite_mode 对本实现不做特殊处理（WEr 统计结果不依赖该差异）。
    """
    n = len(ref)
    m = len(hyp)
    # dp[i][j] = 将 ref[:i] 变为 hyp[:j] 的最小编辑距离
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub_cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,  # delete
                dp[i][j - 1] + 1,  # insert
                dp[i - 1][j - 1] + sub_cost,  # sub / match
            )

    ali: List[Tuple[str, str]] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            sub_cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            if dp[i][j] == dp[i - 1][j - 1] + sub_cost:
                # match or substitution: 这里不需要用 err_token 标记 substitution
                ali.append((ref[i - 1], hyp[j - 1]))
                i -= 1
                j -= 1
                continue
        if i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ali.append((ref[i - 1], err_token))
            i -= 1
            continue
        if j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            ali.append((err_token, hyp[j - 1]))
            j -= 1
            continue
        # 理论上不会走到这里；作为兜底退化为删除
        if i > 0:
            ali.append((ref[i - 1], err_token))
            i -= 1

    ali.reverse()
    return ali


def _align(ref: List[str], hyp: List[str], err_token: str, sclite_mode: bool = False):
    if _HAS_KALDIALIGN and kaldialign is not None:
        return kaldialign.align(ref, hyp, err_token, sclite_mode=sclite_mode)
    return _pure_python_align(ref, hyp, err_token, sclite_mode=sclite_mode)

try:
    import cn2an  # pip install cn2an — 用于「百分之十二」与「12%」对齐
    _HAS_CN2AN = True
except ImportError:
    cn2an = None  # type: ignore
    _HAS_CN2AN = False

LOG_EPS = math.log(1e-10)


def _pct_token_from_number(v: float) -> str:
    """Canonical token for ref/hyp so 12% and 百分之十二 match."""
    if abs(v - round(v)) < 1e-9:
        return f"PCT_{int(round(v))}"
    s = str(v).rstrip("0").rstrip(".")
    return "PCT_" + s.replace(".", "_")


def _normalize_zh_percent(text: str) -> str:
    """
    将「12%」「12.5%」与「百分之十二」「百 分 之 十 二」等统一为同一类 PCT_* 标记，
    避免仅书写形式不同却被计为替换错误。
    """
    # 1) 西文百分号写法
    def repl_ascii(m: re.Match) -> str:
        try:
            v = float(m.group(1))
        except ValueError:
            return m.group(0)
        return f" {_pct_token_from_number(v)} "

    text = re.sub(r"(\d+(?:\.\d+)?)%", repl_ascii, text)

    # 2) ASR 常带空格的「百 分 之」
    text = re.sub(r"百\s*分\s*之\s*", "百分之", text)

    # 3) 中文读法：数字部分仅含中文数字、阿拉伯数字、空格、小数点「点」
    pct_zh = re.compile(
        r"百分之([零一二三四五六七八九十百千万两〇0-9点\s]+)"
    )

    def repl_zh(m: re.Match) -> str:
        raw = m.group(1)
        num_part = re.sub(r"\s+", "", raw)
        if not num_part:
            return m.group(0)
        if not _HAS_CN2AN:
            return m.group(0)
        try:
            v = float(cn2an.cn2an(num_part, "normal"))  # type: ignore[union-attr]
        except Exception:
            return m.group(0)
        return f" {_pct_token_from_number(v)} "

    text = pct_zh.sub(repl_zh, text)
    return text


def _normalize_zh_numbers(text: str) -> str:
    """
    统一常见「数字/读法」差异：
    - Q3 / Q 3 / Q 三 / q三 → Q3
    - 200 / 两百 / 二百 → 200

    说明：中文数字转阿拉伯数字依赖 cn2an；未安装时只做最基础的空格收敛与字母+阿拉伯数字拼接。
    """
    # 1) 将「Q 3」这类字母+阿拉伯数字的空格去掉
    text = re.sub(r"([A-Za-z])\s+(\d)", r"\1\2", text)

    # 2) 将「Q 三」这类字母+中文数字（或带空格）转成「Q3」
    #    仅在安装 cn2an 时启用（避免误处理）
    if _HAS_CN2AN:
        alpha_zh_num = re.compile(
            r"([A-Za-z])\s*([零一二三四五六七八九十百千万两〇0-9点\s]+)"
        )

        def repl_alpha_zh(m: re.Match) -> str:
            a = m.group(1)
            raw = re.sub(r"\s+", "", m.group(2))
            if not raw:
                return m.group(0)
            try:
                # 允许 Q3 / Q三 / Q十二 等
                v = cn2an.cn2an(raw, "normal")  # type: ignore[union-attr]
            except Exception:
                return m.group(0)
            if isinstance(v, float) and abs(v - round(v)) < 1e-9:
                v = int(round(v))
            return f"{a}{v}"

        text = alpha_zh_num.sub(repl_alpha_zh, text)

        # 3) 将常见中文数字整体（如「两百」「二百零一」）转成阿拉伯数字
        #    只匹配明确由中文数字组成的片段，降低误伤。
        zh_num = re.compile(r"([零一二三四五六七八九十百千万两〇]+)")

        def repl_zh_num(m: re.Match) -> str:
            raw = m.group(1)
            # 太短的（如「一」）经常是普通语义词，保守起见不转换
            if len(raw) < 2:
                return raw
            try:
                v = cn2an.cn2an(raw, "normal")  # type: ignore[union-attr]
            except Exception:
                return raw
            if isinstance(v, float) and abs(v - round(v)) < 1e-9:
                v = int(round(v))
            return str(v)

        text = zh_num.sub(repl_zh_num, text)

    return text


def text_normalization(text, lang):
    if lang == 'en':
        text = text.lower().replace("p.m.", "pm").replace("a.m.", "am").replace('@',' at ').replace('/', ' slash ').replace('#', ' pound ').replace('e-mail', 'email').replace("e.g.", "for example").replace('—', ' ')
        words = text.split(' ')
        words2 = []
        for word in words:
            if ':' in word:
                words2.append(convert_time(word))
            else:
                words2.append(word)
        text = ' '.join(words2)

    punctuation = '!,.;:?、！，。；：？》《「」¿"-'
    text = re.sub(r'[{}]+'.format(punctuation), ' ', text).strip()
    text = text.lower()
    if lang == 'zh':
        text = _normalize_zh_percent(text)
        text = _normalize_zh_numbers(text)
        pattern = re.compile(r"([\u4e00-\u9fff])")
        chars = pattern.split(text)
        mix_chars = [w for w in chars if len(w.strip()) > 0]
        tokens = []
        for ch_or_w in mix_chars:
            if pattern.fullmatch(ch_or_w) is not None:
                tokens.append(ch_or_w)
            else:
                tokens.append(ch_or_w.upper())
        text = ' '.join(tokens)
        cn_punc = ["。", "？", "，", "·","—", "‘","’","“","”"]
        for key in cn_punc:
            text = text.replace(key, "")
    pattern_spaces = re.compile(r'\s+')
    text = pattern_spaces.sub(' ', text)
    return text.strip()
def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--label",
        type=str,
        default="label.txt",
    )

    parser.add_argument(
        "--hyp",
        type=str,
        default="hyp.txt",
    )
    parser.add_argument("--language",type=str, default="de",)
    parser.add_argument("--metric", type=str, default="wer")
    parser.add_argument("--detail", type=str, default="")

    return parser
def write_error_stats(
    f: TextIO,
    test_set_name: str,
    results: List[Tuple[str, str]],
    enable_log: bool = True,
    compute_CER: bool = False,
    sclite_mode: bool = False,
) -> float:
    subs: Dict[Tuple[str, str], int] = defaultdict(int)
    ins: Dict[str, int] = defaultdict(int)
    dels: Dict[str, int] = defaultdict(int)
    words: Dict[str, List[int]] = defaultdict(lambda: [0, 0, 0, 0, 0])
    errs: Dict[str, List[float]] = defaultdict(float)
    num_corr = 0
    ERR = "*"

    if compute_CER:
        for i, res in enumerate(results):
            cut_id, ref, hyp = res
            ref = ref.replace(' ', '')
            hyp = hyp.replace(' ', '')
            ref = list("".join(ref))
            hyp = list("".join(hyp))
            results[i] = (cut_id, ref, hyp)
    else:
        for i, res in enumerate(results):
            cut_id, ref, hyp = res
            ref = ref.split(' ')
            hyp  = hyp.split(' ')
            results[i] = (cut_id, ref, hyp)

    for cut_id, ref, hyp in results:
        ali = _align(ref, hyp, ERR, sclite_mode=sclite_mode)
        err = 0
        a_del = 0
        a_ins = 0
        for ref_word, hyp_word in ali:
            if ref_word == ERR:
                ins[hyp_word] += 1
                words[hyp_word][3] += 1
                err += 1
                a_ins += 1
            elif hyp_word == ERR:
                dels[ref_word] += 1
                words[ref_word][4] += 1
                err += 1
                a_del += 1
            elif hyp_word != ref_word:
                subs[(ref_word, hyp_word)] += 1
                words[ref_word][1] += 1
                words[hyp_word][2] += 1
                err += 1
            else:
                words[ref_word][0] += 1
                num_corr += 1
        errs_value = []
        ref_n = len(ref) if len(ref) > 0 else 1
        errs_value.append("%.2f" % (100.0 * err / ref_n))
        errs_value.append(len(ref))
        errs_value.append(err)
        errs_value.append(a_del)
        errs_value.append(a_ins)
        errs[cut_id] = errs_value #"%.2f" % (100.0 * err / len(ref))
    ref_len = sum([len(r) for _, r, _ in results])
    sub_errs = sum(subs.values())
    ins_errs = sum(ins.values())
    del_errs = sum(dels.values())
    tot_errs = sub_errs + ins_errs + del_errs
    if ref_len == 0:
        msg = (
            "Cannot compute WER: total reference length is 0 "
            "(empty evaluation set or all empty references)."
        )
        print(msg, file=f)
        print(msg)
        return 0.0
    tot_err_rate = "%.2f" % (100.0 * tot_errs / ref_len)

    print(f"%WER = {tot_err_rate}")
    print(
        f"Errors: {ins_errs} insertions, {del_errs} deletions, "
        f"{sub_errs} substitutions, over {ref_len} reference "
        f"words ({num_corr} correct)",
    )
    print(
        "Search below for sections starting with PER-UTT DETAILS:, "
        "SUBSTITUTIONS:, DELETIONS:, INSERTIONS:, PER-WORD STATS:",
    )
    
    #print("PER-UTT DETAILS: corr or (ref->hyp)  ", file=f)
    txt = {}
    for cut_id, ref, hyp in results:
        ali = _align(ref, hyp, ERR)
        combine_successive_errors = True
        if combine_successive_errors:
            ali = [[[x], [y]] for x, y in ali]
            for i in range(len(ali) - 1):
                if ali[i][0] != ali[i][1] and ali[i + 1][0] != ali[i + 1][1]:
                    ali[i + 1][0] = ali[i][0] + ali[i + 1][0]
                    ali[i + 1][1] = ali[i][1] + ali[i + 1][1]
                    ali[i] = [[], []]
            ali = [
                [
                    list(filter(lambda a: a != ERR, x)),
                    list(filter(lambda a: a != ERR, y)),
                ]
                for x, y in ali
            ]
            ali = list(filter(lambda x: x != [[], []], ali))
            ali = [
                [
                    ERR if x == [] else " ".join(x),
                    ERR if y == [] else " ".join(y),
                ]
                for x, y in ali
            ]
        txt[cut_id] = " ".join(
                (
                    ref_word if ref_word == hyp_word else f"({ref_word}->{hyp_word})"
                    for ref_word, hyp_word in ali
                )
            )
    tot_err_rate = "%.2f" % (100.0 * tot_errs / ref_len)
    print(tot_err_rate)
    errs = sorted(errs.items(), key=lambda x: x[0])
    print(f"id\twer\tref_words\terr_words\tdel_words\tins_words", file=f, )
    for key, val in errs:
        print(
            f"{key}\t" + str(val[0]) + "\t"+ str(val[1]) + "\t" + str(val[2]) + "\t"+ str(val[3]) + "\t"
            + str(val[4]) + "\t" + txt[key],file=f,)
    return float(tot_err_rate)

def save_results(
    res_dir: str,
    test_set_name: str,
    results_list: List[Tuple[str, List[str], List[str]]],
    metric: str,
):
    test_set_wers = dict()
    results = sorted(results_list)

    # The following prints out WERs, per-word error statistics and aligned
    # ref/hyp pairs.
    errs_filename = f"{res_dir}"
    compute_CER = (metric == "cer")
    
    with open(errs_filename, "w", encoding="utf-8") as f:
        wer = write_error_stats(
            f, f"{test_set_name}", results, enable_log=True, compute_CER=compute_CER,
        )
        test_set_wers["all"] = wer

    logging.info("Wrote detailed error stats to {}".format(errs_filename))

    test_set_wers = sorted(test_set_wers.items(), key=lambda x: x[1])

    s = "\nFor {}, WER of different settings are:\n".format(test_set_name)
    note = "\tbest for {}".format(test_set_name)
    for key, val in test_set_wers:
        s += "{}\t{}{}\n".format(key, val, note)
        note = ""
    logging.info(s)

def main():
    parser = get_parser()
    args = parser.parse_args()
    args.ref_file = Path(args.label)
    args.hyp_file = Path(args.hyp)
    detail_path = args.detail
    if args.detail == "":
        detail_path = args.hyp+"_detail"
    
    ref_map = {}
    hyp2_map = {}
    with open(args.ref_file, encoding="utf-8-sig") as f:
        for line in f:
            l = line.strip().split(' ', 1)
            if len(l) == 2:
                ref_map[l[0]] = l[1]
    results_list = []
    with open(args.hyp_file, encoding="utf-8-sig") as f:
        for line in f:
            l = line.strip().split(' ', 1)
            if l[0] not in ref_map:
                continue
            if len(l) < 2:
                l.append("")
            ref_words = text_normalization(ref_map[l[0]],args.language)
            hyp_words = text_normalization(l[1],args.language)
            results_list.append((l[0], ref_words, hyp_words))
    if not ref_map:
        raise SystemExit(
            "label.txt has no valid lines. Expected format per line: "
            "<utterance_id><space><reference_text>"
        )
    if not results_list:
        raise SystemExit(
            "No matching utterances: hyp.txt first column must match label.txt ids "
            "exactly. Check spelling, spaces, and that both files use the same ids."
        )
    save_results(detail_path, os.path.basename(args.hyp_file), results_list, args.metric)
    logging.info("Done!")


if __name__ == "__main__":
    main()
