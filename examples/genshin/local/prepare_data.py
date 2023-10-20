""" 
- 生成拼音, 拆分 train.txt 等
- 生成 spk2id, lang2id
- 生成 phones.txt speaker.txt
"""
import json
import random
import re
from pathlib import Path
from typing import List
import torchaudio
from tqdm import tqdm
from typing import Dict
from pypinyin import lazy_pinyin, Style
from tn.chinese.normalizer import Normalizer

random.seed(42)


class PyFrontEnd:
    def __init__(self, pinyin_lexicon_path: str):
        self.init_dicts(pinyin_lexicon_path)
        self.text_normalizer = Normalizer()
        print("Finish Init")

    def init_dicts(self, pinyin_lexicon_path):
        self.add_blank = False
        self.blank_id = 0

        self.token_dict = {}

        self.pinyin_lexicon = {}
        with open(pinyin_lexicon_path, "r", encoding="utf8") as fin:
            for line in fin:
                arr = line.strip().split()
                self.pinyin_lexicon[arr[0]] = arr[1:]

    def g2p(self, text):
        text = self.text_normalizer.normalize(text)
        pinyin_seq = lazy_pinyin(
            text,
            style=Style.TONE3,
            neutral_tone_with_five=True,
            errors=lambda punct_and_en_words: list(punct_and_en_words),
        )
        phoneme_seq = []
        for pinyin in pinyin_seq:
            # 12345 represents five tones
            if pinyin == "n2":  # fix for '嗯'
                pinyin = "en2"
            if pinyin[-1] in "12345":
                assert (
                    pinyin in self.pinyin_lexicon
                ), f"pinyin: {pinyin} not in pinyin_lexicon"
                phoneme_seq += self.pinyin_lexicon[pinyin]
            else:
                # Pinyins would end up with a number in 1-5,
                # which represents tones of the pinyin.
                # For symbols which are not pinyin,
                # e.g. English letters, Chinese puncts, we directly use them as inputs.
                phoneme_seq.append(pinyin)
        return " ".join(phoneme_seq)


FILTER_WORDS = [
    "NICKNAME",  # 因为有时候念"旅行者"， 有时候什么都不念
    "PLAYERAVATAR",
    "{MATEAVATAR#SEXPRO[INFO_MALE_PRONOUN_BROTHER|INFO_FEMALE_PRONOUN_SISTER]}",  # 妹妹或者哥哥
    "{MATEAVATAR#SEXPRO[INFO_MALE_PRONOUN_BOYD|INFO_FEMALE_PRONOUN_GIRLD]}",
    "{M#",
    "$UNRELEASED",
    "（test）",
    "{RUBY#",  # ex: 「虚{RUBY#[D]阿卡西}空」
    "♪",
]


def _clean_text(text):
    text = re.sub(r"[「」—『』]", "", text)
    text = re.sub(r"…+", "…", text)
    return text.strip()


def valid_text(text) -> bool:
    """ 根据规则过滤一些文本, 因为原神文本中有一些词槽 """
    for word in FILTER_WORDS:
        if word in text:
            return False

    if re.match(r"^…+$", text):  # 过滤掉全是省略号的文本
        return False
    return True


def main(args):
    ft = PyFrontEnd(args.lexicon_path)

    args.dump_dir.mkdir(exist_ok=True, parents=True)

    with open(args.datalist, "r", encoding="utf-8") as f:
        ori_data = [x.strip().split("|") for x in f.readlines()]
        #  wavpath, speaker, language, text

    raw_data: List[List[str]] = []  # [wavpath, text, speaker, language, phonemes])
    langs = set()

    for item in tqdm(ori_data, desc="g2p"):
        wavpath, speaker, language, text = item

        if Path(wavpath).exists():
            langs.add(language)
            text = _clean_text(text)
            if not valid_text(text):
                continue
            try:
                phonemes = ft.g2p(text)
                phonemes = f"sil {phonemes} sil"
                raw_data.append([str(wavpath), text, speaker, language, phonemes])
            except Exception as e:
                print(f"skip {wavpath}: {text} due to g2p failed, error: {e}")

        else:
            print(f"File not found : {wavpath}")
            pass

    # sort raw_data by spk and wavpath
    raw_data = sorted(raw_data, key=lambda x: (x[2], x[0]))

    # save only text
    with open(args.dump_dir / "only_text.txt", "w", encoding="utf-8") as wf:
        for item in raw_data:
            wf.write(item[1] + "\n")

    # calculate each spk information
    # 1. total duration
    # 2. total number of utterances
    # 3. average duration
    # 4. average text length
    spk_info = {}
    for wavpath, text, spk, lang, phonemes in tqdm(raw_data, desc="calculate spk info"):
        wav_info = torchaudio.info(wavpath)
        dur = wav_info.num_frames / wav_info.sample_rate

        if spk not in spk_info:
            spk_info[spk] = {"dur_sec": 0.0, "num": 0, "text_len": 0}
        spk_info[spk]["dur_sec"] += dur
        spk_info[spk]["num"] += 1
        spk_info[spk]["text_len"] += len(text)
    for spk, info in spk_info.items():
        info["avg_duration_sec"] = info["dur_sec"] / info["num"]
        info["avg_text_len"] = info["text_len"] / info["num"]
        info["dur_hour"] = info["dur_sec"] / 3600

    # sort spk_info by duration
    spk_info = {
        k: v
        for k, v in sorted(
            spk_info.items(), key=lambda x: x[1]["dur_sec"], reverse=True
        )
    }
    with open(args.dump_dir / "spk_info.json", "w", encoding="utf-8") as wf:
        json.dump(spk_info, wf, indent=4, ensure_ascii=False)

    filter_spk_list = spk_info.keys()

    # filter_speaker_summary
    summary = {"dur_sec": 0.0, "num": 0, "text_len": 0}
    for spk, info in spk_info.items():
        if spk in filter_spk_list:
            summary["dur_sec"] += info["dur_sec"]
            summary["num"] += info["num"]
            summary["text_len"] += info["text_len"]
    summary["avg_duration_sec"] = summary["dur_sec"] / summary["num"]
    summary["avg_text_len"] = summary["text_len"] / summary["num"]
    summary["dur_hour"] = summary["dur_sec"] / 3600
    # save summary
    with open(args.dump_dir / "summary.json", "w", encoding="utf-8") as wf:
        json.dump(summary, wf, indent=4, ensure_ascii=False)

    spk2id = {spk: idx for idx, spk in enumerate(filter_spk_list)}
    lang2id = {lang: idx for idx, lang in enumerate(langs)}

    with open(args.dump_dir / "spk2id.json", "w", encoding="utf-8") as wf:
        json.dump(spk2id, wf, indent=4, ensure_ascii=False)
    with open(args.dump_dir / "speaker.txt", "w", encoding="utf-8") as wf:
        for spk, sid in spk2id.items():
            wf.write(f"{spk} {sid}\n")
    with open(args.dump_dir / "lang2id.json", "w", encoding="utf-8") as wf:
        json.dump(lang2id, wf, indent=4, ensure_ascii=False)

    filter_data = [item for item in raw_data if item[2] in filter_spk_list]

    # generate phones.txt (with phone id)
    phones = set()
    for item in filter_data:
        phonemes_list = item[4].split()
        phones.update(phonemes_list)
    phones = sorted(list(phones))
    with open(
        args.dump_dir / "phones.txt", "w", encoding="utf-8"
    ) as wf:
        for i, phone in enumerate(phones):
            wf.write(f"{phone} {i}" + "\n")

    test_data = []
    test_data_spk_utt_num = {}
    exp_data = []
    val_data = []
    for item in filter_data:
        phonemes = item[4]
        spk = item[2]
        if not phonemes:
            continue
        if spk not in test_data_spk_utt_num:
            test_data_spk_utt_num[spk] = 0
        if test_data_spk_utt_num[spk] < 2:
            test_data.append(item)
            test_data_spk_utt_num[spk] += 1
        else:
            exp_data.append(item)

    # sort test_data by spk
    test_data = sorted(test_data, key=lambda x: x[2])

    random.shuffle(exp_data)
    val_data = exp_data[:200]
    train_data = exp_data[200:]

    def save_file(file_path, data):
        with open(file_path, "w", encoding="utf-8") as wf:
            for wavpath, text, spk, lang, phonemes in data:
                wf.write(f"{wavpath}|{spk}|{phonemes}|{lang}" + "\n")

    save_file(args.dump_dir / "test.txt", test_data)
    save_file(args.dump_dir / "train.txt", train_data)
    save_file(args.dump_dir / "val.txt", val_data)
    print(f"See {args.dump_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--datalist", type=str, default=None, help="音频路径列表, wavpath|xxx_other_content"
    )
    parser.add_argument("--dump_dir", type=Path)
    parser.add_argument("--lexicon_path", type=Path)

    args = parser.parse_args()
    main(args)
