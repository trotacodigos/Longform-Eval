# if the inferenced segments are errorless
from utils import read_jsonl

import os
import argparse

from langdetect import detect, detect_langs
from langdetect.lang_detect_exception import LangDetectException

PATH = "data/outputs"

 
LANGUAGE_NAMES = {
    "ko": "Korean", "en": "English", "zh": "Chinese",
}
 
def detect_language(text: str, tgt_lang: str) -> bool:
    try:
        lang_code = detect(text)
        return lang_code == tgt_lang
    except LangDetectException as e:
        print(f"Detection error: {e}")
        return False


def check_errorless(level, lp):
    fdir = os.path.join(PATH, f"{level}/{lp}")
    tgt_lang = lp.split("-")[-2:]
    for file in os.listdir(fdir):
        if file.endswith(".jsonl"):
            data = read_jsonl(os.path.join(fdir, file))

            print("=" * 10)
            print("Checking file:", level, lp, file)

            empty_response = len([d for d in data if d["mt_pe_seg"] is None or not d["mt_pe_seg"]])
            print(f"Empty response count: {empty_response} / {len(data)}")
            lang_detection = [detect_language(d["mt_pe_seg"], tgt_lang) for d in data]
            print(f"Different language count: {sum(lang_detection)} / {len(data)}")

            print("=" * 10)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", required=True)
    parser.add_argument("--lp", required=True)
    args = parser.parse_args()

    check_errorless(args.level, args.lp)