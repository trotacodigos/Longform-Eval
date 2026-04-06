from pathlib import Path
import argparse
import json
import os
import sys
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from script.utils import read_jsonl
from script.rag_prompt import build_prompt
from models import REGISTRY


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--context", default=None, help="Context for the RAG prompt: 'high', 'low', or 'full'")
    ap.add_argument("-k", "--context-size", type=int, default=3, help="Number of segments to retrieve for context (default: 3)")
    ap.add_argument("--input_dir", default="data/wmt25/en-ko_KR/preliminary/output/", help="Input directory containing k?.jsonl files")
    ap.add_argument("--output_dir", default="data/outputs/rag")
    ap.add_argument("--model_params", default="", help="modeling parameters saved in .json")
    args = ap.parse_args()

    # Set dataset paths based on year and language pair
    year, lp = args.input_dir.split("/")[1:3]

    src_lang, tgt_lang = lp.split("-")
    src_lang = "English" if src_lang == "en" else ""
    tgt_lang = "Korean" if tgt_lang == "ko_KR" else "Chinese" if tgt_lang == "zh" else ""

    # set path
    input_dir = Path(args.input_dir) / f"k{args.context_size}"
    out_dir = Path(args.output_dir) / lp / args.context
    os.makedirs(out_dir, exist_ok=True)

    if args.model not in REGISTRY:
        raise ValueError(f"Unknown model '{args.model}'. Available: {list(REGISTRY.keys())}")
    model = REGISTRY[args.model]()

    ref_dic = {(l["doc_id"], l["seg_id"], l["system"]): l for l in read_jsonl(f"data/{year}/{lp}/output.jsonl")}

    for file in os.listdir(input_dir):
        if file.endswith(".jsonl"):
            in_data = read_jsonl(input_dir / file)
            sys_name = file.split(".")[0]

            batch_items = []  # list of (entry, system_prompt, user_prompt)
            for entry in tqdm(in_data, desc="Building prompts"):
                try:
                    system_prompt, user_prompt = build_prompt(entry, src_lang=src_lang, tgt_lang=tgt_lang, context=args.context)
                    batch_items.append((entry, system_prompt, user_prompt))
                except Exception as e:
                    print(f"Prompt error for seg_id={entry.get('seg_id')}: {e}")

            prompts = [(system_prompt, user_prompt) for _, system_prompt, user_prompt in batch_items]
            results = model.generate_batch(prompts)

            out_file = out_dir / f"{year}_{sys_name}.jsonl"
            with open(out_file, "w", encoding="utf-8") as f:
                for (entry, _, _), result in zip(batch_items, results):
                    if result is None:
                        continue
                    text, usage = result
                    line = ref_dic.get((entry["doc_id"], entry["seg_idx"], sys_name))

                    record = {
                        "doc_id": entry["doc_id"],
                        "seg_id": entry["seg_idx"],
                        "bucket_id": line["bucket_id"] if line else None,
                        "mt_pe_seg": text,
                        "ref_seg": line["ref_seg"] if line else None,
                        "usage": {k: usage.get(k) for k in ("input_token", "output_token", "latency_sec")},
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
