import argparse
import json
from pathlib import Path
from tqdm import tqdm

from script.ape_prompt import build_prompt_with_doc
from models import REGISTRY


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--input_file", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--with_doc", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    in_path = Path(args.input_file)
    out_dir = Path(args.output_dir)

    # Verify models from REGISTRY
    if args.models not in REGISTRY:
        raise ValueError(f"Unknown model '{args.models}'. Available: {list(REGISTRY.keys())}")
    model = REGISTRY[args.models]()

    with open(in_path, "r", encoding="utf-8") as f:
        in_data = [json.loads(l) for l in f.readlines()]
    if args.limit:
        in_data = in_data[:args.limit]

    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "doc" if args.with_doc else "seg"
    out_path = out_dir / f"{args.models}_5_samples.{suffix}.jsonl"
    
    with open(out_path, "w", encoding="utf-8") as f:
        for row in tqdm(in_data, desc=f"Running {args.models}"):
            try:
                system, user = build_prompt_with_doc(row, args.with_doc)
                text, usage = model.generate(system, user)
                f.write(json.dumps(
                    {
                    "sample_id": row.get("sample_id"),
                    "output": text,
                    "input_token": usage.get("input_token", 0),
                    "output_token": usage.get("output_token", 0),
                    "latency": usage.get("latency", 0),
                }, ensure_ascii=False
                ) + "\n")
            except Exception as e:
                f.write(json.dumps(
                    {"sample_id": row.get("sample_id"), "output": f"[ERROR] {str(e)}"},
                    ensure_ascii=False
                ) + "\n")


if __name__ == "__main__":
    main()