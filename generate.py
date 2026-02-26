import argparse
import json
import time
from pathlib import Path
from tqdm import tqdm
from prompts.ape_prompt import build_prompt

def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip(): rows.append(json.loads(line.strip()))
    return rows

def write_jsonl(path: Path, records: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True, help="claude 또는 llama 입력")
    ap.add_argument("--input_file", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--with_doc", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    in_path = Path(args.input_file)
    out_dir = Path(args.output_dir)

    # Sonnet 모델 유지
    if args.models == "claude":
        from models.claude import ClaudeModel
        model = ClaudeModel(model_name="claude-3-5-sonnet-20240620")
    elif args.models == "llama":
        from models.llama_ollama import OllamaModel
        model = OllamaModel(model_name="llama3.1:8b")
    else:
        raise ValueError("지원하지 않는 모델명입니다.")

    in_data = read_jsonl(in_path)
    if args.limit:
        in_data = in_data[:args.limit]

    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    
    for row in tqdm(in_data, desc=f"Running {args.models}"):
        try:
            system, user = build_prompt(row, args.with_doc)
            text, usage = model.generate(system, user)
            outputs.append({
                "sample_id": row.get("sample_id"),
                "output": text,
                "input_token": usage.get("input_token", 0),
                "output_token": usage.get("output_token", 0),
                "latency": usage.get("latency", 0),
            })
        except Exception as e:
            outputs.append({"sample_id": row.get("sample_id"), "output": f"[ERROR] {str(e)}"})
        
        if args.models == "claude":
            time.sleep(1)

    suffix = "doc" if args.with_doc else "seg"
    out_path = out_dir / f"{args.models}_5_samples.{suffix}.jsonl"
    write_jsonl(out_path, outputs)
    print(f"[DONE] 물리 파일 생성 완료: {out_path}")

if __name__ == "__main__":
    main()