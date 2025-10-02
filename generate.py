from pathlib import Path
import json
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from prompts.ape_prompt import build_prompt
from models.loaders import load_models_from_yaml


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, records: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            
def normalize_pe(text: str) -> str:
    if not text: return ""
    t = text.strip()
    if "<pe>" in t and "</pe>" in t:
        t = t.replace("<pe>", "").replace("</pe>", "")
    return t.strip().replace("\n"," ").strip('"').strip("'")


def run_one_sample(model, row: dict, has_doc: bool) -> dict:
    system, user = build_prompt(row, has_doc)
    text, usage = model.generate(system, user)
    return {
        "sample_id": row.get("sample_id"),
        "output": text,
        "input_token": usage.get("input_token"),
        "output_token": usage.get("output_token"),
        "latency": usage.get("latency"),
    }


def _parse_model_list(arg: str | None) -> list[str] | None:
    if not arg:
        return None
    # "a,b c , d" → ["a","b","c","d"]
    parts = []
    for chunk in arg.split(","):
        parts.extend(p.strip() for p in chunk.split() if p.strip())
    return parts or None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="models.yaml 경로")
    ap.add_argument("--input_file", required=True, help="입력 JSONL 파일 경로")
    ap.add_argument("--output_dir", required=True, help="모델별 출력 디렉토리")
    ap.add_argument("--models", default=None, help="실행할 모델 이름 목록 (쉼표/공백 구분)")
    ap.add_argument("--with_doc", action="store_true", help="문서 컨텍스트 포함 프롬프트 사용")
    ap.add_argument("--max_workers", type=int, default=4, help="동시 실행 스레드 수")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    in_path = Path(args.input_file)
    out_dir = Path(args.output_dir)

    select_names = _parse_model_list(args.models)

    models = load_models_from_yaml(cfg_path, select_names=select_names)
    in_data = read_jsonl(in_path)

    print(f"[INFO] Loaded {len(models)} models from {cfg_path}")
    print(f"[INFO] Loaded {len(in_data)} samples from {in_path}")
    out_dir.mkdir(parents=True, exist_ok=True)

    for model in models:
        model_display_name = getattr(model, "name", model.__class__.__name__)
        max_workers = args.max_workers

        print(f"[RUN] {model_display_name} ... (workers={max_workers})")
        outputs: list[dict] = []

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            fut2idx = {
                pool.submit(run_one_sample, model, row, args.with_doc): i
                for i, row in enumerate(in_data)
            }
            for fut in tqdm(as_completed(fut2idx), total=len(fut2idx), desc=model_display_name, colour='yellow'):
                i = fut2idx[fut]
                try:
                    rec = fut.result()
                except Exception as e:
                    rec = {
                        "sample_id": in_data[i].get("sample_id"),
                        "output": "",
                        "input_token": None,
                        "output_token": None,
                        "latency": None,
                        "error": str(e),
                    }
                outputs.append(rec)

        outputs.sort(key=lambda r: (r.get("sample_id") is None, r.get("sample_id")))

        suffix = "doc" if args.with_doc else "seg"
        out_path = out_dir / f"{model_display_name}.{suffix}.jsonl"
        write_jsonl(out_path, outputs)
        print(f"[DONE] {out_path}  ({len(outputs)} lines)")

    print("[ALL DONE]")


if __name__ == "__main__":
    main()