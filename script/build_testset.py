import os
import json
from itertools import zip_longest
from typing import Dict, List, Any, Optional

# Configuration and constants
TARGET_PAIRS: List[str] = ['en-ko', 'en-zh']
root = "../../data/wmt25-general-mt/"

MAIN_FILE: str = os.path.join(root, "data/wmt25-genmt.jsonl")
SYS_DIR = os.path.join(root, "data/systems")

BASE_OUT_DIR: str = 'data/wmt25/'

ALLOWED_DOMAINS = {"literary", "social", "news"}

def parse_reference_text(refs_data: Any) -> str:
    """Extract plain text from complex nested reference dictionaries."""
    if isinstance(refs_data, dict) and refs_data:
        ref_val = list(refs_data.values())[0]
        if isinstance(ref_val, dict):
            return str(list(ref_val.values())[0])
        elif isinstance(ref_val, list):
            return '\n'.join(map(str, ref_val))
        return str(ref_val)
    return str(refs_data)

def normalize_lang_pair(doc_id: str) -> Optional[str]:
    """Extract and normalize target language pair from document ID."""
    for lp in TARGET_PAIRS:
        if doc_id.startswith(lp): return lp
    return None

def main():
    # Load system hypotheses into memory
    system_files: List[str] = [f for f in os.listdir(SYS_DIR) if f.endswith('.jsonl')]
    sys_answers: Dict[str, Dict[str, str]] = {}

    for sys_file in system_files:
        sys_name: str = sys_file.replace('.jsonl', '')
        sys_answers[sys_name] = {}
        with open(os.path.join(SYS_DIR, sys_file), 'r', encoding='utf-8') as f:
            for line in f:
                row: Dict[str, Any] = json.loads(line)
                sys_answers[sys_name][row['doc_id']] = row['hypothesis']

    # Display loading results (EDA)
    print(f"[INFO] Successfully loaded {len(sys_answers)} system hypotheses.")
    sample_sys: str = list(sys_answers.keys())[0]
    sample_doc: str = list(sys_answers[sample_sys].keys())[0]
    print(f"\n[DEBUG] Sample output for model '{sample_sys}':")
    print(sys_answers[sample_sys][sample_doc][:150], "...")

    # Pre-create output directories and open output files per lang_pair
    lp_dirs: Dict[str, tuple] = {}
    out_files: Dict[str, tuple] = {}
    sample_ids: Dict[str, int] = {}

    for lp in TARGET_PAIRS:
        src_lang, tgt_lang = lp.split('-')
        src_dir = os.path.join(BASE_OUT_DIR, lp, 'src_docs')
        tgt_dir = os.path.join(BASE_OUT_DIR, lp, 'tgt_docs')
        os.makedirs(src_dir, exist_ok=True)
        os.makedirs(tgt_dir, exist_ok=True)
        lp_dirs[lp] = (src_dir, tgt_dir, src_lang, tgt_lang)

        out_dir = os.path.join(BASE_OUT_DIR, lp)
        f_in = open(os.path.join(out_dir, f'input_{lp}.jsonl'), 'w', encoding='utf-8')
        f_out = open(os.path.join(out_dir, f'output_{lp}.jsonl'), 'w', encoding='utf-8')
        out_files[lp] = (f_in, f_out)
        sample_ids[lp] = 1

    doc_id_maps: Dict[str, Dict[str, int]] = {lp: {} for lp in TARGET_PAIRS}
    numeric_id_counters: Dict[str, int] = {lp: 0 for lp in TARGET_PAIRS}
    preview_rows: List[str] = []
    total_count: int = 0

    try:
        with open(MAIN_FILE, 'r', encoding='utf-8') as f_main:
            for line in f_main:
                data: Dict[str, Any] = json.loads(line)
                original_doc_id: str = data['doc_id']

                # [FILTER 1] Include only 'general' collection (Drop others)
                if data.get('collection_id') != "general":
                    continue

                # [FILTER 2] Include only specific domains (Exclude 'speech', etc.)
                domain: str = data.get('domain', 'unknown')
                if domain not in ALLOWED_DOMAINS:
                    continue

                lang_pair: Optional[str] = normalize_lang_pair(original_doc_id)
                if not lang_pair: continue

                src_text: str = str(data.get('src_text', ''))
                ref_text: str = parse_reference_text(data.get('refs', {}))

                # Map doc_id to integer and save full context documents
                if original_doc_id not in doc_id_maps[lang_pair]:
                    new_num_id: int = numeric_id_counters[lang_pair]
                    doc_id_maps[lang_pair][original_doc_id] = new_num_id
                    numeric_id_counters[lang_pair] += 1

                    src_dir, tgt_dir, _, _ = lp_dirs[lang_pair]
                    with open(os.path.join(src_dir, f"{new_num_id}.txt"), 'w', encoding='utf-8') as sf:
                        sf.write(src_text)
                    with open(os.path.join(tgt_dir, f"{new_num_id}.txt"), 'w', encoding='utf-8') as tf:
                        tf.write(ref_text)

                new_doc_id: int = doc_id_maps[lang_pair][original_doc_id]
                _, _, src_lang, tgt_lang = lp_dirs[lang_pair]
                f_in, f_out = out_files[lang_pair]

                # Segment text and join with system hypotheses
                src_segs: List[str] = src_text.split('\n')
                ref_segs: List[str] = ref_text.split('\n')

                for sys_name, answers_dict in sys_answers.items():
                    hypothesis_text: str = str(answers_dict.get(original_doc_id, ""))
                    if not hypothesis_text.strip(): continue

                    hyp_segs: List[str] = hypothesis_text.split('\n')

                    for src_seg, tgt_seg, ref_seg in zip_longest(src_segs, hyp_segs, ref_segs, fillvalue=""):
                        if not src_seg.strip(): continue

                        sample_id = sample_ids[lang_pair]
                        input_dict: Dict[str, Any] = {
                            "sample_id": sample_id, "doc_id": new_doc_id, "domain": domain,
                            "system": sys_name, "src_lang": src_lang, "tgt_lang": tgt_lang,
                            "src_seg": src_seg, "tgt_seg": tgt_seg
                        }
                        output_dict: Dict[str, Any] = {
                            "sample_id": sample_id, "src_seg": src_seg, "tgt_seg": tgt_seg,
                            "ref_seg": ref_seg, "human_pe_seg": ref_seg,
                            "model_pe_seg": None, "manual": None, "auto": {}
                        }

                        f_in.write(json.dumps(input_dict, ensure_ascii=False) + '\n')
                        f_out.write(json.dumps(output_dict, ensure_ascii=False) + '\n')
                        sample_ids[lang_pair] += 1
                        total_count += 1

                        if len(preview_rows) < 5:
                            preview_rows.append(
                                f"  [{lang_pair}] doc={new_doc_id} sys={sys_name} src={src_seg[:40]!r}"
                            )
    finally:
        for lp, (f_in, f_out) in out_files.items():
            f_in.close()
            f_out.close()

    # Export mapping dictionaries to JSON
    for lp, id_map in doc_id_maps.items():
        if id_map:
            out_dir = os.path.join(BASE_OUT_DIR, lp)
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, 'doc_id.json'), 'w', encoding='utf-8') as f:
                json.dump(id_map, f, ensure_ascii=False, indent=2)

    print("[INFO] Data join and context document extraction completed.")
    print(f"[INFO] Total merged segments: {total_count}")
    print("\n[DEBUG] Preview of first 5 records:")
    print('\n'.join(preview_rows))

    for lp in TARGET_PAIRS:
        count = sample_ids[lp] - 1
        if count:
            print(f"[SUCCESS] Exported {count} records for {lp}.")


if __name__ == "__main__":
    main()