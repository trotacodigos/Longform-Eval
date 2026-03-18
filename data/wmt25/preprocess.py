import json
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, List

@dataclass
class PipelineState:
    """State tracker for sample and document IDs per language pair."""
    sample_id: int = 1
    doc_id: int = 0
    doc_mapping: Dict[str, int] = field(default_factory=dict)

def get_normalized_lang_pair(src_lang: str, tgt_lang: str) -> str:
    """Normalize raw language codes to lab standard."""
    if src_lang == 'en':
        if tgt_lang == 'ko_KR': return 'en-ko'
        if tgt_lang == 'zh_CN': return 'en-zh'
    return ""

def setup_directories(base_dir: Path, lang_pair: str) -> Dict[str, Path]:
    """Create and return directory paths for the lab schema."""
    pair_dir = base_dir / lang_pair
    dirs = {
        "src_docs": pair_dir / "src_docs",
        "tgt_docs": pair_dir / "tgt_docs",
        "base": pair_dir
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs

def process_wmt_data(input_file_path: str, output_base_dir: str) -> None:
    """Process WMT25 jsonl to WMT24++ lab schema."""
    input_path = Path(input_file_path)
    base_out_path = Path(output_base_dir)
    
    # Initialize trackers for target languages
    states: Dict[str, PipelineState] = {
        "en-ko": PipelineState(),
        "en-zh": PipelineState()
    }
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            data: Dict[str, Any] = json.loads(line)
            
            # Filter unsupported languages
            lang_pair = get_normalized_lang_pair(data['src_lang'], data['tgt_lang'])
            if not lang_pair:
                continue
                
            state = states[lang_pair]
            dirs = setup_directories(base_out_path, lang_pair)
            original_doc_id = data['doc_id']
            
            # Check if it's a new document
            if original_doc_id not in state.doc_mapping:
                state.doc_mapping[original_doc_id] = state.doc_id
                
                # Save full raw text to source and target directories
                src_txt_path = dirs["src_docs"] / f"{state.doc_id}.txt"
                tgt_txt_path = dirs["tgt_docs"] / f"{state.doc_id}.txt"
                
                with open(src_txt_path, 'w', encoding='utf-8') as src_file:
                    src_file.write(data['src_text'])
                with open(tgt_txt_path, 'w', encoding='utf-8') as tgt_file:
                    tgt_file.write(data['refs']['refA']['ref'])
                    
                state.doc_id += 1
                
            numeric_doc_id = state.doc_mapping[original_doc_id]
            
            # Segment text by paragraph (newlines)
            src_list = data['src_text'].split('\n')
            tgt_list = data['refs']['refA']['ref'].split('\n')
            
            input_jsonl_path = dirs["base"] / f"input_{lang_pair}.jsonl"
            output_jsonl_path = dirs["base"] / f"output_{lang_pair}.jsonl"
            
            with open(input_jsonl_path, 'a', encoding='utf-8') as in_file, \
                 open(output_jsonl_path, 'a', encoding='utf-8') as out_file:
                 
                for src_seg, tgt_seg in zip(src_list, tgt_list):
                    # Skip empty strings
                    if not src_seg.strip() or not tgt_seg.strip():
                        continue
                        
                    # Format as lab input schema
                    lab_input = {
                        "sample_id": state.sample_id,
                        "doc_id": numeric_doc_id,
                        "domain": data["domain"],
                        "system": "Unbabel-Tower70B",
                        "src_lang": "en",
                        "tgt_lang": lang_pair.split('-')[1],
                        "src_seg": src_seg.strip(),
                        "tgt_seg": tgt_seg.strip()
                    }
                    
                    # Format as lab output schema for inference
                    lab_output = {
                        "sample_id": state.sample_id,
                        "src_seg": src_seg.strip(),
                        "tgt_seg": tgt_seg.strip(),
                        "ref_seg": tgt_seg.strip(),
                        "human_pe_seg": tgt_seg.strip(),
                        "model_pe_seg": None,
                        "manual": None,
                        "auto": {}
                    }
                    
                    in_file.write(json.dumps(lab_input, ensure_ascii=False) + '\n')
                    out_file.write(json.dumps(lab_output, ensure_ascii=False) + '\n')
                    
                    state.sample_id += 1

    # Save mapping dict to json
    for lang_pair, state in states.items():
        if state.doc_mapping:
            mapping_path = base_out_path / lang_pair / "doc_id.json"
            with open(mapping_path, 'w', encoding='utf-8') as map_file:
                json.dump(state.doc_mapping, map_file, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    # Clear generated subdirectories to avoid appending to existing files during testing
    os.system("rm -rf data/wmt25/en-ko data/wmt25/en-zh") 
    
    # Run WMT25 processing pipeline
    process_wmt_data('data/wmt25/wmt25-genmt.jsonl', 'data/wmt25')