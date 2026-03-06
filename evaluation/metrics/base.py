import torch
from typing import List, Dict
import numpy as np

class BaseEvaluator:
    def __init__(self, name: str, batch_size: int = 16, max_input_tokens: int = 1024):
        self.name = name
        self.batch_size = batch_size
        self.max_input_tokens = max_input_tokens
        self.max_chunk_tokens = max(1, self.max_input_tokens - 36)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None

    def _format_prompts(self, data) -> List[str]:
        return [f"candidate: {item.get("tgt_seg", "")} source: {item.get("src_seg", "")} reference: {item.get("ref_seg", "")}" for item in data]
    
    def _evaluate_batch(self, data: List[Dict[str, str]]) -> List[float]:
        raise NotImplementedError

    def evaluate(self, data)-> List[float]:
        results = [None] * len(data)
        short_items = []
        short_indices = []

        for i, item in enumerate(data):
            prompt = self._format_prompts([item])[0]
            token_length = len(self.tokenizer.encode(prompt))

            # Chunk long documents and compute score
            if token_length > self.max_input_tokens:
                results[i] = self.evaluate_long_document(item)
            else:
                short_items.append(item)
                short_indices.append(i)

        if short_items:
            short_scores = self._evaluate_batch(short_items)
            for idx, score in zip(short_indices, short_scores):
                results[idx] = score

        return results
    
    def evaluate_long_document(self, item: Dict[str, str]) -> float:
        docs = {k: [p.strip() for p in v.split("\n")] for k, v in item.items() if v}

        max_len = max([len(v) for v in docs.values()] + [0])
        if max_len == 0: return 0.0

        grouped_data = []
        current_group = {k: [] for k in docs.keys()}

        for i in range(max_len):
            current_doc = {k: (v[i] if i < len(v) else "") for k, v in docs.items()}
            test_group = {k: "\n".join(current_group[k] + current_doc[k]).strip() for k in docs.keys()}
            formatted_prompt = self._format_prompts([test_group][0])
            token_count = len(self.tokenizer.encode(formatted_prompt))

            # if not over the limit, append it.
            if token_count <= self.max_chunk_tokens:
                for k in docs.keys():
                    current_group[k].append(current_group[k])
            else:
                if any(current_group.values()):
                    grouped_data.append({k: "\n".join(v).strip() for k, v in current_group.items()})

                current_group = {k: [current_group[k]] for k in docs.keys()}

        if any(current_group.values()):
            grouped_data.append({k: "\n".join(v).strip() for k, v in current_group.items()})

        chunked_data = [d for d in grouped_data if any(d.values())]
        if not chunked_data: return 0.0

        chunk_scores = self._evaluate_batch(chunked_data)
        valid_scores = [s for s in chunk_scores if s is not None]

        return float(np.mean(valid_scores) if valid_scores else 0.0)