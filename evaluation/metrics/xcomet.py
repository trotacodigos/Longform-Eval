import torch
from typing import List, Dict
from transformers import AutoTokenizer
from comet import download_model, load_from_checkpoint

from .base import BaseEvaluator


class XCOMET(BaseEvaluator):
    def __init__(self, name: str = "Unbable/XCOMET-XXL", batch_size: int = 16, extract_errors: bool = False):
        super().__init__("XCOMET-XXL", batch_size,  max_input_tokens=512) # by XLM-RoBERTa
        self.extract_errors = extract_errors
        print(f"[{self.name}] Model download and loading...")

        model_path = download_model(self.name)
        self.model = load_from_checkpoint(model_path)
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")

    def _format_prompts(self, data) -> List[str]:
        return [f"candidate: {item.get("tgt_seg", "")} source: {item.get("src_seg", "")} reference: {item.get("ref_seg", "")}" for item in data]

    def _evaluate_batch(self, data):
        # From our data to the Unbable style
        data = [{
            "src": item.get("src_seg", ""),
            "ref": item.get("ref_seg", ""),
            "mt": item.get("tgt_seg", "")
        } for item in data]

        predictions = self.model.predict(data, batch_size=self.batch_size, gpus=1)

        if self.extract_errors:
            return [
                {
                    "score": predictions.scores[i],
                    "error_spans": predictions.metadata.error_span[i] if hasattr(predictions.metadata, "error_spans") else []
                }
                for i in range(len(predictions.scores))
            ]
        return predictions.scores

    
class XCOMET_QE(XCOMET):
    def __init__(self, model_name = "Unbabel/wmt23-cometkiwi-da-xxl", batch_size=16, extract_errors: bool = False):
        super().__init__(model_name, batch_size, extract_errors)
        self.name = "XCOMET-XXL-QE"

    def _format_prompts(self, data):
        return [f"{item.get("src_seg", "")} {item.get("tgt_seg", "")}" for item in data]
    
    def _evaluate_batch(self, data):
        data = [{
            "src": item.get("src_seg", ""),
            "mt": item.get("tgt_seg", "")
        } for item in data]

        predictions = self.model.predict(data, batch_size=self.batch_size, gpus=1)

        if self.extract_errors:
            return [
                {
                    "score": predictions.scores[i],
                    "error_spans": predictions.metadata.error_span[i] if hasattr(predictions.metadata, "error_spans") else []
                }
                for i in range(len(predictions.scores))
            ]
        return predictions.scores