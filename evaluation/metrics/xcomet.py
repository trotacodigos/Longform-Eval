import torch
from typing import List, Dict
from transformers import AutoTokenizer
from comet import download_model, load_from_checkpoint

from .base import BaseEvaluator


class XCOMET(BaseEvaluator):
    def __init__(self, name: str = "Unbable/XCOMET-XXL", batch_size: int = 16):
        super().__init__("XCOMET-XXL", batch_size,  max_input_tokens=512) # by XLM-RoBERTa
        print(f"[{self.name}] Model download and loading...")

        model_path = download_model(model_name)
        self.model = load_from_checkpoint(model_path)
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")

    def _format_prompts(self, data):
        return [f"{item.get("src_seg", "")} {item.get("tgt_seg", "")} {item.get("ref_seg", "")}"
                for item in data]
    
    def _evaluate_batch(self, data):
        predictions = self.model.predict(data, batch_size=self.batch_size, gpus=1)
        return predictions.scores
    

class XCOMET_QE(XCOMET):
    def __init__(self, model_name = "Unbabel/wmt23-cometkiwi-da-xxl", batch_size=16):
        super().__init__(model_name, batch_size)
        self.name = "XCOMET-XXL-QE"

    def _format_prompts(self, data):
        return [f"{item.get("src_seg", "")} {item.get("tgt_seg", "")}" for item in data]