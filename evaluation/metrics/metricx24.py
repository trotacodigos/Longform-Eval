from .base import BaseEvaluator

import torch
from typing import List
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class MetricX_Metric(BaseEvaluator):
    def __init__(self, model_name: str = "google/metricx-24-hybrid-xxl-v2p6", batch_size: int = 16, model=None, tokenizer=None):
        super().__init__("MetricX-24", batch_size, max_input_tokens=1536)
        # If loaded, reuse them.
        if model is not None and tokenizer is not None:
            print(f"Reusing the loaded model: [{self.name}]")
            self.model = model
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name, torch_dtype=torch.bfloat16, device_map="auto"
            )
            self.model.eval()

    def _evaluate_batch(self, data) -> List[float]:
        print(f"Start evaluating [{self.name}] {len(data)} (Batch: {self.batch_size})")
        scores = []
        formatted_inputs = self._format_prompts(data)

        with torch.no_grad():
            for i in range(0, len(formatted_inputs), self.batch_size):
                batch_texts = formatted_inputs[i: i+self.batch_size]

                inputs = self.tokenizer(
                    batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_input_tokens,
                ).to(self.device)

                outputs = self.model.generate(**inputs, max_new_tokens=10)
                batch_preds = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

                for pred in batch_preds:
                    try:
                        scores.append(float(pred.strip()))
                    except ValueError:
                        scores.append(None)
        return scores


class MetricX_QE_Metric(MetricX_Metric):
    def __init__(self, model_name: str = "google/metricx-24-hybrid-xxl-v2p6", batch_size=16, model=None, tokenizer=None):
        super().__init__(model_name, batch_size, model, tokenizer)
        self.name = "MetricX-24-QE"

    def _format_prompts(self, data) -> List[str]:
        return [f"candidate: {item.get("tgt_seg", "")} source: {item.get("src_seg", "")}" for item in data]

