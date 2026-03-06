from base import BaseEvaluator

import torch
from typing import List
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

METRICX_SCORE_TOKEN_ID = 250089  # <extra_id_10>


class MetricX_Metric(BaseEvaluator):
    def __init__(self, model_name: str = "google/metricx-24-hybrid-xl-v2p6-bfloat16", batch_size: int = 16, model=None, tokenizer=None):
        super().__init__("MetricX-24", batch_size, max_input_tokens=1536)
        # If loaded, reuse them.
        if model is not None and tokenizer is not None:
            print(f"Reusing the loaded model: [{self.name}]")
            self.model = model
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                tie_word_embeddings=False,
                device_map="auto"
            )
            self.model.eval()

    def _format_prompts(self, data):
        return [
            f"candidate: {item.get('tgt_seg', '')} source: {item.get('src_seg', '')} reference: {item.get('ref_seg', '')}"
            for item in data
        ]

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

                batch_size = inputs.input_ids.shape[0]
                decoder_input_ids = torch.zeros((batch_size, 1), dtype=torch.long, device=self.device)

                outputs = self.model(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    decoder_input_ids=decoder_input_ids,
                )
                predictions = outputs.logits[:, 0, METRICX_SCORE_TOKEN_ID]
                predictions = torch.clamp(predictions, 0, 25)
                scores.extend(predictions.tolist())
        return scores


class MetricX_QE_Metric(MetricX_Metric):
    def __init__(self, model_name: str = "google/metricx-24-hybrid-xxl-v2p6", batch_size=16, model=None, tokenizer=None):
        super().__init__(model_name, batch_size, model, tokenizer)
        self.name = "MetricX-24-QE"

    def _format_prompts(self, data) -> List[str]:
        return [
            f"candidate: {item.get('tgt_seg', '')} source: {item.get('src_seg', '')}"
            for item in data
        ]

