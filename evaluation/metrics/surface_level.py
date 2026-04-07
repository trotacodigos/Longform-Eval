from typing import List, Dict
from sacrebleu.metrics import CHRF, TER

from .base import BaseEvaluator


class SacreBLEU_Metric(BaseEvaluator):
    def __init__(self, name: str, metric_obj, batch_size: int = 32):
        super().__init__(name, batch_size, max_input_tokens=999999)
        self.metric = metric_obj

    def evaluate(self, data) -> List[float]:
        return self._evaluate_batch(data)

    def _evaluate_batch(self, data: List[Dict]) -> List[float]:
        scores = []
        for item in data:
            hyp = item.get("tgt_seg", "")
            ref = item.get("ref_seg", "")
            score = self.metric.sentence_score(hyp, [ref]).score
            scores.append(score)
        return scores


class ChrF_Metric(SacreBLEU_Metric):
    def __init__(self, batch_size: int = 32):
        super().__init__("chrF", CHRF(word_order=2), batch_size)


class TER_Metric(SacreBLEU_Metric):
    def __init__(self, batch_size: int = 32):
        super().__init__("TER", TER(asian_support=True), batch_size)