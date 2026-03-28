import os
import time
from anthropic import Anthropic
#from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
#from anthropic.types.messages.batch_create_params import Request

from .base import BaseModel
from .sampling import ClaudeParams
from .tools import get_keys, extract_token_usage


class AnthropicModel(BaseModel):
    POLL_INTERVAL: int = 30

    def __init__(self, name: str, model_id: str, sampling_params: ClaudeParams | dict | None = None):
        super().__init__(name, model_id, sampling_params or ClaudeParams)
        keys = get_keys("ANTHROPIC_API_KEYS")
        api_key = keys[0] if keys else os.environ.get("ANTHROPIC_API_KEY")
        self.client = Anthropic(api_key=api_key)

    def _call(self, system: str, user: str):
        """single inference (real-time)"""
        resp = self.client.messages.create(
            model=self.model_id,
            system=system,
            messages=[{"role": "user", "content": user}],
            **self.sampling_params.to_kwargs(),
        )
        text = (resp.content[0].text or "").strip() if resp.content else ""
        usage = getattr(resp, "usage", {})
        in_token, out_token = extract_token_usage(usage)
        return text, in_token, out_token

    def generate_batch(
        self,
        prompts: list[tuple[str, str]],
        log_path: str = "inference_result.csv",
    ) -> list[tuple[str, dict] | None]:
        """Submit all prompts as one Anthropic batch; poll until done."""

        # 1. Build requests
        requests = [
            {
                "custom_id": str(i),
                "params": {
                    "model": self.model_id,
                    "system": system,
                    "messages": [{"role": "user", "content": user}],
                    **self.sampling_params.to_kwargs(),
                },
            }
            for i, (system, user) in enumerate(prompts)
        ]
        
        # 2. Submit
        batch = self.client.messages.batches.create(requests=requests)
        batch_id = batch.id
        print(f"[Claude batch] submitted {len(prompts)} requests → batch_id={batch_id}")

        # 3. Poll until ended
        while True:
            time.sleep(self.POLL_INTERVAL)
            status = self.client.messages.batches.retrieve(batch_id)
            counts = status.request_counts
            print(
                f"[Claude batch] processing={counts.processing} "
                f"succeeded={counts.succeeded} "
                f"errored={counts.errored}"
            )
            if status.processing_status == "ended":
                break

        # 4. Collect results (aligned to original index)
        results: list[tuple[str, dict] | None] = [None] * len(prompts)

        for result in self.client.messages.batches.results(batch_id):
            idx = int(result.custom_id)

            if result.result.type != "succeeded":
                print(f"[Claude batch] index {idx} failed: {result.result.type}")
                continue

            msg = result.result.message
            text = (msg.content[0].text or "").strip()
            in_token = msg.usage.input_tokens
            out_token = msg.usage.output_tokens

            metrics = {
                "model_name": self.name,
                "input_token": int(in_token),
                "output_token": int(out_token),
                "latency_sec": None,   # batch API doesn't expose per-request latency
                "tps": None,
            }
            self._log_to_csv(metrics, log_path)
            results[idx] = (text, metrics)

        return results
    
    
class ClaudeSonnet4_5(AnthropicModel):
    def __init__(self):
        super().__init__(name="claude-sonnet-4.5", model_id="claude-sonnet-4-5")