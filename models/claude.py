import os
import time
from anthropic import Anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

from .base import BaseModel
from .sampling import ClaudeParams
from .tools import get_keys, extract_token_usage


class AnthropicModel(BaseModel):
    def __init__(self, name: str, model_id: str, sampling_params: ClaudeParams | dict | None = None):
        super().__init__(name, model_id, sampling_params or ClaudeParams)
        keys = get_keys("ANTHROPIC_API_KEYS")
        api_key = keys[0] if keys else os.environ.get("ANTHROPIC_API_KEY")
        self.client = Anthropic(api_key=api_key)

    def _call(self, system: str, user: str):
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

    def generate_batch(self, prompts: list[tuple[str, str]], poll_interval: int = 60) -> list[tuple[str, dict]]:
        """Submit prompts as a batch and return (text, metrics) per prompt in order."""
        requests = [
            Request(
                custom_id=str(i),
                params=MessageCreateParamsNonStreaming(
                    model=self.model_id,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                    **self.sampling_params.to_kwargs(),
                ),
            )
            for i, (system, user) in enumerate(prompts)
        ]

        batch = self.client.messages.batches.create(requests=requests)
        print(f"Batch submitted: {batch.id}")

        while True:
            batch = self.client.messages.batches.retrieve(batch.id)
            if batch.processing_status == "ended":
                break
            print(f"Batch status: {batch.processing_status} — waiting {poll_interval}s...")
            time.sleep(poll_interval)

        results = {None: None}  # placeholder
        ordered = [None] * len(prompts)
        for result in self.client.messages.batches.results(batch.id):
            i = int(result.custom_id)
            if result.result.type == "succeeded":
                resp = result.result.message
                text = (resp.content[0].text or "").strip() if resp.content else ""
                in_token, out_token = extract_token_usage(getattr(resp, "usage", {}))
                metrics = {
                    "model_name": self.name,
                    "input_token": int(in_token or 0),
                    "output_token": int(out_token or 0),
                    "latency_sec": None,
                    "tps": None,
                }
                ordered[i] = (text, metrics)
            else:
                ordered[i] = ("", {"error": result.result.type})
        return ordered
    
    
class ClaudeSonnet4_5(AnthropicModel):
    def __init__(self):
        super().__init__(name="claude-sonnet-4.5", model_id="claude-sonnet-4-5")