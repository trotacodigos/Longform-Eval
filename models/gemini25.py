import time

from .base import BaseModel
from .sampling import GeminiParams
from .tools import get_keys

from google import genai
from google.genai import types

_TERMINAL_STATES = {
    types.JobState.JOB_STATE_SUCCEEDED,
    types.JobState.JOB_STATE_FAILED,
    types.JobState.JOB_STATE_CANCELLED,
    types.JobState.JOB_STATE_EXPIRED,
    types.JobState.JOB_STATE_PARTIALLY_SUCCEEDED,
}


def _build_config(system: str | None, sampling_kwargs: dict) -> types.GenerateContentConfig:
    kwargs = sampling_kwargs.copy()
    thinking_budget = kwargs.pop("thinking_budget", 0)
    thinking_config = types.ThinkingConfig(thinking_budget=thinking_budget) if thinking_budget != 0 else None
    return types.GenerateContentConfig(
        system_instruction=system,
        thinking_config=thinking_config,
        **kwargs,
    )


class GeminiModel(BaseModel):
    POLL_INTERVAL: int = 10000
    BATCH_API_MIN_SIZE: int = 100

    def __init__(
        self,
        name: str = "gemini-2.5-pro",
        model_id: str = "gemini-2.5-pro",
        sampling_params: GeminiParams | dict | None = None,
        prompt_adapter=None,
        tgt_lang: str | None = None,
    ):
        super().__init__(name, model_id, sampling_params or GeminiParams())
        api_key = get_keys("GEMINI_API_KEYS")[0]
        self.client = genai.Client(api_key=api_key)
        self.prompt_adapter = prompt_adapter
        self.tgt_lang = tgt_lang

    def _complete(self, system: str | None, user: str):
        if self.prompt_adapter:
            user = self.prompt_adapter(user, self.tgt_lang) if self.tgt_lang else self.prompt_adapter(user)

        response = self.client.models.generate_content(
            model=self.model_id,
            contents=user,
            config=_build_config(system, self.sampling_params.to_kwargs()),
        )

        text = response.text.strip()
        usage = response.usage_metadata
        in_token = getattr(usage, "prompt_token_count", None)
        out_token = getattr(usage, "candidates_token_count", None)

        return text, in_token, out_token

    def generate_batch(
        self,
        prompts: list[tuple[str, str]],
    ) -> list[tuple[str, dict] | None]:
        if len(prompts) < self.BATCH_API_MIN_SIZE:
            return super().generate_batch(prompts)
        try:
            return self._generate_batch_api(prompts)
        except Exception as e:
            print(f"[Gemini batch] API failed ({e}); falling back to parallel real-time calls")
            return super().generate_batch(prompts)

    def _generate_batch_api(
        self,
        prompts: list[tuple[str, str]],
    ) -> list[tuple[str, dict] | None]:
        # 1. Build inline requests
        requests = [
            types.InlinedRequest(
                model=self.model_id,
                contents=user,
                config=_build_config(system, self.sampling_params.to_kwargs()),
            )
            for system, user in prompts
        ]

        # 2. Submit
        batch = self.client.batches.create(model=self.model_id, src=requests)
        print(f"[Gemini batch] submitted {len(prompts)} requests → name={batch.name}")

        # 3. Poll until done
        while True:
            time.sleep(self.POLL_INTERVAL)
            batch = self.client.batches.get(name=batch.name)
            print(f"[Gemini batch] state={batch.state}")
            if batch.state in _TERMINAL_STATES:
                break

        if batch.state != types.JobState.JOB_STATE_SUCCEEDED:
            print(f"[Gemini batch] ended with state={batch.state}")
            return [None] * len(prompts)

        # 4. Collect results (aligned to input order)
        results: list[tuple[str, dict] | None] = [None] * len(prompts)

        for idx, inlined in enumerate(batch.dest.inlined_responses):
            if inlined.error:
                print(f"[Gemini batch] index {idx} failed: {inlined.error}")
                continue

            resp = inlined.response
            text = (resp.text or "").strip()
            usage = resp.usage_metadata
            in_token = getattr(usage, "prompt_token_count", None)
            out_token = getattr(usage, "candidates_token_count", None)

            metrics = {
                "model_name": self.name,
                "input_token": int(in_token) if in_token is not None else 0,
                "output_token": int(out_token) if out_token is not None else 0,
                "latency_sec": None,
                "tps": None,
            }
            self._log_to_csv(metrics)
            results[idx] = (text, metrics)

        return results
