from .tools import get_keys, extract_token_usage
from .base import BaseModel
from .sampling import OpenAIParams, OpenAIChatParams

from openai import OpenAI
import requests
import json
import io
import time

class OpenAIModel(BaseModel):
    """SDK-based. generate_batch uses OpenAI Batch API."""
    POLL_INTERVAL: int = 30

    def __init__(self, name: str,
                 model_id: str,
                 sampling_params: OpenAIParams | dict | None = None,
                 log_path: str = "inference_result.csv",
                 ):
        super().__init__(name, model_id, sampling_params or OpenAIParams(), log_path)
        keys = get_keys("OPENAI_API_KEYS")
        self.client = OpenAI(api_key=keys[0])

    def _complete(self, system: str, user: str):
        resp = self.client.chat.completions.create(
            model=self.model_id,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            **self.sampling_params.to_kwargs(),
        )
        content = resp.choices[0].message.content
        text = (content or "").strip()

        usage = getattr(resp, "usage", {})
        in_token, out_token = extract_token_usage(usage)
        return text, in_token, out_token

    def generate_batch(
        self,
        prompts: list[tuple[str, str]],
    ) -> list[tuple[str, dict] | None]:        
        # 1. Build JSONL file in memory
        lines = []
        for i, (system, user) in enumerate(prompts):
            lines.append(json.dumps({
                "custom_id": str(i),
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": self.model_id,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    **self.sampling_params.to_kwargs(),
                },
            }))
        jsonl_bytes = "\n".join(lines).encode("utf-8")

        # 2. Upload file
        file_obj = self.client.files.create(
            file=("batch_input.jsonl", io.BytesIO(jsonl_bytes), "application/jsonl"),
            purpose="batch",
        )
        print(f"[OpenAI batch] uploaded file_id={file_obj.id}")

        # 3. Submit batch
        batch = self.client.batches.create(
            input_file_id=file_obj.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        batch_id = batch.id
        print(f"[OpenAI batch] submitted {len(prompts)} requests → batch_id={batch_id}")

        # 4. Poll until complete
        while True:
            time.sleep(self.POLL_INTERVAL)
            status = self.client.batches.retrieve(batch_id)
            counts = status.request_counts
            print(
                f"[OpenAI batch] completed={counts.completed} "
                f"failed={counts.failed} "
                f"total={counts.total}"
            )
            if status.status in ("completed", "failed", "expired", "cancelled"):
                break

        if status.status != "completed":
            print(f"[OpenAI batch] ended with status={status.status}")
            return [None] * len(prompts)

        # 5. Log errors if any
        if status.error_file_id:
            errors = self.client.files.content(status.error_file_id)
            for line in errors.text.splitlines():
                print(f"[OpenAI batch] error: {line}")

        if not status.output_file_id:
            print("[OpenAI batch] no successful results (output_file_id is None)")
            return [None] * len(prompts)

        # 6. Retrieve and parse results
        results: list[tuple[str, dict] | None] = [None] * len(prompts)

        output = self.client.files.content(status.output_file_id)
        for line in output.text.splitlines():
            item = json.loads(line)
            idx = int(item["custom_id"])

            if item.get("error"):
                print(f"[OpenAI batch] index {idx} failed: {item['error']}")
                continue

            msg = item["response"]["body"]["choices"][0]["message"]
            text = (msg["content"] or "").strip()

            usage = item["response"]["body"].get("usage", {})
            in_token, out_token = extract_token_usage(usage)

            metrics = {
                "model_name": self.name,
                "input_token": int(in_token),
                "output_token": int(out_token),
                "latency_sec": None,  # Batch API does not expose per-request latency
                "tps": None,
            }
            self._log_to_csv(metrics)
            results[idx] = (text, metrics)

        return results
    

class OpenAIChatModel(BaseModel):
    """requests-based"""
    def __init__(
        self,
        name: str,
        model_id: str,
        endpoint: str = "http://localhost:8000/v1/chat/completions",
        sampling_params: OpenAIChatParams | dict | None = None,
        prompt_adapter=None,
        tgt_lang: str | None = None,
        strip_thinking: bool = False,
        merge_system_prompt: bool = False,
    ):
        super().__init__(name, model_id, sampling_params or OpenAIChatParams())
        if not endpoint:
            raise ValueError("OpenAIChatModel requires `endpoint`")
        self.endpoint = endpoint
        self.prompt_adapter = prompt_adapter
        self.tgt_lang = tgt_lang
        self.strip_thinking = strip_thinking
        self.merge_system_prompt = merge_system_prompt

    def _complete(self, system: str | None, user: str):
        
        if self.prompt_adapter:
            user = self.prompt_adapter(user, self.tgt_lang) if self.tgt_lang else self.prompt_adapter(user)

        if self.merge_system_prompt:
            merged = f"{system}\n{user}" if system else user
            messages = [{"role": "user", "content": merged}]
        else:
            messages = [{"role": "user", "content": user}] if system is None else [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]

        payload = {
            "model": self.model_id,
            "messages": messages,
            **self.sampling_params.to_kwargs(),
            **self._extra_payload(),            
        }
        #print(json.dumps(payload, ensure_ascii=False, indent=2))
        response = requests.post(self.endpoint, json=payload, headers=self._headers(), timeout=600)
        response.raise_for_status()
        data = response.json()

        content = data["choices"][0]["message"]["content"] or ""

        if self.strip_thinking and "</think>" in content:
            content = content.split("</think>", 1)[-1]

        text = self._post_process(content).strip()  

        usage = data.get("usage", {})
        in_token, out_token = extract_token_usage(usage)
        return text, in_token, out_token

    def _extra_payload(self) -> dict:
        """To add customized keys to the payload"""
        return {}
    
    def _headers(self) -> dict:
        """Override to add custom request headers (e.g. auth)"""
        return {}

    def _post_process(self, content: str) -> str:
        """To post-process responses"""
        return content

    