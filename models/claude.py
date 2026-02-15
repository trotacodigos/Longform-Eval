import os
import anthropic
import time

class ClaudeModel:
    def __init__(self, model_name="claude-3-5-sonnet-20240620"):
        # API Key는 환경변수에서 로드 (보안 준수)
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is not set.")
        
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model_name = model_name
        self.name = "claude"

    def generate(self, prompt: str) -> str:
        for attempt in range(3):
            try:
                message = self.client.messages.create(
                    model=self.model_name,
                    max_tokens=4096,
                    messages=[{"role": "user", "content": prompt}]
                )
                return message.content[0].text
            except Exception as e:
                print(f"[Claude Error] {e}, retrying ({attempt+1}/3)...")
                time.sleep(2)
        return ""