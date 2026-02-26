import os
import time
import anthropic

class ClaudeModel:
    def __init__(self, model_name="claude-3-5-sonnet-20240620", max_tokens=4096):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.name = "claude"
        
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is not set.")
        self.client = anthropic.Anthropic(api_key=api_key)

    def generate(self, system: str, user: str) -> tuple[str, dict]:
        start_time = time.time()
        for attempt in range(3):
            try:
                # [Fact] Claude Messages API spec: 'system' is separated from 'messages'
                response = self.client.messages.create(
                    model=self.model_name,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                    max_tokens=self.max_tokens
                )
                text = response.content[0].text
                latency = time.time() - start_time
                
                usage = {
                    "input_token": response.usage.input_tokens,
                    "output_token": response.usage.output_tokens,
                    "latency": round(latency, 2)
                }
                return text, usage
            except Exception as e:
                print(f"[Claude Error] {e}. Retrying ({attempt+1}/3)...")
                time.sleep(2)
        
        return "[ERROR] API Failed", {"input_token": 0, "output_token": 0, "latency": 0.0}