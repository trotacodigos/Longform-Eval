import time
import ollama

class OllamaModel:
    def __init__(self, model_name="llama3.1:8b"):
        self.model_name = model_name
        self.name = f"ollama_{model_name.replace(':', '_')}"

    def generate(self, system: str, user: str) -> tuple[str, dict]:
        start_time = time.time()
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user}
                ]
            )
            text = response['message']['content']
            latency = time.time() - start_time
            
            usage = {
                "input_token": response.get('prompt_eval_count', 0),
                "output_token": response.get('eval_count', 0),
                "latency": round(latency, 2)
            }
            return text, usage
        except Exception as e:
            return f"[ERROR] Ollama failed: {str(e)}", {"input_token": 0, "output_token": 0, "latency": 0.0}