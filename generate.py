import argparse
import time
import ollama

# ----------------------------------------------------------------------
# [1. Centralized Config: 파라미터 중앙 통제실]
# ----------------------------------------------------------------------
MODEL_CONFIGS = {
    "kanana": {
        # 주의: 뼈대 테스트를 위해 임시로 이미 깔려있는 qwen:0.5b로 매핑. 테스트 통과 후 실제 태그로 변경할 것.
        "ollama_tag": "qwen:0.5b", 
        "temperature": 0.1,
        "max_tokens": 8192, 
        "top_p": 0.9
    },
    "exaone": {
        "ollama_tag": "exaone4.0-32b:latest",
        "temperature": 0.1,
        "max_tokens": 8192,
        "top_p": 0.9
    },
    "hcx_seed": {
        "ollama_tag": "hcx-seed-think-32b:latest",
        "temperature": 0.1,
        "max_tokens": 16384,
        "top_p": 0.9
    }
    # 100B+급 Solar, K-Exaone은 VRAM 24GB 로컬 구동 물리적 불가. API 전환 보고 요망.
}

# ----------------------------------------------------------------------
# [2. Unified Inference Engine: 범용 Ollama 래퍼]
# ----------------------------------------------------------------------
class UnifiedOllamaEngine:
    def __init__(self, model_id: str):
        if model_id not in MODEL_CONFIGS:
            raise ValueError(f"[ERROR] Invalid Model ID: {model_id}")
        self.config = MODEL_CONFIGS[model_id]
        self.tag = self.config["ollama_tag"]
        print(f"[SYSTEM] Model Engine Initialized: {model_id.upper()} (Tag: {self.tag})")

    def generate(self, system_prompt: str, user_prompt: str):
        start_time = time.time()
        try:
            response = ollama.chat(
                model=self.tag,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                options={
                    "temperature": self.config["temperature"],
                    "num_predict": self.config["max_tokens"],
                    "top_p": self.config["top_p"]
                }
            )
            latency = time.time() - start_time
            return response['message']['content'], latency
        except Exception as e:
            return f"[ERROR] {str(e)}", 0.0

# ----------------------------------------------------------------------
# [3. Main Entry Point]
# ----------------------------------------------------------------------
import os
import json

def main():
    parser = argparse.ArgumentParser(description="Unified LLM Inference Pipeline")
    parser.add_argument("--model", required=True, choices=list(MODEL_CONFIGS.keys()), help="Target Model ID")
    parser.add_argument("--test", action="store_true", help="Run Dummy Data Test")
    parser.add_argument("--input_file", type=str, help="Path to input JSONL file")
    parser.add_argument("--output_dir", type=str, help="Directory to save output JSONL")
    args = parser.parse_args()

    engine = UnifiedOllamaEngine(args.model)

    # [Task A: Dummy Validation]
    if args.test:
        print("\n[SYSTEM] Running Dummy Data Test Mode...")
        dummy_data = [
            {"id": 1, "src": "Hello, how are you?", "tgt": "안녕하세요, 잘 지내시나요?"},
            {"id": 2, "src": "Apple is red.", "tgt": "사과는 빨갛다."}
        ]
        system_prompt = "You are an expert translation evaluator. Reply with a short evaluation."
        
        for row in dummy_data:
            user_prompt = f"Source: {row['src']}\nTarget: {row['tgt']}\nEvaluation:"
            print(f"\n[Input ID: {row['id']}] Inferencing...")
            output, latency = engine.generate(system_prompt, user_prompt)
            print(f"[Output] {output.strip()}")
            print(f"[Latency] {latency:.2f}s")
        
        print("\n[SYSTEM] Dummy Test Completed Successfully. Pipeline is solid.")
        return

    # [Task B: Actual File Processing]
    if args.input_file and args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        output_filename = os.path.basename(args.input_file).replace(".jsonl", f"_{args.model}_out.jsonl")
        output_path = os.path.join(args.output_dir, output_filename)

        # STRICT PROMPT: Prevent chatter/pleasantries
        system_prompt = "You are a professional translator. Translate the following English text to Korean. Output ONLY the translation. Do not add any explanations, greetings, or notes."

        print(f"[SYSTEM] Processing target file: {args.input_file}")
        
        with open(args.input_file, 'r', encoding='utf-8') as f_in, \
             open(output_path, 'w', encoding='utf-8') as f_out:
            
            for line in f_in:
                if not line.strip(): continue
                data = json.loads(line)
                
                # Extract English source
                user_prompt = data.get('src_seg', '')
                print(f"[Inferencing] Sample ID: {data.get('sample_id', 'N/A')}")
                
                # Inference
                output, latency = engine.generate(system_prompt, user_prompt)
                
                # Append results
                data['model_output'] = output.strip()
                data['latency'] = round(latency, 2)
                
                # Write to output file
                f_out.write(json.dumps(data, ensure_ascii=False) + '\n')

        print(f"[SYSTEM] Inference complete. Results saved to: {output_path}")
        return

    print("[SYSTEM] Error: Missing --input_file or --output_dir arguments.")

if __name__ == "__main__":
    main()