import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

class Llama31Model:
    def __init__(self, model_id="NousResearch/Meta-Llama-3-8B-Instruct"):
        self.name = "llama3_1"
        print(f"Loading Model: {model_id}...")
        
        # 4-bit Quantization for efficient resource usage
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto"
        )

    def generate(self, prompt: str) -> str:
        try:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
            
            # Text-based input formatting for stability
            prompt_str = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            inputs = self.tokenizer(prompt_str, return_tensors="pt").to(self.model.device)

            terminators = [
                self.tokenizer.eos_token_id,
                self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=2048,
                eos_token_id=terminators,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            response = outputs[0][inputs.input_ids.shape[-1]:]
            return self.tokenizer.decode(response, skip_special_tokens=True)
            
        except Exception as e:
            print(f"[Llama Error] {e}")
            return ""