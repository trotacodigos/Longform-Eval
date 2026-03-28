#vllm serve CohereLabs/command-a-translate-08-2025 --port 8000 --quantization bitsandbytes --load-format bitsandbytes

python -m vllm.entrypoints.openai.api_server \
  --model tencent/Hunyuan-MT-7B \
  --trust-remote-code \
  --served-model-name hunyuan-mt-7b \
  --port 8000