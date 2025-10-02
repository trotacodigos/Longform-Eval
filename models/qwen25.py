from .basemodel import OllamaModel, Decoding

class Qwen25_32BInstruct(OllamaModel):
    def __init__(self, decoding=None, host=None):
        super().__init__(name="qwen2.5-32b-instruct",
                        model_id="qwen2.5:32b-instruct",
                        decoding=decoding or Decoding(),
                        host=host)