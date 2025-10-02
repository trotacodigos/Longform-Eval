from .basemodel import OllamaModel, Decoding

class LLaMa3_8BInstruct(OllamaModel):
    def __init__(self, decoding=None, host=None):
        super().__init__(name="llama3-8b-instruct", 
                         model_id="llama3:8b-instruct",
                         decoding=decoding or Decoding(),
                         host=host)