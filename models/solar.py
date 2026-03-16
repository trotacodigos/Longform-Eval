from .sampling import HuggingFaceParams
from .base_hf import HFChatModel

class SolarOpenModel(HFChatModel):
    def __init__(self, name, model_id, endpoint, sampling_params: HuggingFaceParams | dict | None = None):
        super().__init__(name, model_id, endpoint, sampling_params)
        