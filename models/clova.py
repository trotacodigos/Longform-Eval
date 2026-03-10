from .base import Decoding
from .base_hf import HFChatModel
from .tools import extract_token_usage
import requests

class ClovaModel(HFChatModel):
    def __init__(self, name="clova", model_id="naver/HyperCLOVAX-SEED-Think-32B", endpoint=None, decoding: Decoding | dict | None = None, thinking: bool = True):
        super().__init__(name, model_id, endpoint, decoding)
        self.thinking = thinking

    def _call(self, system: str, user: str):
        # 1. 페이로드 구조 직접 제어
        payload = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            "max_tokens": self.decoding.max_tokens,
            "top_p": self.decoding.top_p,
        }

        # 2. 파라미터 충돌 방지
        if self.thinking:
            payload["thinking"] = True 
        else:
            payload["temperature"] = self.decoding.temperature

        # 3. 실시간 체크를 위한 의도적 에러 발생
        if not self.endpoint:
            raise ValueError("[Ready] Clova 모델 구조 세팅 완료. 통신을 위한 endpoint 주소가 필요합니다.")

        # 4. 서버 통신 및 응답 처리
        response = requests.post(self.endpoint, json=payload, timeout=600)
        response.raise_for_status()
        
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        text = (content or "").strip()

        usage = data.get("usage", {})
        in_token, out_token = extract_token_usage(usage)

        return text, in_token, out_token