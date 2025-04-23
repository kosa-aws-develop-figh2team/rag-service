import requests
import os
import logging
from typing import List
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# embed-service 주소 (기본값 포함)
EMBED_API_HOST = os.getenv("EMBED_API_HOST", "localhost")
EMBED_API_PORT = os.getenv("EMBED_API_PORT", "5001")

def get_embedding_vector(text: str) -> List[float]:
    """
    embed-service의 /embed/text API를 호출하여 임베딩 벡터를 반환합니다.
    """
    try:
        logger.info(f"[프록시] embed-service 호출: {text[:30]}...")
        url = f"http://{EMBED_API_HOST}:{EMBED_API_PORT}/embed/text"
        response = requests.post(url, json={"raw_text": text})
        response.raise_for_status()

        data = response.json()
        embedding = data.get("embedding_vector")

        if embedding is None:
            raise ValueError("응답에 'embedding_vector' 필드가 없습니다.")

        return embedding

    except Exception as e:
        logger.error(f"[프록시] embed-service 호출 실패: {str(e)}")
        raise