import requests
import os
import logging
from typing import List
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# embed-service 주소 (기본값 포함)
EMBED_API_HOST = os.getenv("EMBED_API_HOST", "embed-service.embed.svc.cluster.local")
EMBED_API_PORT = os.getenv("EMBED_API_PORT", "5001")

def get_retrieve_result(text: str) -> List[dict]:
    """
    embed-service의 /embed/retrieve API를 호출하여 검색된 문서를 반환합니다.
    """
    try:
        logger.info(f"[프록시] embed-service 호출: {text[:30]}...")
        url = f"http://{EMBED_API_HOST}:{EMBED_API_PORT}/embed/retrieve"
        response = requests.post(url, json={"text": text})
        response.raise_for_status()

        data = response.json()
        result_list = data.get("results")

        if result_list is None:
            raise ValueError("응답에 'results' 필드가 없습니다.")

        if len(result_list) == 0:
            return [{"id":"", "service_id":"", "content":""}]
        else:
            return result_list

    except Exception as e:
        logger.error(f"[프록시] embed-service 호출 실패: {str(e)}")
        raise