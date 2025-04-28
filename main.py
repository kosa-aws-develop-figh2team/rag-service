from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import logging
from utils.retrieve import get_retrieve_result
from utils.generator import generate_response_text

# 로깅 설정
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI()

# 요청/응답 모델 정의
class AnswerRequest(BaseModel):
    question_text: str

class AnswerResponse(BaseModel):
    response_text: str

# 유사 문서 검색 API
@app.post("/rag/answer", response_model=AnswerResponse)
def retrieve_documents(request: AnswerRequest):
    try:
        logger.info(f"[문서 검색] 텍스트 임베딩 api 호출")
        content_list = get_retrieve_result(request.question_text)
    except Exception as e:
        logger.error(f"[문서 검색 실패] {str(e)}")
        raise HTTPException(status_code=500, detail="문서 검색 실패")
    
    try:
        logger.info(f"[응답 생성] 질문 및 문서 수신")
        output_text = generate_response_text(request.question_text, content_list)
    except Exception as e:
        logger.error(f"[응답 생성 실패] {str(e)}")
        raise HTTPException(status_code=500, detail="응답 생성 실패")
    
    return {"response_text": output_text}

