from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import logging
from utils.embedding import get_embedding_vector
from utils.retriever import search_similar_documents
from utils.generator import generate_response_text

# 📋 로깅 설정
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI()


# ✅ 요청/응답 모델 정의

class EmbedRequest(BaseModel):
    question_text: str

class EmbedResponse(BaseModel):
    embedding_vector: List[float]


class RetrieveRequest(BaseModel):
    embedding_vector: List[float]

class RetrieveResponse(BaseModel):
    document_list: List[str]


class GenerateRequest(BaseModel):
    question_text: str
    document_list: List[str]

class GenerateResponse(BaseModel):
    response_text: str


# 🔹 1. 질문 임베딩 생성 API
@app.post("/rag/embed-question", response_model=EmbedResponse)
def embed_question(request: EmbedRequest):
    try:
        logger.info(f"[임베딩 요청] 질문 텍스트 수신")
        vector = get_embedding_vector(request.question_text)
        return {"embedding_vector": vector}
    except Exception as e:
        logger.error(f"[임베딩 실패] {str(e)}")
        raise HTTPException(status_code=500, detail="임베딩 생성 실패")


# 🔹 2. 유사 문서 검색 API
@app.post("/rag/retrieve", response_model=RetrieveResponse)
def retrieve_documents(request: RetrieveRequest):
    try:
        logger.info(f"[문서 검색] 임베딩 벡터 수신")
        docs = search_similar_documents(request.embedding_vector)
        return {"document_list": docs}
    except Exception as e:
        logger.error(f"[문서 검색 실패] {str(e)}")
        raise HTTPException(status_code=500, detail="문서 검색 실패")


# 🔹 3. 응답 생성 API
@app.post("/rag/generate", response_model=GenerateResponse)
def generate_answer(request: GenerateRequest):
    try:
        logger.info(f"[응답 생성] 질문 및 문서 수신")
        result = generate_response_text(request.question_text, request.document_list)
        return {"response_text": result}
    except Exception as e:
        logger.error(f"[응답 생성 실패] {str(e)}")
        raise HTTPException(status_code=500, detail="응답 생성 실패")