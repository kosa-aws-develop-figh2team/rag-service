# 🧠 rag-service
> **RAG 기반 응답 생성 API 서비스**  
> 질문 임베딩 생성, 관련 문서 검색, 최종 응답 생성을 담당하는 FastAPI 기반의 RAG API 서버입니다.

## ✅ 개요
이 서비스는 RAG(Retrieval-Augmented Generation) 기반 질문 응답 시스템의 핵심 기능을 담당하며, 다음과 같은 API를 제공합니다:
1. **POST /rag/embed-question**  
   사용자의 질문을 벡터로 임베딩합니다. 내부적으로 `embed-service`의 `/embed/text`를 호출합니다.
2. **POST /rag/retrieve**  
   생성된 임베딩을 기반으로 유사 문서를 검색합니다. `pgvector`를 사용한 벡터 유사도 검색을 수행합니다.
3. **POST /rag/generate**  
   질문과 유사 문서를 입력받아 LLM을 활용해 최종 응답을 생성합니다.

### ⏳ 과정 요약
```
[질문 수신]
   ↓
[임베딩 생성 (/rag/embed-question)]
   ↓
[유사 문서 검색 (/rag/retrieve)]
   ↓
[응답 생성 (/rag/generate)]
   ↓
[프론트엔드에 응답 반환]
```

## 🧩 API 명세

### 🔹 1. 질문 임베딩 생성 API
- **Endpoint**: `POST /rag/embed-question`
- **Request**:
```json
{
  "question_text": "사용자 질문 내용"
}
```
- **Response**:
```json
{
  "embedding_vector": [0.123, 0.456, ...]
}
```
- **Status Codes**:  
  - `200 OK`: 임베딩 생성 성공  
  - `500 Internal Server Error`: 임베딩 호출 실패  


### 🔹 2. 유사 문서 검색 API
- **Endpoint**: `POST /rag/retrieve`
- **Request**:
```json
{
  "embedding_vector": [0.123, 0.456, ...],
  "top_k": 5
}
```
- **Response**:
```json
{
  "document_list": [
    {
      "id": 1,
      "document_id": "DOC0001",
      "content": "문서 내용 예시",
      "distance": 0.01
    }
    // …
  ]
}
```
- **Status Codes**:  
  - `200 OK`: 검색 성공  
  - `500 Internal Server Error`: DB 조회 실패  


### 🔹 3. 응답 생성 API
- **Endpoint**: `POST /rag/generate`
- **Request**:
```json
{
  "question_text": "사용자 질문 내용",
  "document_list": [
    "문서 내용 1",
    "문서 내용 2"
  ]
}
```
- **Response**:
```json
{
  "response_text": "생성된 응답 텍스트"
}
```
- **Status Codes**:  
  - `200 OK`: 응답 생성 성공  
  - `500 Internal Server Error`: LLM 호출 실패  

## 🚀 로컬 실행 방법
```bash
# 1. 의존성 설치
pip install -r requirements.txt

# 2. 서버 실행
uvicorn main:app --reload --port 5201
```

## 🐳 Docker로 빌드 & 실행
```bash
# 이미지 빌드
docker build -t rag-service .

# 컨테이너 실행
docker run --env-file .env -p 5201:5201 rag-service
```

### 📦 .env 예시
```dotenv
EMBED_API_HOST=localhost
EMBED_API_PORT=5001
OPENAI_API_KEY=
POSTGRES_HOST=pgvector
POSTGRES_PORT=5432
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your-strong-password
POSTGRES_DB=yourdb
```

## ⚙️ CI/CD (ECR 배포 - CD 구현 전)
> GitHub Actions를 통해 `main` 브랜치에 push 시 Amazon ECR로 자동 배포됩니다.  
- **Repository**: rag-service  
- **Tag**: Git SHA 또는 `latest`  
> `.github/workflows/deploy.yml` 참고  

## 🛠️ TODO
- 헬스체크 엔드포인트 추가  
- 예외 처리 및 로깅 고도화  
- 배치 처리 최적화  
