from typing import List, Dict
import psycopg2
import logging
from contextlib import contextmanager
from typing import Generator, Dict
import dotenv
import os

dotenv.load_dotenv()

pg_config = {
    "host": os.getenv("POSTGRES_HOST"),
    "port": int(os.getenv("POSTGRES_PORT")),
    "user": os.getenv("POSTGRES_USER"),
    "password": os.getenv("POSTGRES_PASSWORD"),
    "dbname": os.getenv("POSTGRES_DB")
}

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@contextmanager
def get_pg_connection(pg_config: Dict) -> Generator[psycopg2.extensions.connection, None, None]:
    """
    PostgreSQL 연결을 생성하는 context manager.
    사용이 끝나면 자동으로 연결 종료됨.
    """
    conn = None
    try:
        logger.info("PostgreSQL 연결 시도 중...")
        conn = psycopg2.connect(**pg_config)
        logger.info("PostgreSQL 연결 성공")
        yield conn
    except Exception as e:
        logger.error(f"PostgreSQL 연결 실패: {e}")
        raise
    finally:
        if conn:
            conn.close()
            logger.info("PostgreSQL 연결 종료")

def search_similar_documents(
    embedding_vector: List[float],
    top_k: int = 5,
) -> List[Dict]:
    """
    pgvector를 이용해 가장 유사한 문서 청크를 top-k개 검색하여 반환
    """
    results = []

    with get_pg_connection(pg_config) as conn:
        with conn.cursor() as cur:
            query = """
                SELECT id, document_id, content, embedding <=> %s::vector AS distance
                FROM embeddings
                ORDER BY distance
                LIMIT %s;
            """
            cur.execute(query, (embedding_vector, top_k))
            rows = cur.fetchall()

            for row in rows:
                result = {
                    "id": row[0],
                    "document_id": row[1],
                    "content": row[2],
                    "distance": row[3]
                }
                results.append(result)

    return results