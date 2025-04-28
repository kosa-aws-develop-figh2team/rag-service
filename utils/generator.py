from langchain.chat_models import ChatBedrock
from langchain.schema import SystemMessage, HumanMessage
from langchain.prompts import ChatPromptTemplate
from typing import List
import os
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# 프롬프트 템플릿 불러오기
def load_prompt_template(file_path: str) -> ChatPromptTemplate:
    try:
        with open(file_path, encoding="utf-8") as f:
            template_str = f.read()
        return ChatPromptTemplate.from_template(template_str)
    except FileNotFoundError:
        logger.error(f"파일을 찾을 수 없습니다: {file_path}")
        raise

# 질문 + 문서 → LLM 응답 생성 함수
def generate_response_text(question_text: str, docs: List[str]) -> str:
    """
    주어진 질문과 관련 문서 리스트를 기반으로 LLM 응답을 생성합니다.
    :param question_text: 유저의 질문
    :param docs: 관련 문서 리스트
    :return: LLM의 응답 텍스트
    """
    try:
        prompt_template_path = "./prompts/policy_chat_prompt.txt"

        context = "\n".join(docs)
        prompt = load_prompt_template(prompt_template_path)
        messages = prompt.format_messages(question=question_text, context=context)

        # Titan 모델 초기화
        llm = ChatBedrock(
            model_id="amazon.titan-text-lite-v1",    # Titan 모델 ID
            region_name=os.getenv("AWS_REGION", "ap-northeast-2"),  # AWS_REGION 환경변수 또는 기본 서울 리전
            temperature=0.7
        )

        response = llm(messages)

        logger.info(f"LLM 응답 생성 성공: {response.content[:30]}...")
        return response.content
    except Exception as e:
        logger.error(f"LLM 응답 생성 실패: {str(e)}")
        raise RuntimeError(f"LLM 응답 생성 실패: {str(e)}")