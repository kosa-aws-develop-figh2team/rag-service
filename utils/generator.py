from langchain_aws.chat_models.bedrock import ChatBedrock
from langchain.prompts import ChatPromptTemplate
from typing import List
import os
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# 질문 + 문서 → LLM 응답 생성 함수
def generate_response_text(question_text: str, docs: List[str]) -> str:
    """
    주어진 질문과 관련 문서 리스트를 기반으로 LLM 응답을 생성합니다.
    :param question_text: 유저의 질문
    :param docs: 관련 문서 리스트
    :return: LLM의 응답 텍스트
    """
    try:
        # 💬 context 구성
        context = "\n".join(docs)

        # 💬 ChatPromptTemplate 직접 생성
        prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 대한민국의 정부 정책, 복지, 제도 등에 정통한 AI 챗봇입니다.
당신은 사용자가 제공한 질문에 대해 아래의 "참고 문서(context)"를 바탕으로 정확하고 신뢰할 수 있는 한국어 답변을 생성해야 합니다.

💡 안내 사항:
- 반드시 제공된 context 내용을 기반으로만 답변해주세요.
- context에 명시되지 않은 정보는 생성하지 말고, "해당 내용은 자료에 존재하지 않습니다"라고 말해주세요.
- 중요한 정보는 요약하거나 항목으로 나눠 정리해도 좋습니다.
- 질문이 비교/분류/대상자 조건을 포함할 경우, 정확하게 구분해서 설명해주세요.

---

📄 참고 문서 (context):
{context}
"""),
            ("human", "{question}")
        ])

        # 💬 메시지 완성
        messages = prompt.format_messages(
            question=question_text,
            context=context
        )

        # Titan 모델 초기화
        llm = ChatBedrock(
            model_id="amazon.titan-text-express-v1",
            region_name=os.getenv("AWS_REGION", "ap-northeast-2"),
            temperature=0.7
        )

        # LLM 호출
        response = llm.invoke(messages)

        logger.info(f"LLM 응답 생성 성공: {response.content[:30]}...")
        return response.content
    except Exception as e:
        logger.error(f"LLM 응답 생성 실패: {str(e)}")
        raise RuntimeError(f"LLM 응답 생성 실패: {str(e)}")