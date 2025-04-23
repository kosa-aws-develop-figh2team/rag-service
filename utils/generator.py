from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langchain.prompts import ChatPromptTemplate
from typing import List
import os
import dotenv

dotenv.load_dotenv()

# 환경 변수 로드 (예: OpenAI 또는 Bedrock)
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
# os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID")
# os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY")

# 🔹 프롬프트 템플릿 불러오기
def load_prompt_template(file_path: str) -> ChatPromptTemplate:
    with open(file_path, encoding="utf-8") as f:
        template_str = f.read()
    return ChatPromptTemplate.from_template(template_str)


# 🔹 질문 + 문서 → LLM 응답 생성 함수
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

        llm = ChatOpenAI(model="gpt-4.1-nano", temperature=0.7)
        response = llm(messages)

        return response.content
    except Exception as e:
        raise RuntimeError(f"LLM 응답 생성 실패: {str(e)}")