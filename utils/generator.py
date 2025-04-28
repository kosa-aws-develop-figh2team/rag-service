import boto3
import json
import os
import logging
from typing import List
from botocore.exceptions import ClientError

# 로깅 설정
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Bedrock 클라이언트 초기화
def get_bedrock_client(region: str = "ap-northeast-2"):
    return boto3.client("bedrock-runtime", region_name=region)

# 질문 + 문서 → LLM 응답 생성 함수
def generate_response_text(question_text: str, docs: List[str]) -> str:
    """
    주어진 질문과 관련 문서 리스트를 기반으로 Claude 3.5 Sonnet 모델 호출하여 응답을 생성합니다.
    """
    try:
        region = os.getenv("AWS_REGION", "ap-northeast-2")
        model_id = "anthropic.claude-3-5-sonnet-20240620-v1:0"

        client = get_bedrock_client(region=region)

        # 요청 payload 구성
        context = "\n".join(docs)
        prompt = f"""당신은 대한민국의 정부 정책, 복지, 제도 등에 정통한 AI 챗봇입니다.
당신은 사용자가 제공한 질문에 대해 아래의 "참고 문서(context)"를 바탕으로 정확하고 신뢰할 수 있는 한국어 답변을 생성해야 합니다.

💡 안내 사항:
- 반드시 제공된 context 내용을 기반으로만 답변해주세요.
- context에 명시되지 않은 정보는 생성하지 말고, "해당 내용은 자료에 존재하지 않습니다"라고 답해주세요.
- 중요한 정보는 요약하거나 항목으로 나눠 정리해도 좋습니다.
- 질문이 비교/분류/대상자 조건을 포함할 경우, 정확하게 구분해서 설명해주세요.

---
📄 참고 문서 (context):
{context}

---
User 질문:
{question_text}
"""

        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1024,
            "temperature": 0.7,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ]
        }

        logger.info("Claude 3.5 Sonnet 모델 호출 시작")

        response = client.invoke_model(
            modelId=model_id,
            body=json.dumps(request_body),
            contentType="application/json",
            accept="application/json"
        )

        response_body = json.loads(response['body'].read())
        output_text = response_body['content'][0]['text']

        if not output_text:
            logger.warning("Claude 응답 결과가 비어있습니다.")

        logger.info(f"LLM 응답 생성 완료: {output_text[:30]}...")
        return output_text

    except ClientError as e:
        logger.error(f"AWS ClientError: {e.response['Error']['Message']}")
        raise
    except Exception as e:
        logger.error(f"Claude 모델 호출 실패: {str(e)}")
        raise RuntimeError(f"Claude 모델 호출 실패: {str(e)}")