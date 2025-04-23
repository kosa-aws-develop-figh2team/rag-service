FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /app

# 애플리케이션 파일 복사
COPY ./main.py /app
COPY ./utils /app/utils
COPY ./prompts /app/prompts
COPY ./requirements.txt /app

# 설치
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# 포트 설정
EXPOSE 5201

# 애플리케이션 실행
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5201"]