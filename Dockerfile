FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN python -m pip install --upgrade pip setuptools wheel -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn \
    && python -m pip install -r /app/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn

COPY api /app/api
COPY config /app/config
COPY data /app/data
COPY hello_rag_agent /app/hello_rag_agent
COPY prompts /app/prompts
COPY app.py /app/app.py
COPY start_services.py /app/start_services.py
COPY .env.example /app/.env.example

EXPOSE 8000 8501

CMD ["python", "start_services.py", "--host", "0.0.0.0"]
