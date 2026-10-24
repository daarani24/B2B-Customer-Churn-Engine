FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip

COPY requirements.txt .

RUN pip install --no-cache-dir \
    --default-timeout=2000 \
    --retries 30 \
    --prefer-binary \
    -r requirements.txt

COPY . .

EXPOSE 8000 8501