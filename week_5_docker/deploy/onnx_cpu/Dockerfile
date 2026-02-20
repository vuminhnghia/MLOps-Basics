FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

COPY ./ /app
WORKDIR /app

RUN uv sync

RUN uv run python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('google/bert_uncased_L-2_H-128_A-2')"

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV PREDICTOR_TYPE=onnx

EXPOSE 8000
CMD ["uv", "run", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
