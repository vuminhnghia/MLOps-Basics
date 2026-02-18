FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

COPY ./ /app
WORKDIR /app

RUN uv sync

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
EXPOSE 8000
CMD ["uv", "run", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
