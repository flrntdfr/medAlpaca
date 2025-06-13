FROM python:3.10-slim-bookworm
COPY --from=ghcr.io/astral-sh/uv:0.7.13 /uv /uvx /bin/

WORKDIR /app
COPY . .

RUN uv venv --prompt medAlapaca && \
    . .venv/bin/activate && \
    uv pip install -r /app/requirements.txt