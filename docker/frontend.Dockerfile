FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

# Dependencies first, so the (slow) sync layer is cached across source edits.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY frontend/ ./frontend

EXPOSE 8501

CMD ["uv", "run", "streamlit", "run", "frontend/Home.py", "--server.address=0.0.0.0", "--server.port=8501"]
