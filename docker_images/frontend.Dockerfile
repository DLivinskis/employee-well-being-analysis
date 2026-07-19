FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

# Dependencies first, so the (slow) sync layer is cached across source edits.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY frontend/ ./frontend

# Only the markdown docs, not the training/ Python package itself — the
# Metrics Description page (frontend/pages/5_Metrics_Description.py) reads
# these directly rather than duplicating their content.
COPY training/*.md ./training/

EXPOSE 8501

CMD ["uv", "run", "streamlit", "run", "frontend/Home.py", "--server.address=0.0.0.0", "--server.port=8501"]
