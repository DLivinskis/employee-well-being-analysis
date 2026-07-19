FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

# Dependencies first, so the (slow) sync layer is cached across source edits.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# api/ needs exploratory_analysis/ (DescriptiveStats, imported by the
# analysis router) and raw_data/ (the CSV, read by seed.py on first
# startup). It does NOT need training/ — the trained pipelines pickle down
# to plain scikit-learn objects with no reference back to that package.
COPY api/ ./api
COPY exploratory_analysis/ ./exploratory_analysis
COPY raw_data/ ./raw_data

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
