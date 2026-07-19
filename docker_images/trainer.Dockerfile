FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

# Dependencies first, so the (slow) sync layer is cached across source edits.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# This is the one image that actually needs training/ — it's the only
# place run_training.py (and the FeatureEncoder/model classes it uses) is
# imported directly, rather than a pickled artifact loaded elsewhere.
COPY training/ ./training
COPY raw_data/ ./raw_data

# Skip retraining if data/models/ (a volume shared with the api service)
# already has a completed run's output — makes repeated `docker compose up`
# runs fast instead of retraining every time.
CMD ["sh", "-c", "test -f /app/data/models/metadata.json && echo 'Model artifacts already exist — skipping training.' || uv run python -m training.run_training"]
