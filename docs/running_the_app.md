# Running the App End-to-End

## Docker Compose — one command

```bash
docker compose up -d --build
```

This is the whole setup. Four services:

- **`db`** — Postgres.
- **`trainer`** — one-shot: trains both models and writes `data/models/{logistic,tree}.joblib` + `metadata.json` to a volume shared with `api`. Skips retraining if those artifacts already exist (fast on repeat runs). Doesn't touch Postgres — it only reads the CSV and writes local files.
- **`api`** — waits for `db` to be healthy *and* `trainer` to finish successfully (`condition: service_completed_successfully`), then creates tables, seeds `employees`/`models`, and starts serving.
- **`frontend`** — waits for `api`, then starts.

Once it's up: API docs at `http://localhost:8000/docs`, the app at `http://localhost:8501`.

No local Python environment, no manual training step, no manual seeding — a fresh clone with nothing but Docker installed gets a fully working app from this one command. See `docs/architecture_and_design.md`'s Packaging section for how the four services fit together.

> Tested on macOS 14.7.2 (Darwin 23.6.0, arm64), Docker 27.4.0, Docker Compose v2.31.0-desktop.2.

## Running locally without Docker

Useful for iterating on one piece (e.g. the frontend) without rebuilding images. Two ways to install dependencies — with `uv` (how the project is developed) or with plain `pip` + `requirements.txt`.

### 1. Install dependencies

**Option A: uv**

```bash
uv sync
```

Installs everything from `uv.lock` into a project-local `.venv`. Every command below is then run as `uv run <command>`.

**Option B: pip + requirements.txt**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` is exported from `uv.lock` (via `uv export --no-hashes --format requirements-txt`) so both options resolve to identical package versions. It's committed for convenience but not hand-maintained — regenerate it with that same command after any dependency change in `pyproject.toml`. With the venv activated, drop the `uv run` prefix from every command below.

### 2. Start Postgres

```bash
docker compose up -d db
```

Exposed on `localhost:5449` (see `docker-compose.yml`). `api/endpoint_helpers/config.py` and `training/helpers/config.py` both default their `DATABASE_URL`/paths to match this, so no environment variables need setting.

### 3. Fetch the raw dataset

Skip if `raw_data/Impact_of_Remote_Work_on_Mental_Health.csv` already exists.

```bash
uv run python raw_data/fetch_kaggle_dataset.py
```

### 4. Train the models

```bash
uv run python -m training.run_training
```

Writes `data/models/{logistic,tree}.joblib` and `data/models/metadata.json` — required before the API can start.

### 5. Run the API

```bash
uv run uvicorn api.main:app --reload
```

Creates Postgres tables and seeds `employees`/`models` on first run. Visit `http://localhost:8000/docs`.

### 6. Run the frontend

```bash
uv run streamlit run frontend/Home.py
```

Visit `http://localhost:8501`. `frontend/api_client.py` defaults `API_BASE_URL` to `http://localhost:8000`, matching step 5.
