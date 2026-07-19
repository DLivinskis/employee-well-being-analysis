# Execution Plan

Breaks [architecture_and_design.md](architecture_and_design.md) into a concrete repo layout and a sequence of small, independently completable tasks.

## 0. Coding conventions

Every Python file in this plan (`training/`, `api/`, `frontend/`, `exploratory_analysis/descriptive_stats.py`) follows the global conventions in `~/.claude/CLAUDE.md` (`docs/python-code-structure.md`):

- Module-level docstring on every file describing its purpose.
- Full type hints on every function/method signature.
- Google-style docstrings (summary, Args, Returns, Raises) on every function and class.
- Multi-step logic (e.g. `run_training.py`'s orchestration, `evaluation.py`'s metric computation, the FastAPI inference loader) is implemented as a class, not one large function: each distinct step (split data, fit encoder, compute one metric, build one table) becomes its own small, private `_`-prefixed method that can be called and unit-tested in isolation. The class's public entry-point method(s) then just orchestrate those small methods in sequence — the entry point is a thin composition of already-tested building blocks, not where the actual logic lives. The class docstring lists these entry points explicitly (which methods the caller is meant to use directly), while the `_` helpers stay internal and undocumented there.
- Inline comments only for non-obvious WHY (hidden constraints, workarounds), plus brief explanations wherever a non-trivial Python concept (generators, decorators, comprehensions, context managers, etc.) is used.
- Constants and variables that belong to a class's logic live on the class (class attributes, or instance attributes set in `__init__`), not as bare module-level globals — keeps everything the class depends on in one place next to the methods that use it, and makes it overridable per-instance/subclass instead of shared mutable module state.

## 1. Target repo structure

```
employee-well-being-analysis/
├── README.md
├── docker-compose.yml
├── pyproject.toml
├── uv.lock
├── docs/
│   ├── task_description.md
│   ├── ai_suggested_approach.md
│   ├── architecture_and_design.md
│   └── execution_plan.md
├── exploratory_analysis/
│   ├── ad_hoc_exploratory_analysis.ipynb   # manual/personal exploration only, not a source of truth
│   └── descriptive_stats.py                # reusable functions computing the descriptive-analysis numbers the API/frontend actually serve
├── raw_data/
│   ├── fetch_kaggle_dataset.py
│   └── Impact_of_Remote_Work_on_Mental_Health.csv
├── training/
│   ├── __init__.py
│   ├── helpers/                  # supporting building blocks, not entry points themselves
│   │   ├── __init__.py
│   │   ├── config.py             # paths, DB URL, random seed, split ratios
│   │   ├── data_split.py         # stratified train/val/test split
│   │   ├── preprocessing.py      # encoders shared by both models
│   │   └── evaluation.py         # confusion matrix, ROC/AUC, macro-F1
│   ├── train_logistic.py        # fits the logistic model, reports coefficients
│   ├── train_tree.py            # fits the tree model, reports permutation importance
│   └── run_training.py          # entry point: split -> train both -> evaluate -> write artifacts + models-table rows
├── api/
│   ├── __init__.py
│   ├── main.py                  # entry point: FastAPI app + AppStartup + router registration
│   ├── endpoint_helpers/         # supporting building blocks, not entry points themselves
│   │   ├── __init__.py
│   │   ├── config.py             # env-based settings (DB URL, model artifact dir)
│   │   ├── schemas.py            # Pydantic request/response models
│   │   ├── dependencies.py       # FastAPI Depends: get_session, get_inference_service
│   │   └── inference.py          # InferenceService: loads model artifacts, runs prediction
│   ├── database/
│   │   ├── __init__.py           # re-exports Database, DatabaseSeeder
│   │   ├── connection.py         # Database: SQLAlchemy engine/session
│   │   ├── seed.py               # DatabaseSeeder: loads CSV + metadata.json
│   │   └── models/
│   │       ├── __init__.py       # re-exports Base, Employee, TrainedModel
│   │       ├── base.py           # shared declarative Base
│   │       ├── employee.py       # Employee ORM model
│   │       └── trained_model.py  # TrainedModel ORM model
│   └── routers/                  # entry points: one router per resource
│       ├── __init__.py
│       ├── employees.py          # GET /employees/{employee_id}, POST /employees
│       ├── predict.py            # POST /predict
│       └── analysis.py           # GET /analysis/*
├── frontend/
│   ├── Home.py                   # entry point: landing page
│   ├── api_client.py             # entry point: ApiClient, wraps requests to the API
│   ├── helpers/
│   │   └── model_analysis.py     # ModelAnalysisView: shared confusion-matrix/ROC rendering
│   └── pages/
│       ├── 1_Descriptive_Analysis.py
│       ├── 2_Logistic_Model_Features.py
│       ├── 3_Tree_Model_Features.py
│       └── 4_Predict.py
└── docker/
    ├── api.Dockerfile
    └── frontend.Dockerfile
```

`data/` (model artifacts) and Postgres's volume are runtime state, not repo content — created at build/run time, not committed.

## 2. Tasks

### Phase 0 — Cleanup / scaffolding
- [ ] Remove the leftover `main.py`/root `uv init` scaffold once `training/`, `api/`, and `frontend/` exist as the real entry points.
- [ ] Add `training`, `api` (with `fastapi`, `uvicorn`, `sqlalchemy`, `psycopg`), and `frontend` (`streamlit`, `requests`) dependency groups to `pyproject.toml`.
- [ ] Create the empty package directories with `__init__.py` files per the structure above.

### Phase 1 — Finalize EDA findings
- [ ] Turn the ad-hoc notebook's findings into a short written summary (which features showed any signal vs. none) — this determines what `preprocessing.py` actually encodes.
- [ ] Confirm `Employee_ID` uniqueness and no missing values; note either as a EDA-notebook cell, not a new file.
- [ ] `exploratory_analysis/descriptive_stats.py`: extract the actual descriptive-statistics logic (target/feature distributions, feature-vs-target breakdowns) out of the notebook into plain functions taking a DataFrame and returning tables/dicts ready to plot or serialize. The notebook stays for ad-hoc, personal-use exploration; this module is what `api/routers/analysis.py`'s `GET /analysis/descriptive` actually imports and calls, so the served numbers come from tested, reusable code rather than notebook cells.

### Phase 2 — Training pipeline (`training/`)
- [ ] `config.py`: random seed, split ratios (70/15/15), DB URL, artifact output directory — all read from environment with sensible defaults.
- [ ] `data_split.py`: one function that takes the raw DataFrame and returns stratified `(train, val, test)` DataFrames per [architecture_and_design.md §2](architecture_and_design.md#2-train--validation--test-split).
- [ ] `preprocessing.py`: fit-on-train encoder(s) (ordinal for ordered categoricals, one-hot for nominal) as a single object reusable by both models and by the API's inference path.
- [ ] `train_logistic.py`: fit multinomial logistic regression on train, tune on validation, return the fitted pipeline + standardized coefficients.
- [ ] `train_tree.py`: fit gradient boosting/random forest on train, tune on validation, return the fitted pipeline + permutation/SHAP importances.
- [ ] `evaluation.py`: shared functions for confusion matrix, per-class ROC/AUC, and macro-F1, given a fitted model and a DataFrame split — used identically for both models so their metrics are directly comparable.
- [ ] `run_training.py`: orchestrates the above, serializes both fitted pipelines to `data/models/`, and writes/updates the corresponding rows in the `models` table (metrics + importances + artifact path).

### Phase 3 — Database layer (`api/database/`)
- [ ] `models/employee.py`, `models/trained_model.py`: one SQLAlchemy model per file — `Employee` (mirrors CSV columns, `employee_id` primary key so `POST /employees` can't create duplicates) and `TrainedModel` (name as primary key, trained_at, artifact_path, feature_importance JSON, confusion_matrix JSON, roc_auc JSON).
- [ ] `models/base.py`: shared declarative `Base` both models inherit from.
- [ ] `connection.py`: `Database` class — engine/session factory, reading the DB URL from `api/config.py`.
- [ ] `seed.py`: `DatabaseSeeder` — idempotent; creates tables if missing, loads the CSV into `employees` only if the table is empty; called once at API startup. Only seeds initial contents — later inserts via `POST /employees` are untouched by re-running this.
- [x] `docker-compose.yml`: `db` service only for now (postgres + named volume + healthcheck + port 5432 exposed to the host), added early so seeded data can be checked manually (`psql`, a DB client) before `api`/`frontend` exist. `api`/`frontend` services are added in Phase 6.

### Phase 4 — FastAPI backend (`api/`)
- [x] `schemas.py`: Pydantic models for employee response/create request, predict request/response, and analysis responses (coefficients/importances + confusion matrix + ROC/AUC).
- [x] `dependencies.py`: `get_database`/`get_session`/`get_inference_service` — reads the shared `Database`/`InferenceService` off `app.state` for use with `Depends`.
- [x] `inference.py`: `InferenceService` — loads the two serialized pipelines once at startup; `predict(model_name, features) -> {label, class_probabilities}` used by `/predict`.
- [x] `routers/employees.py`: `GET /employees/{employee_id}` (404 if not found) and `POST /employees` (insert a new row, 409 on duplicate `Employee_ID`) — verified end-to-end: inserted a new employee with no target, predicted their satisfaction via `/predict`.
- [x] `routers/predict.py`: `POST /predict`.
- [x] `routers/analysis.py`: `GET /analysis/descriptive` (calls `exploratory_analysis/descriptive_stats.py` against the `employees` table), `GET /analysis/logistic-importance`, `GET /analysis/tree-importance`.
- [x] `main.py`: `AppStartup` class builds the DB (create+seed) and inference service at lifespan startup, attaches both to `app.state`; registers routers.

### Phase 5 — Streamlit frontend (`frontend/`)
- [x] `api_client.py`: `ApiClient` class wrapping each API call (`get_employee`, `create_employee`, `predict`, `get_descriptive`, `get_logistic_importance`, `get_tree_importance`), API base URL from the `API_BASE_URL` environment variable.
- [x] `helpers/model_analysis.py`: `ModelAnalysisView` — shared confusion-matrix/ROC-AUC rendering, since it's identical between the logistic and tree pages (added beyond the original plan to avoid duplicating that rendering code).
- [x] `pages/1_Descriptive_Analysis.py`: target/feature distributions + feature-vs-target breakdowns from `/analysis/descriptive`.
- [x] `pages/2_Logistic_Model_Features.py`: per-class coefficients (`LogisticFeaturesView`, subclasses `ModelAnalysisView`) + confusion matrix + ROC/AUC from `/analysis/logistic-importance`.
- [x] `pages/3_Tree_Model_Features.py`: permutation importances (`TreeFeaturesView`, subclasses `ModelAnalysisView`) + confusion matrix + ROC/AUC from `/analysis/tree-importance`.
- [x] `pages/4_Predict.py`: client-ID lookup (pre-fills form via `/employees/{id}`, offers to save via `POST /employees` if the ID doesn't exist yet) or manual entry, both ending in a call to `/predict`.
- [x] `Home.py`: landing page linking to the four pages.

Verified with Streamlit's `AppTest` harness (executes each page script for real, against the live API): all four pages plus `Home.py` render with zero exceptions; drove the Predict page's actual submit flow (ID lookup → pre-filled form → submit) end-to-end and got back a real prediction.

### Phase 6 — Dockerization
- [x] `docker/api.Dockerfile`: copies `api/`, `exploratory_analysis/` (needed by the analysis router), and `raw_data/` (needed by `seed.py`); runs `uvicorn api.main:app`. Verified `training/` is **not** needed in this image — a pickled pipeline unpickles to plain scikit-learn objects with no reference back to that package (checked by unpickling `logistic.joblib` in a directory with no `training/` present).
- [x] `docker/frontend.Dockerfile`: copies only `frontend/`, runs `streamlit run frontend/Home.py`.
- [x] Both Dockerfiles use the `ghcr.io/astral-sh/uv` base image and a single `uv sync --frozen --no-dev` layer (ordered before the source `COPY` so dependency installs stay cached across code edits); no dependency-group split between images — the full lockfile is installed in both, a deliberate simplicity-over-image-size tradeoff at this project's scale.
- [x] `docker-compose.yml`: added `api` (`depends_on: db, condition: service_healthy`; `DATABASE_URL` pointed at the internal `db:5432` address, distinct from the host-facing `localhost:5449` mapping; `./data:/app/data` volume for model artifacts + metadata) and `frontend` (`depends_on: api`; `API_BASE_URL=http://api:8000`) alongside the existing `db` service.
- [x] `.dockerignore`: excludes `.venv/`, `.git/`, `__pycache__/`, and `data/` (a volume mount, not something to bake into the image) from the build context.
- [x] `data/models/` artifacts already existed on the host from Phase 2/3 testing, so no separate one-off training step was needed before first `docker compose up` — noted here as the prerequisite for anyone running this fresh.

### Phase 7 — End-to-end check
- [x] `docker compose up`, confirm all three services healthy.
- [x] Found and fixed a real bug here: all four pages imported `from frontend.api_client import ...` / `from frontend.helpers.model_analysis import ...`, which works by accident under `streamlit.testing.v1.AppTest` (it inherits the calling process's cwd-based sys.path) but fails under a real `streamlit run` — Streamlit puts the **main script's own directory** (`frontend/`) on `sys.path[0]` for every page, not the repo root, so `frontend` itself isn't importable from within it. Confirmed the exact mechanism by simulating that sys.path in isolation, then fixed by changing every page's imports to bare `from api_client import ...` / `from helpers.model_analysis import ...`. This is why the Phase 5 `AppTest`-based verification passed but the real Docker deployment threw `ModuleNotFoundError: No module named 'frontend'` the first time a page was opened in a browser — `AppTest` is not a fully faithful stand-in for `streamlit run`'s import resolution.
- [x] Verified the fix for real: installed Playwright, launched a real Chromium browser against the running `frontend` container, navigated to all four pages (no error banner on any), and drove the actual Predict flow — typed an employee ID, confirmed the pre-fill message, clicked the real "Predict" button, and got a rendered prediction (`Neutral`) with the probability bar chart.
- [x] Confirmed cross-container networking for real: exec'd into the `frontend` container and fetched `http://api:8000/employees/EMP0001` directly, confirming the Compose-internal DNS/network path works, not just each container in isolation.
