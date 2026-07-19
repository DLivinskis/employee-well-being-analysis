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
│   ├── config.py                # paths, DB URL, random seed, split ratios
│   ├── data_split.py            # stratified train/val/test split
│   ├── preprocessing.py         # encoders shared by both models
│   ├── evaluation.py            # confusion matrix, ROC/AUC, macro-F1
│   ├── train_logistic.py        # fits + serializes the logistic model
│   ├── train_tree.py            # fits + serializes the tree model
│   └── run_training.py          # entry point: split -> train both -> evaluate -> write artifacts + models-table rows
├── api/
│   ├── __init__.py
│   ├── main.py                  # FastAPI app + router registration
│   ├── config.py                # env-based settings (DB URL, model artifact dir)
│   ├── database.py              # SQLAlchemy engine/session
│   ├── db_models.py              # ORM models: Employee, TrainedModel
│   ├── schemas.py                # Pydantic request/response models
│   ├── seed.py                   # loads CSV into `employees` table if empty
│   ├── inference.py               # loads model artifacts, runs prediction
│   └── routers/
│       ├── __init__.py
│       ├── employees.py          # GET /employees/{employee_id}
│       ├── predict.py            # POST /predict
│       └── analysis.py           # GET /analysis/*
├── frontend/
│   ├── Home.py                   # Streamlit entry point / landing page
│   ├── api_client.py             # thin wrapper around requests to the API
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

### Phase 3 — Database layer (`api/database.py`, `api/db_models.py`, `api/seed.py`)
- [ ] `db_models.py`: SQLAlchemy models for `employees` (mirrors CSV columns) and `models` (name, version, trained_at, artifact_path, feature_importance JSON, confusion_matrix JSON, roc_auc JSON).
- [ ] `database.py`: engine/session setup reading the DB URL from `api/config.py`.
- [ ] `seed.py`: idempotent — creates tables if missing, loads the CSV into `employees` only if the table is empty; called once at API startup.

### Phase 4 — FastAPI backend (`api/`)
- [ ] `schemas.py`: Pydantic models for employee response, predict request/response, and analysis responses (coefficients/importances + confusion matrix + ROC/AUC).
- [ ] `inference.py`: loads the two serialized pipelines once at startup; exposes a `predict(features) -> {label, class_probabilities}` function per model type used by `/predict`.
- [ ] `routers/employees.py`: `GET /employees/{employee_id}`.
- [ ] `routers/predict.py`: `POST /predict`.
- [ ] `routers/analysis.py`: `GET /analysis/descriptive` (calls `exploratory_analysis/descriptive_stats.py` against the `employees` table), `GET /analysis/logistic-importance`, `GET /analysis/tree-importance`.
- [ ] `main.py`: app instance, calls `seed.py` and `inference.py` startup hooks, registers routers.

### Phase 5 — Streamlit frontend (`frontend/`)
- [ ] `api_client.py`: thin functions wrapping each API call (`get_employee`, `predict`, `get_descriptive`, `get_logistic_importance`, `get_tree_importance`), API base URL from an environment variable.
- [ ] `pages/1_Descriptive_Analysis.py`: charts from `/analysis/descriptive`.
- [ ] `pages/2_Logistic_Model_Features.py`: coefficients + confusion matrix + ROC/AUC from `/analysis/logistic-importance`.
- [ ] `pages/3_Tree_Model_Features.py`: importances + confusion matrix + ROC/AUC from `/analysis/tree-importance`.
- [ ] `pages/4_Predict.py`: client-ID lookup (pre-fills form via `/employees/{id}`) or manual entry, both ending in a call to `/predict`.
- [ ] `Home.py`: landing page linking to the four pages.

### Phase 6 — Dockerization
- [ ] `docker/api.Dockerfile`: installs `training`+`api` dependencies, copies `training/` and `api/`, runs `uvicorn api.main:app`.
- [ ] `docker/frontend.Dockerfile`: installs `frontend` dependencies, copies `frontend/`, runs `streamlit run frontend/Home.py`.
- [ ] `docker-compose.yml`: `db` (postgres + named volume + healthcheck), `api` (depends_on db healthy, env vars for DB URL, volume for `data/models`), `frontend` (depends_on api, env var for API base URL).
- [ ] Run `run_training.py` locally once to produce `data/models/` artifacts and populate `models` table before first `docker compose up` (or add it as a one-off `docker compose run` step — decide when this phase is reached).

### Phase 7 — End-to-end check
- [ ] `docker compose up`, confirm all three services healthy.
- [ ] Walk through each Streamlit page manually: descriptive charts render, both feature pages show importances + confusion matrix + ROC/AUC, predict works for both client-ID and manual-entry paths.
