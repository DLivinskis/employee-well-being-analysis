# Architecture and Design

This document captures the architecture actually being built, refining ideas from [ai_suggested_approach.md](ai_suggested_approach.md) into concrete interfaces and components.

## 1. Interface

### 1.1 Frontend — Streamlit app

A multi-page Streamlit app is the user-facing surface:

- **Descriptive Analysis** — EDA visuals: target distribution, feature distributions, and feature-vs-target relationships (e.g. satisfaction rate by `Work_Location`, by `Stress_Level`, etc.).
- **Top Features — Logistic Model** — standardized coefficients / odds ratios from the logistic regression baseline, ranked by magnitude, per class; plus its confusion matrix and ROC curve/AUC (per class) as standard evaluation visuals.
- **Top Features — Tree Model** — feature importance (permutation and/or SHAP) from the tree-based model, ranked by magnitude; plus its confusion matrix and ROC curve/AUC (per class).
- **Predict** — a single form covering two input modes:
  - **By client ID**: user enters an `Employee_ID`; the page fetches that employee's stored feature values and pre-fills the form.
  - **Manual entry**: user fills in every feature by hand.
  - Either path ends the same way — the page sends the resulting feature set to the backend and displays the predicted satisfaction class + class probabilities.

Each of the three model pages and the descriptive page calls the FastAPI backend for its data (rather than loading the CSV or model artifacts directly), so the Streamlit app stays a thin presentation layer over one source of truth.

### 1.2 Backend — FastAPI service

Owns all data access and model inference. Candidate endpoints:

- `GET /employees/{employee_id}` — return stored feature values for a given employee (backs the "fetch by client ID" flow).
- `POST /employees` — add a new employee record (simulates a new hire appearing after the initial CSV import); the new row becomes immediately fetchable via `GET /employees/{employee_id}` and eligible for prediction by ID.
- `POST /predict` — accept a feature set (whether fetched by ID or entered manually), return `{label, class_probabilities}`.
- `GET /analysis/descriptive` — summary stats / distributions backing the descriptive-analysis page.
- `GET /analysis/logistic-importance` — logistic model's ranked coefficients, confusion matrix, and ROC curve/AUC.
- `GET /analysis/tree-importance` — tree model's ranked feature importances, confusion matrix, and ROC curve/AUC.

The trained model artifacts (logistic + tree) are loaded once at service startup rather than per-request.

### 1.3 Database — PostgreSQL

A dedicated `db` service, with (at least) two tables:

- **`employees`** — seeded from the raw CSV once at setup time, one row per `Employee_ID`, but treated as a *live, growing* table rather than a frozen snapshot: `POST /employees` inserts new rows for employees that never existed in the CSV. Backs `GET /employees/{employee_id}` and `POST /employees`.
- **`models`** — metadata about each trained model: name/type (`logistic`, `tree`), version/trained-at timestamp, path to its serialized artifact, its ranked feature-importance values, and its evaluation metrics — confusion matrix and ROC curve/AUC (per class, given the 3-class target) — all stored e.g. as JSON columns, so `/analysis/logistic-importance`, `/analysis/tree-importance`, and a model-evaluation endpoint are simple reads rather than recomputation on every request.

Only the FastAPI service talks to Postgres directly; the Streamlit app always goes through the API.

**What the database is (and isn't) for.** Model inference itself never touches Postgres — once a trained pipeline's artifact is loaded into memory, `predict()` only needs the feature values it's called with. But `employees` is now a genuine dependency of the *predict-by-ID* flow specifically, not just a cache of the CSV:

- **`employees`** is the operational store of who can be looked up by ID. The CSV only seeds its initial contents; `POST /employees` is how the app simulates a new employee showing up after that point (a new hire, someone not in the original Kaggle export) and immediately being predictable by ID. This is exactly the case a static CSV handles poorly — appending a row safely under concurrent access, enforcing that `Employee_ID` stays unique, and serving a keyed lookup all need more than "open the file and scan it." A real database is what makes "predict for an employee who didn't exist when the app was built" a first-class, safe operation instead of a file-rewrite hack.
- **`models`** exists so the analysis pages (coefficients, importances, confusion matrix, ROC/AUC) are cheap reads of *already-computed* results, rather than the API recomputing statistics on every request. This one remains closer to a pure cache — it isn't part of any read/write growth story the way `employees` is.

So the database isn't part of the ML computation itself, but `employees` is a real, load-bearing part of the *product* — it's what lets "predict for employee X" mean something for X's the original dataset never saw.

## 2. Train / validation / test split

Given ~5,000 rows and a balanced 3-class target, the split is a single stratified partition rather than anything more elaborate:

1. **Test set (held out, ~15%)** — split off first, stratified on `Satisfaction_with_Remote_Work` so all three classes stay proportionally represented. Untouched until both models are finalized; used exactly once for the final reported metrics (confusion matrix, ROC/AUC) that get stored in the `models` table.
2. **Train / validation (remaining ~85%)** — split again, stratified the same way, roughly 70/15 of the original data (train) and 15% (validation). Validation is used for model comparison and any hyperparameter choices (e.g. tree depth, regularization strength), so those decisions never touch the test set.
3. **Stratification, not plain random split**, is the key detail:
   - A plain random split only preserves class proportions *on average*; on any single draw, small samples (the ~750-row test set, ~750-row validation set) can easily land a few points off the true ~33/33/33 balance just by chance.
   - Stratified splitting partitions each class separately and samples the same proportion from each, so every split (train, validation, test) keeps the same ~33/33/33 balance as the full dataset by construction, not by luck.
   - This matters here specifically because the target is used for macro-F1 and per-class ROC/AUC — if the test set randomly ended up with (say) 30% `Satisfied` vs 36% `Unsatisfied`, per-class metrics computed on it would be noisier and less comparable to validation-set metrics than they need to be, making it harder to tell a genuine model difference from a split artifact.
   - With only 5,000 rows total, this risk is non-trivial (small-sample effects are worse the smaller each split gets), so stratification costs nothing and removes an avoidable source of noise.
4. **Fixed random seed** for reproducibility — the same split is used across both the logistic and tree models, so their validation/test metrics are directly comparable rather than each model seeing different data.

If validation-set size feels too small once the actual splits are in hand, stratified k-fold cross-validation on the train+validation portion is the fallback, keeping the same held-out test set throughout.

## 3. Packaging — Docker Compose

The final application ships as a `docker-compose.yml` with three services:

- **`db`** — official `postgres` image. Holds the `employees` and `models` tables. Uses a named volume (`pgdata:/var/lib/postgresql/data`) so data persists across `docker compose down`/`up`, and exposes port 5432 to the host so it can be inspected directly (`psql`, a DB client, or `training`/`api` scripts run outside Docker) rather than only being reachable from inside the Compose network. Introduced in Phase 3, ahead of `api`/`frontend`, specifically so seeded data could be verified manually before the rest of the app existed.
- **`api`** — the FastAPI service (added in Phase 4/6). On startup, it runs migrations/seed (create tables if missing, load the CSV into `employees` if empty) and loads whichever model artifacts `models` points to. Connects to `db` via the Compose network (e.g. `postgresql://user:pass@db:5432/...`), with credentials passed as environment variables rather than hardcoded.
- **`frontend`** — the Streamlit app (added in Phase 5/6). Only talks to `api` over the Compose network (e.g. `http://api:8000`); never touches `db` or model artifacts directly.

Model artifact *files* (the serialized logistic/tree models the `models` table's paths point to) live on a volume mounted into `api` (e.g. `./data/models:/app/data/models`) — Postgres stores the metadata/importances, not the binary artifacts themselves.

`depends_on` (with a `condition: service_healthy` healthcheck on `db`) ensures `api` waits for Postgres to actually be ready, not just for the container to start; `frontend` similarly depends on `api` and should still retry/backoff on its first calls while the API finishes its own startup (model loading takes longer than plain process start).

Model *training* stays outside Compose — it's a one-off step (notebook or script) that produces the artifacts and populates the `models` table's expected metadata before the images are built/run, not something the running application does.

## 4. Open questions / next sections

- Exact `models` table schema (columns for name, version, trained-at, artifact path, feature-importance JSON) and whether logistic/tree importances need their own table instead of one shared one.
- Request/response schemas for `/predict` and the analysis endpoints.
- How the manual-entry form's fields map to the model's expected feature encoding.
- Dockerfile base images and dependency install strategy for `api` vs `frontend` (likely two separate Dockerfiles sharing a common `requirements`/`uv` lockfile).
