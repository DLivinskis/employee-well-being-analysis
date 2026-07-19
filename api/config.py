"""
Configuration for the FastAPI service.

Centralizes the database URL and artifact/data paths so they're read from
environment variables in one place, with sensible local defaults for
running the API outside Docker Compose.
"""

import os
from pathlib import Path


class APIConfig:
    """Holds settings read by the database layer and the API app.

    Entry points:
        This class has no methods beyond `__init__` — it is a plain
        settings holder read by `Database`, `seed.py`, and (in a later
        phase) `inference.py`.

    Args:
        database_url: SQLAlchemy connection string for Postgres.
        raw_data_path: Path to the source CSV, used to seed `employees`.
        artifacts_dir: Directory containing trained model artifacts and
            `metadata.json`, used to seed `models`.
    """

    def __init__(
        self,
        database_url: str | None = None,
        raw_data_path: Path | None = None,
        artifacts_dir: Path | None = None,
    ) -> None:
        repo_root = Path(__file__).resolve().parent.parent
        # Default matches docker-compose.yml's `db` service credentials and
        # exposed port, so scripts run on the host (outside a container)
        # connect the same way `docker compose exec` / a local DB client
        # would, without needing DATABASE_URL set for local dev.
        self.database_url = database_url or os.environ.get(
            "DATABASE_URL",
            "postgresql+psycopg://ewb:ewb@localhost:5432/employee_wellbeing",
        )
        self.raw_data_path = raw_data_path or Path(
            os.environ.get(
                "RAW_DATA_PATH",
                repo_root / "raw_data" / "Impact_of_Remote_Work_on_Mental_Health.csv",
            )
        )
        self.artifacts_dir = artifacts_dir or Path(
            os.environ.get("ARTIFACTS_DIR", repo_root / "data" / "models")
        )
