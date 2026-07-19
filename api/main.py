"""
FastAPI application entry point.

Wires together the database, seeding, and inference startup steps, then
registers the `employees`, `predict`, and `analysis` routers, per
`docs/architecture_and_design.md`.
"""

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from api.endpoint_helpers.config import APIConfig
from api.database import Database, DatabaseSeeder
from api.endpoint_helpers.inference import InferenceService
from api.routers import analysis, employees, predict


class AppStartup:
    """Runs the API's one-time startup sequence.

    Entry points:
        run: Build the database (creating/seeding tables) and the loaded
            inference service, and attach both to `app.state`.

    Args:
        config: API configuration shared by the database and inference
            service.
    """

    def __init__(self, config: APIConfig) -> None:
        self._config = config

    def _build_database(self) -> Database:
        database = Database(self._config)
        database.create_all()
        DatabaseSeeder(self._config, database).seed_employees()
        DatabaseSeeder(self._config, database).seed_models()
        return database

    def _build_inference_service(self) -> InferenceService:
        inference = InferenceService(self._config)
        inference.load()
        return inference

    def run(self, app: FastAPI) -> None:
        """Build the database and inference service, attaching both to `app.state`.

        Args:
            app: The FastAPI application to attach `state.database` and
                `state.inference` to.
        """
        app.state.database = self._build_database()
        app.state.inference = self._build_inference_service()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    AppStartup(APIConfig()).run(app)
    yield


app = FastAPI(title="Employee Well-Being Analysis API", lifespan=lifespan)
app.include_router(employees.router)
app.include_router(predict.router)
app.include_router(analysis.router)
