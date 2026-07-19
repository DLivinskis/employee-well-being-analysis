"""
FastAPI dependency functions for the shared Database and InferenceService.

Both are created once at startup (see `api/main.py`) and stored on
`app.state`; these functions are how routers access them via `Depends`
without importing global singletons directly.
"""

from typing import Iterator

from fastapi import Request
from sqlalchemy.orm import Session

from api.database import Database
from api.endpoint_helpers.inference import InferenceService


def get_database(request: Request) -> Database:
    """Return the shared `Database` instance stored on `app.state`.

    Args:
        request: The current request, whose `app.state.database` was set
            during startup.

    Returns:
        The API's single `Database` instance.
    """
    return request.app.state.database


def get_session(request: Request) -> Iterator[Session]:
    """Yield a DB session for the duration of one request.

    Args:
        request: The current request, whose `app.state.database` was set
            during startup.

    Yields:
        An open SQLAlchemy Session, committed/closed when the request ends.
    """
    with get_database(request).session() as session:
        yield session


def get_inference_service(request: Request) -> InferenceService:
    """Return the shared `InferenceService` instance stored on `app.state`.

    Args:
        request: The current request, whose `app.state.inference` was set
            during startup.

    Returns:
        The API's single `InferenceService` instance, with models already
        loaded.
    """
    return request.app.state.inference
