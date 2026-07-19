"""
Shared SQLAlchemy declarative base for all ORM models.
"""

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Shared declarative base every ORM model in `api/database/models/` inherits from."""
