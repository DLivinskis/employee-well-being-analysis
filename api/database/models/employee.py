"""
SQLAlchemy ORM model for the `employees` table.

Seeded from the raw CSV but treated as a live, growing table: new rows can
be inserted via the API (simulating a new hire appearing) independently of
the CSV, per `docs/architecture_and_design.md`.
"""

from sqlalchemy import Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from api.database.models.base import Base


class Employee(Base):
    """One row per employee, keyed by `Employee_ID`.

    Columns mirror the raw CSV's schema exactly, so a row loaded from the
    CSV and a row inserted via `POST /employees` are indistinguishable.
    """

    __tablename__ = "employees"

    employee_id: Mapped[str] = mapped_column(String, primary_key=True)
    age: Mapped[int] = mapped_column(Integer)
    gender: Mapped[str] = mapped_column(String)
    job_role: Mapped[str] = mapped_column(String)
    industry: Mapped[str] = mapped_column(String)
    years_of_experience: Mapped[int] = mapped_column(Integer)
    work_location: Mapped[str] = mapped_column(String)
    hours_worked_per_week: Mapped[int] = mapped_column(Integer)
    number_of_virtual_meetings: Mapped[int] = mapped_column(Integer)
    work_life_balance_rating: Mapped[int] = mapped_column(Integer)
    stress_level: Mapped[str] = mapped_column(String)
    mental_health_condition: Mapped[str] = mapped_column(String)
    access_to_mental_health_resources: Mapped[str] = mapped_column(String)
    productivity_change: Mapped[str] = mapped_column(String)
    social_isolation_rating: Mapped[int] = mapped_column(Integer)
    satisfaction_with_remote_work: Mapped[str] = mapped_column(String)
    company_support_for_remote_work: Mapped[int] = mapped_column(Integer)
    physical_activity: Mapped[str] = mapped_column(String)
    sleep_quality: Mapped[str] = mapped_column(String)
    region: Mapped[str] = mapped_column(String)
