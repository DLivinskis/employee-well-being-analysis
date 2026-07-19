"""
SQLAlchemy ORM model for the `employees` table.

Seeded from the raw CSV but treated as a live, growing table: new rows can
be inserted via the API (simulating a new hire appearing) independently of
the CSV, per `docs/architecture_and_design.md`.
"""

from typing import ClassVar

from sqlalchemy import Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from api.database.models.base import Base


class Employee(Base):
    """One row per employee, keyed by `Employee_ID`.

    Columns mirror the raw CSV's schema, except `satisfaction_with_remote_work`
    is nullable — a newly inserted employee (via `POST /employees`) has no
    known satisfaction yet, since predicting it is the whole point of the
    `/predict` flow.
    """

    __tablename__ = "employees"

    # Maps the CSV's (and API schema's) PascalCase/underscore column names
    # to this model's snake_case attribute names. Shared by `seed.py` (CSV
    # row -> Employee) and `api/routers/employees.py` (Employee <->
    # EmployeeCreate/EmployeeResponse), so the mapping is defined once.
    CSV_COLUMN_TO_ATTRIBUTE: ClassVar[dict[str, str]] = {
        "Employee_ID": "employee_id",
        "Age": "age",
        "Gender": "gender",
        "Job_Role": "job_role",
        "Industry": "industry",
        "Years_of_Experience": "years_of_experience",
        "Work_Location": "work_location",
        "Hours_Worked_Per_Week": "hours_worked_per_week",
        "Number_of_Virtual_Meetings": "number_of_virtual_meetings",
        "Work_Life_Balance_Rating": "work_life_balance_rating",
        "Stress_Level": "stress_level",
        "Mental_Health_Condition": "mental_health_condition",
        "Access_to_Mental_Health_Resources": "access_to_mental_health_resources",
        "Productivity_Change": "productivity_change",
        "Social_Isolation_Rating": "social_isolation_rating",
        "Satisfaction_with_Remote_Work": "satisfaction_with_remote_work",
        "Company_Support_for_Remote_Work": "company_support_for_remote_work",
        "Physical_Activity": "physical_activity",
        "Sleep_Quality": "sleep_quality",
        "Region": "region",
    }

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
    satisfaction_with_remote_work: Mapped[str | None] = mapped_column(String, nullable=True)
    company_support_for_remote_work: Mapped[int] = mapped_column(Integer)
    physical_activity: Mapped[str] = mapped_column(String)
    sleep_quality: Mapped[str] = mapped_column(String)
    region: Mapped[str] = mapped_column(String)
