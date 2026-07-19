"""
`GET /employees/{employee_id}` and `POST /employees`.

`POST /employees` is what makes "predict for an employee outside the
original CSV" a real, working path — see `docs/architecture_and_design.md`'s
"What the database is (and isn't) for" note.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from api.database.models import Employee
from api.endpoint_helpers.dependencies import get_session
from api.endpoint_helpers.schemas import EmployeeCreate, EmployeeResponse

router = APIRouter(prefix="/employees", tags=["employees"])


class EmployeeConverter:
    """Converts between `Employee` ORM rows and the API's Pydantic schemas.

    Entry points:
        to_response: Build an `EmployeeResponse` from an `Employee` row.
        to_orm_model: Build an unsaved `Employee` from an `EmployeeCreate`.
    """

    def to_response(self, employee: Employee) -> EmployeeResponse:
        """Build an `EmployeeResponse` from an `Employee` ORM row.

        Args:
            employee: The ORM row to convert.

        Returns:
            An `EmployeeResponse` with one field per CSV column name.
        """
        fields = {
            csv_column: getattr(employee, attribute)
            for csv_column, attribute in Employee.CSV_COLUMN_TO_ATTRIBUTE.items()
        }
        return EmployeeResponse(**fields)

    def to_orm_model(self, employee_create: EmployeeCreate) -> Employee:
        """Build an unsaved `Employee` from a `POST /employees` request body.

        Args:
            employee_create: The validated request body.

        Returns:
            An `Employee` instance, not yet added to a session.
        """
        data = employee_create.model_dump()
        attributes = {
            Employee.CSV_COLUMN_TO_ATTRIBUTE[csv_column]: value
            for csv_column, value in data.items()
        }
        return Employee(**attributes)


_converter = EmployeeConverter()


@router.get("/{employee_id}", response_model=EmployeeResponse)
def get_employee(employee_id: str, session: Session = Depends(get_session)) -> EmployeeResponse:
    """Return stored feature values for one employee, by ID."""
    employee = session.get(Employee, employee_id)
    if employee is None:
        raise HTTPException(status_code=404, detail=f"No employee with ID {employee_id!r}")
    return _converter.to_response(employee)


@router.post("", response_model=EmployeeResponse, status_code=201)
def create_employee(
    employee_create: EmployeeCreate, session: Session = Depends(get_session)
) -> EmployeeResponse:
    """Insert a new employee record (simulates a new hire appearing)."""
    employee = _converter.to_orm_model(employee_create)
    session.add(employee)
    try:
        session.flush()
    except IntegrityError:
        raise HTTPException(
            status_code=409,
            detail=f"Employee {employee_create.Employee_ID!r} already exists",
        )
    return _converter.to_response(employee)
