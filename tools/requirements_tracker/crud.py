from __future__ import annotations

from datetime import datetime
from typing import Iterable, Sequence

from sqlmodel import Session, select

from .models import (
    Requirement,
    RequirementCreate,
    RequirementPriority,
    RequirementStatus,
    RequirementUpdate,
)


def list_requirements(
    session: Session,
    status: RequirementStatus | None = None,
    priority: RequirementPriority | None = None,
) -> Sequence[Requirement]:
    statement = select(Requirement)
    if status:
        statement = statement.where(Requirement.status == status)
    if priority:
        statement = statement.where(Requirement.priority == priority)
    return session.exec(statement).all()


def get_requirement(session: Session, requirement_id: int) -> Requirement | None:
    return session.get(Requirement, requirement_id)


def create_requirement(session: Session, data: RequirementCreate) -> Requirement:
    requirement = Requirement(**data.model_dump())
    session.add(requirement)
    session.commit()
    session.refresh(requirement)
    return requirement


def create_requirements_bulk(session: Session, payloads: Iterable[RequirementCreate]) -> int:
    created = 0
    for payload in payloads:
        requirement = Requirement(**payload.model_dump())
        session.add(requirement)
        created += 1
    session.commit()
    return created


def update_requirement(
    session: Session,
    requirement: Requirement,
    data: RequirementUpdate,
) -> Requirement:
    update_data = data.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(requirement, field, value)
    requirement.updated_at = datetime.utcnow()
    session.add(requirement)
    session.commit()
    session.refresh(requirement)
    return requirement


def delete_requirement(session: Session, requirement: Requirement) -> None:
    session.delete(requirement)
    session.commit()
