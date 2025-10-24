from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from sqlmodel import SQLModel, Field


class RequirementStatus(str, Enum):
    """Workflow states for requirements."""

    PROPOSED = "Proposed"
    IN_PROGRESS = "In Progress"
    REVIEW = "Review"
    DONE = "Done"


class RequirementPriority(str, Enum):
    """Priority levels to aid triage."""

    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


class RequirementBase(SQLModel):
    """Shared attributes for Requirement models."""

    title: str = Field(index=True, max_length=200)
    description: str = Field(default="", max_length=2000)
    priority: RequirementPriority = Field(default=RequirementPriority.MEDIUM)
    status: RequirementStatus = Field(default=RequirementStatus.PROPOSED)
    assignee: str = Field(default="Unassigned", max_length=100, description="Primary driver")
    category: str = Field(default="General", max_length=100)
    acceptance_criteria: str = Field(default="", max_length=2000)
    notes: str = Field(default="", max_length=2000)


class Requirement(RequirementBase, table=True):
    """Persistent Requirement record."""

    id: Optional[int] = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)
    updated_at: datetime = Field(default_factory=datetime.utcnow, nullable=False)


class RequirementCreate(RequirementBase):
    """Payload for creating a requirement."""


class RequirementRead(RequirementBase):
    """Serialized requirement returned to API consumers."""

    id: int
    created_at: datetime
    updated_at: datetime


class RequirementUpdate(SQLModel):
    """Patch payload allowing partial updates."""

    title: Optional[str] = Field(default=None, max_length=200)
    description: Optional[str] = Field(default=None, max_length=2000)
    priority: Optional[RequirementPriority] = None
    status: Optional[RequirementStatus] = None
    assignee: Optional[str] = Field(default=None, max_length=100)
    category: Optional[str] = Field(default=None, max_length=100)
    acceptance_criteria: Optional[str] = Field(default=None, max_length=2000)
    notes: Optional[str] = Field(default=None, max_length=2000)
