from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from sqlmodel import Session, SQLModel, select

from . import db
from .models import Requirement, RequirementCreate, RequirementPriority, RequirementStatus

DATA_PATH = Path(__file__).resolve().parent / "data" / "requirements_seed.json"

logger = logging.getLogger(__name__)

DEFAULT_SEED_ITEMS = [
    {
        "title": "Define analytics requirements backlog",
        "description": "Baseline stories to drive the call analytics system roadmap.",
        "priority": "High",
        "status": "Proposed",
        "assignee": "Product",
        "category": "Planning",
        "acceptance_criteria": "Stakeholders agree on initial backlog scope.",
        "notes": "Generated automatically when requirements seed file is unavailable.",
    }
]


def _build_payloads(raw_items: list[dict]) -> list[RequirementCreate]:
    payloads: list[RequirementCreate] = []
    for item in raw_items:
        payloads.append(
            RequirementCreate(
                title=item["title"],
                description=item.get("description", ""),
                priority=RequirementPriority(item.get("priority", "Medium")),
                status=RequirementStatus(item.get("status", "Proposed")),
                assignee=item.get("assignee", "Unassigned"),
                category=item.get("category", "General"),
                acceptance_criteria=item.get("acceptance_criteria", ""),
                notes=item.get("notes", ""),
            )
        )
    return payloads


def _load_seed_payloads() -> list[RequirementCreate]:
    if DATA_PATH.exists():
        try:
            raw_items = json.loads(DATA_PATH.read_text(encoding="utf-8"))
            if isinstance(raw_items, list):
                return _build_payloads(raw_items)
            logger.warning("Seed file %s must contain a list; received %s", DATA_PATH, type(raw_items))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to load requirements seed data from %s: %s", DATA_PATH, exc)
    else:
        logger.info("Seed file %s missing; using default requirements seed.", DATA_PATH)

    return _build_payloads(DEFAULT_SEED_ITEMS)


def seed_database(reset: bool = False) -> int:
    """Populate the tracker with sample requirements."""
    if reset:
        db.reset_engine()
        SQLModel.metadata.drop_all(db.engine)
    db.init_db()
    payloads = _load_seed_payloads()
    created = 0

    with Session(db.engine) as session:
        existing_titles = set(session.exec(select(Requirement.title)).all())
        for payload in payloads:
            if payload.title in existing_titles:
                continue
            requirement = Requirement(**payload.model_dump())
            session.add(requirement)
            created += 1
        session.commit()
    return created


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Seed the professional requirements tracker with canonical requirements."
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Recreate the SQLite engine before inserting records (clears previous data).",
    )
    args = parser.parse_args()
    created = seed_database(reset=args.reset)
    print(f"Seeded {created} requirement(s).")


if __name__ == "__main__":
    main()
