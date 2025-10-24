from __future__ import annotations

import argparse
import json
from pathlib import Path

from sqlmodel import Session, SQLModel, select

from . import db
from .models import Requirement, RequirementCreate, RequirementPriority, RequirementStatus

DATA_PATH = Path(__file__).resolve().parent / "data" / "requirements_seed.json"


def _load_seed_payloads() -> list[RequirementCreate]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Seed file missing: {DATA_PATH}")
    raw_items = json.loads(DATA_PATH.read_text(encoding="utf-8"))
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
