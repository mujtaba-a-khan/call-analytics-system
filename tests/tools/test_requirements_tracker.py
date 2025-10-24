from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tools.requirements_tracker import db, seed


def _make_client(tmp_path: Path) -> TestClient:
    db_url = f"sqlite:///{tmp_path}"
    os.environ["CALL_ANALYTICS_REQUIREMENTS_DB"] = db_url
    db.reset_engine()
    db.init_db()
    from tools.requirements_tracker.app import app  # imported after engine reset

    return TestClient(app)


@pytest.fixture()
def client(tmp_path: Path) -> TestClient:
    database_file = tmp_path / "requirements.db"
    client = _make_client(database_file)
    # Seed one record for list tests
    seed.seed_database(reset=False)
    return client


def test_list_requirements_returns_seeded_data(client: TestClient) -> None:
    response = client.get("/requirements")
    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload, list)
    assert payload, "Expected at least one requirement from seed data"


def test_create_and_update_requirement(client: TestClient) -> None:
    create_payload = {
        "title": "Test requirement",
        "description": "Created during unit test",
        "priority": "Low",
        "status": "Proposed",
        "assignee": "Test Owner",
        "category": "QA",
        "acceptance_criteria": "When created, it appears in list.",
        "notes": "",
    }
    response = client.post("/requirements", json=create_payload)
    assert response.status_code == 201
    created = response.json()
    assert created["title"] == create_payload["title"]
    assert created["assignee"] == create_payload["assignee"]
    requirement_id = created["id"]

    update_payload = {"status": "Done", "notes": "Marked complete"}
    update_response = client.put(f"/requirements/{requirement_id}", json=update_payload)
    assert update_response.status_code == 200
    updated = update_response.json()
    assert updated["status"] == "Done"
    assert updated["notes"] == "Marked complete"


def test_delete_requirement(client: TestClient) -> None:
    response = client.post(
        "/requirements",
        json={
            "title": "Disposable requirement",
            "description": "",
            "priority": "Low",
            "status": "Proposed",
            "assignee": "Automation",
            "category": "QA",
            "acceptance_criteria": "",
            "notes": "",
        },
    )
    requirement_id = response.json()["id"]

    delete_response = client.delete(f"/requirements/{requirement_id}")
    assert delete_response.status_code == 204

    not_found = client.get(f"/requirements/{requirement_id}")
    assert not_found.status_code == 404
