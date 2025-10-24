from __future__ import annotations

from collections import defaultdict
from contextlib import asynccontextmanager
from math import ceil
from pathlib import Path
from typing import Dict, List
from urllib.parse import urlencode

from fastapi import Depends, FastAPI, HTTPException, Query, Request, status
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from sqlmodel import Session

from . import crud
from .db import get_session, init_db
from .models import (
    Requirement,
    RequirementCreate,
    RequirementPriority,
    RequirementRead,
    RequirementStatus,
    RequirementUpdate,
)

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

@asynccontextmanager
async def lifespan(_: FastAPI) -> None:
    init_db()
    yield


app = FastAPI(
    title="Requirements Tracker",
    description="FastAPI-based tracker used to document project requirements in a professional tool.",
    version="0.1.0",
    lifespan=lifespan,
)

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

STATUS_COLUMNS = [
    RequirementStatus.PROPOSED,
    RequirementStatus.IN_PROGRESS,
    RequirementStatus.REVIEW,
    RequirementStatus.DONE,
]

PRIORITY_ORDER = {
    RequirementPriority.HIGH: 0,
    RequirementPriority.MEDIUM: 1,
    RequirementPriority.LOW: 2,
}

PRIORITY_SEQUENCE = [
    RequirementPriority.HIGH,
    RequirementPriority.MEDIUM,
    RequirementPriority.LOW,
]

FILTER_OPTIONS = ["All"] + [priority.value for priority in PRIORITY_SEQUENCE]

CARDS_PER_PAGE = 1


def _build_url(
    params: Dict[str, str],
    slug: str,
    page: int | None = None,
    priority: str | None = None,
) -> str:
    page_key = f"page_{slug}"
    priority_key = f"priority_{slug}"
    new_params = dict(params)

    if priority is not None:
        if priority == "All":
            new_params.pop(priority_key, None)
        else:
            new_params[priority_key] = priority

    if page is not None:
        if page <= 1:
            new_params.pop(page_key, None)
        else:
            new_params[page_key] = str(page)

    query = urlencode(new_params)
    return f"/board?{query}" if query else "/board"
@app.get("/health", tags=["System"])
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/requirements", response_model=list[RequirementRead], tags=["Requirements"])
def list_requirements(
    status: RequirementStatus | None = Query(default=None),
    priority: RequirementPriority | None = Query(default=None),
    session: Session = Depends(get_session),
) -> list[Requirement]:
    requirements = crud.list_requirements(
        session=session,
        status=status,
        priority=priority,
    )
    return list(requirements)


@app.post(
    "/requirements",
    response_model=RequirementRead,
    status_code=status.HTTP_201_CREATED,
    tags=["Requirements"],
)
def create_requirement(
    payload: RequirementCreate,
    session: Session = Depends(get_session),
) -> Requirement:
    return crud.create_requirement(session, payload)


@app.get("/requirements/{requirement_id}", response_model=RequirementRead, tags=["Requirements"])
def fetch_requirement(
    requirement_id: int,
    session: Session = Depends(get_session),
) -> Requirement:
    requirement = crud.get_requirement(session, requirement_id)
    if not requirement:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Requirement not found")
    return requirement


@app.put("/requirements/{requirement_id}", response_model=RequirementRead, tags=["Requirements"])
def update_requirement(
    requirement_id: int,
    payload: RequirementUpdate,
    session: Session = Depends(get_session),
) -> Requirement:
    requirement = crud.get_requirement(session, requirement_id)
    if not requirement:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Requirement not found")
    return crud.update_requirement(session, requirement, payload)


@app.delete(
    "/requirements/{requirement_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["Requirements"],
)
def delete_requirement(
    requirement_id: int,
    session: Session = Depends(get_session),
) -> None:
    requirement = crud.get_requirement(session, requirement_id)
    if not requirement:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Requirement not found")
    crud.delete_requirement(session, requirement)


@app.get("/board", response_class=HTMLResponse, tags=["Board"])
def board_view(
    request: Request,
    session: Session = Depends(get_session),
) -> HTMLResponse:
    grouped: Dict[RequirementStatus, List[Requirement]] = defaultdict(list)
    for requirement in crud.list_requirements(session):
        grouped[requirement.status].append(requirement)

    for requirements in grouped.values():
        requirements.sort(
            key=lambda req: (
                PRIORITY_ORDER.get(req.priority, 99),
                req.title.lower(),
            )
        )

    query_dict = dict(request.query_params)
    board_columns: List[dict] = []

    for status in STATUS_COLUMNS:
        slug = status.name.lower()
        requirements = grouped.get(status, [])
        priority_counts = {
            priority.value: sum(1 for r in requirements if r.priority == priority)
            for priority in PRIORITY_SEQUENCE
        }
        filter_counts = {"All": len(requirements)}
        filter_counts.update(priority_counts)

        filter_raw = request.query_params.get(f"priority_{slug}", "All")
        selected_priority = "All"
        selected_priority_enum: RequirementPriority | None = None
        if filter_raw and filter_raw.lower() != "all":
            try:
                selected_priority_enum = RequirementPriority(filter_raw)
                selected_priority = selected_priority_enum.value
            except ValueError:
                selected_priority_enum = None
                selected_priority = "All"

        if selected_priority_enum:
            filtered_requirements = [
                req for req in requirements if req.priority == selected_priority_enum
            ]
        elif selected_priority == "All":
            filtered_requirements = list(requirements)
        else:
            filtered_requirements = []

        total = len(filtered_requirements)
        try:
            requested_page = int(request.query_params.get(f"page_{slug}", "1"))
        except ValueError:
            requested_page = 1

        if total:
            total_pages = ceil(total / CARDS_PER_PAGE)
            current_page = max(1, min(requested_page, total_pages))
            start = (current_page - 1) * CARDS_PER_PAGE
            paged_items = filtered_requirements[start : start + CARDS_PER_PAGE]
            prev_page = current_page - 1 if current_page > 1 else None
            next_page = current_page + 1 if current_page < total_pages else None
        else:
            total_pages = 0
            current_page = 0
            paged_items = []
            prev_page = None
            next_page = None

        prev_link = (
            _build_url(query_dict, slug, page=prev_page, priority=selected_priority)
            if prev_page
            else None
        )
        next_link = (
            _build_url(query_dict, slug, page=next_page, priority=selected_priority)
            if next_page
            else None
        )

        filter_links = {
            option: _build_url(
                query_dict,
                slug,
                page=1,
                priority=option,
            )
            for option in FILTER_OPTIONS
        }

        board_columns.append(
            {
                "status": status,
                "slug": slug,
                "cards": paged_items,
                "total": total,
                "page": current_page if total else 0,
                "total_pages": total_pages if total else 0,
                "priority_counts": priority_counts,
                "prev_link": prev_link,
                "next_link": next_link,
                "has_nav": total > CARDS_PER_PAGE,
                "selected_priority": selected_priority,
                "filter_counts": filter_counts,
                "filter_links": filter_links,
            }
        )

    context = {
        "request": request,
        "columns": board_columns,
        "priority_filters": FILTER_OPTIONS,
    }
    return templates.TemplateResponse("board.html", context)
