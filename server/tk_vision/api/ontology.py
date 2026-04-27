from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from ..config import load_ontology, ontology_hash

router = APIRouter(prefix="/api/ontology", tags=["ontology"])


class OntologyResponse(BaseModel):
    path: str
    hash: str
    prompts: list[str]
    labels: list[str]
    mapping: dict[str, str]


class OntologyUpdate(BaseModel):
    mapping: dict[str, str]


def _ontology_path(request: Request) -> Path:
    settings = request.app.state.settings
    return settings.resolve(settings.project.ontology)


@router.get("", response_model=OntologyResponse)
async def get_ontology(request: Request) -> OntologyResponse:
    path = _ontology_path(request)
    if not path.exists():
        raise HTTPException(404, f"Ontology not found at {path}")
    mapping = load_ontology(path)
    return OntologyResponse(
        path=str(path),
        hash=ontology_hash(mapping),
        prompts=list(mapping.keys()),
        labels=list(mapping.values()),
        mapping=mapping,
    )


@router.post("", response_model=OntologyResponse)
async def update_ontology(payload: OntologyUpdate, request: Request) -> OntologyResponse:
    if not payload.mapping:
        raise HTTPException(400, "Ontology must be non-empty")
    path = _ontology_path(request)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload.mapping, f, indent=2, ensure_ascii=False)
    return OntologyResponse(
        path=str(path),
        hash=ontology_hash(payload.mapping),
        prompts=list(payload.mapping.keys()),
        labels=list(payload.mapping.values()),
        mapping=payload.mapping,
    )
