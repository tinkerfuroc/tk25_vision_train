from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

router = APIRouter(prefix="/api/sam3", tags=["sam3"])


ScoreMode = Literal["native", "per_query"]


class Sam3Config(BaseModel):
    score_mode: ScoreMode


class Sam3ConfigUpdate(BaseModel):
    score_mode: ScoreMode


def _engine(request: Request):
    eng = getattr(request.app.state, "sam3_engine", None)
    if eng is None:
        raise HTTPException(503, "Sam3Engine not loaded")
    return eng


@router.get("/config", response_model=Sam3Config)
async def get_config(request: Request) -> Sam3Config:
    eng = _engine(request)
    return Sam3Config(score_mode=eng.score_mode)


@router.post("/config", response_model=Sam3Config)
async def set_config(payload: Sam3ConfigUpdate, request: Request) -> Sam3Config:
    eng = _engine(request)
    eng.score_mode = payload.score_mode
    return Sam3Config(score_mode=eng.score_mode)
