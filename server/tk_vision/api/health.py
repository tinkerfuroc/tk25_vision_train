from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel

router = APIRouter(prefix="/api/healthz", tags=["health"])


class HealthResponse(BaseModel):
    ok: bool
    version: str
    sam3_loaded: bool
    gpu: str | None
    dtype: str | None
    config_path: str | None


@router.get("", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    from .. import __version__

    settings = request.app.state.settings
    sam3 = getattr(request.app.state, "sam3_engine", None)
    return HealthResponse(
        ok=True,
        version=__version__,
        sam3_loaded=sam3 is not None,
        gpu=settings.sam3.device if sam3 is not None else None,
        dtype=settings.sam3.dtype if sam3 is not None else None,
        config_path=settings.config_path,
    )
