from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from contextlib import asynccontextmanager

from .api import augment as augment_api
from .api import clips as clips_api
from .api import export as export_api
from .api import infer as infer_api
from .api import health as health_api
from .api import label as label_api
from .api import ontology as ontology_api
from .api import propagate as propagate_api
from .api import realsense as realsense_api
from .api import sam3 as sam3_api
from .api import train as train_api
from .capture.live_camera import LiveCamera
from .config import Settings
from .data.persistence import ProjectStore
from .services.capture_service import CaptureService
from .services.infer_service import InferManager
from .services.label_service import LabelService
from .services.train_service import TrainManager
from .ws import capture as capture_ws
from .ws import infer as infer_ws
from .ws import label as label_ws
from .ws import propagate as propagate_ws
from .ws import train as train_ws


def create_app(settings: Settings | None = None, *, load_sam3: bool = False) -> FastAPI:
    settings = settings or Settings.load()
    log = logging.getLogger("tk_vision")
    log.info("config: %s", settings.config_path or "<defaults>")

    @asynccontextmanager
    async def lifespan(app_):  # noqa: ANN001
        try:
            yield
        finally:
            cam = getattr(app_.state, "live_camera", None)
            if cam is not None:
                await cam.shutdown()

    app = FastAPI(title="tk_vision", version="0.1.0", lifespan=lifespan)

    @app.exception_handler(ValueError)
    async def _value_error_handler(_request, exc: ValueError) -> JSONResponse:
        # Codebase convention: ValueError == bad user input. Map to HTTP 400
        # so safe_id() and other validators surface as a clear 400 instead of
        # a 500 stack trace.
        return JSONResponse({"detail": str(exc)}, status_code=400)

    app.state.settings = settings
    app.state.sam3_engine = None
    app.state.store = ProjectStore(settings.resolve(settings.project.data_root))
    app.state.live_camera = LiveCamera(
        fps=settings.capture.fps,
        resolution=tuple(settings.capture.resolution),
    )
    app.state.capture = CaptureService(
        app.state.store,
        fps=settings.capture.fps,
        resolution=tuple(settings.capture.resolution),
        live_camera=app.state.live_camera,
    )
    app.state.train = TrainManager(settings)
    app.state.infer = InferManager(app.state.store)

    if settings.server.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.server.cors_origins,
            allow_methods=["*"],
            allow_headers=["*"],
            allow_credentials=True,
        )

    app.include_router(health_api.router)
    app.include_router(ontology_api.router)
    app.include_router(clips_api.router)
    app.include_router(realsense_api.router)
    app.include_router(label_api.router)
    app.include_router(sam3_api.router)
    app.include_router(propagate_api.router)
    app.include_router(export_api.router)
    app.include_router(augment_api.router)
    app.include_router(train_api.router)
    app.include_router(train_ws.router)
    app.include_router(infer_api.router)
    app.include_router(infer_ws.router)
    app.include_router(capture_ws.router)
    app.include_router(label_ws.router)
    app.include_router(propagate_ws.router)

    web_dist = settings.resolve("web/dist")
    if web_dist.exists():
        index = web_dist / "index.html"
        app.mount("/static", StaticFiles(directory=str(web_dist)), name="static")

        @app.get("/", include_in_schema=False)
        async def root_index() -> FileResponse:
            return FileResponse(str(index))

        @app.get("/{full_path:path}", include_in_schema=False, response_model=None)
        async def spa_fallback(full_path: str):
            if full_path.startswith(("api/", "ws/", "static/")):
                return JSONResponse({"detail": "Not found"}, status_code=404)
            candidate = web_dist / full_path
            if candidate.is_file():
                return FileResponse(str(candidate))
            return FileResponse(str(index))
    else:
        @app.get("/", include_in_schema=False)
        async def missing_dist() -> JSONResponse:
            return JSONResponse(
                {
                    "detail": (
                        f"web/dist not found at {web_dist}. Build the SPA "
                        "(cd web && npm ci && npm run build) or serve via Vite dev server."
                    )
                },
                status_code=503,
            )

    if load_sam3:
        try:
            from .annotate.sam3 import Sam3Engine

            engine = Sam3Engine(
                model_dir=settings.resolve(settings.sam3.model_dir),
                device=settings.sam3.device,
                dtype=settings.sam3.dtype,
                text_threshold=settings.sam3.text_threshold,
                box_threshold=settings.sam3.box_threshold,
                score_mode=settings.sam3.score_mode,
            )
            engine.load()
            app.state.sam3_engine = engine
            app.state.label = LabelService(app.state.store, engine, settings)
            log.info("Sam3Engine loaded; LabelService ready")
        except Exception as e:  # noqa: BLE001
            log.exception("Failed to load Sam3Engine: %s", e)
            app.state.sam3_engine = None
            app.state.label = None

    return app


def serve(
    *,
    host: str | None = None,
    port: int | None = None,
    config: str | None = None,
    bind: str | None = None,
    no_sam3: bool = False,
    reload: bool = False,
) -> None:
    import uvicorn

    settings = Settings.load(config)
    if bind:
        if bind == "0.0.0.0" and settings.server.bind_warning:
            print(
                "[tk_vision] WARNING: --bind 0.0.0.0 exposes the server on all interfaces. "
                "There is no auth. Use a firewall or VPN.",
                flush=True,
            )
        settings.server.host = bind
    elif host:
        settings.server.host = host
    if port:
        settings.server.port = port

    if reload:
        # uvicorn reload requires an import string and reads settings on each reload.
        import os
        os.environ.setdefault("TK_VISION_CONFIG", settings.config_path or "")
        os.environ["TK_VISION_NO_SAM3"] = "1" if no_sam3 else "0"
        uvicorn.run(
            "tk_vision.app:_reload_factory",
            factory=True,
            host=settings.server.host,
            port=settings.server.port,
            reload=True,
            log_level="info",
        )
        return

    app = create_app(settings, load_sam3=not no_sam3)
    uvicorn.run(app, host=settings.server.host, port=settings.server.port, log_level="info")


def _reload_factory() -> FastAPI:
    import os

    cfg = os.environ.get("TK_VISION_CONFIG") or None
    no_sam3 = os.environ.get("TK_VISION_NO_SAM3") == "1"
    return create_app(Settings.load(cfg), load_sam3=not no_sam3)
