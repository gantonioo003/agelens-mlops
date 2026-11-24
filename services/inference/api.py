"""Local prototype HTTP API; operational IDs are independent of evaluation IDs."""

import asyncio
import hashlib
import json
import logging
import os
import sqlite3
import time
from contextlib import asynccontextmanager, closing
from uuid import UUID, uuid4

from fastapi import FastAPI, HTTPException, Request
from starlette.concurrency import run_in_threadpool

from scripts.prepare_assets import ROOT
from services.inference.model import MEDIA, Engine
from services.ingest.contract import MAX_IMAGE, RETENTION_DAYS, Rejected

LOG = logging.getLogger("uvicorn.error")


class Results:
    def __init__(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        with closing(sqlite3.connect(path)) as db, db:
            db.execute("""CREATE TABLE IF NOT EXISTS results (
                id TEXT PRIMARY KEY, created_at INTEGER NOT NULL, body TEXT NOT NULL
            )""")
            columns = {row[1] for row in db.execute("PRAGMA table_info(results)")}
            if "fingerprint" not in columns:
                db.execute("ALTER TABLE results ADD COLUMN fingerprint TEXT")

    def save(self, record, fingerprint):
        now = int(time.time())
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("DELETE FROM results WHERE created_at < ?", (now - RETENTION_DAYS * 86400,))
            db.execute(
                "INSERT INTO results (id, created_at, body, fingerprint) VALUES (?, ?, ?, ?)",
                (record["id"], now, json.dumps(record), fingerprint),
            )

    def get(self, request_id):
        result = self.find(request_id)
        return result[0] if result else None

    def find(self, request_id):
        with closing(sqlite3.connect(self.path)) as db:
            row = db.execute(
                "SELECT body, fingerprint FROM results WHERE id = ? AND created_at >= ?",
                (request_id, int(time.time()) - RETENTION_DAYS * 86400),
            ).fetchone()
        return (json.loads(row[0]), row[1]) if row else None


def create_app(engine_factory=Engine, root=ROOT):
    @asynccontextmanager
    async def lifespan(app):
        app.state.engine = engine_factory(root)
        app.state.results = Results(root / "runtime/inference/results.sqlite3")
        app.state.inference_slot = asyncio.Semaphore(1)
        yield

    app = FastAPI(title="AgeLens", version="1.0", lifespan=lifespan, redoc_url=None)

    @app.get("/health")
    def health():
        return {"status": "ready", "model_version": app.state.engine.metadata()["version"]}

    @app.get("/v1/model")
    def model():
        return app.state.engine.metadata()

    @app.post(
        "/v1/predict",
        openapi_extra={
            "requestBody": {
                "required": True,
                "content": {
                    kind: {"schema": {"type": "string", "format": "binary"}} for kind in MEDIA
                },
            }
        },
    )
    async def predict(request: Request):
        request_id = str(uuid4())
        start = time.perf_counter()

        def ms():
            return round((time.perf_counter() - start) * 1000, 3)

        def failure(status, code):
            LOG.warning(
                "prediction_rejected id=%s reason=%s duration_ms=%.3f", request_id, code, ms()
            )
            return HTTPException(status, detail={"id": request_id, "code": code})

        supplied = request.headers.get("x-event-id")
        if supplied is not None:
            try:
                parsed = UUID(supplied)
                if parsed.version != 4 or str(parsed) != supplied:
                    raise ValueError
            except ValueError:
                raise failure(400, "invalid_event_id") from None
            request_id = supplied
        content_type = request.headers.get("content-type", "").split(";")[0].strip().lower()
        if content_type not in MEDIA:
            raise failure(415, "unsupported_media_type")
        data = bytearray()
        async for chunk in request.stream():
            if len(data) + len(chunk) > MAX_IMAGE:
                raise failure(413, "image_too_large")
            data.extend(chunk)
        try:
            fingerprint = hashlib.sha256(content_type.encode() + b"\0" + data).hexdigest()
            async with app.state.inference_slot:
                cached = await run_in_threadpool(app.state.results.find, request_id)
                if cached is not None:
                    record, previous = cached
                    if fingerprint != previous:
                        raise failure(409, "event_payload_conflict")
                    return record
                result = await run_in_threadpool(
                    app.state.engine.predict, bytes(data), content_type
                )
                record = {"id": request_id, "status": "completed", "duration_ms": ms(), **result}
                await run_in_threadpool(app.state.results.save, record, fingerprint)
        except Rejected as exc:
            raise failure(422, str(exc)) from None
        except (RuntimeError, OSError, ValueError, sqlite3.Error):
            raise failure(503, "prediction_unavailable") from None
        LOG.info("prediction_completed id=%s duration_ms=%.3f", request_id, record["duration_ms"])
        return record

    @app.get("/v1/results/{request_id}")
    def result(request_id: UUID):
        value = app.state.results.get(str(request_id))
        if value is None:
            raise HTTPException(404, detail="result_not_found")
        return value

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("AGELENS_API_PORT", "18080")))
