"""Durable event leases and HTTP dispatch; no image bytes in Airflow XCom or logs."""

import hashlib
import json
import math
import os
import re
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener
from uuid import uuid4

ATTEMPTS = 3
LEASE_SECONDS = 120
MEDIA_TYPES = {"image/jpeg", "image/png"}
THRESHOLDS = {"12", "13", "15", "16", "18", "21"}
SCHEMA = """
ALTER TABLE ingest_events ADD COLUMN IF NOT EXISTS inference_attempts integer NOT NULL DEFAULT 0;
ALTER TABLE ingest_events ADD COLUMN IF NOT EXISTS inference_retry_at timestamptz;
ALTER TABLE ingest_events ADD COLUMN IF NOT EXISTS inference_lease_until timestamptz;
ALTER TABLE ingest_events ADD COLUMN IF NOT EXISTS inference_claim uuid;
ALTER TABLE ingest_events ADD COLUMN IF NOT EXISTS inference_error text;
CREATE TABLE IF NOT EXISTS inference_results (
    event_id uuid PRIMARY KEY REFERENCES ingest_events (event_id) ON DELETE CASCADE,
    model_version text NOT NULL,
    model_run_id text NOT NULL,
    age_estimate_years double precision NOT NULL,
    duration_ms double precision NOT NULL,
    decisions jsonb NOT NULL,
    processed_at timestamptz NOT NULL DEFAULT now()
);
"""


class InvalidResponse(ValueError):
    pass


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def connect():
    import psycopg

    return psycopg.connect(connect_timeout=5, options="-c statement_timeout=8000")


def claim(db, token):
    db.execute(
        """UPDATE ingest_events SET status = 'failed', inference_error = 'lease_expired',
            inference_claim = NULL, inference_lease_until = NULL
            WHERE status = 'processing' AND inference_lease_until < now()
              AND inference_attempts >= %s""",
        (ATTEMPTS,),
    )
    return db.execute(
        """WITH candidate AS (
            SELECT event_id FROM ingest_events
            WHERE inference_attempts < %s AND (
                (status = 'pending' AND (inference_retry_at IS NULL OR inference_retry_at <= now()))
                OR (status = 'processing' AND inference_lease_until < now())
            )
            ORDER BY received_at, event_id LIMIT 1 FOR UPDATE SKIP LOCKED
        )
        UPDATE ingest_events AS e SET status = 'processing', inference_attempts = inference_attempts + 1,
            inference_claim = %s, inference_lease_until = now() + make_interval(secs => %s)
        FROM candidate WHERE e.event_id = candidate.event_id
        RETURNING e.event_id, e.media_type, e.image_data, e.image_sha256, e.inference_attempts""",
        (ATTEMPTS, token, LEASE_SECONDS),
    ).fetchone()


def send_prediction(origin, event_id, media, data, digest):
    if media not in MEDIA_TYPES or not 0 < len(data) <= 2 * 1024**2:
        raise InvalidResponse("stored_image_invalid")
    if hashlib.sha256(data).hexdigest() != digest:
        raise InvalidResponse("stored_image_hash_mismatch")
    request = Request(
        origin + "/v1/predict",
        data=data,
        headers={"Content-Type": media, "X-Event-ID": str(event_id)},
        method="POST",
    )
    with build_opener(ProxyHandler({}), NoRedirect()).open(request, timeout=25) as response:
        content = response.read(32769)
    if len(content) > 32768:
        raise InvalidResponse("invalid_api_response")
    try:
        result = json.loads(content)
        model = result["model"]
        points = result["thresholds"]
        age = result["age_estimate_years"]
        duration = result["duration_ms"]
        if (
            result["id"] != str(event_id)
            or result["status"] != "completed"
            or not re.fullmatch(r"[a-z0-9-]{1,64}", model["version"])
            or not re.fullmatch(r"[0-9a-f]{32}", model["mlflow_run_id"])
            or type(age) not in {int, float}
            or not 0 <= age <= 101
            or type(duration) not in {int, float}
            or not math.isfinite(duration)
            or duration < 0
            or set(points) != THRESHOLDS
        ):
            raise ValueError
        for point in points.values():
            score, cutoff, minor = point["score"], point["cutoff"], point["estimated_under_age"]
            if (
                not math.isfinite(score)
                or not math.isfinite(cutoff)
                or score != -age
                or type(minor) is not bool
                or minor != (score >= cutoff)
            ):
                raise ValueError
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        raise InvalidResponse("invalid_api_response") from exc
    return result


def complete(db, event_id, token, result):
    model = result["model"]
    updated = db.execute(
        """UPDATE ingest_events SET status = 'completed', inference_error = NULL,
            inference_claim = NULL, inference_lease_until = NULL, inference_retry_at = NULL
            WHERE event_id = %s AND status = 'processing' AND inference_claim = %s""",
        (event_id, token),
    ).rowcount
    if not updated:
        return False
    db.execute(
        """INSERT INTO inference_results
            (event_id, model_version, model_run_id, age_estimate_years, duration_ms, decisions)
            VALUES (%s, %s, %s, %s, %s, %s::jsonb)
            ON CONFLICT (event_id) DO NOTHING""",
        (
            event_id,
            model["version"],
            model["mlflow_run_id"],
            result["age_estimate_years"],
            result["duration_ms"],
            json.dumps(result["thresholds"]),
        ),
    )
    return True


def fail(db, event_id, token, attempt, code, *, permanent):
    status = "failed" if permanent or attempt >= ATTEMPTS else "pending"
    delay = 15 * 2 ** (attempt - 1)
    updated = db.execute(
        """UPDATE ingest_events SET status = %s, inference_error = %s,
            inference_claim = NULL, inference_lease_until = NULL,
            inference_retry_at = CASE WHEN %s = 'pending' THEN now() + make_interval(secs => %s) END
            WHERE event_id = %s AND status = 'processing' AND inference_claim = %s""",
        (status, code, status, delay, event_id, token),
    ).rowcount
    return status if updated else "stale_claim"


def process_pending(limit=12):
    origin = os.getenv("AGELENS_INFERENCE_URL", "http://host.docker.internal:18080")
    if not re.fullmatch(r"http://host\.docker\.internal:[0-9]{1,5}", origin):
        raise ValueError("Refuse to send images outside the local API")
    with connect() as db:
        db.execute(SCHEMA)
    counts = {"completed": 0, "pending": 0, "failed": 0, "stale_claim": 0}
    for _ in range(limit):
        token = uuid4()
        with connect() as db:
            row = claim(db, token)
        if row is None:
            break
        event_id, media, data, digest, attempt = row
        result = None
        try:
            result = send_prediction(origin, event_id, media, data, digest)
        except HTTPError as exc:
            error, permanent = f"api_http_{exc.code}", exc.code in {400, 409, 413, 415, 422}
            exc.close()
        except (URLError, OSError):
            error, permanent = "api_unavailable", False
        except InvalidResponse as exc:
            error, permanent = str(exc), True
        with connect() as db:
            if result is None:
                state = fail(db, event_id, token, attempt, error, permanent=permanent)
            else:
                state = "completed" if complete(db, event_id, token, result) else "stale_claim"
        counts[state] += 1
    print("Inference batch:", counts)
    return counts
