"""PostgreSQL inbox; writes are committed before the MQTT acknowledgement."""

import hashlib
from contextlib import contextmanager

from services.ingest.contract import RETENTION_DAYS, Rejected, check_time, image_bytes, parse_event

SCHEMA = """
CREATE TABLE IF NOT EXISTS ingest_events (
    event_id uuid PRIMARY KEY,
    observed_at timestamptz NOT NULL,
    received_at timestamptz NOT NULL DEFAULT now(),
    source text NOT NULL,
    image_ref text NOT NULL,
    image_sha256 char(64) NOT NULL,
    payload_sha256 char(64) NOT NULL,
    media_type text NOT NULL,
    image_data bytea NOT NULL CHECK (octet_length(image_data) BETWEEN 1 AND 2097152),
    status text NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'processing', 'completed', 'failed'))
);
CREATE INDEX IF NOT EXISTS ingest_events_status_received_idx ON ingest_events (status, received_at);
CREATE TABLE IF NOT EXISTS ingest_rejections (
    payload_sha256 char(64) PRIMARY KEY,
    reason text NOT NULL,
    received_at timestamptz NOT NULL DEFAULT now(),
    occurrences integer NOT NULL DEFAULT 1
);
"""


@contextmanager
def connection():
    import psycopg

    with psycopg.connect(connect_timeout=5, options="-c statement_timeout=5000") as db:
        yield db


def initialize():
    with connection() as db:
        db.execute(SCHEMA)


def existing(db, event_id, fingerprint):
    row = db.execute(
        "SELECT payload_sha256 FROM ingest_events WHERE event_id = %s", (event_id,)
    ).fetchone()
    if row is None:
        return False
    if row[0] != fingerprint:
        raise Rejected("event_id_conflict")
    return True


def ingest(db, payload, inbox, *, allowed_delivery=True):
    try:
        if not allowed_delivery:
            raise Rejected("invalid_delivery")
        event = parse_event(payload)
        fingerprint = hashlib.sha256(event.payload()).hexdigest()
        if existing(db, event.event_id, fingerprint):
            return "duplicate", event.event_id
        check_time(event)
        image, digest, media_type = image_bytes(inbox, event.image_ref, event.image_sha256)
        row = db.execute(
            """INSERT INTO ingest_events
                (event_id, observed_at, source, image_ref, image_sha256, payload_sha256, media_type, image_data)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (event_id) DO NOTHING RETURNING event_id""",
            (
                event.event_id,
                event.observed_at,
                event.source,
                event.image_ref,
                digest,
                fingerprint,
                media_type,
                image,
            ),
        ).fetchone()
        if row is None:
            if not existing(db, event.event_id, fingerprint):
                raise RuntimeError("Concurrent insert could not be verified")
            return "duplicate", event.event_id
        return "stored", event.event_id
    except Rejected as exc:
        db.execute(
            """INSERT INTO ingest_rejections (payload_sha256, reason) VALUES (%s, %s)
                ON CONFLICT (payload_sha256) DO UPDATE
                SET occurrences = ingest_rejections.occurrences + 1""",
            (hashlib.sha256(payload).hexdigest(), str(exc)),
        )
        return "rejected", str(exc)


def purge_expired():
    with connection() as db:
        counts = {}
        for table in ("ingest_events", "ingest_rejections"):
            from psycopg import sql

            counts[table] = db.execute(
                sql.SQL(
                    "DELETE FROM {} WHERE received_at < now() - make_interval(days => %s)"
                ).format(sql.Identifier(table)),
                (RETENTION_DAYS,),
            ).rowcount
        return counts
