"""Bounded image-reference messages for the local MQTT prototype."""

import hashlib
import io
import json
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from uuid import UUID

from PIL import Image

TOPIC = "agelens/v1/images"
MAX_MESSAGE = 2048
MAX_IMAGE = 2 * 1024 * 1024
MAX_EDGE = 2048
RETENTION_DAYS = 7
NAME = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,100}\.(?:jpg|jpeg|png)")
SOURCE = re.compile(r"sim-[a-z0-9][a-z0-9-]{0,31}")
SHA = re.compile(r"[0-9a-f]{64}")


class Rejected(ValueError):
    """The exception text is a fixed reason code, never message content."""


@dataclass(frozen=True)
class Event:
    version: int
    event_id: str
    observed_at: str
    source: str
    image_ref: str
    image_sha256: str

    def payload(self):
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()


def unique_object(pairs):
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("Duplicate JSON key")
    return result


def parse_event(payload):
    if len(payload) > MAX_MESSAGE:
        raise Rejected("message_too_large")
    try:
        values = json.loads(payload.decode("utf-8"), object_pairs_hook=unique_object)
        event = Event(**values)
        if type(event.version) is not int or event.version != 1:
            raise ValueError
        if not all(isinstance(v, str) for k, v in values.items() if k != "version"):
            raise ValueError
        parsed = UUID(event.event_id)
        if parsed.version != 4 or str(parsed) != event.event_id:
            raise ValueError
        date = datetime.strptime(event.observed_at, "%Y-%m-%dT%H:%M:%SZ")
        if date.strftime("%Y-%m-%dT%H:%M:%SZ") != event.observed_at:
            raise ValueError
        if not all(
            pattern.fullmatch(value)
            for pattern, value in (
                (SOURCE, event.source),
                (NAME, event.image_ref),
                (SHA, event.image_sha256),
            )
        ):
            raise ValueError
    except (ValueError, TypeError, AttributeError, RecursionError) as exc:
        raise Rejected("invalid_contract") from exc
    return event


def check_time(event, now=None):
    now = now or datetime.now(UTC)
    observed = datetime.fromisoformat(event.observed_at)
    if observed > now + timedelta(minutes=5) or observed < now - timedelta(days=RETENTION_DAYS):
        raise Rejected("timestamp_out_of_window")


def image_bytes(inbox, name, sha256=None):
    if not NAME.fullmatch(name):
        raise Rejected("invalid_image_reference")
    root = inbox.resolve()
    path = root / name
    if path.is_symlink() or path.resolve().parent != root:
        raise Rejected("invalid_image_reference")
    try:
        with path.open("rb") as stream:
            data = stream.read(MAX_IMAGE + 1)
    except (FileNotFoundError, IsADirectoryError) as exc:
        raise Rejected("image_unavailable") from exc
    if not data or len(data) > MAX_IMAGE:
        raise Rejected("invalid_image_size")
    digest = hashlib.sha256(data).hexdigest()
    if sha256 is not None and digest != sha256:
        raise Rejected("image_checksum_mismatch")
    expected = "PNG" if name.endswith(".png") else "JPEG"
    try:
        with Image.open(io.BytesIO(data), formats=[expected]) as image:
            if max(image.size) > MAX_EDGE or getattr(image, "n_frames", 1) != 1:
                raise Rejected("invalid_image_dimensions")
            image.verify()
        with Image.open(io.BytesIO(data), formats=[expected]) as image:
            image.load()
    except (OSError, SyntaxError, Image.DecompressionBombError) as exc:
        raise Rejected("invalid_image") from exc
    return data, digest, f"image/{expected.lower()}"
