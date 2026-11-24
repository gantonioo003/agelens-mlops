import hashlib
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from PIL import Image

from services.ingest.contract import (
    MAX_IMAGE,
    Event,
    Rejected,
    check_time,
    image_bytes,
    parse_event,
)

BASE = Event(1, str(uuid4()), "2026-11-24T12:00:00Z", "sim-demo", "sample.png", "a" * 64)


def test_payload_roundtrip_and_operational_time_window():
    assert parse_event(BASE.payload()) == BASE
    now = datetime.fromisoformat(BASE.observed_at).astimezone(UTC)
    check_time(BASE, now)
    with pytest.raises(Rejected, match="timestamp_out_of_window"):
        check_time(BASE, now + timedelta(days=8))
    with pytest.raises(Rejected, match="timestamp_out_of_window"):
        check_time(BASE, now - timedelta(minutes=6))


@pytest.mark.parametrize(
    "updates",
    [
        {"image_ref": "../sample.png"},
        {"image_ref": "/image.jpg"},
        {"image_ref": "sub/image.jpg"},
        {"version": True},
        {"source": "person@example.test"},
        {"observed_at": "2026-11-24T12:00:00"},
        {"image_sha256": "not-a-sha"},
        {"event_id": "no-uuid"},
    ],
)
def test_invalid_contract_fields_are_rejected_without_echoing_input(updates):
    with pytest.raises(Rejected, match="^invalid_contract$"):
        parse_event(replace(BASE, **updates).payload())


def test_bad_json_duplicate_keys_and_message_size():
    for payload in (b'{"version":1,"version":1}', b"{", b'["not-object"]', b"[" * 1000):
        with pytest.raises(Rejected, match="invalid_contract"):
            parse_event(payload)
    with pytest.raises(Rejected, match="message_too_large"):
        parse_event(b"x" * 2049)


def test_image_readback_and_rejection(tmp_path):
    path = tmp_path / "sample.png"
    Image.new("RGB", (16, 12)).save(path)
    actual = path.read_bytes()
    assert image_bytes(tmp_path, path.name) == (
        actual,
        hashlib.sha256(actual).hexdigest(),
        "image/png",
    )
    with pytest.raises(Rejected, match="image_checksum_mismatch"):
        image_bytes(tmp_path, path.name, "0" * 64)
    with pytest.raises(Rejected, match="image_unavailable"):
        image_bytes(tmp_path, "missing.png")
    path.write_bytes(b"x" * (MAX_IMAGE + 1))
    with pytest.raises(Rejected, match="invalid_image_size"):
        image_bytes(tmp_path, path.name)
    path.write_bytes(b"not an image")
    with pytest.raises(Rejected, match="^invalid_image$"):
        image_bytes(tmp_path, path.name)
