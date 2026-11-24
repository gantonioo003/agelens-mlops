"""Publish a staged image reference from a simulated device."""

import argparse
import os
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from paho.mqtt import client as mqtt

from services.ingest.contract import TOPIC, Event, image_bytes, parse_event


def publish(payload, host, port=1883):
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.connect(host, port, keepalive=20)
    client.loop_start()
    try:
        info = client.publish(TOPIC, payload=payload, qos=1, retain=False)
        info.wait_for_publish(timeout=10)
        if not info.is_published():
            raise RuntimeError("No MQTT publication acknowledgement")
    finally:
        client.disconnect()
        client.loop_stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image", required=True, help="Basename of an authorized file staged in the inbox"
    )
    parser.add_argument("--source", default="sim-device-01")
    args = parser.parse_args()
    _, digest, _ = image_bytes(Path("/inbox"), args.image)
    event = Event(
        1,
        str(uuid4()),
        datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        args.source,
        args.image,
        digest,
    )
    payload = event.payload()
    parse_event(payload)
    publish(payload, os.environ.get("MQTT_HOST", "broker"))
    print(payload.decode("utf-8"))


if __name__ == "__main__":
    main()
