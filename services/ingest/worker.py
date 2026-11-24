"""Subscribe with a durable MQTT 3.1.1 session and acknowledge committed records."""

import argparse
import json
import os
from pathlib import Path

import psycopg
from paho.mqtt import client as mqtt

from services.ingest.contract import TOPIC
from services.ingest.store import connection, ingest, initialize, purge_expired

READY = Path("/tmp/agelens-ingestion-ready")


def worker():
    READY.unlink(missing_ok=True)
    initialize()
    client = mqtt.Client(
        mqtt.CallbackAPIVersion.VERSION2,
        client_id="agelens-ingestion-v1",
        clean_session=False,
        manual_ack=True,
    )

    def connected(client, _, flags, reason, properties):
        if reason.is_failure:
            raise RuntimeError("mqtt_connect_refused")
        status, _ = client.subscribe(TOPIC, qos=1)
        if status != mqtt.MQTT_ERR_SUCCESS:
            raise RuntimeError("mqtt_subscribe_failed")

    def subscribed(client, userdata, mid, codes, properties):
        if len(codes) != 1 or codes[0] != 1:
            raise RuntimeError("mqtt_qos1_subscription_refused")
        READY.touch()
        print("mqtt_subscription_ready", flush=True)

    def disconnected(client, userdata, flags, reason, properties):
        READY.unlink(missing_ok=True)

    def received(client, userdata, message):
        try:
            with connection() as db:
                state, detail = ingest(
                    db,
                    message.payload,
                    Path("/inbox"),
                    allowed_delivery=message.qos == 1 and not message.retain,
                )
        except (psycopg.Error, OSError, RuntimeError):
            # Exit without ACK; the durable session will redeliver after container restart.
            raise RuntimeError("ingestion_not_committed") from None
        print(json.dumps({"state": state, "detail": detail}), flush=True)
        if client.ack(message.mid, message.qos) != mqtt.MQTT_ERR_SUCCESS:
            raise RuntimeError("mqtt_ack_failed")

    client.on_connect = connected
    client.on_subscribe = subscribed
    client.on_disconnect = disconnected
    client.on_message = received
    try:
        client.connect(os.environ.get("MQTT_HOST", "broker"), 1883, keepalive=20)
        client.loop_forever()
    finally:
        READY.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--purge-expired", action="store_true")
    if parser.parse_args().purge_expired:
        print(json.dumps(purge_expired()))
    else:
        worker()
