import os
import time
import json
import hashlib
import logging
import requests
from datetime import datetime
from threading import Thread

from fastapi import FastAPI
from sqlalchemy import create_engine, text

from features import fetch_feature_matrix, latest_values
from models.isolation_forest import IsolationForestDetector
from models.correlator import correlate, get_open_incidents

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s"
)
log = logging.getLogger("ai-engine")

PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://prometheus:9090")
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://monitor:monitor123@postgres:5432/monitoring"
)
SCRAPE_INTERVAL = int(os.getenv("SCRAPE_INTERVAL", 60))
CONTAMINATION = float(os.getenv("ANOMALY_CONTAMINATION", 0.03))
HEARTBEAT_URL = os.getenv("HEARTBEAT_URL", "")
MONITORED_INSTANCE = os.getenv("MONITORED_INSTANCE", "node-server")

# Medium anomalies triggered only by Isolation Forest are considered weak signals.
# They must repeat before being accepted.
IF_ONLY_REPEAT_REQUIRED = int(os.getenv("IF_ONLY_REPEAT_REQUIRED", 2))

db_engine = create_engine(DATABASE_URL, pool_pre_ping=True)

detector = IsolationForestDetector(contamination=CONTAMINATION)
model_loaded = detector.load()

state = {
    "last_anomalies": [],
    "last_incidents": [],
    "last_cycle_at": None,
    "last_trained_at": detector.saved_at,
    "cycle_count": 0,
    "last_alert_ts": 0.0,
    "last_alert_keys": set(),
    "model_loaded_from_disk": model_loaded,
    "pending_if_only": {},
}


def anomaly_key(anomaly):
    payload = {
        "family": anomaly.get("family"),
        "top_feature": anomaly.get("top_feature"),
        "severity": anomaly.get("severity"),
        "reason": anomaly.get("reason"),
    }
    raw = json.dumps(payload, sort_keys=True)
    return hashlib.md5(raw.encode()).hexdigest()


def wait_for_db(retries=10, delay=5):
    for attempt in range(retries):
        try:
            with db_engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            log.info("PostgreSQL ready.")
            return
        except Exception as e:
            log.warning(f"DB not ready ({attempt + 1}/{retries}): {e}")
            time.sleep(delay)

    raise RuntimeError("PostgreSQL did not become ready.")


def save_anomaly(instance, anomaly):
    features = anomaly.get("features", {})
    trigger_sources = anomaly.get("trigger_sources", [])

    technical_reason = anomaly.get("reason")
    if trigger_sources:
        technical_reason = f"[{','.join(trigger_sources)}] {technical_reason}"

    try:
        with db_engine.begin() as conn:
            row = conn.execute(text("""
                INSERT INTO anomalies (
                    instance,
                    cpu_pct,
                    mem_pct,
                    disk_fill_pct,
                    disk_io,
                    net_traffic,
                    http_requests,
                    score,
                    severity,
                    status,
                    family,
                    top_feature,
                    technical_reason
                )
                VALUES (
                    :inst,
                    :cpu,
                    :mem,
                    :disk_fill,
                    :disk_io,
                    :net_traffic,
                    :http_requests,
                    :score,
                    :sev,
                    'new',
                    :family,
                    :top_feature,
                    :technical_reason
                )
                RETURNING id
            """), dict(
                inst=instance,
                cpu=features.get("cpu_pct"),
                mem=features.get("mem_pct"),
                disk_fill=features.get("disk_fill_pct"),
                disk_io=features.get("disk_io"),
                net_traffic=features.get("net_traffic"),
                http_requests=features.get("http_requests"),
                score=anomaly.get("score"),
                sev=anomaly.get("severity"),
                family=anomaly.get("family"),
                top_feature=anomaly.get("top_feature"),
                technical_reason=technical_reason,
            ))

            return row.fetchone()[0]

    except Exception as e:
        log.error(f"DB anomaly write failed: {e}", exc_info=True)
        return None


def heartbeat_loop():
    if not HEARTBEAT_URL:
        log.info("HEARTBEAT_URL not set — skipping.")
        return

    while True:
        try:
            response = requests.get(HEARTBEAT_URL, timeout=10)
            status = "OK" if response.status_code == 200 else "FAILED"
            log.info(f"Heartbeat {status}")
        except Exception as e:
            log.error(f"Heartbeat error: {e}", exc_info=True)

        time.sleep(60)


def _dedup_latest(raw_anomalies):
    """
    Keep only the strongest anomaly per family in the latest timestamp bucket.
    This reduces noise in the anomaly table and helps correlation.
    """
    if not raw_anomalies:
        return []

    latest_ts = max(a["timestamp"] for a in raw_anomalies)
    latest_only = [a for a in raw_anomalies if a["timestamp"] == latest_ts]

    by_family = {}
    severity_rank = {"low": 1, "medium": 2, "high": 3, "critical": 4}

    for anomaly in latest_only:
        family = anomaly.get("family", "unknown")
        current = by_family.get(family)

        current_rank = severity_rank.get(current.get("severity"), 0) if current else 0
        new_rank = severity_rank.get(anomaly.get("severity"), 0)

        if current is None:
            by_family[family] = anomaly
        elif new_rank > current_rank:
            by_family[family] = anomaly
        elif new_rank == current_rank and anomaly.get("score", 0) < current.get("score", 0):
            by_family[family] = anomaly

    return list(by_family.values())


def _filter_if_only_medium(anomalies):
    """
    IF-only medium anomalies are weak signals.
    Keep them only if they repeat for IF_ONLY_REPEAT_REQUIRED cycles.
    """
    filtered = []
    new_pending = {}

    for anomaly in anomalies:
        sources = set(anomaly.get("trigger_sources", []))
        key = anomaly_key(anomaly)

        is_if_only_medium = (
            anomaly.get("severity") == "medium" and
            sources == {"iforest"}
        )

        if not is_if_only_medium:
            filtered.append(anomaly)
            continue

        count = state["pending_if_only"].get(key, 0) + 1

        if count >= IF_ONLY_REPEAT_REQUIRED:
            filtered.append(anomaly)
            new_pending[key] = 0
        else:
            new_pending[key] = count

    state["pending_if_only"] = new_pending
    return filtered


def analysis_loop():
    while True:
        try:
            matrix = fetch_feature_matrix(
                PROMETHEUS_URL,
                instance=MONITORED_INSTANCE
            )

            if matrix is None:
                log.info("Waiting for enough data...")
                time.sleep(SCRAPE_INTERVAL)
                continue

            X, names, timestamps = (
                matrix.values,
                matrix.feature_names,
                matrix.timestamps
            )

            if not detector.trained:
                detector.train(X, names)
                state["last_trained_at"] = detector.saved_at
                state["model_loaded_from_disk"] = False
                log.info("Initial training complete.")
                time.sleep(SCRAPE_INTERVAL)
                continue

            raw = detector.predict(X, names, timestamps, matrix.families)
            latest_compact = _dedup_latest(raw)
            filtered = _filter_if_only_medium(latest_compact)

            new = []
            new_keys = set()

            for anomaly in filtered:
                key = anomaly_key(anomaly)

                if (
                    anomaly["timestamp"] > state["last_alert_ts"]
                    or key not in state["last_alert_keys"]
                ):
                    new.append(anomaly)
                    new_keys.add(key)

            log.info(
                "Cycle stats | "
                f"samples={len(X)} raw={len(raw)} "
                f"latest_compact={len(latest_compact)} "
                f"filtered={len(filtered)} new={len(new)}"
            )

            if not new:
                log.info(f"Normal. {latest_values(matrix)}")
                state["last_anomalies"] = []
                state["last_incidents"] = []
                state["last_cycle_at"] = datetime.utcnow().isoformat()
                state["cycle_count"] += 1
                time.sleep(SCRAPE_INTERVAL)
                continue

            state["last_alert_ts"] = max(a["timestamp"] for a in new)
            state["last_alert_keys"] = new_keys

            for anomaly in new:
                db_id = save_anomaly(matrix.instance, anomaly)
                anomaly["db_id"] = db_id
                anomaly["instance"] = matrix.instance

            incidents = correlate(new)

            state["last_anomalies"] = new
            state["last_incidents"] = incidents

            for incident in incidents:
                log.warning(
                    f'INCIDENT [{incident["severity"].upper()}] '
                    f'{incident["root_cause"]} — '
                    f'{incident["anomaly_count"]} signal(s) | '
                    f'{incident["reason"]}'
                )

            state["last_cycle_at"] = datetime.utcnow().isoformat()
            state["cycle_count"] += 1

        except Exception as e:
            log.error(f"Analysis loop error: {e}", exc_info=True)

        time.sleep(SCRAPE_INTERVAL)


app = FastAPI(title="AI Monitoring Engine", version="5.1.0")


@app.on_event("startup")
def startup():
    wait_for_db()
    Thread(target=heartbeat_loop, daemon=True).start()
    Thread(target=analysis_loop, daemon=True).start()
    log.info("AI engine v5.1.0 started.")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_trained": detector.trained,
        "cycle_count": state["cycle_count"],
    }


@app.get("/anomalies")
def get_anomalies():
    return {
        "count": len(state["last_anomalies"]),
        "anomalies": state["last_anomalies"],
    }


@app.get("/incidents")
def get_incidents():
    return {
        "open_incidents": get_open_incidents(),
        "last_cycle": state["last_incidents"],
    }


@app.get("/alerts")
def get_alerts():
    return {
        "anomalies": state["last_anomalies"],
        "incidents": state["last_incidents"],
        "total": len(state["last_anomalies"]),
        "cycle": state["cycle_count"],
        "last_cycle": state["last_cycle_at"],
    }


@app.get("/status")
def status():
    return {
        "model_trained": detector.trained,
        "last_trained_at": state["last_trained_at"],
        "last_cycle_at": state["last_cycle_at"],
        "cycle_count": state["cycle_count"],
        "baselines": detector.baselines,
        "scrape_interval": SCRAPE_INTERVAL,
        "instance": MONITORED_INSTANCE,
        "model_loaded_from_disk": state["model_loaded_from_disk"],
    }
