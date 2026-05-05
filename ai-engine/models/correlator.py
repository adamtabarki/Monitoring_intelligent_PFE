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
    format="%(asctime)s [%(name)s] %(message)s",
)
log = logging.getLogger("ai-engine")

PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://prometheus:9090")
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://monitor:monitor123@postgres:5432/monitoring",
)
SCRAPE_INTERVAL = int(os.getenv("SCRAPE_INTERVAL", 60))
CONTAMINATION = float(os.getenv("ANOMALY_CONTAMINATION", 0.03))
HEARTBEAT_URL = os.getenv("HEARTBEAT_URL", "")
MONITORED_INSTANCE = os.getenv("MONITORED_INSTANCE", "node-server")

# Prevent training on old Prometheus history after an AI engine restart.
# The model will use only samples collected after this process started.
ENGINE_START_TS = time.time()

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
}


def anomaly_key(anomaly):
    payload = {
        "family": anomaly.get("family"),
        "top_feature": anomaly.get("top_feature"),
        "severity": anomaly.get("severity"),
        "reason": anomaly.get("reason"),
        "trigger_sources": anomaly.get("trigger_sources", []),
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
                    :instance,
                    :cpu_pct,
                    :mem_pct,
                    :disk_fill_pct,
                    :disk_io,
                    :net_traffic,
                    :http_requests,
                    :score,
                    :severity,
                    'new',
                    :family,
                    :top_feature,
                    :technical_reason
                )
                RETURNING id
            """), dict(
                instance=instance,
                cpu_pct=features.get("cpu_pct"),
                mem_pct=features.get("mem_pct"),
                disk_fill_pct=features.get("disk_fill_pct"),
                disk_io=features.get("disk_io"),
                net_traffic=features.get("net_traffic"),
                http_requests=features.get("http_requests"),
                score=anomaly.get("score"),
                severity=anomaly.get("severity"),
                family=anomaly.get("family"),
                top_feature=anomaly.get("top_feature"),
                technical_reason=technical_reason,
            ))

            anomaly_id = row.fetchone()[0]
            return anomaly_id

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
    Keep only the strongest anomaly per family for the latest timestamp.
    This avoids creating many rows for the same incident cycle.
    """
    if not raw_anomalies:
        return []

    latest_ts = max(anomaly["timestamp"] for anomaly in raw_anomalies)
    latest_only = [
        anomaly
        for anomaly in raw_anomalies
        if anomaly["timestamp"] == latest_ts
    ]

    severity_rank = {
        "low": 1,
        "medium": 2,
        "high": 3,
        "critical": 4,
    }

    by_family = {}

    for anomaly in latest_only:
        family = anomaly.get("family", "unknown")
        current = by_family.get(family)

        if current is None:
            by_family[family] = anomaly
            continue

        current_rank = severity_rank.get(current.get("severity"), 0)
        new_rank = severity_rank.get(anomaly.get("severity"), 0)

        if new_rank > current_rank:
            by_family[family] = anomaly
        elif new_rank == current_rank:
            if anomaly.get("score", 0) < current.get("score", 0):
                by_family[family] = anomaly

    return list(by_family.values())


def _filter_if_only_anomalies(anomalies):
    """
    Isolation Forest alone is treated as a weak exploratory signal.
    In the final system, real incidents require z-score or hard-threshold
    confirmation. This prevents false CPU/storage incidents from IF-only scores.
    """
    filtered = []

    for anomaly in anomalies:
        sources = set(anomaly.get("trigger_sources", []))

        if sources == {"iforest"}:
            log.info(
                "Dropped IF-only anomaly: "
                f"feature={anomaly.get('top_feature')} "
                f"family={anomaly.get('family')} "
                f"score={anomaly.get('score')}"
            )
            continue

        filtered.append(anomaly)

    return filtered


def analysis_loop():
    while True:
        try:
            matrix = fetch_feature_matrix(
                PROMETHEUS_URL,
                instance=MONITORED_INSTANCE,
                min_timestamp=ENGINE_START_TS,
            )

            if matrix is None:
                log.info("Waiting for enough data...")
                time.sleep(SCRAPE_INTERVAL)
                continue

            X = matrix.values
            names = matrix.feature_names
            timestamps = matrix.timestamps

            if not detector.trained:
                detector.train(X, names)
                state["last_trained_at"] = detector.saved_at
                state["model_loaded_from_disk"] = False
                log.info("Initial training complete.")
                time.sleep(SCRAPE_INTERVAL)
                continue

            raw = detector.predict(X, names, timestamps, matrix.families)
            latest_compact = _dedup_latest(raw)
            filtered = _filter_if_only_anomalies(latest_compact)

            new_anomalies = []
            new_keys = set()

            for anomaly in filtered:
                key = anomaly_key(anomaly)

                if (
                    anomaly["timestamp"] > state["last_alert_ts"]
                    or key not in state["last_alert_keys"]
                ):
                    new_anomalies.append(anomaly)
                    new_keys.add(key)

            log.info(
                "Cycle stats | "
                f"samples={len(X)} "
                f"raw={len(raw)} "
                f"latest_compact={len(latest_compact)} "
                f"filtered={len(filtered)} "
                f"new={len(new_anomalies)}"
            )

            if not new_anomalies:
                log.info(f"Normal. {latest_values(matrix)}")
                state["last_anomalies"] = []
                state["last_incidents"] = []
                state["last_cycle_at"] = datetime.utcnow().isoformat()
                state["cycle_count"] += 1
                time.sleep(SCRAPE_INTERVAL)
                continue

            state["last_alert_ts"] = max(
                anomaly["timestamp"]
                for anomaly in new_anomalies
            )
            state["last_alert_keys"] = new_keys

            for anomaly in new_anomalies:
                db_id = save_anomaly(matrix.instance, anomaly)
                anomaly["db_id"] = db_id
                anomaly["instance"] = matrix.instance

            incidents = correlate(new_anomalies)

            state["last_anomalies"] = new_anomalies
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


app = FastAPI(title="AI Monitoring Engine", version="5.2.0")


@app.on_event("startup")
def startup():
    wait_for_db()
    Thread(target=heartbeat_loop, daemon=True).start()
    Thread(target=analysis_loop, daemon=True).start()
    log.info("AI engine v5.2.0 started.")


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
        "engine_start_ts": ENGINE_START_TS,
    }
