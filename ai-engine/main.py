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
from models.isolation_forest import IsolationForestDetector, required_min_samples
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

# FIX 2: Warmup period — ignore all data collected during system startup.
# Docker services, image pulls, and init scripts produce artificial spikes
# that corrupt the training baseline if included.
# 600 seconds (10 minutes) is the minimum safe warmup for a full stack.
WARMUP_SECONDS = int(os.getenv("WARMUP_SECONDS", 600))

# FIX 6: Alert cooldown per family — minimum seconds between incidents
# of the same family. Prevents the same issue from firing every 60 seconds.
ALERT_COOLDOWN_SECONDS = {
    "cpu_pressure": 300,
    "memory_pressure": 300,
    "storage_pressure": 600,
    "network_pressure": 300,
    "application_pressure": 180,
    "unknown": 300,
}

# FIX 5: Consecutive anomaly confirmation per family.
# Require N consecutive anomalous cycles before creating an incident.
# This eliminates single-spike false positives entirely.
CONFIRMATION_REQUIRED = {
    "cpu_pressure": 2,
    "memory_pressure": 2,
    "storage_pressure": 3,   # disk metrics are noisy — require more confirmation
    "network_pressure": 3,
    "application_pressure": 2,
    "unknown": 2,
}

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

    # FIX 5: Track consecutive anomaly counts per family
    # Resets to 0 when a family has no anomaly in a cycle
    "consecutive_anomaly_counts": {},

    # FIX 6: Track last alert timestamp per family
    "last_family_alert_ts": {},
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
    Real incidents require z-score or hard-threshold confirmation.
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


def _apply_confirmation_window(anomalies):
    """
    FIX 5: Consecutive confirmation window.

    Require N consecutive anomalous cycles per family before firing.
    This eliminates single-spike false positives entirely.

    Logic:
    - For each family present in this cycle's anomalies → increment counter
    - For each family NOT present this cycle → reset counter to 0
    - Only pass anomalies whose family counter >= CONFIRMATION_REQUIRED
    """
    current_families = {a.get("family") for a in anomalies}

    # Increment counters for active families
    for family in current_families:
        state["consecutive_anomaly_counts"][family] = (
            state["consecutive_anomaly_counts"].get(family, 0) + 1
        )

    # Reset counters for families that had no anomaly this cycle
    for family in list(state["consecutive_anomaly_counts"]):
        if family not in current_families:
            if state["consecutive_anomaly_counts"][family] > 0:
                log.info(
                    f"Family {family} cleared — "
                    f"resetting consecutive count from "
                    f"{state['consecutive_anomaly_counts'][family]} to 0"
                )
            state["consecutive_anomaly_counts"][family] = 0

    # Filter: only pass anomalies that have hit their confirmation threshold
    confirmed = []
    for anomaly in anomalies:
        family = anomaly.get("family", "unknown")
        count = state["consecutive_anomaly_counts"].get(family, 0)
        required = CONFIRMATION_REQUIRED.get(family, 2)

        if count >= required:
            confirmed.append(anomaly)
        else:
            log.info(
                f"Confirmation pending for {family}: "
                f"{count}/{required} consecutive cycles"
            )

    return confirmed


def _apply_cooldown_filter(anomalies):
    """
    FIX 6: Alert cooldown per family.

    Suppress anomalies whose family was alerted within the cooldown window.
    This prevents the same incident from firing every 60 seconds.
    """
    now = time.time()
    passed = []

    for anomaly in anomalies:
        family = anomaly.get("family", "unknown")
        last_ts = state["last_family_alert_ts"].get(family, 0)
        cooldown = ALERT_COOLDOWN_SECONDS.get(family, 300)
        elapsed = now - last_ts

        if elapsed < cooldown:
            remaining = int(cooldown - elapsed)
            log.info(
                f"Cooldown active for {family} — "
                f"{remaining}s remaining, suppressing alert"
            )
            continue

        passed.append(anomaly)

    return passed


def _update_cooldown_timestamps(anomalies):
    """Update last alert timestamps for families that passed cooldown."""
    now = time.time()
    for anomaly in anomalies:
        family = anomaly.get("family", "unknown")
        state["last_family_alert_ts"][family] = now


def analysis_loop():
    while True:
        try:
            # FIX 2: Warmup period — skip data collection and training
            # during the initial startup window to avoid boot artifacts.
            elapsed_since_start = time.time() - ENGINE_START_TS
            if elapsed_since_start < WARMUP_SECONDS:
                remaining = int(WARMUP_SECONDS - elapsed_since_start)
                log.info(
                    f"Warmup period active — {remaining}s remaining. "
                    "Skipping to avoid boot artifact contamination."
                )
                time.sleep(SCRAPE_INTERVAL)
                continue

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

            # FIX 8: Check per-metric minimum sample requirement
            min_required = required_min_samples(names)

            if not detector.trained:
                if len(X) < min_required:
                    log.info(
                        f"Collecting samples: {len(X)}/{min_required} "
                        f"(per-metric minimum for present features)"
                    )
                    time.sleep(SCRAPE_INTERVAL)
                    continue

                # FIX 1: train() now validates CV and returns True/False
                success = detector.train(X, names)

                if not success:
                    log.warning(
                        "Training rejected due to poor baseline quality. "
                        f"Collecting more samples and retrying in {SCRAPE_INTERVAL}s."
                    )
                    time.sleep(SCRAPE_INTERVAL)
                    continue

                state["last_trained_at"] = detector.saved_at
                state["model_loaded_from_disk"] = False
                log.info(
                    f"Initial training complete on {len(X)} samples. "
                    "Detection is now active."
                )
                time.sleep(SCRAPE_INTERVAL)
                continue

            raw = detector.predict(X, names, timestamps, matrix.families)
            latest_compact = _dedup_latest(raw)
            filtered = _filter_if_only_anomalies(latest_compact)

            # FIX 5: Apply consecutive confirmation window
            confirmed = _apply_confirmation_window(filtered)

            # FIX 6: Apply cooldown filter
            cooldown_passed = _apply_cooldown_filter(confirmed)

            new_anomalies = []
            new_keys = set()

            for anomaly in cooldown_passed:
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
                f"confirmed={len(confirmed)} "
                f"cooldown_passed={len(cooldown_passed)} "
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

            # Update cooldown timestamps for families that are firing
            _update_cooldown_timestamps(new_anomalies)

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


app = FastAPI(title="AI Monitoring Engine", version="6.0.0")


@app.on_event("startup")
def startup():
    wait_for_db()
    Thread(target=heartbeat_loop, daemon=True).start()
    Thread(target=analysis_loop, daemon=True).start()
    log.info(
        f"AI engine v6.0.0 started. "
        f"Warmup: {WARMUP_SECONDS}s. "
        f"Detection active after warmup + {required_min_samples(['cpu_pct','mem_pct','disk_fill_pct','disk_io','net_traffic','http_requests'])} samples."
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_trained": detector.trained,
        "baseline_quality_passed": detector.baseline_quality_passed,
        "cycle_count": state["cycle_count"],
        "warmup_remaining": max(0, int(WARMUP_SECONDS - (time.time() - ENGINE_START_TS))),
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
        "baseline_quality_passed": detector.baseline_quality_passed,
        "last_trained_at": state["last_trained_at"],
        "last_cycle_at": state["last_cycle_at"],
        "cycle_count": state["cycle_count"],
        "baselines": detector.baselines,
        "scrape_interval": SCRAPE_INTERVAL,
        "instance": MONITORED_INSTANCE,
        "model_loaded_from_disk": state["model_loaded_from_disk"],
        "engine_start_ts": ENGINE_START_TS,
        "warmup_seconds": WARMUP_SECONDS,
        "warmup_remaining": max(0, int(WARMUP_SECONDS - (time.time() - ENGINE_START_TS))),
        "consecutive_anomaly_counts": state["consecutive_anomaly_counts"],
        "last_family_alert_ts": {
            k: datetime.utcfromtimestamp(v).isoformat()
            for k, v in state["last_family_alert_ts"].items()
        },
    }
