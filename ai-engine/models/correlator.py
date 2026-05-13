import logging
import os
from datetime import datetime

from sqlalchemy import create_engine, text

log = logging.getLogger("correlator")

engine = create_engine(
    os.getenv(
        "DATABASE_URL",
        "postgresql://monitor:monitor123@postgres:5432/monitoring",
    ),
    pool_pre_ping=True,
)

ROOT_CAUSES = {
    "cpu_pressure": "CPU overload",
    "memory_pressure": "Memory exhaustion",
    "storage_pressure": "Storage pressure",
    "network_pressure": "Network saturation",
    "application_pressure": "Application anomaly",
    "unknown": "System anomaly",
}

SEVERITY_RANK = {
    "low": 1,
    "medium": 2,
    "high": 3,
    "critical": 4,
}

RANK_SEVERITY = {
    value: key
    for key, value in SEVERITY_RANK.items()
}

OPEN_STATUSES = (
    "new",
    "processing",
    "alerted",
    "in_progress",
    "manual_required",
)

# CHANGE 1: Per-family incident reuse window in minutes.
# Previously all families used a flat 20-minute window.
# Storage pressure changes slowly — reuse window is longer to avoid duplicates.
# Application anomalies are fast-changing — shorter reuse window.
# These are aligned with the cooldown windows in main.py.
INCIDENT_REUSE_MINUTES = {
    "CPU overload": 30,
    "Memory exhaustion": 30,
    "Storage pressure": 60,       # disk changes slowly, long reuse window
    "Network saturation": 45,
    "Application anomaly": 20,
    "System anomaly": 30,
}

DEFAULT_REUSE_MINUTES = 30


def correlate(anomalies):
    if not anomalies:
        return []

    groups = {}

    for anomaly in anomalies:
        family = anomaly.get("family", "unknown")
        groups.setdefault(family, []).append(anomaly)

    incidents = []

    for family, group in groups.items():
        worst_rank = max(
            SEVERITY_RANK.get(anomaly.get("severity", "low"), 1)
            for anomaly in group
        )

        severity = RANK_SEVERITY.get(worst_rank, "low")
        root_cause = ROOT_CAUSES.get(
            family,
            f"Anomaly in {family}",
        )

        seen_reasons = set()
        reasons = []

        for anomaly in group:
            reason = anomaly.get("reason", "")

            if reason and reason not in seen_reasons:
                seen_reasons.add(reason)
                reasons.append(reason)

        technical_reason = " | ".join(reasons[:3])
        title = f"[{severity.upper()}] {root_cause} — {len(group)} signal(s)"

        # CHANGE 2: Aggregate max confidence from the anomaly group
        max_confidence = max(
            anomaly.get("confidence", 50)
            for anomaly in group
        )

        try:
            # CHANGE 1: Use per-family reuse window
            reuse_minutes = INCIDENT_REUSE_MINUTES.get(root_cause, DEFAULT_REUSE_MINUTES)
            open_incident = _find_matching_open_incident(root_cause, reuse_minutes)

            if open_incident:
                incident_id = open_incident["id"]

                _update_incident_if_needed(
                    incident_id=incident_id,
                    current_severity=open_incident["severity"],
                    new_severity=severity,
                    title=title,
                    technical_reason=technical_reason,
                    confidence=max_confidence,
                )
            else:
                incident_id = _save(
                    title=title,
                    severity=severity,
                    root_cause=root_cause,
                    technical_reason=technical_reason,
                    confidence=max_confidence,
                )

            if not incident_id:
                log.warning(
                    f"Skipping correlation for family={family}; "
                    "incident id is missing."
                )
                continue

            _link_anomalies(incident_id, group)

            incidents.append({
                "incident_id": incident_id,
                "title": title,
                "severity": severity,
                "root_cause": root_cause,
                "reason": technical_reason,
                "anomaly_count": len(group),
                "family": family,
                "confidence": max_confidence,
                "created_at": datetime.utcnow().isoformat(),
            })

            log.warning(
                f"INCIDENT [{severity.upper()}] "
                f"{root_cause} — {len(group)} anomaly(ies) | "
                f"confidence={max_confidence}"
            )

        except Exception as e:
            log.error(
                f"Correlate group failed for family={family}: {e}",
                exc_info=True,
            )

    return incidents


def _find_matching_open_incident(root_cause, reuse_minutes):
    """
    CHANGE 1: Reuse window is now per-family instead of flat 20 minutes.

    Finds the most recent open incident for the same root cause
    within the family-specific reuse window.
    """
    try:
        with engine.connect() as conn:
            row = conn.execute(text("""
                SELECT id, severity
                FROM incidents
                WHERE root_cause = :root_cause
                  AND status IN (
                      'new',
                      'processing',
                      'alerted',
                      'in_progress',
                      'manual_required'
                  )
                  AND created_at >= NOW() - (:minutes * INTERVAL '1 minute')
                ORDER BY created_at DESC
                LIMIT 1
            """), dict(
                root_cause=root_cause,
                minutes=reuse_minutes,
            )).fetchone()

            if not row:
                return None

            return {
                "id": row[0],
                "severity": row[1],
            }

    except Exception as e:
        log.error(f"Find open incident failed: {e}", exc_info=True)
        return None


def _save(title, severity, root_cause, technical_reason, confidence=50):
    """
    CHANGE 2: confidence parameter added and stored in DB.
    Requires the incidents table to have a confidence column (integer).
    Add via migration if not present:
        ALTER TABLE incidents ADD COLUMN IF NOT EXISTS confidence INTEGER DEFAULT 50;
    """
    try:
        with engine.begin() as conn:
            row = conn.execute(text("""
                INSERT INTO incidents (
                    title,
                    severity,
                    root_cause,
                    technical_reason,
                    confidence,
                    status
                )
                VALUES (
                    :title,
                    :severity,
                    :root_cause,
                    :technical_reason,
                    :confidence,
                    'new'
                )
                RETURNING id
            """), dict(
                title=title,
                severity=severity,
                root_cause=root_cause,
                technical_reason=technical_reason,
                confidence=confidence,
            ))

            incident_id = row.fetchone()[0]

            log.info(
                f"Created incident id={incident_id} "
                f"title={title} confidence={confidence}"
            )
            return incident_id

    except Exception as e:
        log.error(f"Save incident failed: {e}", exc_info=True)
        return None


def _update_incident_if_needed(
    incident_id,
    current_severity,
    new_severity,
    title,
    technical_reason,
    confidence=50,
):
    """
    Escalate/update an existing open incident instead of creating duplicates.
    CHANGE 2: Also updates confidence if the new value is higher.
    """
    current_rank = SEVERITY_RANK.get(current_severity, 1)
    new_rank = SEVERITY_RANK.get(new_severity, 1)

    if new_rank <= current_rank:
        return

    try:
        with engine.begin() as conn:
            conn.execute(text("""
                UPDATE incidents
                SET severity = :severity,
                    title = :title,
                    technical_reason = :technical_reason,
                    confidence = GREATEST(COALESCE(confidence, 0), :confidence)
                WHERE id = :id
            """), dict(
                id=incident_id,
                severity=new_severity,
                title=title,
                technical_reason=technical_reason,
                confidence=confidence,
            ))

        log.info(
            f"Escalated incident id={incident_id} "
            f"from {current_severity} to {new_severity} | "
            f"confidence={confidence}"
        )

    except Exception as e:
        log.error(f"Update incident failed: {e}", exc_info=True)


def _link_anomalies(incident_id, anomalies):
    if not incident_id:
        log.warning("Skipping anomaly linking because incident id is missing.")
        return

    try:
        with engine.begin() as conn:
            for anomaly in anomalies:
                anomaly_id = anomaly.get("db_id")

                if not anomaly_id:
                    continue

                conn.execute(text("""
                    INSERT INTO incident_anomalies (
                        incident_id,
                        anomaly_id
                    )
                    VALUES (
                        :incident_id,
                        :anomaly_id
                    )
                    ON CONFLICT DO NOTHING
                """), dict(
                    incident_id=incident_id,
                    anomaly_id=anomaly_id,
                ))

    except Exception as e:
        log.error(f"Link anomalies failed: {e}", exc_info=True)


def get_open_incidents():
    try:
        with engine.connect() as conn:
            rows = conn.execute(text("""
                SELECT
                    id,
                    title,
                    severity,
                    root_cause,
                    technical_reason,
                    llm_summary,
                    confidence,
                    status,
                    created_at
                FROM incidents
                WHERE status IN (
                    'new',
                    'processing',
                    'alerted',
                    'in_progress',
                    'manual_required'
                )
                ORDER BY created_at DESC
                LIMIT 20
            """)).fetchall()

            return [
                dict(
                    incident_id=row[0],
                    title=row[1],
                    severity=row[2],
                    root_cause=row[3],
                    technical_reason=row[4],
                    llm_summary=row[5],
                    confidence=row[6],
                    status=row[7],
                    created_at=row[8].isoformat() if row[8] else None,
                )
                for row in rows
            ]

    except Exception as e:
        log.error(f"Fetch incidents failed: {e}", exc_info=True)
        return []
