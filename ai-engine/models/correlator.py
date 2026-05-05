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

        try:
            open_incident = _find_matching_open_incident(root_cause)

            if open_incident:
                incident_id = open_incident["id"]

                _update_incident_if_needed(
                    incident_id=incident_id,
                    current_severity=open_incident["severity"],
                    new_severity=severity,
                    title=title,
                    technical_reason=technical_reason,
                )
            else:
                incident_id = _save(
                    title=title,
                    severity=severity,
                    root_cause=root_cause,
                    technical_reason=technical_reason,
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
                "created_at": datetime.utcnow().isoformat(),
            })

            log.warning(
                f"INCIDENT [{severity.upper()}] "
                f"{root_cause} — {len(group)} anomaly(ies)"
            )

        except Exception as e:
            log.error(
                f"Correlate group failed for family={family}: {e}",
                exc_info=True,
            )

    return incidents


def _find_matching_open_incident(root_cause):
    """
    Reuse a recent open incident for the same root cause.
    Severity is not part of the match because the same incident can escalate.
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
                  AND created_at >= NOW() - INTERVAL '20 minutes'
                ORDER BY created_at DESC
                LIMIT 1
            """), dict(root_cause=root_cause)).fetchone()

            if not row:
                return None

            return {
                "id": row[0],
                "severity": row[1],
            }

    except Exception as e:
        log.error(f"Find open incident failed: {e}", exc_info=True)
        return None


def _save(title, severity, root_cause, technical_reason):
    try:
        with engine.begin() as conn:
            row = conn.execute(text("""
                INSERT INTO incidents (
                    title,
                    severity,
                    root_cause,
                    technical_reason,
                    status
                )
                VALUES (
                    :title,
                    :severity,
                    :root_cause,
                    :technical_reason,
                    'new'
                )
                RETURNING id
            """), dict(
                title=title,
                severity=severity,
                root_cause=root_cause,
                technical_reason=technical_reason,
            ))

            incident_id = row.fetchone()[0]

            log.info(f"Created incident id={incident_id} title={title}")
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
):
    """
    Escalate/update an existing open incident instead of creating duplicates.
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
                    technical_reason = :technical_reason
                WHERE id = :id
            """), dict(
                id=incident_id,
                severity=new_severity,
                title=title,
                technical_reason=technical_reason,
            ))

        log.info(
            f"Escalated incident id={incident_id} "
            f"from {current_severity} to {new_severity}"
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
                    status=row[6],
                    created_at=row[7].isoformat() if row[7] else None,
                )
                for row in rows
            ]

    except Exception as e:
        log.error(f"Fetch incidents failed: {e}", exc_info=True)
        return []
