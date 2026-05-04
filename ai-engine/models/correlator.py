import logging
import os
from datetime import datetime

from sqlalchemy import create_engine, text

log = logging.getLogger("correlator")

engine = create_engine(
    os.getenv(
        "DATABASE_URL",
        "postgresql://monitor:monitor123@postgres:5432/monitoring"
    ),
    pool_pre_ping=True
)

ROOT_CAUSES = {
    "cpu_pressure": "CPU overload",
    "memory_pressure": "Memory exhaustion",
    "storage_pressure": "Storage pressure",
    "network_pressure": "Network saturation",
    "application_pressure": "Application anomaly",
    "unknown": "System anomaly",
}

SEVERITY_RANK = {"low": 1, "medium": 2, "high": 3, "critical": 4}
RANK_SEVERITY = {v: k for k, v in SEVERITY_RANK.items()}


def correlate(anomalies):
    if not anomalies:
        return []

    groups = {}
    for a in anomalies:
        groups.setdefault(a.get("family", "unknown"), []).append(a)

    incidents = []

    for family, group in groups.items():
        worst = max(SEVERITY_RANK.get(a.get("severity", "low"), 1) for a in group)
        severity = RANK_SEVERITY.get(worst, "low")
        root = ROOT_CAUSES.get(family, f"Anomaly in {family}")

        seen, reasons = set(), []
        for a in group:
            r = a.get("reason", "")
            if r and r not in seen:
                seen.add(r)
                reasons.append(r)

        reason = " | ".join(reasons[:3])
        title = f"[{severity.upper()}] {root} — {len(group)} signal(s)"

        try:
            inc_id = _find_matching_open_incident(root, severity)
            if not inc_id:
                inc_id = _save(title, severity, root, reason)

            _link_anomalies(inc_id, group)

            incidents.append({
                "incident_id": inc_id,
                "title": title,
                "severity": severity,
                "root_cause": root,
                "reason": reason,
                "anomaly_count": len(group),
                "family": family,
                "created_at": datetime.utcnow().isoformat(),
            })

            log.warning(
                f"INCIDENT [{severity.upper()}] {root} — {len(group)} anomaly(ies)"
            )

        except Exception as e:
            log.error(f"Correlate group failed for family={family}: {e}", exc_info=True)

    return incidents


def _find_matching_open_incident(root_cause, severity):
    try:
        with engine.connect() as conn:
            row = conn.execute(text("""
                SELECT id
                FROM incidents
                WHERE root_cause = :r
                  AND severity = :s
                  AND status IN ('new', 'processing', 'alerted', 'in_progress', 'manual_required')
                  AND created_at >= NOW() - INTERVAL '10 minutes'
                ORDER BY created_at DESC
                LIMIT 1
            """), dict(r=root_cause, s=severity)).fetchone()

            return row[0] if row else None

    except Exception as e:
        log.error(f"Find open incident failed: {e}", exc_info=True)
        return None


def _save(title, severity, root_cause, reason):
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
                VALUES (:t, :s, :r, :tr, 'new')
                RETURNING id
            """), dict(
                t=title,
                s=severity,
                r=root_cause,
                tr=reason
            ))
            inc_id = row.fetchone()[0]
            log.info(f"Created incident id={inc_id} title={title}")
            return inc_id

    except Exception as e:
        log.error(f"Save incident failed: {e}", exc_info=True)
        return None


def _link_anomalies(inc_id, anomalies):
    if not inc_id:
        log.warning("Skipping anomaly linking because incident id is missing.")
        return

    try:
        with engine.begin() as conn:
            for a in anomalies:
                if a.get("db_id"):
                    conn.execute(text("""
                        INSERT INTO incident_anomalies (incident_id, anomaly_id)
                        VALUES (:i, :a)
                        ON CONFLICT DO NOTHING
                    """), dict(i=inc_id, a=a["db_id"]))

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
                WHERE status IN ('new', 'processing', 'alerted', 'in_progress', 'manual_required')
                ORDER BY created_at DESC
                LIMIT 20
            """)).fetchall()

            return [
                dict(
                    incident_id=r[0],
                    title=r[1],
                    severity=r[2],
                    root_cause=r[3],
                    technical_reason=r[4],
                    llm_summary=r[5],
                    status=r[6],
                    created_at=r[7].isoformat() if r[7] else None
                )
                for r in rows
            ]

    except Exception as e:
        log.error(f"Fetch incidents failed: {e}", exc_info=True)
        return []


def resolve_incident(incident_id, note=""):
    try:
        with engine.begin() as conn:
            conn.execute(text("""
                UPDATE incidents
                SET status = 'resolved',
                    resolved_at = NOW()
                WHERE id = :id
            """), dict(id=incident_id))

        log.info(f"Incident {incident_id} resolved. note={note}")

    except Exception as e:
        log.error(f"Resolve failed: {e}", exc_info=True)
