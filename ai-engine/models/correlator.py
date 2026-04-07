import logging, os
from datetime import datetime
from sqlalchemy import create_engine, text

log = logging.getLogger('correlator')
engine = create_engine(
    os.getenv('DATABASE_URL',
              'postgresql://monitor:monitor123@postgres:5432/monitoring'),
    pool_pre_ping=True)

ROOT_CAUSES = {
    'cpu_pressure':     'CPU overload',
    'memory_pressure':  'Memory exhaustion',
    'storage_pressure': 'Storage pressure',
    'network_pressure': 'Network saturation',
    'unknown':          'System anomaly',
}
SEVERITY_RANK = {'low':1,'medium':2,'high':3,'critical':4}
RANK_SEVERITY = {v:k for k,v in SEVERITY_RANK.items()}


def correlate(anomalies):
    if not anomalies: return []
    groups = {}
    for a in anomalies:
        groups.setdefault(a.get('family','unknown'), []).append(a)
    incidents = []
    for family, group in groups.items():
        worst    = max(SEVERITY_RANK.get(a.get('severity','low'),1) for a in group)
        severity = RANK_SEVERITY.get(worst, 'low')
        root     = ROOT_CAUSES.get(family, f'Anomaly in {family}')
        seen, reasons = set(), []
        for a in group:
            r = a.get('reason','')
            if r and r not in seen: seen.add(r); reasons.append(r)
        reason = ' | '.join(reasons[:3])
        title  = f'[{severity.upper()}] {root} — {len(group)} signal(s)'
        inc_id = _save(title, severity, root, reason, group)
        incidents.append({
            'incident_id':   inc_id,
            'title':         title,
            'severity':      severity,
            'root_cause':    root,
            'reason':        reason,
            'anomaly_count': len(group),
            'family':        family,
            'created_at':    datetime.utcnow().isoformat(),
        })
        log.warning(f'INCIDENT [{severity.upper()}] {root} — {len(group)} anomaly(ies)')
    return incidents


def _save(title, severity, root_cause, reason, anomalies):
    try:
        with engine.begin() as conn:
            row = conn.execute(text(
                'INSERT INTO incidents (title, severity, root_cause, llm_summary, status)'
                ' VALUES (:t,:s,:r,:l,\'new\') RETURNING id'),
                dict(t=title, s=severity, r=root_cause, l=reason))
            inc_id = row.fetchone()[0]
            for a in anomalies:
                if a.get('db_id'):
                    conn.execute(text(
                        'INSERT INTO incident_anomalies (incident_id, anomaly_id)'
                        ' VALUES (:i,:a) ON CONFLICT DO NOTHING'),
                        dict(i=inc_id, a=a['db_id']))
        return inc_id
    except Exception as e:
        log.error(f'Save incident failed: {e}')
        return None


def get_open_incidents():
    try:
        with engine.connect() as conn:
            rows = conn.execute(text(
                'SELECT id,title,severity,root_cause,llm_summary,status,created_at'
                ' FROM incidents WHERE status IN (\'new\', \'open\')'
                ' ORDER BY created_at DESC LIMIT 20')).fetchall()
            return [dict(incident_id=r[0],title=r[1],severity=r[2],
                         root_cause=r[3],reason=r[4],status=r[5],
                         created_at=r[6].isoformat() if r[6] else None)
                    for r in rows]
    except Exception as e:
        log.error(f'Fetch incidents: {e}'); return []


def resolve_incident(incident_id, note=''):
    try:
        with engine.begin() as conn:
            conn.execute(text(
                'UPDATE incidents SET status=\'resolved\', resolved_at=NOW(),'
                ' llm_summary=COALESCE(:n, llm_summary) WHERE id=:id'),
                dict(id=incident_id, n=note or None))
        log.info(f'Incident {incident_id} resolved.')
    except Exception as e:
        log.error(f'Resolve failed: {e}')
