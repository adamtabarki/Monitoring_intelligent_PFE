import os, time, logging, requests
from datetime import datetime
from threading import Thread
from fastapi import FastAPI
from sqlalchemy import create_engine, text

from features import fetch_feature_matrix, latest_values
from models.isolation_forest import IsolationForestDetector
from models.correlator import correlate, get_open_incidents, resolve_incident

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(name)s] %(message)s')
log = logging.getLogger('ai-engine')

PROMETHEUS_URL  = os.getenv('PROMETHEUS_URL',  'http://prometheus:9090')
DATABASE_URL    = os.getenv('DATABASE_URL',
                  'postgresql://monitor:monitor123@postgres:5432/monitoring')
SCRAPE_INTERVAL = int(os.getenv('SCRAPE_INTERVAL', 60))
CONTAMINATION   = float(os.getenv('ANOMALY_CONTAMINATION', 0.05))
HEARTBEAT_URL   = os.getenv('HEARTBEAT_URL', '')

db_engine  = create_engine(DATABASE_URL, pool_pre_ping=True)
detector   = IsolationForestDetector(contamination=CONTAMINATION)
detector.load()

state = {
    'last_anomalies':  [],
    'last_incidents':  [],
    'last_cycle_at':   None,
    'last_trained_at': None,
    'cycle_count':     0,
    'last_alert_ts':   0.0,
}


def wait_for_db(retries=10, delay=5):
    for attempt in range(retries):
        try:
            with db_engine.connect() as conn:
                conn.execute(text('SELECT 1'))
            log.info('PostgreSQL ready.')
            return
        except Exception as e:
            log.warning(f'DB not ready ({attempt+1}/{retries}): {e}')
            time.sleep(delay)
    raise RuntimeError('PostgreSQL did not become ready.')


def save_anomaly(instance, features, score, severity):
    try:
        with db_engine.begin() as conn:
            row = conn.execute(text(
                'INSERT INTO anomalies (instance,cpu_pct,mem_pct,disk_io,score,severity,status)'
                ' VALUES (:inst,:cpu,:mem,:disk,:score,:sev,\'new\') RETURNING id'),
                dict(inst=instance, cpu=features.get('cpu_pct',0),
                     mem=features.get('mem_pct',0), disk=features.get('disk_io',0),
                     score=score, sev=severity))
            return row.fetchone()[0]
    except Exception as e:
        log.error(f'DB write failed: {e}'); return None


def heartbeat_loop():
    if not HEARTBEAT_URL:
        log.info('HEARTBEAT_URL not set — skipping.'); return
    while True:
        try:
            r = requests.get(HEARTBEAT_URL, timeout=10)
            log.info(f'Heartbeat {"OK" if r.status_code==200 else "FAILED"}')
        except Exception as e:
            log.error(f'Heartbeat error: {e}')
        time.sleep(60)


def analysis_loop():
    while True:
        try:
            matrix = fetch_feature_matrix(PROMETHEUS_URL, instance='server-vm')
            if matrix is None:
                log.info('Waiting for enough data...')
                time.sleep(SCRAPE_INTERVAL); continue

            X, names, timestamps = (
                matrix.values, matrix.feature_names, matrix.timestamps)

            if not detector.trained:
                detector.train(X, names)
                state['last_trained_at'] = datetime.utcnow().isoformat()
                log.info('Initial training complete.')
                time.sleep(SCRAPE_INTERVAL); continue

            raw = detector.predict(X, names, timestamps, matrix.families)
            new = [a for a in raw if a['timestamp'] > state['last_alert_ts']]

            if not new:
                log.info(f'Normal. {latest_values(matrix)}')
                state['last_cycle_at'] = datetime.utcnow().isoformat()
                state['cycle_count'] += 1
                time.sleep(SCRAPE_INTERVAL); continue

            state['last_alert_ts'] = max(a['timestamp'] for a in new)
            for a in new:
                db_id = save_anomaly(matrix.instance, a['features'],
                                     a['score'], a['severity'])
                a['db_id'] = db_id
                a['instance'] = matrix.instance

            incidents = correlate(new)
            state['last_anomalies'] = new
            state['last_incidents'] = incidents

            for inc in incidents:
                log.warning(f'INCIDENT [{inc["severity"].upper()}]'
                            f' {inc["root_cause"]} — {inc["anomaly_count"]} signal(s)'
                            f' | {inc["reason"]}')

            state['last_cycle_at'] = datetime.utcnow().isoformat()
            state['cycle_count'] += 1

        except Exception as e:
            log.error(f'Analysis loop error: {e}')
        time.sleep(SCRAPE_INTERVAL)


app = FastAPI(title='AI Monitoring Engine', version='5.0.0')

@app.on_event('startup')
def startup():
    wait_for_db()
    Thread(target=heartbeat_loop, daemon=True).start()
    Thread(target=analysis_loop,  daemon=True).start()
    log.info('AI engine v5.0.0 started.')

@app.get('/health')
def health():
    return {'status':'ok','model_trained':detector.trained,
            'cycle_count':state['cycle_count']}

@app.get('/anomalies')
def get_anomalies():
    return {'count':len(state['last_anomalies']),
            'anomalies':state['last_anomalies']}

@app.get('/incidents')
def get_incidents():
    return {'open_incidents':get_open_incidents(),
            'last_cycle':state['last_incidents']}

@app.get('/alerts')
def get_alerts():
    return {'anomalies':state['last_anomalies'],
            'incidents':state['last_incidents'],
            'total':len(state['last_anomalies']),
            'cycle':state['cycle_count'],
            'last_cycle':state['last_cycle_at']}

@app.get('/status')
def status():
    return {'model_trained':detector.trained,
            'last_trained_at':state['last_trained_at'],
            'last_cycle_at':state['last_cycle_at'],
            'cycle_count':state['cycle_count'],
            'baselines':detector.baselines,
            'scrape_interval':SCRAPE_INTERVAL}

@app.post('/incidents/{incident_id}/resolve')
def resolve(incident_id: int, note: str = ''):
    resolve_incident(incident_id, note)
    return {'result': f'Incident {incident_id} resolved.'}
