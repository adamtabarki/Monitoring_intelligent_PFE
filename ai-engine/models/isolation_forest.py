import logging
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

log = logging.getLogger('isolation-forest')

ZSCORE_THRESHOLD = 3.0
HARD_THRESHOLDS  = {
    'cpu_pct':       90.0,
    'mem_pct':       90.0,
    'disk_fill_pct': 85.0,
}
NIGHT_HOURS  = set(range(0, 6)) | {23}
SAVE_PATH    = Path('/app/persistence/isolation_forest.joblib')
SEVERITY_RANK = {'low':1,'medium':2,'high':3,'critical':4}
RANK_SEVERITY = {v:k for k,v in SEVERITY_RANK.items()}


class IsolationForestDetector:

    def __init__(self, contamination=0.05):
        self.contamination = contamination
        self.model   = IsolationForest(contamination=contamination,
            n_estimators=200, max_samples='auto', random_state=42)
        self.scaler  = StandardScaler()
        self.trained = False
        self.baselines = {}

    def train(self, X, feature_names):
        Xs = self.scaler.fit_transform(X)
        self.model.fit(Xs)
        self.trained = True
        for i, name in enumerate(feature_names):
            col = X[:, i]
            self.baselines[name] = {
                'mean': float(np.mean(col)),
                'std':  float(np.std(col)),
            }
        log.info(f'Trained on {len(X)} samples. Baselines: '
                 f'{ {k: round(v["mean"],2) for k,v in self.baselines.items()} }')
        self.save()

    def predict(self, X, feature_names, timestamps, families):
        if not self.trained:
            return []
        Xs     = self.scaler.transform(X)
        preds  = self.model.predict(Xs)
        scores = self.model.score_samples(Xs)
        anomalies = []
        for i, (pred, score) in enumerate(zip(preds, scores)):
            features = {name: round(float(X[i][j]), 2)
                        for j, name in enumerate(feature_names)}
            reasons, severity = [], None
            hour = datetime.utcfromtimestamp(timestamps[i]).hour
            if pred == -1:
                reasons.append(f'IF score={round(float(score),3)}')
                severity = _escalate(severity, 'medium')
            top_z, top_feat = 0.0, None
            for name, val in features.items():
                b = self.baselines.get(name)
                if not b or b['std'] < 0.05: continue
                z = (val - b['mean']) / b['std']
                if abs(z) > top_z: top_z, top_feat = abs(z), name
                if z >= 5:
                    reasons.append(f'{name} z={round(z,1)} spike')
                    severity = _escalate(severity, 'critical')
                elif z >= 4:
                    reasons.append(f'{name} z={round(z,1)} spike')
                    severity = _escalate(severity, 'high')
                elif z >= ZSCORE_THRESHOLD:
                    reasons.append(f'{name} z={round(z,1)} spike')
                    severity = _escalate(severity, 'medium')
            for name, threshold in HARD_THRESHOLDS.items():
                val = features.get(name, 0)
                if val >= threshold:
                    reasons.append(f'{name}={val}% >= {threshold}%')
                    severity = _escalate(severity,
                        'high' if name=='disk_fill_pct' else 'critical')
            if not severity: continue
            if hour in NIGHT_HOURS:
                severity = _escalate_one(severity)
                reasons.append(f'night hour ({hour}:00) escalated')
            dominant = top_feat or max(features, key=features.get)
            anomalies.append({
                'timestamp':   timestamps[i],
                'score':       round(float(score), 4),
                'severity':    severity,
                'features':    features,
                'top_feature': dominant,
                'family':      families.get(dominant, 'unknown'),
                'reason':      '; '.join(reasons) or 'anomaly detected',
                'hour':        hour,
            })
        return anomalies

    def save(self):
        try:
            SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump({'model':self.model,'scaler':self.scaler,
                         'baselines':self.baselines}, SAVE_PATH)
            log.info(f'Model saved to {SAVE_PATH}')
        except Exception as e:
            log.error(f'Save failed: {e}')

    def load(self):
        if not SAVE_PATH.exists():
            log.info('No saved model — will train from scratch.')
            return False
        try:
            data = joblib.load(SAVE_PATH)
            self.model, self.scaler, self.baselines = (
                data['model'], data['scaler'], data['baselines'])
            self.trained = True
            log.info('Model loaded from disk — skipping retraining.')
            return True
        except Exception as e:
            log.warning(f'Load failed: {e} — will retrain.')
            return False


def _escalate(current, new):
    if current is None: return new
    return new if SEVERITY_RANK[new] > SEVERITY_RANK[current] else current

def _escalate_one(severity):
    rank = SEVERITY_RANK.get(severity, 1)
    return RANK_SEVERITY.get(min(rank+1, 4), 'critical')
