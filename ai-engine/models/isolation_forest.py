import logging
from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

log = logging.getLogger("isolation-forest")

ZSCORE_THRESHOLDS = {
    "cpu_pct": {"medium": 3.0, "high": 4.0, "critical": 5.0},
    "mem_pct": {"medium": 4.5, "high": 5.5, "critical": 6.5},
    "disk_fill_pct": {"medium": 2.5, "high": 3.5, "critical": 4.5},
    "disk_io": {"medium": 3.5, "high": 4.5, "critical": 5.5},
    "net_traffic": {"medium": 5.0, "high": 7.0, "critical": 9.0},
    "http_requests": {"medium": 6.0, "high": 8.0, "critical": 10.0},
}

# Operational hard thresholds.
# These are the most reliable triggers for strong demo scenarios.
HARD_THRESHOLDS = {
    "cpu_pct": 90.0,
    "mem_pct": 90.0,
    "disk_fill_pct": 85.0,
}

# Minimum std to avoid impossible z-scores on almost constant metrics.
MIN_STD_FOR_ZSCORE = {
    "cpu_pct": 0.10,
    "mem_pct": 0.05,
    "disk_fill_pct": 0.01,
    "disk_io": 100.0,
    "net_traffic": 20.0,
    "http_requests": 0.05,
}

# Prevent small operational values from creating noisy positive z-score spikes.
# Example: CPU moving from 2% to 5% can be statistically high but not operationally serious.
ZSCORE_SPIKE_MIN_ABS_VALUE = {
    "cpu_pct": 50.0,
    "mem_pct": 35.0,
    "disk_fill_pct": 70.0,
    "disk_io": 5000.0,
    "net_traffic": 5000.0,
    "http_requests": 1.0,
}

# For drops, only alert if the baseline itself was meaningful.
# This keeps application/HTTP drop detection possible without treating near-zero traffic as critical.
ZSCORE_DROP_MIN_BASELINE_MEAN = {
    "disk_io": 1000.0,
    "net_traffic": 1000.0,
    "http_requests": 0.02,
}

NIGHT_HOURS = set(range(0, 6)) | {23}
SAVE_PATH = Path("/app/persistence/isolation_forest.joblib")

SEVERITY_RANK = {"low": 1, "medium": 2, "high": 3, "critical": 4}
RANK_SEVERITY = {v: k for k, v in SEVERITY_RANK.items()}


class IsolationForestDetector:
    def __init__(self, contamination=0.03):
        self.contamination = contamination
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=200,
            max_samples="auto",
            random_state=42,
        )
        self.scaler = StandardScaler()
        self.trained = False
        self.baselines = {}
        self.feature_names = []
        self.saved_at = None

    def train(self, X, feature_names):
        transformed = self.scaler.fit_transform(X)
        self.model.fit(transformed)

        self.trained = True
        self.feature_names = list(feature_names)
        self.saved_at = datetime.utcnow().isoformat()

        for index, name in enumerate(feature_names):
            column = X[:, index]
            self.baselines[name] = {
                "mean": float(np.mean(column)),
                "std": float(np.std(column)),
            }

        short_baselines = {
            key: round(value["mean"], 2)
            for key, value in self.baselines.items()
        }

        log.info(
            f"Trained on {len(X)} samples. Baselines: {short_baselines}"
        )

        self.save()

    def predict(self, X, feature_names, timestamps, families):
        if not self.trained:
            return []

        transformed = self.scaler.transform(X)
        predictions = self.model.predict(transformed)
        scores = self.model.score_samples(transformed)

        anomalies = []

        for row_index, (prediction, score) in enumerate(zip(predictions, scores)):
            features = {
                name: round(float(X[row_index][feature_index]), 2)
                for feature_index, name in enumerate(feature_names)
            }

            reasons = []
            trigger_sources = []
            severity = None

            hour = datetime.utcfromtimestamp(timestamps[row_index]).hour

            if prediction == -1:
                reasons.append(f"IF score={round(float(score), 3)}")
                trigger_sources.append("iforest")
                severity = _escalate(severity, "medium")

            top_abs_z = 0.0
            top_feat_z = None
            z_trigger_feature = None
            z_trigger_abs = 0.0

            for name, value in features.items():
                baseline = self.baselines.get(name)

                if not baseline:
                    continue

                std = baseline.get("std", 0.0)
                mean = baseline.get("mean", 0.0)

                min_std = MIN_STD_FOR_ZSCORE.get(name, 0.05)
                if std < min_std:
                    continue

                z_value = (value - mean) / std
                abs_z = abs(z_value)

                if abs_z > top_abs_z:
                    top_abs_z = abs_z
                    top_feat_z = name

                thresholds = ZSCORE_THRESHOLDS.get(
                    name,
                    ZSCORE_THRESHOLDS["cpu_pct"],
                )

                if z_value >= thresholds["medium"]:
                    min_value = ZSCORE_SPIKE_MIN_ABS_VALUE.get(name)

                    if min_value is not None and value < min_value:
                        continue

                    if abs_z > z_trigger_abs:
                        z_trigger_abs = abs_z
                        z_trigger_feature = name

                    if z_value >= thresholds["critical"]:
                        reasons.append(f"{name} z={round(z_value, 1)} spike")
                        trigger_sources.append("zscore")
                        severity = _escalate(severity, "critical")
                    elif z_value >= thresholds["high"]:
                        reasons.append(f"{name} z={round(z_value, 1)} spike")
                        trigger_sources.append("zscore")
                        severity = _escalate(severity, "high")
                    else:
                        reasons.append(f"{name} z={round(z_value, 1)} spike")
                        trigger_sources.append("zscore")
                        severity = _escalate(severity, "medium")

                elif z_value <= -thresholds["medium"]:
                    min_baseline = ZSCORE_DROP_MIN_BASELINE_MEAN.get(name)

                    if min_baseline is not None and mean < min_baseline:
                        continue

                    if abs_z > z_trigger_abs:
                        z_trigger_abs = abs_z
                        z_trigger_feature = name

                    if z_value <= -thresholds["critical"]:
                        reasons.append(f"{name} z={round(z_value, 1)} drop")
                        trigger_sources.append("zscore")
                        severity = _escalate(severity, "critical")
                    elif z_value <= -thresholds["high"]:
                        reasons.append(f"{name} z={round(z_value, 1)} drop")
                        trigger_sources.append("zscore")
                        severity = _escalate(severity, "high")
                    else:
                        reasons.append(f"{name} z={round(z_value, 1)} drop")
                        trigger_sources.append("zscore")
                        severity = _escalate(severity, "medium")

            hard_trigger_feature = None

            for name, threshold in HARD_THRESHOLDS.items():
                value = features.get(name, 0.0)

                if value >= threshold:
                    reasons.append(f"{name}={value}% >= {threshold}%")
                    trigger_sources.append("hard_threshold")

                    hard_severity = (
                        "high"
                        if name == "disk_fill_pct"
                        else "critical"
                    )

                    severity = _escalate(severity, hard_severity)
                    hard_trigger_feature = name

            if not severity:
                continue

            if hour in NIGHT_HOURS:
                severity = _escalate_one(severity)
                reasons.append(f"night hour ({hour}:00) escalated")
                trigger_sources.append("night_escalation")

            dominant_feature = (
                hard_trigger_feature
                or z_trigger_feature
                or top_feat_z
                or _best_if_only_feature(features)
            )

            anomalies.append({
                "timestamp": timestamps[row_index],
                "score": round(float(score), 4),
                "severity": severity,
                "features": features,
                "top_feature": dominant_feature,
                "family": families.get(dominant_feature, "unknown"),
                "reason": "; ".join(reasons) if reasons else "anomaly detected",
                "trigger_sources": sorted(set(trigger_sources)),
                "hour": hour,
            })

        return anomalies

    def save(self):
        try:
            SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(
                {
                    "model": self.model,
                    "scaler": self.scaler,
                    "baselines": self.baselines,
                    "feature_names": self.feature_names,
                    "saved_at": self.saved_at,
                },
                SAVE_PATH,
            )
            log.info(f"Model saved to {SAVE_PATH}")

        except Exception as e:
            log.error(f"Save failed: {e}", exc_info=True)

    def load(self):
        if not SAVE_PATH.exists():
            log.info("No saved model — will train from scratch.")
            return False

        try:
            data = joblib.load(SAVE_PATH)

            self.model = data["model"]
            self.scaler = data["scaler"]
            self.baselines = data["baselines"]
            self.feature_names = data.get("feature_names", [])
            self.saved_at = data.get("saved_at")
            self.trained = True

            log.info("Model loaded from disk — skipping retraining.")
            return True

        except Exception as e:
            log.warning(f"Load failed: {e} — will retrain.", exc_info=True)
            return False


def _best_if_only_feature(features):
    """
    Fallback dominant feature for IF-only anomalies.
    Prefer operationally meaningful metrics rather than arbitrary max().
    """
    priority = [
        "cpu_pct",
        "disk_io",
        "net_traffic",
        "mem_pct",
        "disk_fill_pct",
        "http_requests",
    ]

    for name in priority:
        if name in features:
            return name

    return next(iter(features.keys()))


def _escalate(current, new):
    if current is None:
        return new

    return (
        new
        if SEVERITY_RANK[new] > SEVERITY_RANK[current]
        else current
    )


def _escalate_one(severity):
    rank = SEVERITY_RANK.get(severity, 1)
    return RANK_SEVERITY.get(min(rank + 1, 4), "critical")
