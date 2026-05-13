import logging
from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

log = logging.getLogger("isolation-forest")

# ── Z-score thresholds per metric ────────────────────────────────────────────
ZSCORE_THRESHOLDS = {
    "cpu_pct": {
        "medium": 3.0,
        "high": 4.0,
        "critical": 5.0,
    },
    "mem_pct": {
        "medium": 4.5,
        "high": 5.5,
        "critical": 6.5,
    },
    "disk_fill_pct": {
        "medium": 2.5,
        "high": 3.5,
        "critical": 4.5,
    },
    "disk_io": {
        "medium": 4.0,
        "high": 6.0,
        "critical": 999.0,
    },
    "net_traffic": {
        "medium": 5.0,
        "high": 7.0,
        "critical": 999.0,
    },
    "http_requests": {
        "medium": 6.0,
        "high": 8.0,
        "critical": 10.0,
    },
}

# ── Hard thresholds — deterministic guardrails ────────────────────────────────
HARD_THRESHOLDS = {
    "cpu_pct": 90.0,
    "mem_pct": 90.0,
    "disk_fill_pct": 85.0,
}

# ── Minimum std before z-score is computed ───────────────────────────────────
MIN_STD_FOR_ZSCORE = {
    "cpu_pct": 0.10,
    "mem_pct": 0.05,
    "disk_fill_pct": 0.01,
    "disk_io": 100.0,
    "net_traffic": 20.0,
    "http_requests": 0.05,
}

# ── FIX 7: Raised absolute value floors for volatile metrics ─────────────────
# disk_io and net_traffic require meaningful absolute values before z-score fires.
# Previous values (50_000 and 5_000) were too low — normal idle spikes triggered them.
ZSCORE_SPIKE_MIN_ABS_VALUE = {
    "cpu_pct": 50.0,
    "mem_pct": 35.0,
    "disk_fill_pct": 70.0,
    "disk_io": 5_000_000.0,    # FIX 7: raised from 50_000 → 5MB/s minimum
    "net_traffic": 500_000.0,  # FIX 7: raised from 5_000 → 500KB/s minimum
    "http_requests": 1.0,
}

# ── FIX 4: Volatile metrics require IF confirmation for z-score to count ──────
# disk_io and net_traffic are noisy by nature.
# Z-score alone on these metrics produces too many false positives.
# IF must also flag the sample as anomalous before z-score is accepted.
REQUIRE_IF_CONFIRMATION = {
    "disk_io",
    "net_traffic",
}

# ── FIX 1: CV thresholds for baseline quality gating ─────────────────────────
# CV = std / mean. High CV means the training window was too noisy.
# A model trained on a high-CV window produces unreliable baselines.
CV_THRESHOLDS = {
    "cpu_pct": 2.0,
    "mem_pct": 1.0,
    "disk_fill_pct": 0.5,
    "disk_io": 3.0,
    "net_traffic": 3.0,
    "http_requests": 5.0,
}

# ── FIX 8: Per-metric minimum training samples ───────────────────────────────
# Volatile metrics need more history to produce stable baselines.
# Global minimum was 30 samples (30 min) — insufficient for disk_io and net_traffic.
MIN_SAMPLES_PER_METRIC = {
    "cpu_pct": 60,
    "mem_pct": 60,
    "disk_fill_pct": 120,
    "disk_io": 120,
    "net_traffic": 120,
    "http_requests": 90,
}

# ── Drop detection config ─────────────────────────────────────────────────────
ZSCORE_DROP_FEATURES = {
    "http_requests",
}

ZSCORE_DROP_MIN_BASELINE_MEAN = {
    "http_requests": 0.02,
}

NIGHT_HOURS = set(range(0, 6)) | {23}
SAVE_PATH = Path("/app/persistence/isolation_forest.joblib")

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


def baseline_quality_check(baselines):
    """
    FIX 1: Validate baseline quality before committing a trained model.

    Computes CV = std / mean per feature.
    Returns (True, "ok") if all features pass.
    Returns (False, reason) if any feature has a CV exceeding its threshold.

    Called by IsolationForestDetector.train() before saving the model.
    If quality check fails, the model is NOT saved and detection is NOT activated.
    """
    for feature, stats in baselines.items():
        mean = stats.get("mean", 0.0)
        std = stats.get("std", 0.0)

        if mean == 0:
            continue

        cv = std / abs(mean)
        threshold = CV_THRESHOLDS.get(feature, 2.0)

        if cv > threshold:
            return False, (
                f"{feature} CV={round(cv, 2)} exceeds threshold={threshold} "
                f"(mean={round(mean, 2)}, std={round(std, 2)}) — "
                "training window is too noisy"
            )

    return True, "ok"


def required_min_samples(feature_names):
    """
    FIX 8: Compute the minimum number of samples required to train,
    based on which features are present in the matrix.

    Returns the maximum of per-metric minimums for present features.
    Falls back to 60 if a feature is not listed.
    """
    minimums = [
        MIN_SAMPLES_PER_METRIC.get(f, 60)
        for f in feature_names
    ]
    return max(minimums) if minimums else 60


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
        self.baseline_quality_passed = False

    def train(self, X, feature_names):
        """
        Train the model on matrix X.

        FIX 3: Uses MAD (Median Absolute Deviation) instead of std
        for volatile metrics (disk_io, net_traffic). MAD is resistant
        to outliers — a single spike does not inflate the baseline std.

        FIX 1: Runs baseline quality check after computing baselines.
        If CV exceeds threshold for any feature, training is rejected
        and the model stays untrained (or keeps old trained state).
        """
        transformed = self.scaler.fit_transform(X)
        self.model.fit(transformed)

        feature_names = list(feature_names)
        candidate_baselines = {}

        for index, name in enumerate(feature_names):
            column = X[:, index]

            # FIX 3: Use MAD for volatile metrics
            if name in {"disk_io", "net_traffic"}:
                median_val = float(np.median(column))
                # 1.4826 normalizes MAD to be comparable to std
                # for normally distributed data
                mad = float(np.median(np.abs(column - median_val))) * 1.4826
                candidate_baselines[name] = {
                    "mean": median_val,
                    "std": max(mad, 1.0),  # floor at 1 to avoid division by zero
                    "robust": True,        # flag: MAD was used
                }
            else:
                candidate_baselines[name] = {
                    "mean": float(np.mean(column)),
                    "std": float(np.std(column)),
                    "robust": False,
                }

        # FIX 1: Validate baseline quality before accepting this model
        quality_ok, reason = baseline_quality_check(candidate_baselines)

        if not quality_ok:
            log.warning(
                f"Baseline quality check FAILED — model rejected. Reason: {reason}. "
                "Will collect more samples and retry."
            )
            self.baseline_quality_passed = False
            return False  # caller should retry later

        self.baselines = candidate_baselines
        self.feature_names = feature_names
        self.trained = True
        self.baseline_quality_passed = True
        self.saved_at = datetime.utcnow().isoformat()

        short_baselines = {
            key: {
                "mean": round(value["mean"], 2),
                "std": round(value["std"], 2),
                "robust": value.get("robust", False),
            }
            for key, value in self.baselines.items()
        }

        log.info(
            f"Trained on {len(X)} samples. Quality: PASSED. "
            f"Baselines: {short_baselines}"
        )

        self.save()
        return True  # training succeeded

    def predict(self, X, feature_names, timestamps, families):
        if not self.trained:
            return []

        transformed = self.scaler.transform(X)
        predictions = self.model.predict(transformed)
        scores = self.model.score_samples(transformed)

        anomalies = []

        for row_index, (prediction, score) in enumerate(
            zip(predictions, scores)
        ):
            features = {
                name: round(float(X[row_index][feature_index]), 2)
                for feature_index, name in enumerate(feature_names)
            }

            reasons = []
            trigger_sources = []
            severity = None
            hour = datetime.utcfromtimestamp(timestamps[row_index]).hour

            # Isolation Forest signal — supporting only, never sole trigger.
            if_triggered = prediction == -1
            if if_triggered:
                reasons.append(f"IF score={round(float(score), 3)}")
                trigger_sources.append("iforest")

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

                thresholds = ZSCORE_THRESHOLDS.get(name)
                if not thresholds:
                    continue

                # Positive spike detection
                if z_value >= thresholds["medium"]:
                    min_abs_value = ZSCORE_SPIKE_MIN_ABS_VALUE.get(name)

                    # FIX 7: Absolute value floor check
                    if (
                        min_abs_value is not None
                        and value < min_abs_value
                    ):
                        log.debug(
                            f"Suppressed {name} z={round(z_value,1)} — "
                            f"value={value} below floor={min_abs_value}"
                        )
                        continue

                    # FIX 4: Volatile metrics require IF confirmation
                    if name in REQUIRE_IF_CONFIRMATION and not if_triggered:
                        log.info(
                            f"Suppressed {name} z-score — "
                            "IF did not confirm, volatile metric requires dual signal"
                        )
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

                # Negative drop detection
                elif z_value <= -thresholds["medium"]:
                    if name not in ZSCORE_DROP_FEATURES:
                        continue

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

            # Hard threshold checks — always fire regardless of IF
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

            # Safety rule: IF alone never creates an anomaly
            if set(trigger_sources) == {"iforest"}:
                continue

            if hour in NIGHT_HOURS:
                severity = _escalate_one(severity)
                reasons.append(f"night hour ({hour}:00) escalated")
                trigger_sources.append("night_escalation")

            dominant_feature = hard_trigger_feature or z_trigger_feature

            if not dominant_feature:
                continue

            # Confidence score: how many independent signal types fired
            confidence = _confidence_score(trigger_sources, score)

            anomalies.append({
                "timestamp": timestamps[row_index],
                "score": round(float(score), 4),
                "confidence": confidence,
                "severity": severity,
                "features": features,
                "top_feature": dominant_feature,
                "family": families.get(dominant_feature, "unknown"),
                "reason": "; ".join(reasons),
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
            self.baseline_quality_passed = True

            log.info("Model loaded from disk — skipping retraining.")
            return True

        except Exception as e:
            log.warning(f"Load failed: {e} — will retrain.", exc_info=True)
            return False


def _confidence_score(trigger_sources, if_score):
    """
    Compute a 0-100 confidence score based on how many independent
    signal types confirmed the anomaly.

    - hard_threshold alone → 70 (deterministic, always reliable)
    - zscore + iforest → 80 (two independent statistical signals)
    - hard_threshold + zscore → 90
    - all three → 100
    - zscore alone → 40 (weakest, only fires on stable metrics)
    """
    sources = set(trigger_sources) - {"night_escalation"}

    if sources == {"hard_threshold"}:
        return 70
    if sources == {"zscore"}:
        return 40
    if sources == {"iforest", "zscore"}:
        return 80
    if "hard_threshold" in sources and "zscore" in sources:
        return 90
    if "hard_threshold" in sources and "iforest" in sources and "zscore" in sources:
        return 100

    return 50  # fallback


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
