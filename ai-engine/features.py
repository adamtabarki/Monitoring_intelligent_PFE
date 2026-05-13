import time
import logging
from dataclasses import dataclass

import pandas as pd
import requests

log = logging.getLogger("features")

# ── Feature definitions ───────────────────────────────────────────────────────
# Each entry: (feature_name, prometheus_query, unit, family)
FEATURE_QUERIES = [
    (
        "cpu_pct",
        '100 - (avg by (instance) (irate(node_cpu_seconds_total{{mode="idle",instance="{instance}"}}[5m])) * 100)',
        "percent",
        "cpu_pressure",
    ),
    (
        "mem_pct",
        '(1 - (node_memory_MemAvailable_bytes{{instance="{instance}"}} / node_memory_MemTotal_bytes{{instance="{instance}"}})) * 100',
        "percent",
        "memory_pressure",
    ),
    (
        "disk_fill_pct",
        '(1 - (node_filesystem_avail_bytes{{mountpoint="/",instance="{instance}"}} / node_filesystem_size_bytes{{mountpoint="/",instance="{instance}"}})) * 100',
        "percent",
        "storage_pressure",
    ),
    (
        "disk_io",
        'sum by (instance) (irate(node_disk_read_bytes_total{{instance="{instance}",device!~"loop.*|ram.*"}}[5m]) + irate(node_disk_written_bytes_total{{instance="{instance}",device!~"loop.*|ram.*"}}[5m]))',
        "bytes_per_sec",
        "storage_pressure",
    ),
    (
        "net_traffic",
        'sum by (instance) (irate(node_network_receive_bytes_total{{device!~"lo|docker.*|veth.*",instance="{instance}"}}[5m]) + irate(node_network_transmit_bytes_total{{device!~"lo|docker.*|veth.*",instance="{instance}"}}[5m]))',
        "bytes_per_sec",
        "network_pressure",
    ),
    (
        "http_requests",
        'sum by (instance) (irate(apache_accesses_total{{instance="{instance}"}}[5m]))',
        "requests_per_sec",
        "application_pressure",
    ),
]

FEATURE_NAMES = [feature[0] for feature in FEATURE_QUERIES]
FEATURE_FAMILIES = {
    feature[0]: feature[3]
    for feature in FEATURE_QUERIES
}

# Core metrics — must be present and non-null for training to proceed.
# If any of these are missing, the matrix is rejected.
REQUIRED_FEATURES = [
    "cpu_pct",
    "mem_pct",
    "disk_fill_pct",
]

# Optional metrics — missing values are zero-filled.
# Their absence does not block training or detection.
OPTIONAL_ZERO_FILL_FEATURES = [
    "disk_io",
    "net_traffic",
    "http_requests",
]

# CHANGE 2: Increased lookback from 3600s (1h) to 7200s (2h).
# With volatile metrics requiring 120 samples minimum (one per 60s step),
# a 1h lookback can only ever provide 60 samples — not enough.
# 2h lookback gives up to 120 samples, satisfying the per-metric minimums.
DEFAULT_LOOKBACK = 7200  # 2 hours


@dataclass
class FeatureMatrix:
    values: object
    timestamps: list
    feature_names: list
    families: dict
    instance: str


def _query_range(prometheus_url, query, lookback=DEFAULT_LOOKBACK, step=60):
    end = int(time.time())
    start = end - lookback

    try:
        response = requests.get(
            f"{prometheus_url}/api/v1/query_range",
            params={
                "query": query,
                "start": start,
                "end": end,
                "step": str(step),
            },
            timeout=10,
        )
        response.raise_for_status()

        results = response.json()["data"]["result"]
        if not results:
            return pd.Series(dtype=float)

        values = results[0]["values"]

        return pd.Series(
            {
                float(timestamp): float(value)
                for timestamp, value in values
            },
            dtype=float,
        )

    except Exception as e:
        log.warning(f"Query failed [{query[:120]}]: {e}")
        return pd.Series(dtype=float)


def fetch_feature_matrix(
    prometheus_url,
    instance="node-server",
    lookback=DEFAULT_LOOKBACK,
    step=60,
    min_timestamp=None,
):
    """
    Fetch all feature metrics from Prometheus and return a FeatureMatrix.

    CHANGE 1: Removed the min_samples parameter and its internal check.
    Minimum sample validation is now handled by main.py via
    required_min_samples() from isolation_forest.py — keeping the
    responsibility in one place and making it per-metric aware.

    CHANGE 2: Default lookback is now 7200s (2h) instead of 3600s (1h)
    to support the new 120-sample minimum for volatile metrics.

    CHANGE 3: Logs which individual features returned empty from Prometheus,
    making it easier to diagnose missing exporters or scrape config issues.

    Returns None if required features are missing or matrix is empty.
    Returns FeatureMatrix on success.
    """
    series = {}
    empty_features = []

    for name, query, _, _ in FEATURE_QUERIES:
        formatted_query = query.format(instance=instance)
        result = _query_range(
            prometheus_url,
            formatted_query,
            lookback=lookback,
            step=step,
        )
        series[name] = result

        # CHANGE 3: Track which features returned no data from Prometheus
        if result.empty:
            empty_features.append(name)

    # CHANGE 3: Log missing features clearly
    if empty_features:
        required_missing = [f for f in empty_features if f in REQUIRED_FEATURES]
        optional_missing = [f for f in empty_features if f in OPTIONAL_ZERO_FILL_FEATURES]

        if required_missing:
            log.warning(
                f"Required features returned no data from Prometheus: {required_missing}. "
                "Check node_exporter is running and Prometheus scrape config is correct."
            )
        if optional_missing:
            log.info(
                f"Optional features returned no data (will be zero-filled): {optional_missing}"
            )

    df = pd.DataFrame(series).sort_index()

    if df.empty:
        log.warning("No Prometheus samples returned for any feature.")
        return None

    # Filter to only samples collected after engine start (avoids pre-startup history)
    if min_timestamp is not None:
        df = df[df.index >= float(min_timestamp)]

    if df.empty:
        log.warning("No fresh samples after engine startup timestamp.")
        return None

    # Smooth short gaps (up to 3 consecutive missing points) with forward/back fill
    df = df.ffill(limit=3).bfill(limit=3)

    # Optional metrics zero-filled — their absence must not break core detection
    for feature in OPTIONAL_ZERO_FILL_FEATURES:
        if feature in df.columns:
            df[feature] = df[feature].fillna(0.0)

    existing_required = [
        feature
        for feature in REQUIRED_FEATURES
        if feature in df.columns
    ]

    if not existing_required:
        log.warning(
            "No required core metrics (cpu_pct, mem_pct, disk_fill_pct) found. "
            "Cannot build feature matrix."
        )
        return None

    # Drop rows where any required feature is null
    df = df.dropna(subset=existing_required)

    # Zero-fill any remaining optional gaps
    df = df.fillna(0.0)

    if df.empty:
        log.warning("Feature matrix is empty after cleaning.")
        return None

    log.debug(
        f"Feature matrix built: {len(df)} samples, "
        f"features={list(df.columns)}, "
        f"lookback={lookback}s"
    )

    return FeatureMatrix(
        values=df.values.astype("float32"),
        timestamps=list(df.index),
        feature_names=list(df.columns),
        families=FEATURE_FAMILIES,
        instance=instance,
    )


def latest_values(matrix):
    """Return the most recent sample as a dict of feature → value."""
    if matrix is None or len(matrix.values) == 0:
        return {}

    latest = matrix.values[-1]

    return {
        name: round(float(value), 2)
        for name, value in zip(matrix.feature_names, latest)
    }
