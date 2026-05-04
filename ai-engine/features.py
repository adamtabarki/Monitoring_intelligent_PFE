import time
import logging
from dataclasses import dataclass

import pandas as pd
import requests

log = logging.getLogger("features")

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

FEATURE_NAMES = [f[0] for f in FEATURE_QUERIES]
FEATURE_FAMILIES = {f[0]: f[3] for f in FEATURE_QUERIES}


@dataclass
class FeatureMatrix:
    values: object
    timestamps: list
    feature_names: list
    families: dict
    instance: str


def _query_range(prometheus_url, query, lookback=3600, step=60):
    end = int(time.time())
    start = end - lookback

    try:
        r = requests.get(
            f"{prometheus_url}/api/v1/query_range",
            params={
                "query": query,
                "start": start,
                "end": end,
                "step": str(step),
            },
            timeout=10,
        )
        r.raise_for_status()

        results = r.json()["data"]["result"]
        if not results:
            return pd.Series(dtype=float)

        vals = results[0]["values"]
        return pd.Series({float(ts): float(v) for ts, v in vals}, dtype=float)

    except Exception as e:
        log.warning(f"Query failed [{query[:120]}]: {e}")
        return pd.Series(dtype=float)


def fetch_feature_matrix(
    prometheus_url,
    instance="node-server",
    lookback=3600,
    step=60,
    min_samples=30,
):
    series = {}

    for name, query, _, _ in FEATURE_QUERIES:
        formatted_query = query.format(instance=instance)
        s = _query_range(prometheus_url, formatted_query, lookback, step)
        series[name] = s

    df = pd.DataFrame(series).sort_index()
    df = df.ffill(limit=3).bfill(limit=3)

    if "http_requests" in df.columns:
        df["http_requests"] = df["http_requests"].fillna(0.0)

    df = df.dropna()

    if len(df) < min_samples:
        log.warning(f"Only {len(df)} samples — need {min_samples}.")
        return None

    return FeatureMatrix(
        values=df.values.astype("float32"),
        timestamps=list(df.index),
        feature_names=list(df.columns),
        families=FEATURE_FAMILIES,
        instance=instance,
    )


def latest_values(matrix):
    if matrix is None or len(matrix.values) == 0:
        return {}

    latest = matrix.values[-1]
    return {
        name: round(float(val), 2)
        for name, val in zip(matrix.feature_names, latest)
    }
