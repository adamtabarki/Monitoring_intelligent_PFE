CREATE TABLE IF NOT EXISTS anomalies (
    id           SERIAL PRIMARY KEY,
    detected_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    instance     VARCHAR(100) NOT NULL DEFAULT 'server-vm',
    cpu_pct      FLOAT,
    mem_pct      FLOAT,
    disk_io      FLOAT,
    score        FLOAT,
    severity     VARCHAR(20) DEFAULT 'low',
    status       VARCHAR(20) DEFAULT 'new'
);

CREATE TABLE IF NOT EXISTS incidents (
    id            SERIAL PRIMARY KEY,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    resolved_at   TIMESTAMPTZ,
    title         VARCHAR(255) NOT NULL,
    severity      VARCHAR(20)  NOT NULL DEFAULT 'low',
    root_cause    VARCHAR(255),
    llm_summary   TEXT,
    status        VARCHAR(20)  NOT NULL DEFAULT 'new',
    whatsapp_sent BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE TABLE IF NOT EXISTS incident_anomalies (
    incident_id INT NOT NULL REFERENCES incidents(id) ON DELETE CASCADE,
    anomaly_id  INT NOT NULL REFERENCES anomalies(id) ON DELETE CASCADE,
    PRIMARY KEY (incident_id, anomaly_id)
);

CREATE TABLE IF NOT EXISTS remediation_actions (
    id           SERIAL PRIMARY KEY,
    executed_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    incident_id  INT REFERENCES incidents(id) ON DELETE SET NULL,
    action_type  VARCHAR(100) NOT NULL,
    target       VARCHAR(100),
    triggered_by VARCHAR(50)  DEFAULT 'auto',
    result       TEXT,
    success      BOOLEAN DEFAULT TRUE
);

CREATE INDEX IF NOT EXISTS idx_anomalies_detected ON anomalies (detected_at);
CREATE INDEX IF NOT EXISTS idx_incidents_status   ON incidents  (status);
