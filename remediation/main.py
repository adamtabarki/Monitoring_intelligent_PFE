"""
Remediation Service

Receives commands from two sources:
  1. n8n / auto-remediation  → POST /auto-remediate
  2. WhatsApp bot commands   → POST /remediate

Supported actions:
  - restart   : restart a named Docker container
  - clear_cache : drop Linux page cache on the server VM
  - kill_process: kill a named process by name
  - status    : return current system metrics snapshot

Every action is logged to PostgreSQL remediation_actions table.
"""
import os
import logging
import subprocess
import docker
import requests
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import create_engine, text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [remediation] %(message)s",
)
log = logging.getLogger("remediation")

DATABASE_URL  = os.getenv(
    "DATABASE_URL",
    "postgresql://monitor:monitor123@postgres:5432/monitoring"
)
AI_ENGINE_URL = os.getenv("AI_ENGINE_URL", "http://ai-engine:8000")

db_engine     = create_engine(DATABASE_URL, pool_pre_ping=True)

try:
    docker_client = docker.from_env()
    log.info("Docker socket connected.")
except Exception as e:
    docker_client = None
    log.warning(f"Docker socket unavailable: {e}")

app = FastAPI(title="Remediation Service", version="2.0.0")


# ── Request models ────────────────────────────────────

class RemediateRequest(BaseModel):
    action:       str            # restart | clear_cache | kill_process | status
    target:       str = ""       # container name or process name
    incident_id:  int | None = None
    triggered_by: str = "manual" # manual | whatsapp | auto


# ── DB helper ─────────────────────────────────────────

def log_action(
    action_type:  str,
    target:       str,
    triggered_by: str,
    result:       str,
    success:      bool,
    incident_id:  int | None = None,
):
    try:
        with db_engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO remediation_actions
                  (incident_id, action_type, target,
                   triggered_by, result, success)
                VALUES
                  (:inc, :action, :target,
                   :by, :result, :success)
            """), {
                "inc":     incident_id,
                "action":  action_type,
                "target":  target,
                "by":      triggered_by,
                "result":  result,
                "success": success,
            })
    except Exception as e:
        log.error(f"Failed to log action: {e}")


def mark_incident_resolved(incident_id: int, note: str):
    try:
        with db_engine.begin() as conn:
            conn.execute(text("""
                UPDATE incidents
                SET status      = 'resolved',
                    resolved_at = NOW(),
                    llm_summary = COALESCE(:note || ' | ' || llm_summary, :note)
                WHERE id = :id
            """), {"id": incident_id, "note": note})
        log.info(f"Incident {incident_id} marked resolved.")
    except Exception as e:
        log.error(f"Failed to resolve incident: {e}")


# ── Actions ───────────────────────────────────────────

def action_restart(target: str) -> tuple[str, bool]:
    if not docker_client:
        return "Docker socket not available.", False
    try:
        container = docker_client.containers.get(target)
        container.restart()
        result = f"Container '{target}' restarted successfully."
        log.info(result)
        return result, True
    except docker.errors.NotFound:
        return f"Container '{target}' not found.", False
    except Exception as e:
        return f"Restart failed: {e}", False


def action_clear_cache() -> tuple[str, bool]:
    """
    Drops the Linux page cache.
    Works on the monitoring VM (where this container runs).
    For the server VM, use SSH — not implemented here.
    """
    try:
        result = subprocess.run(
            ["sh", "-c", "sync && echo 3 > /proc/sys/vm/drop_caches"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            msg = "Page cache cleared successfully."
            log.info(msg)
            return msg, True
        else:
            msg = f"Cache clear failed: {result.stderr}"
            log.warning(msg)
            return msg, False
    except Exception as e:
        return f"Cache clear error: {e}", False


def action_kill_process(target: str) -> tuple[str, bool]:
    """Kill a process by name using pkill."""
    try:
        result = subprocess.run(
            ["pkill", "-f", target],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            msg = f"Process '{target}' killed."
            log.info(msg)
            return msg, True
        else:
            msg = f"No process named '{target}' found."
            log.warning(msg)
            return msg, False
    except Exception as e:
        return f"Kill process error: {e}", False


def action_status() -> tuple[str, bool]:
    """Return a quick system status snapshot from the AI engine."""
    try:
        r = requests.get(f"{AI_ENGINE_URL}/status", timeout=5)
        data = r.json()
        msg = (
            f"AI engine status — "
            f"trained={data.get('model_trained')}, "
            f"cycles={data.get('cycle_count')}, "
            f"last_cycle={data.get('last_cycle_at')}"
        )
        return msg, True
    except Exception as e:
        return f"Could not reach AI engine: {e}", False


# ── API endpoints ─────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "docker_connected": docker_client is not None}


@app.post("/remediate")
def remediate(req: RemediateRequest):
    """
    Main endpoint. Called by n8n or the WhatsApp bot.
    Executes the requested action and logs everything.
    """
    log.info(
        f"Action requested: {req.action} | "
        f"target={req.target} | "
        f"triggered_by={req.triggered_by}"
    )

    if req.action == "restart":
        if not req.target:
            raise HTTPException(400, "target (container name) required for restart")
        result, success = action_restart(req.target)

    elif req.action == "clear_cache":
        result, success = action_clear_cache()

    elif req.action == "kill_process":
        if not req.target:
            raise HTTPException(400, "target (process name) required for kill_process")
        result, success = action_kill_process(req.target)

    elif req.action == "status":
        result, success = action_status()

    else:
        raise HTTPException(400, f"Unknown action: {req.action}. "
                                  f"Allowed: restart, clear_cache, kill_process, status")

    log_action(
        action_type=req.action,
        target=req.target,
        triggered_by=req.triggered_by,
        result=result,
        success=success,
        incident_id=req.incident_id,
    )

    if success and req.incident_id:
        mark_incident_resolved(
            req.incident_id,
            f"Resolved by {req.triggered_by}: {req.action} on {req.target}"
        )

    return {
        "action":  req.action,
        "target":  req.target,
        "success": success,
        "result":  result,
    }


@app.post("/auto-remediate")
def auto_remediate():
    """
    Called by n8n after detecting a critical incident.
    Automatically picks the right action based on root cause.
    """
    try:
        r = requests.get(f"{AI_ENGINE_URL}/incidents", timeout=5)
        data = r.json()
        incidents = data.get("open_incidents", [])
    except Exception as e:
        raise HTTPException(503, f"Cannot reach AI engine: {e}")

    if not incidents:
        return {"result": "No open incidents — nothing to remediate."}

    actions_taken = []
    for inc in incidents:
        severity   = inc.get("severity", "low")
        root_cause = inc.get("root_cause", "")
        inc_id     = inc.get("incident_id")

        # Only auto-remediate critical incidents
        if severity not in ("critical", "high"):
            continue

        # Pick action based on root cause
        if "CPU" in root_cause:
            result, success = action_restart("ai-engine")
            action = "restart ai-engine"
        elif "Memory" in root_cause:
            result, success = action_clear_cache()
            action = "clear_cache"
        elif "Storage" in root_cause:
            result = "Disk fill detected — manual cleanup required."
            success = False
            action = "manual_required"
        else:
            result = f"No auto-remediation rule for: {root_cause}"
            success = False
            action = "no_rule"

        log_action(
            action_type=action,
            target="auto",
            triggered_by="auto",
            result=result,
            success=success,
            incident_id=inc_id,
        )

        if success and inc_id:
            mark_incident_resolved(inc_id, f"Auto-remediated: {action}")

        actions_taken.append({
            "incident_id": inc_id,
            "action":      action,
            "result":      result,
            "success":     success,
        })

    return {
        "actions_taken": actions_taken or [
            {"result": "Open incidents found but no auto-remediation rule matched."}
        ]
    }


@app.get("/history")
def history(limit: int = 20):
    """Return recent remediation actions — shown in Grafana dashboard."""
    try:
        with db_engine.connect() as conn:
            rows = conn.execute(text("""
                SELECT id, executed_at, action_type, target,
                       triggered_by, result, success, incident_id
                FROM remediation_actions
                ORDER BY executed_at DESC
                LIMIT :limit
            """), {"limit": limit}).fetchall()
            return {
                "actions": [
                    {
                        "id":          r[0],
                        "executed_at": r[1].isoformat() if r[1] else None,
                        "action_type": r[2],
                        "target":      r[3],
                        "triggered_by":r[4],
                        "result":      r[5],
                        "success":     r[6],
                        "incident_id": r[7],
                    }
                    for r in rows
                ]
            }
    except Exception as e:
        raise HTTPException(500, f"DB error: {e}")
