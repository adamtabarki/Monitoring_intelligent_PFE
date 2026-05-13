import os
import logging
import subprocess
import requests

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import create_engine, text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [remediation] %(message)s",
)
log = logging.getLogger("remediation")

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://monitor:monitor123@postgres:5432/monitoring"
)
AI_ENGINE_URL = os.getenv("AI_ENGINE_URL", "http://ai-engine:8000")

SSH_HOST = os.getenv("SSH_HOST", "")
SSH_USER = os.getenv("SSH_USER", "remediator")
SSH_PORT = int(os.getenv("SSH_PORT", "22"))
SSH_KEY_PATH = os.getenv("SSH_KEY_PATH", "/app/keys/remediation_ed25519")

db_engine = create_engine(DATABASE_URL, pool_pre_ping=True)

app = FastAPI(title="Remediation Service", version="3.0.0")


# ── Request models ────────────────────────────────────

class RemediateRequest(BaseModel):
    action: str            # restart | clear_cache | kill_process | status
    target: str = ""       # service name or process name
    incident_id: int | None = None
    triggered_by: str = "manual"  # manual | telegram | auto


# ── DB helpers ────────────────────────────────────────

def log_action(
    action_type: str,
    target: str,
    triggered_by: str,
    result: str,
    success: bool,
    incident_id: int | None = None,
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
                "inc": incident_id,
                "action": action_type,
                "target": target,
                "by": triggered_by,
                "result": result,
                "success": success,
            })
    except Exception as e:
        log.error(f"Failed to log action: {e}")




# ── SSH helper ────────────────────────────────────────

def ssh_run(remote_cmd: str) -> tuple[str, bool]:
    if not SSH_HOST:
        return "SSH_HOST is not configured.", False

    try:
        result = subprocess.run(
            [
                "ssh",
                "-i", SSH_KEY_PATH,
                "-p", str(SSH_PORT),
                "-o", "StrictHostKeyChecking=no",
                f"{SSH_USER}@{SSH_HOST}",
                remote_cmd,
            ],
            capture_output=True,
            text=True,
            timeout=20,
        )

        stdout = (result.stdout or "").strip()
        stderr = (result.stderr or "").strip()

        if result.returncode == 0:
            return (stdout or "Command executed successfully."), True

        return (stderr or stdout or "Remote command failed."), False

    except Exception as e:
        return f"SSH execution error: {e}", False


# ── Allowed target policy ─────────────────────────────

ALLOWED_RESTART_SERVICES = {
    "apache2",
}

ALLOWED_KILL_PROCESSES = {
    "stress",
    "stress-ng",
    "python3",
    "python",
    "apache2",
}


# ── Actions via SSH ───────────────────────────────────

def action_restart(target: str) -> tuple[str, bool]:
    if target != "apache2":
        return f"Restart target '{target}' is not allowed.", False
    result, success = ssh_run("sudo /usr/local/bin/remediate_restart_apache")
    if success:
        return f"Service 'apache2' restarted successfully. Status: {result}", True
    return result, False


def action_clear_cache() -> tuple[str, bool]:
    return ssh_run("sudo /usr/local/bin/remediate_clear_cache")


def action_kill_process(target: str) -> tuple[str, bool]:
    if target != "stress":
        return f"Process target '{target}' is not allowed.", False
    return ssh_run("sudo /usr/local/bin/remediate_kill_stress")


def action_status() -> tuple[str, bool]:
    return ssh_run("sudo /usr/local/bin/remediate_status")


# ── API endpoints ─────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "ssh_host": SSH_HOST,
        "ssh_user": SSH_USER,
        "ssh_port": SSH_PORT,
    }


@app.post("/remediate")
def remediate(req: RemediateRequest):
    log.info(
        f"Action requested: {req.action} | "
        f"target={req.target} | "
        f"triggered_by={req.triggered_by}"
    )

    if req.action == "restart":
        result, success = action_restart(req.target)

    elif req.action == "clear_cache":
        result, success = action_clear_cache()

    elif req.action == "kill_process":
        result, success = action_kill_process(req.target)

    elif req.action == "status":
        result, success = action_status()

    else:
        raise HTTPException(
            400,
            "Unknown action. Allowed: restart, clear_cache, kill_process, status"
        )

    log_action(
        action_type=req.action,
        target=req.target,
        triggered_by=req.triggered_by,
        result=result,
        success=success,
        incident_id=req.incident_id,
    )



    return {
        "action": req.action,
        "target": req.target,
        "success": success,
        "result": result,
    }


@app.post("/auto-remediate")
def auto_remediate():
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
        severity = inc.get("severity", "low")
        root_cause = inc.get("root_cause", "")
        inc_id = inc.get("incident_id")

        if severity not in ("critical", "high"):
            continue

        if "CPU" in root_cause or "Application" in root_cause:
            result, success = action_restart("apache2")
            action = "restart"
            target = "apache2"
        elif "Memory" in root_cause:
            result, success = action_clear_cache()
            action = "clear_cache"
            target = ""
        elif "Storage" in root_cause:
            result = "Disk/storage issue detected — manual cleanup required."
            success = False
            action = "manual_required"
            target = ""
        else:
            result = f"No auto-remediation rule for: {root_cause}"
            success = False
            action = "no_rule"
            target = ""

        log_action(
            action_type=action,
            target=target,
            triggered_by="auto",
            result=result,
            success=success,
            incident_id=inc_id,
        )


        actions_taken.append({
            "incident_id": inc_id,
            "action": action,
            "target": target,
            "result": result,
            "success": success,
        })

    return {
        "actions_taken": actions_taken or [
            {"result": "Open incidents found but no auto-remediation rule matched."}
        ]
    }


@app.get("/history")
def history(limit: int = 20):
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
                        "id": r[0],
                        "executed_at": r[1].isoformat() if r[1] else None,
                        "action_type": r[2],
                        "target": r[3],
                        "triggered_by": r[4],
                        "result": r[5],
                        "success": r[6],
                        "incident_id": r[7],
                    }
                    for r in rows
                ]
            }
    except Exception as e:
        raise HTTPException(500, f"DB error: {e}")
