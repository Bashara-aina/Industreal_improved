"""Process health monitor — PID alive, heartbeat staleness, zombie detection."""
from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List

from rf2_swarm import config as C
from rf2_swarm.base_agent import AgentResult, BaseAgent, CheckResult, Verdict

logger = logging.getLogger("swarm.process_health")


class ProcessHealthAgent(BaseAgent):
    """Monitors training process: PID alive, heartbeat freshness, subprocess health."""

    def __init__(self) -> None:
        super().__init__("process_health", "PID alive, heartbeat staleness, subprocess existence")

    def run(self, datastore: Dict[str, Any]) -> AgentResult:
        checks: List[CheckResult] = []
        state = datastore.get("state", {})
        pid_alive = datastore.get("pid_alive", False)
        pid = datastore.get("pid")
        log_tail = datastore.get("log_tail", [])

        # 1. PID alive check
        if pid is not None:
            if pid_alive:
                v = Verdict.PASS
            else:
                v = Verdict.CRIT
            checks.append(CheckResult(
                name="pid_alive",
                verdict=v,
                summary=f"PID {pid}: {'ALIVE' if pid_alive else 'DEAD'}",
                detail=f"Checked via /proc/{pid}",
                dimension="pid_alive",
            ))
        else:
            checks.append(CheckResult(name="pid_alive", verdict=Verdict.WARN,
                                       summary="No PID in state JSON",
                                       detail="Training may not have started or state file stale"))

        # 2. Heartbeat freshness
        hb_timestamp = state.get("heartbeat") or state.get("timestamp") or state.get("last_update")
        if hb_timestamp:
            try:
                if isinstance(hb_timestamp, (int, float)):
                    hb_age = time.time() - hb_timestamp
                else:
                    hb_time_str = str(hb_timestamp)
                    import datetime
                    hb_dt = datetime.datetime.fromisoformat(hb_time_str)
                    hb_age = time.time() - hb_dt.timestamp()
                if hb_age < C.HEARTBEAT_WARN_SECONDS:
                    v = Verdict.PASS
                elif hb_age < C.HEARTBEAT_FAIL_SECONDS:
                    v = Verdict.WARN
                else:
                    v = Verdict.FAIL
                checks.append(CheckResult(
                    name="heartbeat_freshness",
                    verdict=v,
                    summary=f"Heartbeat: {hb_age:.0f}s ago (WARN>{C.HEARTBEAT_WARN_SECONDS}s, FAIL>{C.HEARTBEAT_FAIL_SECONDS}s)",
                    metric=hb_age,
                    threshold=C.HEARTBEAT_WARN_SECONDS,
                    dimension="heartbeat_age_s",
                ))
            except (ValueError, TypeError, OSError) as e:
                checks.append(CheckResult(name="heartbeat_freshness", verdict=Verdict.WARN,
                                           summary=f"Could not parse heartbeat: {e}"))
        else:
            checks.append(CheckResult(name="heartbeat_freshness", verdict=Verdict.WARN,
                                       summary="No heartbeat timestamp in state",
                                       detail="State file may not include heartbeat field"))

        # 3. Subprocess log activity
        liveness_lines = datastore.get("liveness_lines", [])
        if liveness_lines:
            checks.append(CheckResult(
                name="training_activity",
                verdict=Verdict.PASS,
                summary=f"{len(liveness_lines)} LIVENESS lines in log tail",
                detail=f"Latest: {liveness_lines[-1][:100] if liveness_lines else 'N/A'}",
            ))
        else:
            # Check general log activity
            log_lines = len(log_tail)
            if log_lines > 10:
                checks.append(CheckResult(
                    name="training_activity",
                    verdict=Verdict.WARN,
                    summary=f"No LIVENESS lines but {log_lines} log lines present",
                    detail="Training may be running without liveness logging",
                ))
            else:
                checks.append(CheckResult(
                    name="training_activity",
                    verdict=Verdict.FAIL,
                    summary=f"No training activity (only {log_lines} log lines)",
                    detail="Training process may be stalled or not producing output",
                ))

        # 4. Zombie / defunct process detection
        if pid is not None:
            try:
                status_path = f"/proc/{pid}/status"
                if os.path.isfile(status_path):
                    with open(status_path) as f:
                        status_text = f.read()
                    if "zombie" in status_text.lower() or "defunct" in status_text.lower():
                        checks.append(CheckResult(
                            name="zombie_process",
                            verdict=Verdict.CRIT,
                            summary=f"PID {pid} is ZOMBIE/DEFUNCT",
                            detail="Process is dead but not reaped",
                        ))
                    else:
                        checks.append(CheckResult(name="zombie_process", verdict=Verdict.PASS,
                                                   summary=f"PID {pid} is alive and running"))
            except (OSError, PermissionError):
                checks.append(CheckResult(name="zombie_process", verdict=Verdict.SKIP,
                                           summary="Could not check /proc status"))

        return AgentResult(agent_name=self.name, checks=checks)
