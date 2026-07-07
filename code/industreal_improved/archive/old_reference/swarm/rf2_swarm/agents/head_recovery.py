"""Head recovery monitor — freezing/unfreezing, reinit tracking, LR changes."""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List

from rf2_swarm.base_agent import AgentResult, BaseAgent, CheckResult, Verdict

logger = logging.getLogger("swarm.head_recovery")


class HeadRecoveryAgent(BaseAgent):
    """Tracks head recovery events: freeze/unfreeze, reinit, LR adjustments."""

    def __init__(self) -> None:
        super().__init__("head_recovery", "Freezing/unfreezing detection, reinit tracking, LR changes")

    def run(self, datastore: Dict[str, Any]) -> AgentResult:
        checks: List[CheckResult] = []
        log_tail = datastore.get("log_tail", [])
        state = datastore.get("state", {})

        log_text = "\n".join(log_tail)

        # 1. Head freeze/unfreeze events
        freeze_events = re.findall(r"(?:freez|unfreez|thaw).*?(?:head|det|pose|psr)", log_text, re.IGNORECASE)
        if freeze_events:
            checks.append(CheckResult(
                name="head_freeze_events",
                verdict=Verdict.WARN,
                summary=f"{len(freeze_events)} freeze/unfreeze events detected",
                detail=f"Recent: {freeze_events[-3:]}",
                metric=float(len(freeze_events)),
                dimension="freeze_events",
            ))
        else:
            checks.append(CheckResult(name="head_freeze_events", verdict=Verdict.PASS,
                                       summary="No head freeze/unfreeze events"))

        # 2. Head reinitialization
        reinit_events = re.findall(r"(?:reinit|reset|re-initialize).*?(?:head|det|pose|psr)", log_text, re.IGNORECASE)
        if reinit_events:
            checks.append(CheckResult(
                name="head_reinit_events",
                verdict=Verdict.WARN,
                summary=f"{len(reinit_events)} head reinit events",
                detail=f"Events: {reinit_events[-3:]}",
                metric=float(len(reinit_events)),
                dimension="reinit_events",
            ))
        else:
            checks.append(CheckResult(name="head_reinit_events", verdict=Verdict.PASS,
                                       summary="No head reinitializations"))

        # 3. LR schedule changes
        lr_changes = re.findall(r"LR.*?(?:change|update|set|adjust)", log_text, re.IGNORECASE)
        if lr_changes:
            checks.append(CheckResult(
                name="lr_changes",
                verdict=Verdict.WARN if len(lr_changes) > 3 else Verdict.PASS,
                summary=f"{len(lr_changes)} LR changes detected",
                detail=f"Frequent LR changes may indicate tuning",
                metric=float(len(lr_changes)),
                dimension="lr_changes",
            ))
        else:
            checks.append(CheckResult(name="lr_changes", verdict=Verdict.SKIP,
                                       summary="No LR change events"))

        # 4. Dead head recovery attempts
        dead_recovery = re.findall(r"(?:dead|DEAD).*?(?:recover|revive|restore)", log_text, re.IGNORECASE)
        if dead_recovery:
            checks.append(CheckResult(
                name="dead_head_recovery",
                verdict=Verdict.WARN,
                summary=f"{len(dead_recovery)} dead head recovery attempts",
                detail=f"Events: {dead_recovery[-3:]}",
                metric=float(len(dead_recovery)),
                dimension="dead_head_recovery",
            ))
        else:
            checks.append(CheckResult(name="dead_head_recovery", verdict=Verdict.PASS,
                                       summary="No dead head recovery needed"))

        # 5. Gradient clipping events
        clip_events = re.findall(r"(?:grad|gradient).*?clip", log_text, re.IGNORECASE)
        if clip_events:
            if len(clip_events) > 50:
                v = Verdict.FAIL
            elif len(clip_events) > 10:
                v = Verdict.WARN
            else:
                v = Verdict.PASS
            checks.append(CheckResult(
                name="grad_clipping",
                verdict=v,
                summary=f"{len(clip_events)} gradient clipping events",
                detail="Excessive clipping = unstable gradients",
                metric=float(len(clip_events)),
                threshold=10,
                dimension="grad_clip_events",
            ))
        else:
            checks.append(CheckResult(name="grad_clipping", verdict=Verdict.PASS,
                                       summary="No gradient clipping events"))

        # 6. Head status from state
        det_health = state.get("det_health", "")
        hp_health = state.get("hp_health", "")
        psr_health = state.get("psr_health", "")

        for head_name, health in [("DET", det_health), ("HeadPose", hp_health), ("PSR", psr_health)]:
            if health:
                is_dead = "DEAD" in str(health).upper()
                checks.append(CheckResult(
                    name=f"{head_name.lower()}_health_state",
                    verdict=Verdict.CRIT if is_dead else Verdict.PASS,
                    summary=f"{head_name} head: {health}",
                    dimension=f"{head_name.lower()}_health",
                ))
            else:
                checks.append(CheckResult(name=f"{head_name.lower()}_health_state", verdict=Verdict.SKIP,
                                           summary=f"{head_name} health not in state"))

        # 7. LR scheduler type
        sched_lines = [l for l in log_tail if "scheduler" in l.lower() or "lr_sched" in l.lower()]
        if sched_lines:
            checks.append(CheckResult(
                name="lr_scheduler",
                verdict=Verdict.PASS,
                summary=f"LR scheduler active: {sched_lines[-1][:80]}",
                dimension="lr_scheduler",
            ))

        # 8. Optimizer state
        opt_lines = [l for l in log_tail if "optim" in l.lower() and ("adam" in l.lower() or "sgd" in l.lower() or "step" in l.lower())]
        if opt_lines:
            checks.append(CheckResult(
                name="optimizer_active",
                verdict=Verdict.PASS,
                summary=f"Optimizer activity: {len(opt_lines)} log references",
            ))
        else:
            checks.append(CheckResult(name="optimizer_active", verdict=Verdict.WARN,
                                       summary="No optimizer references in log tail"))

        return AgentResult(agent_name=self.name, checks=checks)
