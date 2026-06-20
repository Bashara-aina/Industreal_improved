"""Blocker assessment — cross-cutting blocker summary, P0-P3 classification."""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List

from rf2_swarm.base_agent import AgentResult, BaseAgent, CheckResult, Verdict

logger = logging.getLogger("swarm.blocker_assessment")


class BlockerAssessmentAgent(BaseAgent):
    """Cross-cutting blocker analysis — classifies issues by severity P0-P3."""

    # P0: Immediate stop — training cannot continue
    P0_PATTERNS = [
        r"out of memory", r"CUDA_ERROR", r"CUDA error",
        r"NaN.*loss", r"loss.*NaN",
        r"no module named", r"ImportError",
        r"cannot open shared object",
        r"training.*not.*start",
        r"PID.*DEAD",
    ]
    # P1: Serious — gate at risk
    P1_PATTERNS = [
        r"det_mAP50.*0\.0",
        r"gradient.*explod",
        r"dead.*head", r"head.*dead",
        r"metric.*stagnat",
        r"loss.*diverg",
        r"plateau.*fail",
    ]
    # P2: Concerning — needs attention
    P2_PATTERNS = [
        r"VRAM.*FAIL",
        r"GPU.*temp",
        r"slow.*batch",
        r"DataLoader.*error",
        r"checkpoint.*fail",
    ]
    # P3: Informational — non-blocking
    P3_PATTERNS = [
        r"epoch.*ETA.*high",
        r"variance.*high",
        r"warning.*freq",
        r"subprocess.*error",
    ]

    def __init__(self) -> None:
        super().__init__("blocker_assessment", "Cross-cutting blocker summary, P0-P3 classification")

    def run(self, datastore: Dict[str, Any]) -> AgentResult:
        checks: List[CheckResult] = []
        log_tail = datastore.get("log_tail", [])
        state = datastore.get("state", {})
        pid_alive = datastore.get("pid_alive", False)

        log_text = "\n".join(log_tail)

        # 1. P0 blocker detection
        p0_matches = []
        for pat in self.P0_PATTERNS:
            matches = re.findall(pat, log_text, re.IGNORECASE)
            p0_matches.extend(matches)
        p0_count = len(p0_matches)

        if not pid_alive:
            p0_count += 1  # dead process = P0

        if p0_count > 0:
            v = Verdict.CRIT
        elif not pid_alive:
            v = Verdict.CRIT
        else:
            v = Verdict.PASS

        checks.append(CheckResult(
            name="p0_blockers",
            verdict=v,
            summary=f"P0 blockers: {p0_count}" + ("" if p0_count == 0 else f" — {p0_matches[:3]}"),
            detail="P0 = training cannot continue without intervention",
            metric=float(p0_count),
            threshold=0,
            dimension="p0_blockers",
        ))

        # 2. P1-P3 issue counts
        p1_count = len(re.findall("|".join(self.P1_PATTERNS), log_text, re.IGNORECASE))
        p2_count = len(re.findall("|".join(self.P2_PATTERNS), log_text, re.IGNORECASE))
        p3_count = len(re.findall("|".join(self.P3_PATTERNS), log_text, re.IGNORECASE))

        total_issues = p1_count + p2_count + p3_count
        if p1_count > 0:
            v = Verdict.FAIL
        elif p2_count > 5:
            v = Verdict.WARN
        elif total_issues > 0:
            v = Verdict.PASS
        else:
            v = Verdict.PASS

        checks.append(CheckResult(
            name="blocker_summary",
            verdict=v,
            summary=f"P1={p1_count}, P2={p2_count}, P3={p3_count} (total={total_issues})",
            detail="Cross-cutting issue classification across all log sources",
            metric=float(p1_count),
            threshold=0,
            dimension="blockers_total",
        ))

        # 3. Overall training health (composite assessment)
        best_metric = state.get("best_metric", 0.0) or 0.0
        epoch = datastore.get("epoch", 0) or 0
        from rf2_swarm import config as C

        if p0_count > 0:
            health_status = "CRITICAL"
            health_v = Verdict.CRIT
        elif p1_count > 0:
            health_status = "AT_RISK"
            health_v = Verdict.FAIL
        elif best_metric >= C.GATE.det_mAP50:
            health_status = "GOOD"
            health_v = Verdict.PASS
        elif epoch >= 25 and best_metric < C.GATE.det_mAP50 * 0.5:
            health_status = "UNLIKELY"
            health_v = Verdict.FAIL
        elif epoch >= 15 and best_metric < C.GATE.det_mAP50 * 0.3:
            health_status = "CONCERNING"
            health_v = Verdict.WARN
        elif best_metric > C.GATE.det_mAP50 * 0.5:
            health_status = "ON_TRACK"
            health_v = Verdict.PASS
        else:
            health_status = "EARLY"
            health_v = Verdict.PASS

        checks.append(CheckResult(
            name="overall_health",
            verdict=health_v,
            summary=f"Training health: {health_status} (epoch {epoch}, best={best_metric:.3f}, gate={C.GATE.det_mAP50})",
            detail=f"P0={p0_count}, P1={p1_count}, P2={p2_count}, P3={p3_count}",
            dimension="overall_health",
        ))

        return AgentResult(agent_name=self.name, checks=checks)
