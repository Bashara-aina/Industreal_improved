"""Executive summary — trend direction, recommended actions."""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from rf2_swarm import config as C
from rf2_swarm.base_agent import AgentResult, BaseAgent, CheckResult, Verdict

logger = logging.getLogger("swarm.summary")


class SummaryAgent(BaseAgent):
    """Executive summary aggregator — trend direction, recommended actions."""

    def __init__(self) -> None:
        super().__init__("summary", "Executive summary, trend direction, recommended actions")

    def run(self, datastore: Dict[str, Any]) -> AgentResult:
        checks: List[CheckResult] = []
        state = datastore.get("state", {})
        e4 = datastore.get("e4_grad_norms", {})

        best_metric = state.get("best_metric", 0.0) or 0.0
        epoch = datastore.get("epoch", 0) or 0
        pid_alive = datastore.get("pid_alive", False)
        gate_passed = state.get("gate_passed", False)

        # 1. Trend direction
        state_history = state.get("metric_history", [])
        if isinstance(state_history, list) and len(state_history) >= 3:
            recent = state_history[-3:]
            if recent[-1] > recent[0] * 1.05:
                trend = "IMPROVING"
                tv = Verdict.PASS
            elif recent[-1] >= recent[0] * 0.95:
                trend = "FLAT"
                tv = Verdict.WARN
            else:
                trend = "DEGRADING"
                tv = Verdict.FAIL
        else:
            trend = "INSUFFICIENT_DATA"
            tv = Verdict.SKIP
        checks.append(CheckResult(
            name="trend_direction",
            verdict=tv,
            summary=f"Trend: {trend}",
            detail=f"Recent metric history: {[f'{x:.4f}' for x in (state_history[-5:] if isinstance(state_history, list) and len(state_history) >= 5 else state_history)]}",
            dimension="trend_direction",
        ))

        # 2. Recommended action
        actions: List[str] = []

        if not pid_alive:
            actions.append("CRITICAL: Training process is dead — restart immediately")

        if gate_passed:
            actions.append("Gate already passed — consider advancing to next stage")
        else:
            gap = C.GATE.det_mAP50 - best_metric
            if gap > 0.3:
                actions.append(f"Large gate gap ({gap:.3f}) — investigate head health and data quality")
            elif gap > 0.1:
                actions.append(f"Moderate gate gap ({gap:.3f}) — review LR schedule and head balance")
            elif gap > 0.0:
                actions.append(f"Small gate gap ({gap:.3f}) — continue monitoring, may converge naturally")
            if epoch >= 25 and gap > 0.05:
                actions.append(f"CRITICAL: Only {30 - epoch} epochs remaining but gap={gap:.3f} — may need restart with tuned config")

        det_norm = e4.get("det", 0)
        if det_norm is not None and det_norm < C.HEALTH.min_grad_norm_det:
            actions.append(f"DET head gradient near zero ({det_norm:.2e}) — head may be dead")

        if best_metric <= 0:
            actions.append("Best metric is 0 — model not learning, check data pipeline and config")

        action_text = "; ".join(actions) if actions else "No action needed — training is proceeding normally"
        action_v = Verdict.CRIT if "CRITICAL" in action_text else (Verdict.FAIL if any(a.startswith("CRITICAL") or a.startswith("Large") or a.startswith("Moderate") for a in actions) else Verdict.PASS)
        checks.append(CheckResult(
            name="recommended_actions",
            verdict=action_v,
            summary=f"{len(actions)} action items: {action_text[:200]}",
            detail=action_text,
            dimension="actions",
        ))

        # 3. Executive health score (0-100)
        score = 100
        if not pid_alive:
            score -= 50
        if gate_passed:
            score = 100  # gate passed = goal achieved
        else:
            metric_ratio = min(1.0, best_metric / C.GATE.det_mAP50) if C.GATE.det_mAP50 > 0 else 0
            score = int(metric_ratio * 70)  # metric contributes up to 70 points
            if not pid_alive:
                score -= 30
            epoch_ratio = epoch / 30
            if epoch_ratio > 0.8 and metric_ratio < 0.5:
                score -= 20  # running out of time
            if det_norm is not None and det_norm < 1e-6:
                score -= 15  # dead head penalty
            score = max(0, min(100, score))

        if score >= 70:
            sv = Verdict.PASS
        elif score >= 40:
            sv = Verdict.WARN
        elif score > 0:
            sv = Verdict.FAIL
        else:
            sv = Verdict.CRIT
        checks.append(CheckResult(
            name="executive_health_score",
            verdict=sv,
            summary=f"Health score: {score}/100 (epoch {epoch}, best={best_metric:.3f}, gate={C.GATE.det_mAP50})",
            detail=f"Score components: metric_ratio={min(1.0, best_metric / C.GATE.det_mAP50) if C.GATE.det_mAP50 > 0 else 0:.2f}, pid_alive={pid_alive}, epoch_ratio={epoch / 30:.2f}",
            metric=float(score),
            threshold=40,
            dimension="health_score",
        ))

        return AgentResult(agent_name=self.name, checks=checks)
