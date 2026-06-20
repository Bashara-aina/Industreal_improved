"""Head health monitor — DET/ASD/PSR head liveness, gradient norms, NaN weights."""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List

from rf2_swarm import config as C
from rf2_swarm.base_agent import AgentResult, BaseAgent, CheckResult, Verdict

logger = logging.getLogger("swarm.head_health")


class HeadHealthAgent(BaseAgent):
    """Monitors all network heads for liveness, gradient health, and balance."""

    def __init__(self) -> None:
        super().__init__("head_health", "DET/ASD/PSR heads ALIVE/DEAD, NaN weights, gradient norms")

    def run(self, datastore: Dict[str, Any]) -> AgentResult:
        checks: List[CheckResult] = []
        e4 = datastore.get("e4_grad_norms", {})
        log_tail = datastore.get("log_tail", [])
        state = datastore.get("state", {})

        # 1. DET head grad norm
        det_norm = e4.get("det")
        if det_norm is not None:
            if det_norm > C.HEALTH.min_grad_norm_det:
                v = Verdict.PASS
            else:
                v = Verdict.CRIT
            checks.append(CheckResult(
                name="det_head_grad_norm",
                verdict=v,
                summary=f"DET grad norm: {det_norm:.2e}",
                detail=f"Threshold: >={C.HEALTH.min_grad_norm_det:.1e}",
                metric=det_norm,
                threshold=C.HEALTH.min_grad_norm_det,
                dimension="grad_norm_det",
            ))
        else:
            checks.append(CheckResult(name="det_head_grad_norm", verdict=Verdict.SKIP,
                                       summary="No E4 grad norm data for DET head",
                                       detail="E4 diagnostics may not be enabled in training loop"))

        # 2. Head pose (HP) grad norm
        hp_norm = e4.get("hp") or e4.get("head_pose")
        if hp_norm is not None:
            if hp_norm > C.HEALTH.min_grad_norm_pose:
                v = Verdict.PASS
            else:
                v = Verdict.CRIT
            checks.append(CheckResult(
                name="hp_head_grad_norm",
                verdict=v,
                summary=f"Head-Pose grad norm: {hp_norm:.2e}",
                metric=hp_norm,
                threshold=C.HEALTH.min_grad_norm_pose,
                dimension="grad_norm_hp",
            ))
        else:
            checks.append(CheckResult(name="hp_head_grad_norm", verdict=Verdict.SKIP,
                                       summary="No HP grad norm data"))
        # 3. Activity head grad norm
        act_norm = e4.get("act")
        if act_norm is not None:
            v = Verdict.PASS if act_norm > 1e-8 else Verdict.WARN
            checks.append(CheckResult(
                name="act_head_grad_norm",
                verdict=v,
                summary=f"Activity grad norm: {act_norm:.2e}",
                metric=act_norm,
                threshold=1e-8,
                dimension="grad_norm_act",
            ))
        else:
            checks.append(CheckResult(name="act_head_grad_norm", verdict=Verdict.SKIP,
                                       summary="No activity head grad norm"))

        # 4. PSR head grad norm
        psr_norm = e4.get("psr")
        if psr_norm is not None:
            v = Verdict.PASS if psr_norm > 1e-8 else Verdict.WARN
            checks.append(CheckResult(
                name="psr_head_grad_norm",
                verdict=v,
                summary=f"PSR grad norm: {psr_norm:.2e}",
                metric=psr_norm,
                threshold=1e-8,
                dimension="grad_norm_psr",
            ))
        else:
            checks.append(CheckResult(name="psr_head_grad_norm", verdict=Verdict.SKIP,
                                       summary="No PSR head grad norm"))

        # 5. Backbone grad norm
        bb_norm = e4.get("backbone")
        if bb_norm is not None:
            v = Verdict.PASS if bb_norm > 1e-5 else Verdict.WARN
            checks.append(CheckResult(
                name="backbone_grad_norm",
                verdict=v,
                summary=f"Backbone grad norm: {bb_norm:.2e}",
                metric=bb_norm,
                threshold=1e-5,
                dimension="grad_norm_backbone",
            ))
        else:
            checks.append(CheckResult(name="backbone_grad_norm", verdict=Verdict.SKIP,
                                       summary="No backbone grad norm"))

        # 6. Head balance ratio (DET vs HP) — WARN only, never FAIL
        # DET (24-class classification) and HP (6-DoF regression) have fundamentally
        # different architectures — their raw gradient magnitudes differ naturally.
        # The individual liveness checks above (1-2) are the real signal.
        # A ratio imbalance only matters if ONE head is actually DEAD (< threshold).
        if det_norm is not None and hp_norm is not None and hp_norm > 0:
            ratio = det_norm / hp_norm
            both_alive = det_norm > C.HEALTH.min_grad_norm_det and hp_norm > C.HEALTH.min_grad_norm_pose
            if both_alive and ratio > 100:
                # Both alive, ratio is architectural — WARN only
                v = Verdict.WARN
            elif ratio < 0.1 or ratio > 10:
                v = Verdict.FAIL
            else:
                v = Verdict.PASS
            checks.append(CheckResult(
                name="head_grad_balance",
                verdict=v,
                summary=f"DET/HP grad ratio: {ratio:.2f} (both alive={both_alive})",
                detail=f"WARN only when both alive — different archs have different grad magnitudes",
                metric=ratio,
                threshold=(10.0, 0.1),
                dimension="head_grad_ratio",
            ))
        else:
            checks.append(CheckResult(name="head_grad_balance", verdict=Verdict.SKIP,
                                       summary="Insufficient data for head balance"))

        # 7. NaN detection in gradients via log patterns
        nan_grad_lines = [l for l in log_tail if "NaN" in l and "grad" in l.lower()]
        if nan_grad_lines:
            checks.append(CheckResult(
                name="nan_gradients",
                verdict=Verdict.CRIT,
                summary=f"{len(nan_grad_lines)} NaN gradient events in log",
                detail=f"Recent: {nan_grad_lines[-3:]}",
            ))
        else:
            checks.append(CheckResult(name="nan_gradients", verdict=Verdict.PASS,
                                       summary="No NaN gradients detected"))

        # 8. Head liveness from state
        det_health = state.get("det_health", "")
        if det_health:
            if "DEAD" in str(det_health).upper():
                v = Verdict.CRIT
            elif "ALIVE" in str(det_health).upper():
                v = Verdict.PASS
            else:
                v = Verdict.WARN
            checks.append(CheckResult(
                name="det_head_liveness",
                verdict=v,
                summary=f"DET head: {det_health}",
                dimension="det_head_liveness",
            ))
        else:
            checks.append(CheckResult(name="det_head_liveness", verdict=Verdict.SKIP,
                                       summary="det_health not in state"))

        # 9. Consecutive dead epochs
        dead_history = state.get("det_health_history", [])
        if dead_history:
            recent_dead = sum(1 for h in dead_history[-5:] if "DEAD" in str(h).upper())
            if recent_dead >= C.HEALTH.max_consecutive_dead:
                v = Verdict.CRIT
            elif recent_dead >= 3:
                v = Verdict.FAIL
            else:
                v = Verdict.PASS
            checks.append(CheckResult(
                name="consecutive_dead_epochs",
                verdict=v,
                summary=f"{recent_dead} dead in last {min(5, len(dead_history))} epochs",
                detail=f"Threshold: max {C.HEALTH.max_consecutive_dead} consecutive dead",
                metric=float(recent_dead),
                threshold=C.HEALTH.max_consecutive_dead,
                dimension="consecutive_dead",
            ))
        else:
            checks.append(CheckResult(name="consecutive_dead_epochs", verdict=Verdict.SKIP,
                                       summary="det_health_history not available"))

        # 10. Kendall precision weighting check
        kendall_active = state.get("kendall_hp_prec_cap", None)
        if kendall_active is not None:
            checks.append(CheckResult(
                name="kendall_precision_cap",
                verdict=Verdict.PASS if kendall_active else Verdict.WARN,
                summary=f"Kendall HP precision cap: {kendall_active}",
                detail="Cap prevents head_pose from dominating",
            ))
        else:
            checks.append(CheckResult(name="kendall_precision_cap", verdict=Verdict.SKIP,
                                       summary="Kendall cap status unknown"))

        return AgentResult(agent_name=self.name, checks=checks)
