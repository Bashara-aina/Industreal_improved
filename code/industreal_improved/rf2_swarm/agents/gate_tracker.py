"""Gate target tracker — compares current metrics against RF2 gate targets."""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from rf2_swarm import config as C
from rf2_swarm.base_agent import AgentResult, BaseAgent, CheckResult, Verdict

logger = logging.getLogger("swarm.gate_tracker")


class GateTrackerAgent(BaseAgent):
    """Checks det_mAP50, mAP50_95, MAE against gate targets."""

    def __init__(self) -> None:
        super().__init__("gate_tracker", "det_mAP50, mAP50_95, MAE thresholds, best-vs-current, gate_passed flag")

    def run(self, datastore: Dict[str, Any]) -> AgentResult:
        checks: List[CheckResult] = []
        state = datastore.get("state", {})
        metrics = datastore.get("metrics", [])
        val_lines = datastore.get("val_lines", [])

        # 1. Gate passed flag
        gate_passed = state.get("gate_passed", False)
        checks.append(CheckResult(
            name="gate_passed",
            verdict=Verdict.PASS if gate_passed else Verdict.WARN,
            summary=f"Gate passed: {gate_passed}",
            detail=f"gate_passed flag from rf_stage_state.json",
            metric=1.0 if gate_passed else 0.0,
            threshold=1.0,
            dimension="gate_passed",
        ))

        # 2. Best mAP50 vs gate target (check both flat key and best_metrics nested)
        best_metric = state.get("best_metric", 0.0)
        # best_metric may be a combined score; prefer best_metrics.det_mAP50 if available
        best_metrics_nested = state.get("best_metrics", {})
        if best_metrics_nested.get("det_mAP50") is not None:
            best_metric = float(best_metrics_nested["det_mAP50"])
        if isinstance(best_metric, (int, float)):
            gap = C.GATE.det_mAP50 - best_metric
            if best_metric >= C.GATE.det_mAP50:
                v = Verdict.PASS
            elif best_metric >= C.GATE.det_mAP50 * 0.75:
                v = Verdict.WARN
            else:
                v = Verdict.FAIL
            checks.append(CheckResult(
                name="best_det_mAP50_vs_gate",
                verdict=v,
                summary=f"best_mAP50={best_metric:.3f}, target={C.GATE.det_mAP50}, gap={gap:.3f}",
                detail=f"Need {C.GATE.det_mAP50} to pass gate, currently {gap:+.3f} away",
                metric=best_metric,
                threshold=C.GATE.det_mAP50,
                dimension="det_mAP50",
            ))
        else:
            checks.append(CheckResult(name="best_det_mAP50_vs_gate", verdict=Verdict.SKIP, summary="best_metric not available"))

        # 3. Current mAP50 from latest metrics
        latest_metric = self._latest_val_metric(metrics)
        if latest_metric is not None:
            if latest_metric >= C.GATE.det_mAP50 * 0.5:
                v = Verdict.PASS
            elif latest_metric >= C.GATE.det_mAP50 * 0.3:
                v = Verdict.WARN
            else:
                v = Verdict.FAIL
            checks.append(CheckResult(
                name="current_det_mAP50",
                verdict=v,
                summary=f"latest mAP50={latest_metric:.3f}",
                detail=f"From metrics.jsonl latest entry",
                metric=latest_metric,
                threshold=C.GATE.det_mAP50 * 0.3,
                dimension="det_mAP50_current",
            ))
        else:
            checks.append(CheckResult(name="current_det_mAP50", verdict=Verdict.SKIP, summary="No val metrics available"))

        # 4. mAP50_95 vs gate target
        best_map50_95 = state.get("best_map50_95", 0.0)
        if isinstance(best_map50_95, (int, float)) and best_map50_95 > 0:
            gap95 = C.GATE.det_mAP50_95 - best_map50_95
            if best_map50_95 >= C.GATE.det_mAP50_95:
                v = Verdict.PASS
            elif best_map50_95 >= C.GATE.det_mAP50_95 * 0.5:
                v = Verdict.WARN
            else:
                v = Verdict.FAIL
            checks.append(CheckResult(
                name="best_mAP50_95_vs_gate",
                verdict=v,
                summary=f"best_mAP50_95={best_map50_95:.3f}, target={C.GATE.det_mAP50_95}, gap={gap95:.3f}",
                metric=best_map50_95,
                threshold=C.GATE.det_mAP50_95,
                dimension="det_mAP50_95",
            ))
        else:
            checks.append(CheckResult(name="best_mAP50_95_vs_gate", verdict=Verdict.WARN,
                                       summary="mAP50_95 not tracked in state yet",
                                       detail="State JSON missing best_map50_95 field"))

        # 5. MAE vs gate target (check both flat key and best_metrics nested)
        best_mae = state.get("best_mae")
        if best_mae is None:
            best_metrics = state.get("best_metrics", {})
            best_mae = best_metrics.get("forward_angular_MAE_deg")
        if best_mae is None:
            best_mae = float("inf")
        if isinstance(best_mae, (int, float)) and best_mae < float("inf"):
            if best_mae <= C.GATE.forward_angular_MAE_deg:
                v = Verdict.PASS
            elif best_mae <= C.GATE.forward_angular_MAE_deg * 1.5:
                v = Verdict.WARN
            else:
                v = Verdict.FAIL
            checks.append(CheckResult(
                name="best_MAE_vs_gate",
                verdict=v,
                summary=f"best_MAE={best_mae:.1f}°, target≤{C.GATE.forward_angular_MAE_deg}°",
                metric=best_mae,
                threshold=C.GATE.forward_angular_MAE_deg,
                dimension="MAE",
            ))
        else:
            checks.append(CheckResult(name="best_MAE_vs_gate", verdict=Verdict.SKIP,
                                       summary="MAE not available in state"))

        # 6. Best-vs-current gap (overfitting / regression signal)
        current_metric = latest_metric or best_metric
        if isinstance(best_metric, (int, float)) and isinstance(current_metric, (int, float)) and current_metric > 0:
            regress = best_metric - current_metric
            if regress > 0.05:
                v = Verdict.FAIL
            elif regress > 0.02:
                v = Verdict.WARN
            else:
                v = Verdict.PASS
            checks.append(CheckResult(
                name="best_vs_current_gap",
                verdict=v,
                summary=f"regression={regress:.3f} (best={best_metric:.3f}, current={current_metric:.3f})",
                detail="Large gap between best and current = training instability",
                metric=regress,
                threshold=0.02,
                dimension="regression_gap",
            ))
        else:
            checks.append(CheckResult(name="best_vs_current_gap", verdict=Verdict.SKIP,
                                       summary="Best or current not available"))

        # 7. Progress rate estimate (per-epoch improvement)
        if latest_metric is not None and isinstance(best_metric, (int, float)) and best_metric > 0:
            epoch = datastore.get("epoch", 0) or 0
            if epoch > 0:
                rate = best_metric / epoch
                needed_improvement = C.GATE.det_mAP50 - best_metric
                est_epochs = needed_improvement / rate if rate > 0 else float("inf")
                if est_epochs <= C.RF2GateTargets().det_mAP50 / rate if rate > 0 else False:  # noqa
                    v = Verdict.PASS
                elif est_epochs < 20:
                    v = Verdict.WARN
                else:
                    v = Verdict.FAIL
                checks.append(CheckResult(
                    name="progress_rate",
                    verdict=v,
                    summary=f"~{est_epochs:.0f} more epochs at current rate (epoch {epoch})",
                    detail=f"Best={best_metric:.3f}, rate={rate:.4f}/epoch, need {C.GATE.det_mAP50}",
                    metric=est_epochs,
                    threshold=20,
                    dimension="est_epochs_to_gate",
                ))
            else:
                checks.append(CheckResult(name="progress_rate", verdict=Verdict.SKIP,
                                           summary="Epoch 0, can't estimate rate"))
        else:
            checks.append(CheckResult(name="progress_rate", verdict=Verdict.SKIP,
                                       summary="Not enough data for rate estimation"))

        # 8. Epoch count vs max
        epoch = datastore.get("epoch", 0) or 0
        max_epochs = int(state.get("max_epochs", C.DEFAULT_MAX_EPOCHS))
        epoch_ratio = epoch / max_epochs if max_epochs > 0 else 0
        if epoch_ratio >= 0.9:
            v = Verdict.CRIT
        elif epoch_ratio >= 0.75:
            v = Verdict.WARN
        else:
            v = Verdict.PASS
        checks.append(CheckResult(
            name="epoch_budget",
            verdict=v,
            summary=f"Epoch {epoch}/{max_epochs} ({epoch_ratio:.0%})",
            detail=f"At {epoch_ratio:.0%} of max epoch budget",
            metric=float(epoch_ratio),
            threshold=0.75,
            dimension="epoch_ratio",
        ))

        # 9. Val lines present (confirms validation is happening)
        if val_lines:
            checks.append(CheckResult(name="val_activity", verdict=Verdict.PASS,
                                       summary=f"{len(val_lines)} validation log lines found"))
        else:
            checks.append(CheckResult(name="val_activity", verdict=Verdict.WARN,
                                       summary="No validation log lines in tail",
                                       detail="Validation may not be running"))

        # 10. Gate gap trend (widening or shrinking)
        gap_trend = state.get("gap_trend", "")
        if gap_trend == "shrinking":
            v = Verdict.PASS
        elif gap_trend == "widening":
            v = Verdict.FAIL
        else:
            v = Verdict.SKIP
        checks.append(CheckResult(
            name="gate_gap_trend",
            verdict=v,
            summary=f"Gate gap trend: {gap_trend or 'unknown'}",
            dimension="gap_trend",
        ))

        return AgentResult(agent_name=self.name, checks=checks)

    def _latest_val_metric(self, metrics: List[Dict[str, Any]]) -> float | None:
        for m in metrics:
            val = m.get("val", {})
            mm = val.get("det_mAP50") or val.get("mAP50") or m.get("det_mAP50")
            if mm is not None:
                try:
                    return float(mm)
                except (ValueError, TypeError):
                    continue
        return None
