"""Gate predictor — extrapolate from recent val epochs to estimate gate ETA."""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Tuple

from rf2_swarm import config as C
from rf2_swarm.base_agent import AgentResult, BaseAgent, CheckResult, Verdict

logger = logging.getLogger("swarm.gate_predictor")


class GatePredictorAgent(BaseAgent):
    """Linear extrapolation from last N val epochs to gate targets."""

    def __init__(self) -> None:
        super().__init__("gate_predictor", "Linear extrapolation from last N val epochs to gate targets")

    def run(self, datastore: Dict[str, Any]) -> AgentResult:
        checks: List[CheckResult] = []
        state = datastore.get("state", {})
        val_lines = datastore.get("val_lines", [])
        metrics = datastore.get("metrics", [])

        # Parse epoch-vs-metric pairs from val lines
        epoch_map_pairs = self._parse_epoch_metric_pairs(val_lines)
        best_metric = state.get("best_metric", 0.0) or 0.0
        best_metrics_nested = state.get("best_metrics", {})
        if best_metrics_nested.get("det_mAP50") is not None:
            best_metric = float(best_metrics_nested["det_mAP50"])
        current_epoch = datastore.get("epoch", 0) or 0
        max_epochs = int(state.get("max_epochs", C.DEFAULT_MAX_EPOCHS))

        # Also check metrics.jsonl for val data
        if not epoch_map_pairs and metrics:
            for m in metrics:
                val = m.get("val", m)
                me = val.get("epoch") or m.get("epoch")
                mm = val.get("det_mAP50") or val.get("mAP50") or m.get("det_mAP50")
                if me is not None and mm is not None:
                    try:
                        epoch_map_pairs.append((int(me), float(mm)))
                    except (ValueError, TypeError):
                        pass

        # 1. Linear extrapolation to gate
        if len(epoch_map_pairs) >= C.EPOCH_EXTRAPOLATION_WINDOW:
            recent = epoch_map_pairs[-C.EPOCH_EXTRAPOLATION_WINDOW:]
            slope, intercept = self._linear_fit(recent)
            current_epoch_val = current_epoch or recent[-1][0]
            eta_epochs = (C.GATE.det_mAP50 - (slope * current_epoch_val + intercept)) / slope if slope > 0 else float("inf")
            eta_epochs = max(0, eta_epochs)

            if eta_epochs <= 5:
                v = Verdict.PASS
            elif eta_epochs <= 20:
                v = Verdict.WARN
            elif eta_epochs < float("inf"):
                v = Verdict.FAIL
            else:
                v = Verdict.CRIT

            projected_val = slope * (current_epoch_val + 10) + intercept
            checks.append(CheckResult(
                name="gate_eta_prediction",
                verdict=v,
                summary=f"ETA: ~{eta_epochs:.0f} epochs to gate={C.GATE.det_mAP50} (slope={slope:.4f}/epoch)",
                detail=f"Recent: {[(e, f'{m:.4f}') for e, m in recent]}, projected@+10ep={projected_val:.4f}",
                metric=eta_epochs,
                threshold=20,
                dimension="gate_eta_epochs",
            ))
        else:
            checks.append(CheckResult(name="gate_eta_prediction", verdict=Verdict.SKIP,
                                       summary=f"Need {C.EPOCH_EXTRAPOLATION_WINDOW} val points, have {len(epoch_map_pairs)}"))

        # 2. Confidence in reaching gate
        if best_metric > 0 and current_epoch > 0:
            progress_per_epoch = best_metric / current_epoch
            if progress_per_epoch > 0:
                epochs_to_gate = (C.GATE.det_mAP50 - best_metric) / progress_per_epoch
                max_remaining = max(0, max_epochs - current_epoch)
                if epochs_to_gate <= max_remaining:
                    confidence = "HIGH" if epochs_to_gate <= max_remaining * 0.5 else "MEDIUM"
                    cv = Verdict.PASS if confidence == "HIGH" else Verdict.WARN
                else:
                    confidence = "LOW"
                    cv = Verdict.FAIL
                checks.append(CheckResult(
                    name="gate_confidence",
                    verdict=cv,
                    summary=f"Gate confidence: {confidence} (need {epochs_to_gate:.0f}ep, have {max_remaining}ep remaining)",
                    detail=f"Progress: {progress_per_epoch:.4f}/epoch, current best={best_metric:.3f}",
                    dimension="gate_confidence",
                ))
            else:
                checks.append(CheckResult(name="gate_confidence", verdict=Verdict.FAIL,
                                           summary="No positive progress, cannot reach gate"))
        else:
            checks.append(CheckResult(name="gate_confidence", verdict=Verdict.SKIP,
                                       summary="Insufficient data"))

        # 3. MAE gate prediction (check both flat key and best_metrics nested)
        best_mae = state.get("best_mae")
        if best_mae is None:
            best_metrics_n = state.get("best_metrics", {})
            best_mae = best_metrics_n.get("forward_angular_MAE_deg")
        if best_mae is None:
            best_mae = float("inf")
        if isinstance(best_mae, (int, float)) and best_mae < float("inf") and current_epoch > 0:
            mae_progress = max(0, (best_mae - C.GATE.forward_angular_MAE_deg) / current_epoch) if best_mae > C.GATE.forward_angular_MAE_deg else 0
            if best_mae <= C.GATE.forward_angular_MAE_deg:
                v = Verdict.PASS
            elif mae_progress > 0:
                v = Verdict.WARN
            else:
                v = Verdict.FAIL
            checks.append(CheckResult(
                name="mae_gate_prediction",
                verdict=v,
                summary=f"Best MAE={best_mae:.1f}°, target≤{C.GATE.forward_angular_MAE_deg}°",
                detail=f"Improvement rate: {mae_progress:.1f}°/epoch",
                metric=best_mae,
                threshold=C.GATE.forward_angular_MAE_deg,
                dimension="mae_gate",
            ))
        else:
            checks.append(CheckResult(name="mae_gate_prediction", verdict=Verdict.SKIP,
                                       summary="MAE data not available"))

        # 4. mAP50_95 gate prediction
        best_map50_95 = state.get("best_map50_95", 0.0)
        if isinstance(best_map50_95, (int, float)) and best_map50_95 > 0 and current_epoch > 0:
            progress_95 = best_map50_95 / current_epoch
            if progress_95 > 0:
                eta_95 = (C.GATE.det_mAP50_95 - best_map50_95) / progress_95
                checks.append(CheckResult(
                    name="map50_95_gate_eta",
                    verdict=Verdict.PASS if eta_95 <= 15 else Verdict.WARN,
                    summary=f"mAP50_95 ETA: ~{eta_95:.0f} epochs (current={best_map50_95:.3f}, target={C.GATE.det_mAP50_95})",
                    metric=eta_95,
                    dimension="map50_95_eta",
                ))
            else:
                checks.append(CheckResult(name="map50_95_gate_eta", verdict=Verdict.SKIP,
                                           summary="No improvement in mAP50_95"))
        else:
            checks.append(CheckResult(name="map50_95_gate_eta", verdict=Verdict.SKIP,
                                       summary="Insufficient mAP50_95 data"))

        # 5. Required improvement rate
        if best_metric > 0 and current_epoch > 0:
            remaining = C.GATE.det_mAP50 - best_metric
            max_epochs_left = max(1, max_epochs - current_epoch)
            required_rate = remaining / max_epochs_left
            if required_rate <= 0:
                v = Verdict.PASS
            elif required_rate <= 0.02:
                v = Verdict.WARN
            elif required_rate <= 0.05:
                v = Verdict.FAIL
            else:
                v = Verdict.CRIT
            checks.append(CheckResult(
                name="required_improvement_rate",
                verdict=v,
                summary=f"Need {required_rate:.4f}/epoch for {max_epochs_left} epochs to reach gate",
                detail=f"Remaining: {remaining:.3f} mAP50, {max_epochs_left} epochs left",
                metric=required_rate,
                threshold=0.02,
                dimension="required_rate",
            ))
        else:
            checks.append(CheckResult(name="required_improvement_rate", verdict=Verdict.SKIP,
                                       summary="Insufficient data"))

        # 6. Trend direction (accelerating, steady, decelerating)
        if len(epoch_map_pairs) >= 4:
            first_half = epoch_map_pairs[:len(epoch_map_pairs) // 2]
            second_half = epoch_map_pairs[len(epoch_map_pairs) // 2:]
            slope1, _ = self._linear_fit(first_half)
            slope2, _ = self._linear_fit(second_half)
            if slope2 > slope1 * 1.1:
                trend = "accelerating"
                v = Verdict.PASS
            elif slope2 > slope1 * 0.9:
                trend = "steady"
                v = Verdict.WARN
            else:
                trend = "decelerating"
                v = Verdict.FAIL
            checks.append(CheckResult(
                name="trend_direction",
                verdict=v,
                summary=f"Trend: {trend} (early slope={slope1:.4f}, recent slope={slope2:.4f})",
                detail="Accelerating = good, decelerating = may stall before gate",
                dimension="trend_direction",
            ))
        else:
            checks.append(CheckResult(name="trend_direction", verdict=Verdict.SKIP,
                                       summary=f"Need ≥4 data points, have {len(epoch_map_pairs)}"))

        return AgentResult(agent_name=self.name, checks=checks)

    def _parse_epoch_metric_pairs(self, lines: List[str]) -> List[Tuple[int, float]]:
        pairs: List[Tuple[int, float]] = []
        for line in lines:
            em = re.search(r"(?:epoch|Epoch)\s*[#=]?\s*(\d+)", line)
            mm = re.search(r"(?:mAP|det_mAP|mAP50)[_\s]*[=:]\s*([\d.]+)", line)
            if em and mm:
                try:
                    pairs.append((int(em.group(1)), float(mm.group(1))))
                except ValueError:
                    pass
        pairs.sort(key=lambda x: x[0])
        return pairs

    def _linear_fit(self, points: List[Tuple[int, float]]) -> Tuple[float, float]:
        n = len(points)
        if n < 2:
            return 0.0, points[0][1] if points else 0.0
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        import statistics
        try:
            x_mean = statistics.mean(xs)
            y_mean = statistics.mean(ys)
            num = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
            den = sum((x - x_mean) ** 2 for x in xs)
            slope = num / den if den != 0 else 0.0
            intercept = y_mean - slope * x_mean
        except statistics.StatisticsError:
            slope, intercept = 0.0, ys[-1] if ys else 0.0
        return slope, intercept
