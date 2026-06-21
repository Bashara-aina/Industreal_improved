"""Convergence monitor — loss plateau, metric stagnation, oscillation detection."""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List

from rf2_swarm import config as C
from rf2_swarm.base_agent import AgentResult, BaseAgent, CheckResult, Verdict

logger = logging.getLogger("swarm.convergence")


class ConvergenceAgent(BaseAgent):
    """Detects training convergence issues: plateaus, stagnation, oscillation."""

    def __init__(self) -> None:
        super().__init__("convergence", "Loss plateau over N epochs, metric stagnation, oscillation")

    def run(self, datastore: Dict[str, Any]) -> AgentResult:
        checks: List[CheckResult] = []
        metrics = datastore.get("metrics", [])
        log_tail = datastore.get("log_tail", [])
        state = datastore.get("state", {})

        best_metric = state.get("best_metric", 0.0) or 0.0

        # 1. Metric stagnation — best not improving over patience window
        state_history = state.get("metric_history", [])
        # Normalize: entries may be floats or dicts like {'epoch': N, 'det_mAP50': V}
        def _extract_val(entry):
            return entry.get("det_mAP50", 0.0) if isinstance(entry, dict) else float(entry)
        hist_vals = [_extract_val(e) for e in (state_history if isinstance(state_history, list) else [])]
        if len(hist_vals) >= 3:
            recent = hist_vals[-C.CONVERGENCE.patience_epochs:] if len(hist_vals) >= C.CONVERGENCE.patience_epochs else hist_vals
            oldest = recent[0]
            newest = recent[-1]
            improvement = newest - oldest
            if improvement < C.CONVERGENCE.min_improvement:
                v = Verdict.FAIL if improvement < 0 else Verdict.WARN
            else:
                v = Verdict.PASS
            checks.append(CheckResult(
                name="metric_stagnation",
                verdict=v,
                summary=f"Metric{'↑' if improvement >= 0 else '↓'} {improvement:+.4f} over {len(recent)} epochs",
                detail=f"From {oldest:.4f} to {newest:.4f}, threshold={C.CONVERGENCE.min_improvement}",
                metric=improvement,
                threshold=C.CONVERGENCE.min_improvement,
                dimension="metric_stagnation",
            ))
        else:
            checks.append(CheckResult(name="metric_stagnation", verdict=Verdict.SKIP,
                                       summary=f"Need ≥{C.CONVERGENCE.patience_epochs} history entries"))

        # 2. Metric oscillation (zigzag pattern)
        if len(hist_vals) >= 5:
            oscillations = sum(1 for i in range(2, len(hist_vals))
                               if (hist_vals[i] - hist_vals[i - 1]) *
                                  (hist_vals[i - 1] - hist_vals[i - 2]) < 0)
            osc_ratio = oscillations / (len(hist_vals) - 2)
            if osc_ratio > 0.6:
                v = Verdict.FAIL
            elif osc_ratio > 0.4:
                v = Verdict.WARN
            else:
                v = Verdict.PASS
            checks.append(CheckResult(
                name="metric_oscillation",
                verdict=v,
                summary=f"Oscillation: {osc_ratio:.0%} direction changes ({oscillations}/{len(hist_vals) - 2})",
                detail="High oscillation = unstable training dynamics",
                metric=osc_ratio,
                threshold=0.4,
                dimension="metric_oscillation",
            ))
        else:
            checks.append(CheckResult(name="metric_oscillation", verdict=Verdict.SKIP,
                                       summary="Need ≥5 history entries for oscillation detection"))

        # 3. Convergence velocity (rate of improvement)
        if len(hist_vals) >= 3:
            recent3 = hist_vals[-3:]
            if recent3[-1] > 0 and recent3[0] > 0:
                velocity = (recent3[-1] - recent3[0]) / 3
                if velocity > 0.01:
                    v = Verdict.PASS
                elif velocity > 0.0:
                    v = Verdict.WARN
                else:
                    v = Verdict.FAIL
                checks.append(CheckResult(
                    name="convergence_velocity",
                    verdict=v,
                    summary=f"Velocity: {velocity:+.4f}/epoch over last 3 epochs",
                    detail=f"Values: {[f'{x:.4f}' for x in recent3]}",
                    metric=velocity,
                    threshold=0.0,
                    dimension="convergence_velocity",
                ))
            else:
                checks.append(CheckResult(name="convergence_velocity", verdict=Verdict.SKIP,
                                           summary="Metrics contain zeros"))
        else:
            checks.append(CheckResult(name="convergence_velocity", verdict=Verdict.SKIP,
                                       summary="Need ≥3 entries"))

        # 4. Best metric vs random/initial baseline
        if best_metric > 0:
            baseline = 0.01
            if best_metric > baseline * 5:
                v = Verdict.PASS
            elif best_metric > baseline:
                v = Verdict.WARN
            else:
                v = Verdict.FAIL
            checks.append(CheckResult(
                name="above_baseline",
                verdict=v,
                summary=f"Best={best_metric:.4f} vs baseline={baseline}",
                detail="Model is learning if significantly above random baseline",
                metric=best_metric,
                threshold=baseline,
                dimension="above_baseline",
            ))
        else:
            checks.append(CheckResult(name="above_baseline", verdict=Verdict.FAIL,
                                       summary="Best metric is 0 or negative",
                                       detail="Model has not produced any positive metric"))

        # 5. Loss plateau from log
        loss_vals = self._parse_loss_sequence(log_tail)
        if len(loss_vals) >= 10:
            recent_loss = loss_vals[-10:]
            flat = (max(recent_loss) - min(recent_loss)) / (sum(recent_loss) / len(recent_loss))
            if flat < 0.05:
                v = Verdict.FAIL
            elif flat < 0.1:
                v = Verdict.WARN
            else:
                v = Verdict.PASS
            checks.append(CheckResult(
                name="loss_plateau_log",
                verdict=v,
                summary=f"Loss flatness: {flat:.3%} over last 10 entries",
                detail="Very flat loss suggests stalled training",
                metric=flat,
                threshold=0.05,
                dimension="loss_flatness",
            ))
        else:
            checks.append(CheckResult(name="loss_plateau_log", verdict=Verdict.SKIP,
                                       summary=f"Only {len(loss_vals)} loss entries"))

        # 6. LR status
        lr_lines = [l for l in log_tail if "LR" in l or "lr" in l or "learning_rate" in l.lower()]
        if lr_lines:
            lr_val = self._parse_last_lr(lr_lines)
            if lr_val is not None:
                if lr_val > 0:
                    v = Verdict.PASS if lr_val >= 1e-6 else Verdict.WARN
                else:
                    v = Verdict.CRIT
                checks.append(CheckResult(
                    name="learning_rate",
                    verdict=v,
                    summary=f"LR={lr_val:.2e}",
                    metric=lr_val,
                    dimension="learning_rate",
                ))
            else:
                checks.append(CheckResult(name="learning_rate", verdict=Verdict.SKIP,
                                           summary="Could not parse LR value"))
        else:
            checks.append(CheckResult(name="learning_rate", verdict=Verdict.WARN,
                                       summary="No LR mentioned in log tail"))

        # 7. Parse metrics.jsonl for LR
        if metrics:
            latest = metrics[0]
            lr = latest.get("lr")
            if lr is not None:
                checks.append(CheckResult(
                    name="lr_from_metrics",
                    verdict=Verdict.PASS if float(lr) > 0 else Verdict.CRIT,
                    summary=f"LR={float(lr):.2e} from metrics.jsonl",
                    metric=float(lr),
                    dimension="lr_metrics",
                ))

        return AgentResult(agent_name=self.name, checks=checks)

    def _parse_loss_sequence(self, lines: List[str]) -> List[float]:
        vals = []
        for line in lines:
            m = re.search(r"(?:loss|total)[=:]\s*([\d.]+)", line)
            if m:
                try:
                    vals.append(float(m.group(1)))
                except ValueError:
                    pass
        return vals

    def _parse_last_lr(self, lines: List[str]) -> float | None:
        for line in reversed(lines):
            for pat in [r"LR[=:]\s*([\d.e+\-]+)", r"lr[=:]\s*([\d.e+\-]+)",
                        r"learning_rate[=:]\s*([\d.e+\-]+)"]:
                m = re.search(pat, line)
                if m:
                    try:
                        return float(m.group(1))
                    except ValueError:
                        pass
        return None
