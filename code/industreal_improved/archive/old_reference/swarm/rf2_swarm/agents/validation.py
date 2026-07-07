"""Validation monitor — val runs, metric consistency, NaN detection."""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List

from rf2_swarm import config as C
from rf2_swarm.base_agent import AgentResult, BaseAgent, CheckResult, Verdict

logger = logging.getLogger("swarm.validation")


class ValidationAgent(BaseAgent):
    """Monitors validation health: completion, consistency, frequency."""

    def __init__(self) -> None:
        super().__init__("validation", "Val runs completed, metric consistency, NaN in val metrics")

    def run(self, datastore: Dict[str, Any]) -> AgentResult:
        checks: List[CheckResult] = []
        metrics = datastore.get("metrics", [])
        val_lines = datastore.get("val_lines", [])
        full_val_lines = datastore.get("full_val_lines", [])
        log_tail = datastore.get("log_tail", [])
        state = datastore.get("state", {})

        # Use full-log val grep when available (catches val data outside log tail)
        all_val = full_val_lines or val_lines

        # 1. Validation is running
        if all_val:
            checks.append(CheckResult(
                name="validation_active",
                verdict=Verdict.PASS,
                summary=f"{len(val_lines)} validation log lines in tail",
                detail=f"Latest: {val_lines[-1][:120] if val_lines else 'N/A'}",
            ))
        else:
            checks.append(CheckResult(
                name="validation_active",
                verdict=Verdict.FAIL,
                summary="No validation log lines",
                detail="Validation may not be scheduled or the eval loop isn't running",
            ))

        # 2. Latest val metrics not all zero
        if metrics:
            latest_m = metrics[0]
            val = latest_m.get("val", latest_m)
            val_metrics = ["det_mAP50", "det_mAP50_95", "mAP50", "mAP"]
            zero_metrics = []
            non_zero = False
            for key in val_metrics:
                mv = val.get(key)
                if mv is not None:
                    try:
                        fv = float(mv)
                        if fv == 0.0:
                            zero_metrics.append(key)
                        else:
                            non_zero = True
                    except (ValueError, TypeError):
                        pass
            if non_zero:
                v = Verdict.PASS
            elif zero_metrics:
                v = Verdict.CRIT
            else:
                v = Verdict.SKIP
            checks.append(CheckResult(
                name="val_metrics_nonzero",
                verdict=v,
                summary=f"Zero metrics: {zero_metrics}" if zero_metrics else "All val metrics non-zero",
                detail=f"Latest metrics from metrics.jsonl: {len(val_metrics)} checked",
                dimension="val_metrics_zero",
            ))
        else:
            checks.append(CheckResult(name="val_metrics_nonzero", verdict=Verdict.SKIP,
                                       summary="No metrics data"))

        # 3. Val frequency (should be every N epochs)
        val_epochs = self._parse_val_epochs(all_val)
        # Fallback: use metrics.jsonl epoch count as val frequency indicator
        if len(val_epochs) < 2 and metrics:
            ep_vals = []
            prev_ep = None
            for r in metrics:  # newest-first
                v = r.get("val", {})
                m50 = v.get("det_mAP50")
                ep = r.get("epoch", 0)
                if m50 is not None:
                    if prev_ep is not None and ep > prev_ep + 1:
                        break  # epoch reset
                    ep_vals.append(ep)
                    prev_ep = ep
            if len(ep_vals) >= 2:
                val_epochs = ep_vals  # newest-first
        # Sort chronologically for gap computation
        if len(val_epochs) >= 2:
            val_epochs_sorted = sorted(val_epochs)
            gaps = [val_epochs_sorted[i + 1] - val_epochs_sorted[i] for i in range(len(val_epochs_sorted) - 1)]
            avg_gap = sum(gaps) / len(gaps)
            if avg_gap <= 3:
                v = Verdict.PASS
            elif avg_gap <= 5:
                v = Verdict.WARN
            else:
                v = Verdict.FAIL
            checks.append(CheckResult(
                name="val_frequency",
                verdict=v,
                summary=f"Val every {avg_gap:.1f} epochs on avg (recent gaps: {[f'{g:.0f}' for g in gaps[-3:]]})",
                metric=avg_gap,
                threshold=3.0,
                dimension="val_frequency_epochs",
            ))
        else:
            checks.append(CheckResult(name="val_frequency", verdict=Verdict.SKIP,
                                       summary=f"Only {len(val_epochs)} val epochs, need 2"))

        # 4. Val metric consistency (std)
        val_map = self._parse_val_map(all_val)
        # Fallback: extract mAP50 values from metrics.jsonl val dicts
        # Only use the current run (latest contiguous epoch block)
        if len(val_map) < 3 and metrics:
            current_vals = []
            prev_epoch = None
            for r in metrics:  # newest-first
                v = r.get("val", {})
                m50 = v.get("det_mAP50")
                ep = r.get("epoch", 0)
                if m50 is not None:
                    # Detect epoch reset: jumps up (new run) or drops >2 (big gap)
                    if prev_epoch is not None and (ep > prev_epoch + 1 or prev_epoch - ep > 2):
                        break
                    try:
                        current_vals.append(float(m50))
                    except (ValueError, TypeError):
                        pass
                    prev_epoch = ep
            val_map = current_vals  # newest-first, same as metrics
        if len(val_map) >= 3:
            import statistics
            std = statistics.stdev(val_map) if len(val_map) >= 2 else 0
            if std < 0.03:
                v = Verdict.PASS
            elif std < 0.08:
                v = Verdict.WARN
            else:
                v = Verdict.FAIL
            checks.append(CheckResult(
                name="val_metric_consistency",
                verdict=v,
                summary=f"Val mAP std: {std:.4f} over {len(val_map)} measurements",
                detail=f"Values: {[f'{x:.4f}' for x in val_map]}",
                metric=std,
                threshold=0.03,
                dimension="val_metric_consistency",
            ))
        else:
            checks.append(CheckResult(name="val_metric_consistency", verdict=Verdict.SKIP,
                                       summary=f"Need ≥3 val measurements, have {len(val_map)}"))

        # 5. NaN in val metrics from log (skip expected efficiency eval NaNs)
        nan_val_lines = [
            l for l in val_lines if ("nan" in l.lower() or "inf" in l.lower())
            and "[EVAL NaN/Inf]" not in l
            and "Efficiency" not in l
            and "pipeline_" not in l
            and "eff_" not in l
        ]
        if nan_val_lines:
            checks.append(CheckResult(
                name="val_nan_metrics",
                verdict=Verdict.CRIT,
                summary=f"{len(nan_val_lines)} NaN/Inf in val output",
                detail=f"Lines: {nan_val_lines[-3:]}",
            ))
        else:
            checks.append(CheckResult(name="val_nan_metrics", verdict=Verdict.PASS,
                                       summary="No NaN/Inf in val metrics"))

        # 6. Validation step count (sufficient images evaluated)
        val_steps = self._parse_val_steps(log_tail)
        if val_steps is not None:
            if val_steps >= 200:
                v = Verdict.PASS
            elif val_steps >= 50:
                v = Verdict.WARN
            else:
                v = Verdict.FAIL
            checks.append(CheckResult(
                name="val_step_count",
                verdict=v,
                summary=f"Val steps: {val_steps}",
                detail="Too few val steps = unreliable metrics",
                metric=float(val_steps),
                threshold=200,
                dimension="val_steps",
            ))
        else:
            checks.append(CheckResult(name="val_step_count", verdict=Verdict.SKIP,
                                       summary="Val step count not found"))

        # 7. Best val metric trend
        best_metric = state.get("best_metric", 0.0) or 0.0
        current_epoch = datastore.get("epoch", 0) or 0
        if current_epoch > 0 and best_metric > 0:
            eta_epochs = (C.GATE.det_mAP50 - best_metric) / (best_metric / current_epoch) if best_metric / current_epoch > 0 else float("inf")
            checks.append(CheckResult(
                name="val_gate_eta",
                verdict=Verdict.WARN if eta_epochs > 20 else Verdict.PASS,
                summary=f"ETA to gate: ~{eta_epochs:.0f} epochs (epoch {current_epoch}, best={best_metric:.3f})",
                metric=eta_epochs,
                dimension="gate_eta_epochs",
            ))
        else:
            checks.append(CheckResult(name="val_gate_eta", verdict=Verdict.SKIP,
                                       summary="Insufficient data"))

        return AgentResult(agent_name=self.name, checks=checks)

    def _parse_val_epochs(self, lines: List[str]) -> List[int]:
        epochs = []
        for line in lines:
            m = re.search(r"(?:epoch|Epoch)\s*[#=]?\s*(\d+)", line)
            if m:
                epochs.append(int(m.group(1)))
        return epochs

    def _parse_val_map(self, lines: List[str]) -> List[float]:
        vals = []
        for line in lines:
            for pat in [r"mAP[50_]*[=:]\s*([\d.]+)", r"det_mAP50[=:]\s*([\d.]+)"]:
                m = re.search(pat, line)
                if m:
                    try:
                        vals.append(float(m.group(1)))
                    except ValueError:
                        pass
                    break
        return vals

    def _parse_val_steps(self, lines: List[str]) -> int | None:
        for line in reversed(lines):
            m = re.search(r"(?:val|eval).*?step[=:]?\s*(\d+)(?:\s*/\s*\d+)?", line, re.IGNORECASE)
            if m:
                return int(m.group(1))
            m = re.search(r"(\d+)\s*validation\s*batches", line, re.IGNORECASE)
            if m:
                return int(m.group(1))
        return None
