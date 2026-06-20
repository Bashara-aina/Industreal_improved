"""Loss health monitor — per-head losses, plateau detection, divergence."""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from rf2_swarm.base_agent import AgentResult, BaseAgent, CheckResult, Verdict

logger = logging.getLogger("swarm.loss_health")


class LossHealthAgent(BaseAgent):
    """Monitors loss values for all heads, detects plateaus and divergence."""

    def __init__(self) -> None:
        super().__init__("loss_health", "det_cls/det_box/ASD/PSR loss values, plateau, divergence")

    def run(self, datastore: Dict[str, Any]) -> AgentResult:
        checks: List[CheckResult] = []
        log_tail = datastore.get("log_tail", [])
        metrics = datastore.get("metrics", [])

        # Use full-loss grep for DEBUG loss lines; fallback to log_tail
        loss_log = datastore.get("full_loss_lines", []) or log_tail
        if not loss_log:
            loss_log = log_tail

        # Parse all loss lines from log
        loss_records = self._parse_loss_lines(loss_log)
        latest_losses = loss_records[-1] if loss_records else {}

        # 1. Detection classification loss
        det_cls = latest_losses.get("det_cls") or latest_losses.get("loss_cls")
        if det_cls is not None:
            v = Verdict.PASS if det_cls < 10.0 else (Verdict.WARN if det_cls < 50.0 else Verdict.FAIL)
            checks.append(CheckResult(
                name="det_cls_loss",
                verdict=v,
                summary=f"det_cls={det_cls:.4f}",
                metric=det_cls,
                threshold=10.0,
                dimension="loss_det_cls",
            ))
        else:
            checks.append(CheckResult(name="det_cls_loss", verdict=Verdict.SKIP,
                                       summary="det_cls loss not in log tail"))

        # 2. Detection box loss
        det_box = latest_losses.get("det_box") or latest_losses.get("loss_box")
        if det_box is not None:
            v = Verdict.PASS if det_box < 10.0 else (Verdict.WARN if det_box < 50.0 else Verdict.FAIL)
            checks.append(CheckResult(
                name="det_box_loss",
                verdict=v,
                summary=f"det_box={det_box:.4f}",
                metric=det_box,
                threshold=10.0,
                dimension="loss_det_box",
            ))
        else:
            checks.append(CheckResult(name="det_box_loss", verdict=Verdict.SKIP,
                                       summary="det_box loss not in log tail"))

        # 3. Total loss
        total_loss = latest_losses.get("total") or latest_losses.get("loss")
        if total_loss is not None:
            v = Verdict.PASS if total_loss < 20.0 else (Verdict.WARN if total_loss < 100.0 else Verdict.FAIL)
            checks.append(CheckResult(
                name="total_loss",
                verdict=v,
                summary=f"total_loss={total_loss:.4f}",
                metric=total_loss,
                threshold=20.0,
                dimension="loss_total",
            ))
        else:
            checks.append(CheckResult(name="total_loss", verdict=Verdict.SKIP,
                                       summary="Total loss not in log tail"))

        # 4. Loss plateau detection (last N records)
        total_vals: List[float] = []
        if len(loss_records) >= 5:
            total_vals = [r.get("total") or r.get("loss") for r in loss_records[-5:] if r.get("total") or r.get("loss")]
            total_vals = [v for v in total_vals if v is not None]
            if len(total_vals) >= 3:
                plateau = max(total_vals) - min(total_vals) < 0.01 * max(total_vals)
                checks.append(CheckResult(
                    name="loss_plateau",
                    verdict=Verdict.WARN if plateau else Verdict.PASS,
                    summary=f"Loss plateau: {plateau} (range={max(total_vals) - min(total_vals):.4f})",
                    detail=f"Loss values: {[f'{x:.4f}' for x in total_vals]}",
                    dimension="loss_plateau",
                ))
            else:
                checks.append(CheckResult(name="loss_plateau", verdict=Verdict.SKIP,
                                           summary="Not enough total loss samples"))
        else:
            checks.append(CheckResult(name="loss_plateau", verdict=Verdict.SKIP,
                                       summary=f"Only {len(loss_records)} loss records, need 5"))

        # 5. Loss divergence (sudden spike)
        if len(total_vals or []) >= 3:
            mean_val = sum(total_vals[:-1]) / len(total_vals[:-1])
            latest_val = total_vals[-1]
            if mean_val > 0 and latest_val > mean_val * 5:
                v = Verdict.CRIT
            elif mean_val > 0 and latest_val > mean_val * 2:
                v = Verdict.FAIL
            else:
                v = Verdict.PASS
            checks.append(CheckResult(
                name="loss_divergence",
                verdict=v,
                summary=f"Latest loss vs mean: {latest_val:.4f} vs {mean_val:.4f} (ratio={latest_val/mean_val:.1f}x)",
                metric=latest_val / mean_val if mean_val > 0 else 0,
                threshold=2.0,
                dimension="loss_divergence",
            ))
        else:
            checks.append(CheckResult(name="loss_divergence", verdict=Verdict.SKIP,
                                       summary="Need ≥3 total loss samples"))

        # 6. NaN/Inf in loss via metrics.jsonl
        if metrics:
            latest_m = metrics[0]
            for key in ("loss", "total_loss", "det_loss"):
                val = latest_m.get(key)
                if val is not None:
                    import math
                    if isinstance(val, (int, float)) and (math.isnan(val) or math.isinf(val)):
                        checks.append(CheckResult(
                            name=f"{key}_nan",
                            verdict=Verdict.CRIT,
                            summary=f"{key}={val} (NaN/Inf detected!)",
                            dimension="loss_nan",
                        ))
                        break
                    else:
                        checks.append(CheckResult(
                            name=f"{key}_valid",
                            verdict=Verdict.PASS,
                            summary=f"{key}={val:.4f}",
                            metric=float(val),
                            dimension=f"loss_{key}",
                        ))
                        break
            else:
                checks.append(CheckResult(name="metric_loss_valid", verdict=Verdict.SKIP,
                                           summary="No loss in latest metrics"))
        else:
            checks.append(CheckResult(name="metric_loss_valid", verdict=Verdict.SKIP,
                                       summary="No metrics available"))

        # 7. Head-specific loss ratio (DET vs HP)
        det_total = (det_cls or 0) + (det_box or 0)
        hp_loss = latest_losses.get("hp") or latest_losses.get("head_pose")
        if det_total > 0 and hp_loss is not None:
            ratio = hp_loss / det_total
            if 0.05 <= ratio <= 5.0:
                v = Verdict.PASS
            elif 0.01 <= ratio <= 10.0:
                v = Verdict.WARN
            else:
                v = Verdict.FAIL
            checks.append(CheckResult(
                name="loss_head_balance",
                verdict=v,
                summary=f"HP/DET loss ratio: {ratio:.2f} (HP={hp_loss:.4f}, DET={det_total:.4f})",
                metric=ratio,
                dimension="loss_head_ratio",
            ))
        else:
            checks.append(CheckResult(name="loss_head_balance", verdict=Verdict.SKIP,
                                       summary="Insufficient data"))

        return AgentResult(agent_name=self.name, checks=checks)

    def _parse_loss_lines(self, lines: List[str]) -> List[Dict[str, float]]:
        records = []
        for line in lines:
            # Match either explicit "loss" or "DEBUG" lines (which contain loss values)
            if "loss" not in line.lower() and "DEBUG" not in line:
                continue
            rec: Dict[str, float] = {}
            # Named capture groups — avoids fragile key-derivation from pattern strings
            patterns = [
                (r"total[=_]\s*([\d.]+)", "total"),
                (r"\bdet\b(?![_-])(?:[=:]\s*)([\d.]+)", "det"),
                (r"det_cls[=:]\s*([\d.]+)", "det_cls"),
                (r"det_reg[=:]\s*([\d.]+)", "det_reg"),
                (r"det_box[=:]\s*([\d.]+)", "det_box"),
                (r"loss[:_]\s*([\d.]+)", "loss"),
                (r"(?<!\w)cls[=:]\s*([\d.]+)", "cls"),
                (r"(?<!\w)box[=:]\s*([\d.]+)", "box"),
                (r"head_pose[=:]\s*([\d.]+)", "head_pose"),
                (r"hp[=:]\s*([\d.]+)", "hp"),
                (r"pose[=:]\s*([\d.]+)", "pose"),
                (r"act[=:]\s*([\d.]+)", "act"),
                (r"psr[=:]\s*([\d.]+)", "psr"),
            ]
            for pat, key in patterns:
                m = re.search(pat, line)
                if m:
                    try:
                        rec[key] = float(m.group(1))
                    except ValueError:
                        pass
            if rec:
                records.append(rec)
        return records
