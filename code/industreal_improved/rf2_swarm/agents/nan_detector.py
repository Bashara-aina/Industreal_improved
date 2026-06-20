"""NaN detector — scan for NaN/inf in losses, metrics, weights via log."""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List

from rf2_swarm.base_agent import AgentResult, BaseAgent, CheckResult, Verdict

logger = logging.getLogger("swarm.nan_detector")


class NanDetectorAgent(BaseAgent):
    """Detects NaN/Inf values across all training signals."""

    def __init__(self) -> None:
        super().__init__("nan_detector", "NaN/inf in loss values, metrics, weights via log patterns")

    def run(self, datastore: Dict[str, Any]) -> AgentResult:
        checks: List[CheckResult] = []
        log_tail = datastore.get("log_tail", [])
        metrics = datastore.get("metrics", [])
        val_lines = datastore.get("val_lines", [])

        log_text = "\n".join(log_tail)

        # 1. NaN/Inf in log
        nan_log_lines = [l for l in log_tail if re.search(r"\bNaN\b|\binf\b|\binfinity\b", l, re.IGNORECASE)]
        nan_count = len(nan_log_lines)
        if nan_count > 0:
            v = Verdict.CRIT if nan_count > 5 else Verdict.FAIL
            checks.append(CheckResult(
                name="nan_in_log",
                verdict=v,
                summary=f"{nan_count} NaN/Inf references in log tail",
                detail=f"Recent: {nan_log_lines[-5:]}",
                metric=float(nan_count),
                threshold=0,
                dimension="nan_log_count",
            ))
        else:
            checks.append(CheckResult(name="nan_in_log", verdict=Verdict.PASS,
                                       summary="No NaN/Inf in log tail"))

        # 2. NaN/Inf in metrics.jsonl
        nan_metrics = 0
        for m in metrics[:20]:
            for k, v in m.items():
                if isinstance(v, (int, float)):
                    import math
                    if math.isnan(v) or math.isinf(v):
                        nan_metrics += 1
                        break
        if nan_metrics > 0:
            checks.append(CheckResult(
                name="nan_in_metrics",
                verdict=Verdict.CRIT,
                summary=f"{nan_metrics} entries with NaN/Inf in metrics.jsonl",
                detail="NaN in persisted metrics = training data corruption",
            ))
        else:
            checks.append(CheckResult(name="nan_in_metrics", verdict=Verdict.PASS,
                                       summary="No NaN/Inf in metrics.jsonl"))

        # 3. NaN/Inf in validation metrics
        nan_val = 0
        for line in val_lines:
            if re.search(r"\bNaN\b|\binf\b|\binfinity\b", line, re.IGNORECASE):
                nan_val += 1
        if nan_val > 0:
            checks.append(CheckResult(
                name="nan_in_validation",
                verdict=Verdict.CRIT,
                summary=f"{nan_val} NaN/Inf in validation output",
                detail="NaN in validation = model producing invalid outputs",
            ))
        else:
            checks.append(CheckResult(name="nan_in_validation", verdict=Verdict.PASS,
                                       summary="No NaN/Inf in validation"))

        # 4. E4 gradient norm anomalies
        e4 = datastore.get("e4_grad_norms", {})
        nan_grads = [k for k, v in e4.items() if isinstance(v, float) and (v != v or v == float("inf"))]
        if nan_grads:
            checks.append(CheckResult(
                name="nan_gradient_norms",
                verdict=Verdict.CRIT,
                summary=f"NaN/Inf in grad norms: {nan_grads}",
                detail="Gradient norms should always be finite positive values",
            ))
        else:
            checks.append(CheckResult(name="nan_gradient_norms", verdict=Verdict.PASS,
                                       summary=f"All {len(e4)} grad norms finite"))

        return AgentResult(agent_name=self.name, checks=checks)
