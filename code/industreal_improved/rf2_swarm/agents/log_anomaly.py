"""Log anomaly detector — warning patterns, error frequency, unexpected lines."""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List

from rf2_swarm.base_agent import AgentResult, BaseAgent, CheckResult, Verdict

logger = logging.getLogger("swarm.log_anomaly")


class LogAnomalyAgent(BaseAgent):
    """Detects anomalous patterns in training logs."""

    # Patterns that indicate problems
    WARNING_PATTERNS = [
        r"\bwarning\b", r"\bWARNING\b",
        r"unexpected", r"unusual",
        r"deprecated", r"deprecat",
        r"fallback", r"FALLBACK",
        r"retrying", r"RETRY",
        r"timeout", r"TIMEOUT",
        r"skip", r"SKIP",
    ]
    ERROR_PATTERNS = [
        r"\bError\b", r"\bERROR\b",
        r"exception", r"EXCEPTION",
        r"traceback", r"TRACEBACK",
        r"failed", r"FAILED",
        r"abort", r"ABORT",
        r"interrupt", r"INTERRUPT",
    ]

    def __init__(self) -> None:
        super().__init__("log_anomaly", "Warning patterns, error frequency, unexpected log lines")

    def run(self, datastore: Dict[str, Any]) -> AgentResult:
        checks: List[CheckResult] = []
        log_tail = datastore.get("log_tail", [])
        error_lines = datastore.get("error_lines", [])

        total_lines = len(log_tail)

        # 1. Error frequency
        if error_lines:
            error_count = len(error_lines)
            if error_count > 20:
                v = Verdict.CRIT
            elif error_count > 5:
                v = Verdict.FAIL
            else:
                v = Verdict.WARN
            checks.append(CheckResult(
                name="error_frequency",
                verdict=v,
                summary=f"{error_count} ERROR/CRITICAL lines in {total_lines} log lines",
                detail=f"Recent errors: {error_lines[-5:]}",
                metric=float(error_count),
                threshold=5,
                dimension="error_frequency",
            ))
        else:
            checks.append(CheckResult(name="error_frequency", verdict=Verdict.PASS,
                                       summary="No ERROR/CRITICAL lines in log tail"))

        # 2. Warning frequency
        warning_lines = []
        for pat in self.WARNING_PATTERNS:
            matches = [l for l in log_tail if re.search(pat, l)]
            warning_lines.extend(matches)
        warning_count = len(set(warning_lines))
        if warning_count > 30:
            v = Verdict.FAIL
        elif warning_count > 10:
            v = Verdict.WARN
        else:
            v = Verdict.PASS
        checks.append(CheckResult(
            name="warning_frequency",
            verdict=v,
            summary=f"{warning_count} warning-pattern lines in {total_lines} log lines",
            metric=float(warning_count),
            threshold=10,
            dimension="warning_frequency",
        ))

        # 3. Log silence detection
        liveness = datastore.get("liveness_lines", [])
        if not liveness and total_lines > 0:
            # Check for any recent log activity
            has_recent = any(
                re.search(r"(?:step|batch|epoch|loss|iter)", l, re.IGNORECASE)
                for l in log_tail[-10:]
            )
            if not has_recent:
                checks.append(CheckResult(
                    name="log_silence",
                    verdict=Verdict.FAIL,
                    summary="No active training signals in last log lines",
                    detail="Log may be stalled or training frozen",
                ))
            else:
                checks.append(CheckResult(name="log_silence", verdict=Verdict.PASS,
                                           summary="Active log signals present"))
        elif liveness:
            checks.append(CheckResult(name="log_silence", verdict=Verdict.PASS,
                                       summary=f"{len(liveness)} liveness signals present"))

        # 4. Unexpected pattern: NaN or Inf propagated (skip expected efficiency eval NaNs)
        # Filter known efficiency eval lines BEFORE regex (re.findall matches don't contain full line prefix)
        efficiency_filtered = [
            l for l in log_tail
            if "[EVAL NaN/Inf]" not in l
            and "Efficiency" not in l
            and "pipeline_" not in l
            and "eff_" not in l
        ]
        all_log_text = "\n".join(efficiency_filtered)
        nan_propagation = re.findall(r"NaN.*?(?:loss|grad|metric|weight|param)", all_log_text, re.IGNORECASE)
        if nan_propagation:
            checks.append(CheckResult(
                name="nan_propagation",
                verdict=Verdict.CRIT,
                summary=f"{len(nan_propagation)} NaN propagation events",
                detail="NaN in critical training signals = data corruption",
            ))
        else:
            checks.append(CheckResult(name="nan_propagation", verdict=Verdict.PASS,
                                       summary="No NaN propagation detected"))

        # 5. Unexpected training interruptions
        interrupt_patterns = [
            r"KeyboardInterrupt", r"SIGTERM", r"SIGINT",
            r"killed", r"KILLED", r"signal \d+",
            r"received signal", r"shutting down",
        ]
        interrupt_lines = []
        for pat in interrupt_patterns:
            matches = [l for l in log_tail if re.search(pat, l)]
            interrupt_lines.extend(matches)
        if interrupt_lines:
            checks.append(CheckResult(
                name="training_interruptions",
                verdict=Verdict.WARN,
                summary=f"{len(interrupt_lines)} interruption signals detected",
                detail=f"Recent: {interrupt_lines[-3:]}",
                metric=float(len(interrupt_lines)),
                dimension="interruptions",
            ))
        else:
            checks.append(CheckResult(name="training_interruptions", verdict=Verdict.PASS,
                                       summary="No training interruptions"))

        # 6. Log rate (lines per step estimate)
        if liveness:
            checks.append(CheckResult(name="log_activity_rate", verdict=Verdict.PASS,
                                       summary=f"{len(liveness)} LIVENESS lines indicate active training"))
        else:
            checks.append(CheckResult(name="log_activity_rate", verdict=Verdict.WARN,
                                       summary="No LIVENESS lines — log format may have changed"))

        return AgentResult(agent_name=self.name, checks=checks)
