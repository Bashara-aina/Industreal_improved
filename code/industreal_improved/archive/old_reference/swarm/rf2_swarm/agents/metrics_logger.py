"""Metrics logger monitor — metrics.jsonl completeness, field consistency, drift."""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from rf2_swarm.base_agent import AgentResult, BaseAgent, CheckResult, Verdict

logger = logging.getLogger("swarm.metrics_logger")


class MetricsLoggerAgent(BaseAgent):
    """Monitors the metrics logging subsystem for completeness and consistency."""

    REQUIRED_FIELDS = {"step", "loss", "lr", "epoch"}
    OPTIONAL_FIELDS = {"det_mAP50", "mAP50", "val_loss", "det_cls_loss", "det_box_loss"}

    def __init__(self) -> None:
        super().__init__("metrics_logger", "Metrics.jsonl completeness, field drift, consistency")

    def run(self, datastore: Dict[str, Any]) -> AgentResult:
        checks: List[CheckResult] = []
        metrics = datastore.get("metrics", [])

        if not metrics:
            checks.append(CheckResult(
                name="metrics_data_available",
                verdict=Verdict.FAIL,
                summary="No metrics.jsonl entries found",
                detail="Metrics logging may not be active",
            ))
            return AgentResult(agent_name=self.name, checks=checks)

        # 1. Field completeness across entries
        if len(metrics) >= 5:
            sample = metrics[:5]  # latest 5
            missing_counts: Dict[str, int] = {}
            for entry in sample:
                for field in self.REQUIRED_FIELDS:
                    if field not in entry and field not in entry.get("val", {}):
                        missing_counts[field] = missing_counts.get(field, 0) + 1
            if missing_counts:
                msg = "; ".join(f"{f} missing in {c}/{len(sample)}" for f, c in missing_counts.items())
                checks.append(CheckResult(
                    name="metrics_field_completeness",
                    verdict=Verdict.WARN,
                    summary=f"Missing fields: {msg}",
                    dimension="metrics_completeness",
                ))
            else:
                checks.append(CheckResult(name="metrics_field_completeness", verdict=Verdict.PASS,
                                           summary="All required fields present"))
        else:
            checks.append(CheckResult(name="metrics_field_completeness", verdict=Verdict.SKIP,
                                       summary=f"Only {len(metrics)} entries, need 5"))

        # 2. Step monotonicity (no going backwards)
        if len(metrics) >= 3:
            steps = []
            for m in metrics[:10]:
                s = m.get("step") or m.get("global_step")
                if s is not None:
                    steps.append(int(s))
            if len(steps) >= 2:
                monotonic = all(steps[i] >= steps[i + 1] for i in range(len(steps) - 1))
                if not monotonic:
                    checks.append(CheckResult(
                        name="metrics_step_monotonic",
                        verdict=Verdict.FAIL,
                        summary=f"Steps not monotonic: {steps[:5]}",
                        detail="Non-monotonic steps = data corruption or reset",
                    ))
                else:
                    checks.append(CheckResult(name="metrics_step_monotonic", verdict=Verdict.PASS,
                                               summary=f"Steps monotonic: {steps[0]} → {steps[-1]}"))
            else:
                checks.append(CheckResult(name="metrics_step_monotonic", verdict=Verdict.SKIP,
                                           summary="No step field in metrics"))
        else:
            checks.append(CheckResult(name="metrics_step_monotonic", verdict=Verdict.SKIP,
                                       summary="Need ≥3 entries"))

        # 3. Entry count (enough logging)
        if len(metrics) >= 10:
            checks.append(CheckResult(
                name="metrics_entry_count",
                verdict=Verdict.PASS,
                summary=f"{len(metrics)} entries in metrics.jsonl",
                metric=float(len(metrics)),
                dimension="metrics_count",
            ))
        else:
            checks.append(CheckResult(name="metrics_entry_count", verdict=Verdict.WARN,
                                       summary=f"Only {len(metrics)} entries, may be incomplete",
                                       metric=float(len(metrics))))

        # 4. Timestamp freshness
        latest_ts = metrics[0].get("timestamp") or metrics[0].get("time")
        if latest_ts:
            checks.append(CheckResult(
                name="metrics_timestamp",
                verdict=Verdict.PASS,
                summary=f"Latest metric: {str(latest_ts)[:19]}",
                dimension="metrics_timestamp",
            ))
        else:
            checks.append(CheckResult(name="metrics_timestamp", verdict=Verdict.WARN,
                                       summary="No timestamp in latest entry",
                                       detail="Timestamps help track metric freshness"))

        # 5. Drift in field schemas across entries
        if len(metrics) >= 5:
            field_sets = [set(m.keys()) for m in metrics[:10]]
            base = field_sets[0]
            divergent = [fs for fs in field_sets[1:] if fs != base]
            if divergent:
                checks.append(CheckResult(
                    name="metrics_schema_drift",
                    verdict=Verdict.WARN,
                    summary=f"{len(divergent)} entries with different schema than first",
                    detail="Schema drift = logging format changed mid-training",
                ))
            else:
                checks.append(CheckResult(name="metrics_schema_drift", verdict=Verdict.PASS,
                                           summary="All entries share consistent schema"))
        else:
            checks.append(CheckResult(name="metrics_schema_drift", verdict=Verdict.SKIP,
                                       summary="Need ≥5 entries"))

        # 6. Subprocess log health (parent logging process)
        subproc_lines = datastore.get("log_tail", [])
        subproc_errors = [l for l in subproc_lines if "subprocess" in l.lower() and "error" in l.lower()]
        if subproc_errors:
            checks.append(CheckResult(
                name="subprocess_log_errors",
                verdict=Verdict.WARN,
                summary=f"{len(subproc_errors)} subprocess errors in log",
                detail=f"Errors: {subproc_errors[-3:]}",
            ))
        else:
            checks.append(CheckResult(name="subprocess_log_errors", verdict=Verdict.PASS,
                                       summary="No subprocess errors"))

        return AgentResult(agent_name=self.name, checks=checks)
