"""Report generators — text format and JSON output."""
from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from rf2_swarm.base_agent import AgentResult, Verdict
from rf2_swarm.alerting import Alert


def generate_text_report(
    results: List[AgentResult],
    datastore: Dict[str, Any],
    cycle_num: int,
    alerts: Optional[List[Alert]] = None,
    deltas: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Generate a human-readable monitoring report."""
    lines: List[str] = []
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    lines.append(f"{'=' * 72}")
    lines.append(f"RF2 SWARM REPORT  —  Cycle #{cycle_num}  —  {ts}")
    lines.append(f"{'=' * 72}")
    lines.append("")

    # Datastore snapshot
    state = datastore.get("state", {})
    lines.append(f"Epoch:     {datastore.get('epoch', '?')}")
    lines.append(f"Step:      {datastore.get('step', '?')}")
    lines.append(f"PID alive: {datastore.get('pid_alive', '?')}")
    lines.append(f"Best mAP50: {state.get('best_metric', '?')}")
    lines.append(f"GPU util:  {datastore.get('gpu', {}).get('util_pct', '?')}%")
    lines.append(f"GPU mem:   {datastore.get('gpu', {}).get('mem_used_gb', '?')} GB")
    lines.append(f"E4 grad:   {datastore.get('e4_grad_norms', {})}")
    lines.append("")

    # Deltas
    if deltas:
        lines.append(f"{'─' * 40}  CHANGES SINCE LAST CYCLE  {'─' * 40}")
        for d in deltas:
            arrow = "⬆" if SEVERITY_ORDER.get(d["to"], -1) > SEVERITY_ORDER.get(d["from"], -1) else "⬇"
            lines.append(f"  {arrow} {d['key']}: {d['from']} → {d['to']}  —  {d['summary']}")
        lines.append("")

    # Alerts
    if alerts:
        lines.append(f"{'─' * 40}  ALERTS  {'─' * 40}")
        for a in alerts:
            lines.append(f"  [{a.verdict.value}] {a.agent}/{a.check}: {a.summary}")
        lines.append("")

    # Per-agent results
    lines.append(f"{'─' * 40}  AGENT RESULTS  {'─' * 40}")
    for ar in results:
        if ar.error:
            lines.append(f"  🔴 {ar.agent_name}: ERROR — {ar.error}")
            continue
        verdict_str = ar.worst.value
        lines.append(f"  {ar.agent_name}: {len(ar.checks)} checks, worst={verdict_str}, {ar.elapsed:.1f}s")
        for c in ar.checks:
            if c.verdict != Verdict.PASS:
                lines.append(f"    [{c.verdict.value}] {c.name}: {c.summary}")
        lines.append("")

    # Summary counts
    total = sum(len(ar.checks) for ar in results if not ar.error)
    passed = sum(1 for ar in results for c in ar.checks if c.verdict == Verdict.PASS)
    warns = sum(1 for ar in results for c in ar.checks if c.verdict == Verdict.WARN)
    fails = sum(1 for ar in results for c in ar.checks if c.verdict in (Verdict.FAIL, Verdict.CRIT))
    skipped = sum(1 for ar in results for c in ar.checks if c.verdict == Verdict.SKIP)
    lines.append(f"{'─' * 72}")
    lines.append(f"Total: {total}  |  PASS: {passed}  |  WARN: {warns}  |  FAIL: {fails}  |  SKIP: {skipped}")
    lines.append(f"{'=' * 72}")
    return "\n".join(lines)


SEVERITY_ORDER = {"PASS": 0, "SKIP": 1, "WARN": 2, "FAIL": 3, "CRIT": 4}


def generate_json_report(
    results: List[AgentResult],
    datastore: Dict[str, Any],
    cycle_num: int,
    alerts: Optional[List[Alert]] = None,
    deltas: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Generate a structured JSON report."""
    total_checks = sum(len(ar.checks) for ar in results if not ar.error)
    fail_count = sum(1 for ar in results for c in ar.checks if c.verdict in (Verdict.FAIL, Verdict.CRIT))
    warn_count = sum(1 for ar in results for c in ar.checks if c.verdict == Verdict.WARN)

    return {
        "cycle": cycle_num,
        "timestamp": datetime.utcnow().isoformat(),
        "summary": {
            "total_agents": len(results),
            "total_checks": total_checks,
            "fail": fail_count,
            "warn": warn_count,
            "agents_with_errors": sum(1 for ar in results if ar.error),
        },
        "datastore_snapshot": {
            "epoch": datastore.get("epoch"),
            "step": datastore.get("step"),
            "pid_alive": datastore.get("pid_alive"),
            "best_metric": datastore.get("state", {}).get("best_metric"),
            "gpu_util": datastore.get("gpu", {}).get("util_pct"),
            "gpu_mem_gb": datastore.get("gpu", {}).get("mem_used_gb"),
            "e4_grad_norms": datastore.get("e4_grad_norms"),
        },
        "alerts": [a.to_dict() for a in alerts] if alerts else [],
        "deltas": deltas if deltas else [],
        "agents": [ar.to_dict() for ar in results],
    }


def write_reports(
    results: List[AgentResult],
    datastore: Dict[str, Any],
    cycle_num: int,
    json_path: str,
    txt_path: str,
    alerts: Optional[List[Alert]] = None,
    deltas: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """Write both text and JSON reports to disk."""
    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    txt = generate_text_report(results, datastore, cycle_num, alerts, deltas)
    with open(txt_path, "w") as f:
        f.write(txt + "\n")

    j = generate_json_report(results, datastore, cycle_num, alerts, deltas)
    with open(json_path, "w") as f:
        json.dump(j, f, indent=2, default=str)

    print(txt[-2000:])  # tail to stdout for quick glance
