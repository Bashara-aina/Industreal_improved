"""Atomic file reloaders for log, state, metrics, config.

Each function reads a single source and returns parsed data.
Callers get a fresh snapshot every cycle from coordinator.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from typing import Any, Dict, List, Optional, Tuple


def load_state_json(path: str) -> Dict[str, Any]:
    """Parse rf_stage_state.json into a dict."""
    if not os.path.isfile(path):
        return {"_error": f"File not found: {path}"}
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        return {"_error": str(e)}


def load_metrics_jsonl(path: str) -> List[Dict[str, Any]]:
    """Parse metrics.jsonl into a list of dicts (latest first)."""
    if not os.path.isfile(path):
        return []
    records: List[Dict[str, Any]] = []
    try:
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    except (json.JSONDecodeError, OSError):
        pass
    records.reverse()
    return records


def tail_log(path: str, n_lines: int = 200) -> List[str]:
    """Return last n_lines of a log file."""
    if not os.path.isfile(path):
        return [f"File not found: {path}"]
    try:
        result = subprocess.run(
            ["tail", "-n", str(n_lines), path],
            capture_output=True, text=True, timeout=10,
        )
        return result.stdout.splitlines()
    except (subprocess.TimeoutExpired, OSError) as e:
        return [f"Error reading {path}: {e}"]


def search_log(path: str, pattern: str, max_matches: int = 20) -> List[str]:
    """Grep for pattern in log file, return matching lines."""
    if not os.path.isfile(path):
        return []
    try:
        result = subprocess.run(
            ["grep", "-aE", pattern, path],
            capture_output=True, text=True, timeout=30,
        )
        lines = result.stdout.splitlines()
        return lines[-max_matches:] if len(lines) > max_matches else lines
    except (subprocess.TimeoutExpired, OSError):
        return []


def load_ckpt_info(ckpt_dir: str) -> Dict[str, Any]:
    """Return info about checkpoint files: path, size, mtime."""
    info: Dict[str, Any] = {"files": [], "total_size_mb": 0.0}
    if not os.path.isdir(ckpt_dir):
        return info
    total = 0
    for fname in os.listdir(ckpt_dir):
        fpath = os.path.join(ckpt_dir, fname)
        if not os.path.isfile(fpath):
            continue
        stat = os.stat(fpath)
        size_mb = stat.st_size / (1024 * 1024)
        total += stat.st_size
        info["files"].append({
            "name": fname,
            "size_mb": round(size_mb, 1),
            "mtime": stat.st_mtime,
            "age_hours": (os.path.getmtime(fpath) - stat.st_mtime) / 3600 if False else 0,
        })
    info["total_size_mb"] = round(total / (1024 * 1024), 1)
    info["total_files"] = len(info["files"])
    info["files"].sort(key=lambda x: x["mtime"], reverse=True)
    return info


def get_training_pid(state: Dict[str, Any]) -> Optional[int]:
    """Extract training PID from state or probe via ps."""
    pid = state.get("training_pid")
    if pid and isinstance(pid, int):
        return pid
    return _probe_pid_via_ps()


def _probe_pid_via_ps() -> Optional[int]:
    """Fallback: find training PID via ps aux when no state file."""
    try:
        result = subprocess.run(
            ["ps", "aux"],
            capture_output=True, text=True, timeout=10,
        )
        for line in result.stdout.splitlines():
            if "train.py" in line and "--preset stage_rf2" in line:
                parts = line.split()
                if len(parts) >= 2:
                    return int(parts[1])
    except (subprocess.TimeoutExpired, OSError, ValueError):
        pass
    return None


def is_pid_alive(pid: int) -> bool:
    """Check if a process is alive via /proc."""
    try:
        return os.path.isdir(f"/proc/{pid}")
    except OSError:
        return False


def _compute_state_from_metrics(metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Infer training state from metrics.jsonl when rf_stage_state.json is missing."""
    state: Dict[str, Any] = {}
    if not metrics:
        return state

    records = list(reversed(metrics))  # chronological

    best_mAP50 = 0.0
    best_mAP50_95 = 0.0
    best_MAE = float("inf")
    best_epoch = 0
    metric_history: List[float] = []

    for r in records:
        ep = r.get("epoch", 0)
        v = r.get("val", {})
        m50 = v.get("det_mAP50", 0.0) or 0.0
        m95 = v.get("det_mAP_50_95", v.get("det_mAP50_95", 0.0)) or 0.0
        mae = v.get("forward_angular_MAE_deg", float("inf")) or float("inf")

        if isinstance(m50, (int, float)) and m50 > best_mAP50:
            best_mAP50 = m50
            best_mAP50_95 = m95 if isinstance(m95, (int, float)) else 0.0
            best_MAE = mae if isinstance(mae, (int, float)) else float("inf")
            best_epoch = ep

        if isinstance(m50, (int, float)):
            metric_history.append(m50)

    state["best_metric"] = best_mAP50
    state["best_map50_95"] = best_mAP50_95
    state["best_mae"] = best_MAE if best_MAE != float("inf") else 0.0
    state["best_metrics"] = {
        "det_mAP50": best_mAP50,
        "det_mAP50_95": best_mAP50_95 if best_mAP50_95 != float("inf") else 0.0,
        "forward_angular_MAE_deg": best_MAE if best_MAE != float("inf") else 0.0,
    }
    state["best_epoch"] = best_epoch
    state["metric_history"] = metric_history
    state["gate_passed"] = best_mAP50 >= 0.40
    state["total_epochs"] = len(records)
    state["subprocess_errors"] = 0
    state["_source"] = "inferred_from_metrics"
    return state


def get_gpu_info() -> Dict[str, Any]:
    """Query nvidia-smi for GPU stats. Returns defaults on failure."""
    default = {
        "util_pct": -1,
        "mem_used_gb": -1.0,
        "mem_total_gb": 12.0,
        "temp_c": -1,
        "power_w": -1,
        "error": None,
    }
    try:
        result = subprocess.run(
            [
                "nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,"
                "temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=10,
        )
        parts = result.stdout.strip().split(", ")
        if len(parts) >= 5:
            return {
                "util_pct": float(parts[0]),
                "mem_used_gb": round(float(parts[1]) / 1024, 2),
                "mem_total_gb": round(float(parts[2]) / 1024, 2),
                "temp_c": float(parts[3]),
                "power_w": float(parts[4]),
                "error": None,
            }
    except Exception as e:
        default["error"] = str(e)
    return default


def get_cpu_ram_gb() -> float:
    """Return available RAM in GB from /proc/meminfo."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    kb = int(line.split()[1])
                    return round(kb / (1024 * 1024), 1)
    except OSError:
        pass
    return -1.0


def parse_epoch_from_log(lines: List[str]) -> Optional[int]:
    """Extract current epoch number from log tail.

    Handles formats like:
      Epoch 17 [no-staging]
      --- Epoch 17/30 ---
    """
    for line in reversed(lines):
        m = re.search(r"(?:Epoch|epoch)\s+(\d+)", line)
        if m:
            return int(m.group(1))
    return None


def parse_step_from_log(lines: List[str]) -> Optional[int]:
    """Extract current step/batch from log tail."""
    for line in reversed(lines):
        m = re.search(r"step=(\d+)", line)
        if m:
            return int(m.group(1))
    return None


def parse_liveness_from_log(lines: List[str]) -> List[Dict[str, Any]]:
    """Extract LIVENESS lines from log."""
    results = []
    for line in lines:
        if "LIVENESS step=" not in line:
            continue
        results.append(line)
    return results


def _parse_liveness_grad(line: str) -> Optional[Dict[str, float]]:
    """Parse gradient norms from LIVENESS_GRAD or compact LIVENESS lines.

    Format 1 (LIVENESS_GRAD):
      detection_head:ALIVE[1.95e-02]/ALIVE[7.16e-02] | pose_head:ALIVE[...] | ...
      backbone:ALIVE[2.361e+00|n=178] | fpn:ALIVE[4.617e-01|n=16]

    Format 2 (compact LIVENESS):
      det=9.12e-01 ALIVE | head_pose=1.40e-03 ALIVE | pose=1.96e+00 ALIVE

    Extract: det, head_pose/pose, backbone, act, psr
    """
    result: Dict[str, float] = {}

    # Format 1: detection_head:ALIVE[1.95e-02]
    m = re.search(r"detection_head:ALIVE\[([\d.e+\-]+)", line)
    if m:
        result["det"] = float(m.group(1))
    m = re.search(r"backbone:ALIVE\[([\d.e+\-]+)", line)
    if m:
        result["backbone"] = float(m.group(1))
    m = re.search(r"head_pose_head:ALIVE\[([\d.e+\-]+)", line)
    if m:
        result["head_pose"] = float(m.group(1))
    m = re.search(r"pose_head:ALIVE\[([\d.e+\-]+)", line)
    if m:
        result["pose"] = float(m.group(1))
    # NO_GRAD heads
    if "activity_head" in line:
        result["act"] = 0.0
    if "psr_head" in line:
        result["psr"] = 0.0

    # Format 2: det=9.12e-01 ALIVE (compact)
    if "det=" in line:
        m = re.search(r"det=([\d.e+\-]+)", line)
        if m:
            result["det"] = float(m.group(1))
        m = re.search(r"pose=([\d.e+\-]+)", line)
        if m:
            result["pose"] = float(m.group(1))
        m = re.search(r"head_pose=([\d.e+\-]+)", line)
        if m:
            result["head_pose"] = float(m.group(1))

    return result or None


def gather_all() -> Dict[str, Any]:
    """Gather all data sources into one dict for agent consumption.

    This is called once per monitoring cycle.
    """
    from rf2_swarm import config as C

    state = load_state_json(C.STATE_JSON)
    log_tail = tail_log(C.TRAIN_LOG, n_lines=500)
    metrics = load_metrics_jsonl(C.METRICS_JSONL)

    # Infer state from metrics when state file is missing
    if "_error" in state:
        inferred = _compute_state_from_metrics(metrics)
        if inferred:
            state.update(inferred)

    # Probe PID from ps aux fallback (works w/o state file)
    pid = get_training_pid(state)

    epoch = parse_epoch_from_log(log_tail)
    if epoch is None:
        epoch = len(metrics)  # fallback: each metrics record = one completed epoch

    ds: Dict[str, Any] = {
        "state": state,
        "log_tail": log_tail,
        "metrics": metrics,
        "gpu": get_gpu_info(),
        "ram_gb": get_cpu_ram_gb(),
        "ckpt": load_ckpt_info(C.CKPT_DIR),
        "pid": pid,
        "pid_alive": pid is not None and is_pid_alive(pid),
        "epoch": epoch,
        "step": parse_step_from_log(log_tail),
        "liveness_lines": parse_liveness_from_log(log_tail),
        "e4_lines": [l for l in log_tail if "LIVENESS_GRAD" in l],
        "val_lines": [l for l in log_tail if "STEP VAL" in l or "EVAL" in l],
        "error_lines": [l for l in log_tail if "ERROR" in l or "CRITICAL" in l],
        # Targeted full-log searches (not limited to tail)
        "full_det_probe": search_log(C.TRAIN_LOG, r"DET_PROBE", max_matches=100),
        "full_val_lines": search_log(C.TRAIN_LOG, r"\[EVAL", max_matches=50),
        "full_val_metrics": search_log(C.TRAIN_LOG, r"(det_mAP50|det_mAP_50_95|forward_angular_MAE)", max_matches=30),
        "full_loss_lines": search_log(C.TRAIN_LOG, r"\[DEBUG epoch=[0-9]+ step=[0-9]+\]", max_matches=50),
        "_timestamp": __import__("time").time(),
    }

    # E4 gradient norms — try LIVENESS_GRAD first, fallback to compact LIVENESS
    ds["e4_grad_norms"] = {}
    for line in ds["e4_lines"]:
        parsed = _parse_liveness_grad(line)
        if parsed:
            ds["e4_grad_norms"] = parsed
    if not ds["e4_grad_norms"]:
        for line in ds["liveness_lines"]:
            parsed = _parse_liveness_grad(line)
            if parsed:
                ds["e4_grad_norms"] = parsed
                break

    return ds
