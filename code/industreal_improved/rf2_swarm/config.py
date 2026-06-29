"""Central configuration for the RF2 monitoring swarm.

All paths, thresholds, intervals and gate targets in one place.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = "/media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved"
RUNS_DIR = os.path.join(PROJECT_ROOT, "src", "runs", "rf_stages")
LOGS_DIR = os.path.join(RUNS_DIR, "logs")
CKPT_DIR = os.path.join(RUNS_DIR, "checkpoints")

TRAIN_LOG = os.path.join(LOGS_DIR, "train.log")
STATE_JSON = os.path.join(PROJECT_ROOT, "src", "runs", "rf_stage_state.json")
METRICS_JSONL = os.path.join(LOGS_DIR, "metrics.jsonl")
SUBPROCESS_LOG = os.path.join(LOGS_DIR, "subprocess.log")

# Swarm output
SWARM_OUTPUT_DIR = os.path.join(RUNS_DIR, "swarm_reports")
SWARM_RESULTS_JSON = os.path.join(SWARM_OUTPUT_DIR, "swarm_results.json")
SWARM_REPORT_TXT = os.path.join(SWARM_OUTPUT_DIR, "swarm_report.txt")

# ---------------------------------------------------------------------------
# Thresholds & Gate Targets  (mirrors stage_manager.py RF_STAGES[1] = rf2)
# ---------------------------------------------------------------------------
@dataclass
class RF2GateTargets:
    det_mAP50: float = 0.22  # present-class — IDEA PROVEN
    det_mAP50_95: float = 0.10
    forward_angular_MAE_deg: float = 70.0  # lower is better — must be ≤ this


@dataclass
class RF2HealthThresholds:
    min_grad_norm_det: float = 1e-8
    min_grad_norm_pose: float = 1e-8
    max_consecutive_dead: int = 10
    max_loss_spike_factor: float = 20.0


@dataclass
class RF2ConvergenceThresholds:
    patience_epochs: int = 12
    min_improvement: float = 0.002  # per 3-epoch window


@dataclass
class RF2ValidationThresholds:
    det_mAP50_min: float = 0.15  # absolute floor (warning)
    forward_angular_MAE_deg_max: float = 80.0


@dataclass
class RF2StabilityThresholds:
    max_grad_spike_epochs: int = 5
    min_liveness_ratio: float = 0.5


GATE = RF2GateTargets()
HEALTH = RF2HealthThresholds()
CONVERGENCE = RF2ConvergenceThresholds()
VALIDATION = RF2ValidationThresholds()
STABILITY = RF2StabilityThresholds()

# ---------------------------------------------------------------------------
# Epoch limits
# ---------------------------------------------------------------------------
DEFAULT_MAX_EPOCHS: int = 36  # actual max from training config; state overrides

# ---------------------------------------------------------------------------
# Monitoring intervals
# ---------------------------------------------------------------------------
POLL_INTERVAL_SECONDS: int = 300        # 5 min between cycles
AGENT_TIMEOUT_SECONDS: int = 120        # max per agent
HEARTBEAT_WARN_SECONDS: int = 180       # 3 min stale → WARN
HEARTBEAT_FAIL_SECONDS: int = 300       # 5 min stale → FAIL

# ---------------------------------------------------------------------------
# GPU  (swarm monitors primary training GPU: RTX 5060 Ti 16GB on CUDA 0)
# ---------------------------------------------------------------------------
GPU_TOTAL_MEM_GB: float = 16.0
VRAM_WARN_FRACTION: float = 0.85       # 10.2 GB → WARN
VRAM_FAIL_FRACTION: float = 0.95       # 11.4 GB → FAIL

# ---------------------------------------------------------------------------
# Epoch extrapolation
# ---------------------------------------------------------------------------
EPOCH_EXTRAPOLATION_WINDOW: int = 3    # last N epochs for trend

# ---------------------------------------------------------------------------
# Agent definitions  (name → description)
# ---------------------------------------------------------------------------
AGENT_DEFINITIONS: Dict[str, str] = {
    "gate_tracker": "det_mAP50, mAP50_95, MAE thresholds, best-vs-current, gate_passed flag",
    "probe_analyzer": "DET_PROBE results per epoch, mAP progress, class-level APs",
    "head_health": "DET/ASD/PSR heads ALIVE/DEAD, NaN weights, gradient norms",
    "loss_health": "det_cls/det_box/ASD/PSR loss values, plateau, divergence",
    "convergence": "Loss plateau over N epochs, metric stagnation, oscillation",
    "data_pipeline": "DataLoader workers, batch timing, cache hits, dataset sizes",
    "checkpoint": "File age, sizes, disk usage, corruption check, cleanup",
    "gpu_resource": "VRAM usage, util %, temperature, power, ECC errors",
    "validation": "Val runs completed, metric consistency, NaN in val metrics",
    "head_recovery": "Freezing/unfreezing detection, reinit tracking, LR changes",
    "metrics_logger": "Subprocess.log parser, metrics.jsonl completeness, drift",
    "gate_predictor": "Linear extrapolation from last N val epochs to gate targets",
    "process_health": "PID alive, heartbeat staleness, subprocess existence",
    "epoch_tracker": "Epoch progression rate, ETA, batch throughput",
    "nan_detector": "NaN/inf in loss values, metrics, weights via log patterns",
    "cuda_health": "CUDA errors, OOM events, NCCL failures, GPU visibility",
    "config_validator": "Training config consistency, model arch params",
    "log_anomaly": "Warning patterns, error frequency, unexpected log lines",
    "blocker_assessment": "Cross-cutting blocker summary, P0-P3 classification",
    "summary": "Executive summary, trend direction, recommended actions",
}

# Severity labels used across all agents
SEVERITY_LABELS: Dict[str, str] = {
    "PASS": "All checks OK",
    "WARN": "Non-blocking concern",
    "FAIL": "Blocker — requires intervention",
    "CRIT": "Immediate action required",
}

# ---------------------------------------------------------------------------
# Paths that must exist at startup
# ---------------------------------------------------------------------------
REQUIRED_PATHS: List[str] = [
    PROJECT_ROOT,
    RUNS_DIR,
    LOGS_DIR,
    CKPT_DIR,
]

# ---------------------------------------------------------------------------
# Allowed missing fields in state.json (expected during bootstrap)
# ---------------------------------------------------------------------------
ALLOWED_MISSING_STATE_FIELDS: set = {
    "det_health_history",
    "issues_log",
    "cross_stage_memory",
}
