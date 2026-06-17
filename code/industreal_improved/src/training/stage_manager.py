#!/usr/bin/env python3
"""
RF1-RF10 Stage Manager — Progressive multi-task training orchestration.

Usage:
  python -m src.training.stage_manager --check       # Cron: evaluate & step
  python -m src.training.stage_manager --status      # Show current state
  python -m src.training.stage_manager --abort       # Kill current training
  python -m src.training.stage_manager --launch RF1  # Force-launch a stage
  python -m src.training.stage_manager --reset       # Reset state (fresh start)

Architecture:
  - Stateless cron invocation: reads state from JSON, acts, writes state back
  - Process management: subprocess.Popen + PID file for training subprocess
  - Log parsing: reads train.log for loss/LIVENESS/validation metrics
  - Decision engine: 5-category checklists (Gate/Health/Convergence/Validation/Stability)
  - 20-why root cause analysis: rule-based from failure patterns
"""

import argparse
import json
import logging
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

# =========================================================================
# Constants
# =========================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent  # industreal_improved/
TRAIN_SCRIPT = PROJECT_ROOT / 'src' / 'training' / 'train.py'
RUNS_DIR = PROJECT_ROOT / 'src' / 'runs'
STATE_FILE = RUNS_DIR / 'rf_stage_state.json'
RF_RUN_DIR = RUNS_DIR / 'rf_stages'
TRAIN_LOG = RF_RUN_DIR / 'logs' / 'train.log'
CKPT_DIR = RF_RUN_DIR / 'checkpoints'
LATEST_CKPT = CKPT_DIR / 'latest.pth'
BEST_CKPT = CKPT_DIR / 'best.pth'

# Environment marker so the training subprocess knows it's under stage_manager
STAGE_MANAGER_ENV_VAR = '_STAGE_MANAGER_ACTIVE=1'

# Signal file: train.py writes this when all gate thresholds are met
TARGET_MET_FILE = RF_RUN_DIR / '.stage_target_met'

# PID lock file: tracks the training PID to prevent duplicate launches
PID_LOCK_FILE = RF_RUN_DIR / '.training_pid'

# Max number of train.py processes allowed (1 main + DataLoader workers)
MAX_TRAIN_PIDS = 1

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger('stage_manager')

# =========================================================================
# Stage Definitions
# =========================================================================

RF_STAGES = [
    {
        'name': 'rf1',
        'description': 'Detection only',
        'preset': 'stage_rf1',
        'subset_ratio': 0.20,
        'max_epochs': 20,
        'active_heads': 'det',
        # GATE: must achieve these to transition
        'gate': {
            'det_mAP50': 0.30,       # min mAP@0.50 on val
            'det_mAP50_95': 0.12,    # min mAP@0.50:0.95
        },
        # HEALTH: liveness & gradient sanity
        'health': {
            'min_grad_norm_det': 1e-6,
            'max_consecutive_dead': 5,        # epochs of DEAD det before kill
            'max_loss_spike_factor': 10.0,     # loss spike > 10x rolling mean
        },
        # CONVERGENCE: rate of improvement
        'convergence': {
            'patience_epochs': 8,             # epochs without det_mAP50 improvement
            'min_improvement': 0.005,          # min delta per 3-epoch window
        },
        # VALIDATION: metric floors (below these = FAIL)
        'validation': {
            'det_mAP50_min': 0.25,            # absolute floor (not gate, but warning)
            'max_pose_MAE': float('inf'),     # no pose yet
        },
        # STABILITY: training dynamics
        'stability': {
            'max_grad_spike_epochs': 3,       # epochs with grad > 10x median
            'min_liveness_ratio': 0.7,         # fraction of steps with ALL heads alive
        },
        'reinit_heads': True,  # reinit detection head for fresh start
        # [RF1 FIX 2026-06-17] Do NOT detach the regression gradient for the
        # detection-bootstrap stage. Without this override, launch_training()
        # defaults detach_reg_fpn to reinit_heads (True) and passes
        # --detach-reg-fpn, which stops the GIoU/box-regression gradient from
        # reaching the shared FPN/backbone. With a freshly reinit head and all
        # other heads off, that leaves the backbone with ONLY the sparse
        # classification path — features never become object-discriminative, so
        # the cls conv cannot separate fg from bg and the head sticks at the
        # pi=0.01 "predict-background-everywhere" equilibrium (localizes but won't
        # fire). The reg-loss warmup (REINIT_REG_WARMUP_STEPS) is the correct,
        # sufficient guard against reinit gradient shock; the detach was redundant
        # overkill that starved the trunk. This matches recovery_det_only, which
        # trains detection from reinit WITHOUT detaching regression.
        'detach_reg_fpn': False,
        'resume_source': 'latest',  # resume from previous stage's latest.pth
    },
    {
        'name': 'rf2',
        'description': 'Detection + Body/Head Pose',
        'preset': 'stage_rf2',
        'subset_ratio': 0.35,
        'max_epochs': 15,
        'active_heads': 'det+pose',
        'gate': {
            'det_mAP50': 0.40,
            'det_mAP50_95': 0.18,
            'forward_angular_MAE_deg': 60.0,  # max MAE (lower is better)
        },
        'health': {
            'min_grad_norm_det': 1e-6,
            'min_grad_norm_pose': 1e-6,
            'max_consecutive_dead': 5,
            'max_loss_spike_factor': 10.0,
        },
        'convergence': {
            'patience_epochs': 6,
            'min_improvement': 0.003,
        },
        'validation': {
            'det_mAP50_min': 0.35,
            'forward_angular_MAE_deg_max': 70.0,
        },
        'stability': {
            'max_grad_spike_epochs': 3,
            'min_liveness_ratio': 0.7,
        },
        'resume_source': 'best',
    },
    {
        'name': 'rf3',
        'description': 'Detection + Pose + Activity',
        'preset': 'stage_rf3',
        'subset_ratio': 0.35,
        'max_epochs': 15,
        'active_heads': 'det+pose+act',
        'gate': {
            'det_mAP50': 0.45,
            'det_mAP50_95': 0.20,
            'act_top1': 0.40,              # clip-level top-1 accuracy
            'forward_angular_MAE_deg': 55.0,
        },
        'health': {
            'min_grad_norm_det': 1e-6,
            'min_grad_norm_pose': 1e-6,
            'min_grad_norm_act': 1e-6,
            'max_consecutive_dead': 5,
            'max_loss_spike_factor': 10.0,
        },
        'convergence': {
            'patience_epochs': 6,
            'min_improvement': 0.003,
        },
        'validation': {
            'det_mAP50_min': 0.40,
            'act_top1_min': 0.30,
            'forward_angular_MAE_deg_max': 65.0,
        },
        'stability': {
            'max_grad_spike_epochs': 3,
            'min_liveness_ratio': 0.65,
        },
        'resume_source': 'best',
    },
    {
        'name': 'rf4',
        'description': 'All heads + PSR (transition enabled)',
        'preset': 'stage_rf4',
        'subset_ratio': 0.50,
        'max_epochs': 20,
        'active_heads': 'all',
        'gate': {
            'det_mAP50': 0.50,
            'act_top1': 0.45,
            'psr_f1_at_t': 0.25,           # F1 at threshold
            'forward_angular_MAE_deg': 50.0,
        },
        'health': {
            'min_grad_norm_det': 1e-6,
            'min_grad_norm_pose': 1e-6,
            'min_grad_norm_act': 1e-6,
            'min_grad_norm_psr': 1e-6,
            'max_consecutive_dead': 5,
            'max_loss_spike_factor': 10.0,
            'psr_bias_gradient_check': True,  # verify psr bias gets non-zero grad
        },
        'convergence': {
            'patience_epochs': 8,
            'min_improvement': 0.002,
        },
        'validation': {
            'det_mAP50_min': 0.45,
            'act_top1_min': 0.35,
            'psr_f1_min': 0.15,
            'forward_angular_MAE_deg_max': 60.0,
        },
        'stability': {
            'max_grad_spike_epochs': 4,
            'min_liveness_ratio': 0.6,
        },
        'resume_source': 'best',
    },
    {
        'name': 'rf5',
        'description': 'Consolidate all heads',
        'preset': 'stage_rf5',
        'subset_ratio': 0.50,
        'max_epochs': 10,
        'active_heads': 'all',
        'gate': {
            'det_mAP50': 0.55,
            'act_top1': 0.50,
            'psr_f1_at_t': 0.30,
            'forward_angular_MAE_deg': 45.0,
        },
        'health': {
            'min_grad_norm_det': 1e-6,
            'min_grad_norm_pose': 1e-6,
            'min_grad_norm_act': 1e-6,
            'min_grad_norm_psr': 1e-6,
            'max_consecutive_dead': 5,
            'max_loss_spike_factor': 8.0,
        },
        'convergence': {
            'patience_epochs': 5,
            'min_improvement': 0.002,
        },
        'validation': {
            'det_mAP50_min': 0.50,
            'act_top1_min': 0.40,
            'psr_f1_min': 0.20,
            'forward_angular_MAE_deg_max': 55.0,
        },
        'stability': {
            'max_grad_spike_epochs': 3,
            'min_liveness_ratio': 0.65,
        },
        'resume_source': 'best',
    },
    {
        'name': 'rf6',
        'description': 'Scale data to 65%',
        'preset': 'stage_rf6',
        'subset_ratio': 0.65,
        'max_epochs': 10,
        'active_heads': 'all',
        'gate': {
            'det_mAP50': 0.58,
            'act_top1': 0.52,
            'psr_f1_at_t': 0.35,
            'forward_angular_MAE_deg': 42.0,
        },
        'health': {
            'min_grad_norm_det': 1e-6,
            'min_grad_norm_pose': 1e-6,
            'min_grad_norm_act': 1e-6,
            'min_grad_norm_psr': 1e-6,
            'max_consecutive_dead': 3,
            'max_loss_spike_factor': 8.0,
        },
        'convergence': {
            'patience_epochs': 5,
            'min_improvement': 0.003,
        },
        'validation': {
            'det_mAP50_min': 0.52,
            'act_top1_min': 0.42,
            'psr_f1_min': 0.25,
            'forward_angular_MAE_deg_max': 50.0,
        },
        'stability': {
            'max_grad_spike_epochs': 3,
            'min_liveness_ratio': 0.7,
        },
        'resume_source': 'best',
    },
    {
        'name': 'rf7',
        'description': 'Continue at 65%',
        'preset': 'stage_rf7',
        'subset_ratio': 0.65,
        'max_epochs': 10,
        'active_heads': 'all',
        'gate': {
            'det_mAP50': 0.62,
            'act_top1': 0.55,
            'psr_f1_at_t': 0.40,
            'forward_angular_MAE_deg': 40.0,
        },
        'health': {
            'min_grad_norm_det': 1e-6,
            'min_grad_norm_pose': 1e-6,
            'min_grad_norm_act': 1e-6,
            'min_grad_norm_psr': 1e-6,
            'max_consecutive_dead': 3,
            'max_loss_spike_factor': 8.0,
        },
        'convergence': {
            'patience_epochs': 5,
            'min_improvement': 0.003,
        },
        'validation': {
            'det_mAP50_min': 0.55,
            'act_top1_min': 0.45,
            'psr_f1_min': 0.30,
            'forward_angular_MAE_deg_max': 48.0,
        },
        'stability': {
            'max_grad_spike_epochs': 3,
            'min_liveness_ratio': 0.7,
        },
        'resume_source': 'best',
    },
    {
        'name': 'rf8',
        'description': 'Scale data to 80%',
        'preset': 'stage_rf8',
        'subset_ratio': 0.80,
        'max_epochs': 10,
        'active_heads': 'all',
        'gate': {
            'det_mAP50': 0.65,
            'act_top1': 0.58,
            'psr_f1_at_t': 0.45,
            'forward_angular_MAE_deg': 38.0,
        },
        'health': {
            'min_grad_norm_det': 1e-6,
            'min_grad_norm_pose': 1e-6,
            'min_grad_norm_act': 1e-6,
            'min_grad_norm_psr': 1e-6,
            'max_consecutive_dead': 3,
            'max_loss_spike_factor': 8.0,
        },
        'convergence': {
            'patience_epochs': 5,
            'min_improvement': 0.003,
        },
        'validation': {
            'det_mAP50_min': 0.58,
            'act_top1_min': 0.48,
            'psr_f1_min': 0.35,
            'forward_angular_MAE_deg_max': 45.0,
        },
        'stability': {
            'max_grad_spike_epochs': 3,
            'min_liveness_ratio': 0.7,
        },
        'resume_source': 'best',
    },
    {
        'name': 'rf9',
        'description': 'Scale data to 90%',
        'preset': 'stage_rf9',
        'subset_ratio': 0.90,
        'max_epochs': 10,
        'active_heads': 'all',
        'gate': {
            'det_mAP50': 0.70,
            'act_top1': 0.60,
            'psr_f1_at_t': 0.50,
            'forward_angular_MAE_deg': 35.0,
        },
        'health': {
            'min_grad_norm_det': 1e-6,
            'min_grad_norm_pose': 1e-6,
            'min_grad_norm_act': 1e-6,
            'min_grad_norm_psr': 1e-6,
            'max_consecutive_dead': 3,
            'max_loss_spike_factor': 8.0,
        },
        'convergence': {
            'patience_epochs': 5,
            'min_improvement': 0.003,
        },
        'validation': {
            'det_mAP50_min': 0.62,
            'act_top1_min': 0.50,
            'psr_f1_min': 0.38,
            'forward_angular_MAE_deg_max': 42.0,
        },
        'stability': {
            'max_grad_spike_epochs': 3,
            'min_liveness_ratio': 0.7,
        },
        'resume_source': 'best',
    },
    {
        'name': 'rf10',
        'description': 'Final full-data push — paper results',
        'preset': 'stage_rf10',
        'subset_ratio': 1.0,
        'max_epochs': 15,
        'active_heads': 'all',
        'gate': {
            'det_mAP50': 0.75,
            'det_mAP50_95': 0.35,
            'act_top1': 0.63,
            'psr_f1_at_t': 0.55,
            'forward_angular_MAE_deg': 30.0,
        },
        'health': {
            'min_grad_norm_det': 1e-6,
            'min_grad_norm_pose': 1e-6,
            'min_grad_norm_act': 1e-6,
            'min_grad_norm_psr': 1e-6,
            'max_consecutive_dead': 3,
            'max_loss_spike_factor': 8.0,
        },
        'convergence': {
            'patience_epochs': 5,
            'min_improvement': 0.003,
        },
        'validation': {
            'det_mAP50_min': 0.68,
            'det_mAP50_95_min': 0.30,
            'act_top1_min': 0.55,
            'psr_f1_min': 0.45,
            'forward_angular_MAE_deg_max': 38.0,
        },
        'stability': {
            'max_grad_spike_epochs': 3,
            'min_liveness_ratio': 0.75,
        },
        'resume_source': 'best',
    },
]

_STAGE_BY_NAME = {s['name']: s for s in RF_STAGES}

# Paper target baselines (for final comparison)
PAPER_BASELINES = {
    'det_mAP50': 0.838,       # YOLOv8m
    'det_mAP50_95': 0.45,     # estimated
    'act_top1': 0.6525,       # MViTv2
    'act_top5': 0.90,         # estimated
    'psr_f1_at_t': 0.731,     # B2
    'forward_angular_MAE_deg': 10.0,  # estimated SOTA
}


# =========================================================================
# State Management
# =========================================================================

@dataclass
class StageState:
    current_stage: str = 'rf1'
    stage_index: int = 0
    status: str = 'idle'  # idle | running | paused | completed | failed | retrying
    training_pid: Optional[int] = None
    epoch: int = 0
    best_metric: float = 0.0
    best_metrics: Dict[str, float] = field(default_factory=dict)
    gate_passed: bool = False
    checklist_results: Dict[str, Any] = field(default_factory=lambda: {
        'gate': {'passed': False, 'details': {}},
        'health': {'passed': False, 'details': {}},
        'convergence': {'passed': False, 'details': {}},
        'validation': {'passed': False, 'details': {}},
        'stability': {'passed': False, 'details': {}},
    })
    # Per-stage metric history for trend analysis
    metric_history: List[Dict[str, float]] = field(default_factory=list)
    # Retry tracking for auto-tuning
    retry_count: int = 0
    current_strategy: str = 'default'
    strategies_tried: List[str] = field(default_factory=list)
    # DET-HEALTH tracking
    det_health_history: List[Dict[str, Any]] = field(default_factory=list)
    # Stage completion history
    stage_history: List[Dict[str, Any]] = field(default_factory=list)
    issues_log: List[Dict[str, Any]] = field(default_factory=list)
    last_check_time: Optional[str] = None
    run_start_time: Optional[str] = None
    # Log cursor for incremental reading — tracks bytes consumed in train.log
    log_cursor: int = 0


@dataclass
class LogSnapshot:
    """Single-pass log parse result — avoids repeated line iteration across 5 checklists.

    All fields extracted from one pass over the raw lines.
    """
    epoch: int = 0
    metrics: Dict[str, float] = field(default_factory=dict)
    liveness: Dict[str, Any] = field(default_factory=dict)
    det_health_lines: List[Dict[str, Any]] = field(default_factory=list)
    det_debug_lines: List[Dict[str, Any]] = field(default_factory=list)
    loss_trend: Dict[str, float] = field(default_factory=dict)
    crash_count: int = 0


def load_state() -> StageState:
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE) as f:
                data = json.load(f)
            known_fields = set(StageState.__dataclass_fields__.keys())
            filtered = {k: v for k, v in data.items() if k in known_fields}
            return StageState(**filtered)
        except Exception as e:
            logger.warning(f'Failed to load state file ({e}), starting fresh.')
    return StageState()


def save_state(state: StageState) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, 'w') as f:
        json.dump(asdict(state), f, indent=2, default=str)


# =========================================================================
# Log Parsing
# =========================================================================

def read_train_log(tail_lines: int = 200) -> List[str]:
    """Read the last N lines of train.log (one-off use)."""
    if not TRAIN_LOG.exists():
        return []
    try:
        with open(TRAIN_LOG) as f:
            lines = f.readlines()
        return lines[-tail_lines:]
    except Exception as e:
        logger.warning(f'Failed to read train.log: {e}')
        return []


def read_train_log_incr(cursor: int) -> Tuple[List[str], int]:
    """Incremental log read: only bytes after *cursor*, returns (lines, new_cursor).

    On first call (cursor=0) or truncation, falls back to tail 500 so
    trend analysis has enough data. Stores cursor in state for next check.
    """
    if not TRAIN_LOG.exists():
        return [], 0
    try:
        with open(TRAIN_LOG) as f:
            size = f.seek(0, 2)  # seek to end, get size
            if cursor <= 0 or cursor > size:
                # First check or truncated log — read tail
                cursor = max(0, size - 32768)  # ~32KB tail
                f.seek(cursor)
                # Skip partial line
                remaining = f.readlines()
                # Always ensure at least 500 lines for trend context
                if len(remaining) < 500:
                    f.seek(0)
                    remaining = f.readlines()
                return remaining[-500:], size
            f.seek(cursor)
            new_lines = f.readlines()
            return new_lines, size
    except Exception as e:
        logger.warning(f'Failed incremental read: {e}')
        return [], cursor


_PARSE_LOSS_RE = re.compile(
    r'det=([\d.e+-]+)\(c=([\d.e+-]+)\s+g=([\d.e+-]+)\)\s+'
    r'pose=([\d.e+-]+)\s+'
    r'act=([\d.e+-]+)\s+'
    r'psr=([\d.e+-]+)\s+'
    r'wd=([\d.e+-]+)'
)

# RF1-RF10 5-head liveness format:
#   det=6.25e+00 ALIVE | act=0.00e+00 DEAD | psr=0.00e+00 DEAD | head_pose=1.00e-06 DEAD | pose=1.00e-06 DEAD
_PARSE_LIVENESS_RE = re.compile(
    r'det=([\d.e+-]+)\s+(DEAD|ALIVE)\s*\|\s*'
    r'act=([\d.e+-]+)\s+(DEAD|ALIVE)\s*\|\s*'
    r'psr=([\d.e+-]+)\s+(DEAD|ALIVE)\s*\|\s*'
    r'head_pose=([\d.e+-]+)\s+(DEAD|ALIVE)\s*\|\s*'
    r'pose=([\d.e+-]+)\s+(DEAD|ALIVE)'
)



def parse_loss_line(line: str) -> Optional[Dict[str, float]]:
    m = _PARSE_LOSS_RE.search(line)
    if m:
        return {
            'det_total': float(m.group(1)),
            'det_cls': float(m.group(2)),
            'det_reg': float(m.group(3)),
            'pose': float(m.group(4)),
            'act': float(m.group(5)),
            'psr': float(m.group(6)),
            'wd': float(m.group(7)),
        }
    return None


def parse_liveness_line(line: str) -> Optional[Dict[str, Any]]:
    m = _PARSE_LIVENESS_RE.search(line)
    if m:
        return {
            'det_grad': float(m.group(1)),
            'det_status': m.group(2),
            'act_grad': float(m.group(3)),
            'act_status': m.group(4),
            'psr_grad': float(m.group(5)),
            'psr_status': m.group(6),
            'head_pose_grad': float(m.group(7)),
            'head_pose_status': m.group(8),
            'pose_grad': float(m.group(9)),
            'pose_status': m.group(10),
        }
    return None


def parse_val_metrics(lines: List[str]) -> Optional[Dict[str, float]]:
    """Extract the most recent validation metrics from log lines."""
    metrics = {}
    for line in reversed(lines):
        if 'Val:' in line and 'det_mAP50' in line:
            # Extract all metric=value pairs
            parts = line.strip().split()
            for part in parts:
                if '=' in part:
                    key, val = part.split('=', 1)
                    try:
                        metrics[key] = float(val)
                    except ValueError:
                        pass
            if metrics:
                return metrics
    return None


def parse_current_epoch(lines: List[str]) -> int:
    """Extract the most recent epoch number from log lines."""
    for line in reversed(lines):
        m = re.search(r'--- Epoch (\d+)/(\d+) ---', line)
        if m:
            return int(m.group(1))
    return 0


def parse_liveness_summary(lines: List[str]) -> Dict[str, Any]:
    """Aggregate liveness info from all LIVENESS lines in the log tail."""
    det_statuses: List[str] = []
    act_statuses: List[str] = []
    psr_statuses: List[str] = []
    head_pose_statuses: List[str] = []
    pose_statuses: List[str] = []

    for line in lines:
        if 'LIVENESS' in line:
            l = parse_liveness_line(line)
            if l:
                det_statuses.append(l['det_status'])
                act_statuses.append(l['act_status'])
                psr_statuses.append(l['psr_status'])
                head_pose_statuses.append(l['head_pose_status'])
                pose_statuses.append(l['pose_status'])

    total = len(det_statuses) or 1
    return {
        'det_alive_ratio': sum(1 for s in det_statuses if s == 'ALIVE') / total,
        'act_alive_ratio': sum(1 for s in act_statuses if s == 'ALIVE') / total,
        'psr_alive_ratio': sum(1 for s in psr_statuses if s == 'ALIVE') / total,
        'head_pose_alive_ratio': sum(1 for s in head_pose_statuses if s == 'ALIVE') / total,
        'pose_alive_ratio': sum(1 for s in pose_statuses if s == 'ALIVE') / total,
        'total_checks': len(det_statuses),
    }


def parse_loss_trend(lines: List[str]) -> Dict[str, float]:
    """Compute recent loss stats for stability checking."""
    det_losses = []
    act_losses = []
    for line in lines:
        p = parse_loss_line(line)
        if p:
            det_losses.append(p['det_total'])
            act_losses.append(p['act'])
    if not det_losses:
        return {}
    recent = det_losses[-50:] if len(det_losses) > 50 else det_losses
    mean_loss = sum(recent) / len(recent)
    return {
        'det_loss_mean': mean_loss,
        'det_loss_max': max(recent),
        'det_loss_min': min(recent),
        'det_loss_spike_factor': max(recent) / (mean_loss + 1e-10) if mean_loss > 0 else 1.0,
    }


def parse_det_health_lines(lines: List[str]) -> List[Dict[str, Any]]:
    """Extract DET-HEALTH snapshots from log lines.

    Matches: [DET-HEALTH step=N] cls_preds: mean=X std=Y near_zero=Z
    Returns chronologically ordered list.
    """
    results = []
    pat = re.compile(
        r'\[DET-HEALTH step=(\d+)\]\s+cls_preds:\s+'
        r'mean=([\d.-]+)\s+std=([\d.-]+)\s+'
        r'near_zero=([\d.]+)'
    )
    for line in lines:
        m = pat.search(line)
        if m:
            results.append({
                'step': int(m.group(1)),
                'cls_mean': float(m.group(2)),
                'cls_std': float(m.group(3)),
                'near_zero': float(m.group(4)),
            })
    return results


def parse_det_debug_lines(lines: List[str]) -> List[Dict[str, Any]]:
    """Extract DET-DEBUG lines for logit distribution stats."""
    results = []
    pat = re.compile(
        r'\[DET-DEBUG step=(\d+)\]\s+det tally:\s+'
        r'floor\(<1e-5\)=([\d.]+)\s+'
        r'alive\(>0\.1\)=([\d.]+)\s+'
        r'total_window=(\d+)\s+'
        r'floor_frac=([\d.]+)'
    )
    for line in lines:
        m = pat.search(line)
        if m:
            results.append({
                'step': int(m.group(1)),
                'floor_count': float(m.group(2)),
                'alive_count': float(m.group(3)),
                'total_window': int(m.group(4)),
                'floor_frac': float(m.group(5)),
            })
    return results


def parse_log_snapshot(lines: List[str]) -> LogSnapshot:
    """Single pass over log lines: extract epoch, metrics, liveness, det_health, loss trend, crashes.

    Replaces 7 separate line iterations with one. Returns a LogSnapshot
    consumed by all checklist evaluations.
    """
    snap = LogSnapshot()
    if not lines:
        return snap

    epoch_pat = re.compile(r'--- Epoch (\d+)/(\d+) ---')
    det_health_pat = re.compile(
        r'\[DET-HEALTH step=(\d+)\]\s+cls_preds:\s+'
        r'mean=([\d.-]+)\s+std=([\d.-]+)\s+'
        r'near_zero=([\d.]+)'
    )
    crash_pats = ['CUDA out of memory', 'RuntimeError', 'Traceback', 'Killed']

    det_losses: List[float] = []
    det_statuses: List[str] = []
    act_statuses: List[str] = []
    psr_statuses: List[str] = []
    head_pose_statuses: List[str] = []
    pose_statuses: List[str] = []

    for line in lines:
        # ── Epoch (last match wins, reversed order not needed) ──
        m = epoch_pat.search(line)
        if m:
            snap.epoch = int(m.group(1))

        # ── Validation metrics (last occurrence = most recent) ──
        if 'Val:' in line and 'det_mAP50' in line:
            parts = line.strip().split()
            for part in parts:
                if '=' in part:
                    k, v = part.split('=', 1)
                    try:
                        snap.metrics[k] = float(v)
                    except ValueError:
                        pass

        # ── DET-HEALTH ──
        m = det_health_pat.search(line)
        if m:
            snap.det_health_lines.append({
                'step': int(m.group(1)),
                'cls_mean': float(m.group(2)),
                'cls_std': float(m.group(3)),
                'near_zero': float(m.group(4)),
            })

        # ── DET-DEBUG tally (floor-fraction tracking) ──
        if '[DET-DEBUG' in line and 'det tally:' in line:
            dd_pat = re.compile(
                r'floor\(<1e-5\)=([\d.]+)\s+'
                r'alive\(>0\.1\)=([\d.]+)\s+'
                r'total_window=(\d+)\s+'
                r'floor_frac=([\d.]+)'
            )
            dm = dd_pat.search(line)
            if dm:
                snap.det_debug_lines.append({
                    'floor_count': float(dm.group(1)),
                    'alive_count': float(dm.group(2)),
                    'total_window': int(dm.group(3)),
                    'floor_frac': float(dm.group(4)),
                })

        # ── Loss values (det=... lines) ──
        p = parse_loss_line(line)
        if p:
            det_losses.append(p['det_total'])

        # ── Liveness ──
        if 'LIVENESS' in line:
            l = parse_liveness_line(line)
            if l:
                det_statuses.append(l['det_status'])
                act_statuses.append(l['act_status'])
                psr_statuses.append(l['psr_status'])
                head_pose_statuses.append(l['head_pose_status'])
                pose_statuses.append(l['pose_status'])

        # ── Crash patterns ──
        for pat in crash_pats:
            if pat in line:
                snap.crash_count += 1
                break

    # ── Compute aggregate stats after loop ──
    total = len(det_statuses) or 1
    snap.liveness = {
        'det_alive_ratio': sum(1 for s in det_statuses if s == 'ALIVE') / total,
        'act_alive_ratio': sum(1 for s in act_statuses if s == 'ALIVE') / total,
        'psr_alive_ratio': sum(1 for s in psr_statuses if s == 'ALIVE') / total,
        'head_pose_alive_ratio': sum(1 for s in head_pose_statuses if s == 'ALIVE') / total,
        'pose_alive_ratio': sum(1 for s in pose_statuses if s == 'ALIVE') / total,
        'total_checks': len(det_statuses),
    }

    if det_losses:
        recent = det_losses[-50:]
        mean_loss = sum(recent) / len(recent)
        snap.loss_trend = {
            'det_loss_mean': mean_loss,
            'det_loss_max': max(recent),
            'det_loss_min': min(recent),
            'det_loss_spike_factor': max(recent) / (mean_loss + 1e-10) if mean_loss > 0 else 1.0,
        }

    return snap


def assess_det_health(det_health_lines: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Assess detection head health from DET-HEALTH history."""
    if not det_health_lines:
        return {'status': 'UNKNOWN', 'reason': 'No DET-HEALTH data found'}
    latest = det_health_lines[-1]
    cls_mean = latest['cls_mean']
    cls_std = latest['cls_std']
    near_zero = latest['near_zero']

    issues = []
    # Healthy cls_mean target: ≈ -2.2 (pi=0.1 sigmoid init)
    if cls_mean < -15:
        issues.append(f'cls_mean={cls_mean:.2f} — severely negative, logits collapsed')
    elif cls_mean < -8:
        issues.append(f'cls_mean={cls_mean:.2f} — very negative, weak positive signal')
    elif cls_mean < -4:
        issues.append(f'cls_mean={cls_mean:.2f} — moderately negative, may slow convergence')
    elif -3 < cls_mean < -1.5:
        pass  # ideal range

    # near_zero ratio: fraction of logits with abs < 0.01
    if near_zero > 0.5:
        issues.append(f'near_zero={near_zero:.2%} — majority logits near zero, head may be collapsing')
    elif near_zero > 0.2:
        issues.append(f'near_zero={near_zero:.2%} — significant logit mass near zero')

    # std: too low means logits are all the same (degenerate)
    if cls_std < 0.1:
        issues.append(f'cls_std={cls_std:.4f} — extremely low, all logits nearly equal (degenerate)')
    elif cls_std < 0.5:
        issues.append(f'cls_std={cls_std:.4f} — low variance, watch for collapse')

    status = 'HEALTHY' if not issues else 'WARN' if len(issues) <= 1 else 'UNHEALTHY'
    return {
        'status': status,
        'cls_mean': cls_mean,
        'cls_std': cls_std,
        'near_zero': near_zero,
        'issues': issues,
        'trend': _det_health_trend(det_health_lines),
    }


def _det_health_trend(history: List[Dict[str, Any]]) -> str:
    """Analyze DET-HEALTH trend: improving, stable, worsening."""
    if len(history) < 3:
        return 'insufficient_data'
    recent = history[-3:]
    means = [h['cls_mean'] for h in recent]
    # cls_mean should move toward -2.2 from negative
    if all(m >= -3 for m in means):
        return 'healthy'
    if means[-1] > means[0] + 1:
        return 'improving'
    if means[-1] < means[0] - 1:
        return 'worsening'
    return 'stable'


# =========================================================================
# Retry Strategies — Auto-tuning for when checklists fail
# =========================================================================

RETRY_STRATEGIES = [
    {
        'name': 'default',
        'description': 'Same config, fresh retry',
        'lr_mult': 1.0,
        'warmup_mult': 1.0,
        'reinit_heads': True,
        'seed_offset': 0,
    },
    {
        'name': 'reduce_lr_5x',
        'description': 'Reduce learning rate 5×, reinit heads',
        'lr_mult': 0.2,
        'warmup_mult': 1.0,
        'reinit_heads': True,
        'seed_offset': 1,
    },
    {
        'name': 'reduce_lr_2x_warmup_2x',
        'description': 'Reduce LR 2×, double warmup, reinit heads',
        'lr_mult': 0.5,
        'warmup_mult': 2.0,
        'reinit_heads': True,
        'seed_offset': 2,
    },
    {
        'name': 'reduce_lr_10x_warmup_2x',
        'description': 'Reduce LR 10×, double warmup, reinit heads',
        'lr_mult': 0.1,
        'warmup_mult': 2.0,
        'reinit_heads': True,
        'seed_offset': 3,
    },
    {
        'name': 'reduce_lr_20x_warmup_3x',
        'description': 'Reduce LR 20×, triple warmup, reinit heads, new seed',
        'lr_mult': 0.05,
        'warmup_mult': 3.0,
        'reinit_heads': True,
        'seed_offset': 5,
    },
]

RETRY_ESCALATION_THRESHOLD = len(RETRY_STRATEGIES)  # after exhausting all strategies -> escalate


def get_active_heads(stage_cfg: Dict[str, Any]) -> List[str]:
    """Resolve active_heads string to full list of liveness keys."""
    raw = stage_cfg.get('active_heads', 'det')
    heads = set(raw.split('+'))
    # 'pose' in stage config encompasses both body_pose and head_pose
    if 'pose' in heads:
        heads.add('head_pose')
    return sorted(heads)


def select_retry_strategy(state: StageState) -> Dict[str, Any]:
    """Select the next retry strategy based on retry_count."""
    idx = min(state.retry_count, len(RETRY_STRATEGIES) - 1)
    return RETRY_STRATEGIES[idx]


# =========================================================================
# Process Management
# =========================================================================

def is_pid_alive(pid: Optional[int]) -> bool:
    """Check if a process with the given PID is running."""
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
        return True
    except (OSError, PermissionError):
        return False


def get_existing_train_pids(preset: str) -> List[int]:
    """Scan for any existing train.py processes with the given preset.

    This is the primary defense against duplicate training launches.
    Returns list of PIDs (may include DataLoader worker children).
    """
    import subprocess as sp
    try:
        result = sp.run(
            ['pgrep', '-f', f'train.py.*--preset {preset}'],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            return [int(p) for p in result.stdout.strip().split()]
    except Exception:
        pass
    return []


def kill_all_train_pids(preset: str, exclude_pid: Optional[int] = None) -> int:
    """Kill ALL processes matching the training preset, optionally excluding a PID.

    Uses SIGTERM first, then SIGKILL after 3s delay.
    Returns count of processes killed.
    """
    pids = get_existing_train_pids(preset)
    if exclude_pid is not None:
        pids = [p for p in pids if p != exclude_pid]

    killed = 0
    for pid in pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except (OSError, PermissionError):
            continue

    if pids:
        time.sleep(3)
        for pid in pids:
            try:
                os.kill(pid, 0)
                os.kill(pid, signal.SIGKILL)
                killed += 1
            except (OSError, PermissionError):
                pass

    return killed


def read_lock_pid() -> Optional[int]:
    """Read the PID lock file. Returns None if missing or invalid."""
    try:
        if PID_LOCK_FILE.exists():
            pid_str = PID_LOCK_FILE.read_text().strip()
            return int(pid_str) if pid_str else None
    except (ValueError, OSError):
        pass
    return None


def write_lock_pid(pid: int) -> None:
    """Write PID to lock file."""
    try:
        PID_LOCK_FILE.write_text(str(pid))
    except OSError as e:
        logger.warning(f'Could not write PID lock file: {e}')


def clear_lock_pid() -> None:
    """Remove the PID lock file."""
    try:
        PID_LOCK_FILE.unlink(missing_ok=True)
    except OSError:
        pass


def kill_training(state: StageState, force: bool = False) -> bool:
    """Kill the training subprocess(es). Returns True if any were killed."""
    # Get the preset from the stage definition
    preset = 'stage_rf1'
    idx = get_stage_index(state.current_stage)
    if idx >= 0 and idx < len(RF_STAGES):
        preset = RF_STAGES[idx].get('preset', 'stage_rf1')

    # Kill ALL processes matching this training preset
    killed_any = kill_all_train_pids(preset)

    if state.training_pid and not is_pid_alive(state.training_pid):
        state.training_pid = None

    if killed_any > 0 or state.training_pid is None:
        clear_lock_pid()
        return killed_any > 0

    # Fallback: kill just the tracked PID
    try:
        os.kill(state.training_pid, signal.SIGTERM)
        if not force:
            for _ in range(10):
                time.sleep(0.5)
                if not is_pid_alive(state.training_pid):
                    state.training_pid = None
                    clear_lock_pid()
                    return True
        os.kill(state.training_pid, signal.SIGKILL)
        state.training_pid = None
        clear_lock_pid()
        return True
    except Exception as e:
        logger.error(f'Failed to kill PID {state.training_pid}: {e}')
        return False


def launch_training(stage_cfg: Dict[str, Any], resume_from: Optional[Path],
                   strategy: Optional[Dict[str, Any]] = None) -> Optional[int]:
    """Launch a training subprocess. Returns PID or None."""
    RF_RUN_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    strategy = strategy or RETRY_STRATEGIES[0]

    # Compute max-epochs accounting for resume epoch.
    # train.py's --max-epochs sets the TOTAL epoch count (not remaining).
    # If resuming from epoch N, we need max_epochs = N + stage_epochs.
    max_epochs = stage_cfg['max_epochs']
    if resume_from and resume_from.exists():
        try:
            ckpt = torch.load(resume_from, map_location='cpu', weights_only=False)
            resume_epoch = ckpt.get('epoch', 0) + 1  # train.py resumes at epoch+1
            max_epochs = resume_epoch + stage_cfg['max_epochs']
            logger.info(f'  Checkpoint epoch={ckpt.get("epoch", 0)} → '
                        f'training for {stage_cfg["max_epochs"]} more → total --max-epochs={max_epochs}')
        except Exception as e:
            logger.warning(f'  Could not read checkpoint epoch ({e}), using raw max_epochs={max_epochs}')

    cmd = [
        sys.executable, str(TRAIN_SCRIPT),
        '--preset', stage_cfg['preset'],
        '--max-epochs', str(max_epochs),
        '--subset-ratio', str(stage_cfg['subset_ratio']),
    ]
    if stage_cfg.get('reinit_heads'):
        cmd += ['--reinit-heads']
    if stage_cfg.get('detach_reg_fpn', stage_cfg.get('reinit_heads')):
        cmd += ['--detach-reg-fpn']
    if stage_cfg.get('detach_psr_fpn', stage_cfg.get('reinit_heads')):
        cmd += ['--detach-psr-fpn']
    if resume_from and resume_from.exists():
        cmd += ['--resume', str(resume_from)]

    env = os.environ.copy()
    env['OUTPUT_ROOT_OVERRIDE'] = str(RF_RUN_DIR)
    env['_STAGE_MANAGER_ACTIVE'] = '1'
    env['_STAGE_GATE_JSON'] = json.dumps(stage_cfg.get('gate', {}))
    env['_STAGE_TARGET_MET_FILE'] = str(TARGET_MET_FILE)

    # Apply retry strategy overrides (train.py may respect these via env var)
    env['_STAGE_LR_MULT'] = str(strategy['lr_mult'])
    env['_STAGE_WARMUP_MULT'] = str(strategy['warmup_mult'])
    env['_STAGE_SEED_OFFSET'] = str(strategy['seed_offset'])
    if strategy['lr_mult'] != 1.0:
        logger.info(f'  [STRATEGY] LR mult={strategy["lr_mult"]}× (effective LR={5e-4 * strategy["lr_mult"]:.2e})')
    if strategy['warmup_mult'] != 1.0:
        logger.info(f'  [STRATEGY] Warmup mult={strategy["warmup_mult"]}× (effective warmup={5 * strategy["warmup_mult"]:.0f} epochs)')
    if strategy['seed_offset']:
        logger.info(f'  [STRATEGY] Seed offset={strategy["seed_offset"]}')

    logger.info(f'Launching: {" ".join(cmd)}')
    logger.info(f'  OUTPUT_ROOT={RF_RUN_DIR}')
    if resume_from and resume_from.exists():
        logger.info(f'  Resume from: {resume_from}')

    try:
        log_file = RF_RUN_DIR / 'logs' / 'subprocess.log'
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, 'a') as f:
            proc = subprocess.Popen(
                cmd,
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
                cwd=str(PROJECT_ROOT),
                text=True,
            )
        # Detach — we don't call proc.wait() here
        # The training runs independently; we check on it via --check
        logger.info(f'  PID={proc.pid} (stdout → {log_file})')
        return proc.pid
    except Exception as e:
        logger.error(f'Failed to launch training: {e}')
        return None


# =========================================================================
# Checklist Evaluation
# =========================================================================

def evaluate_gate(stage_cfg: Dict[str, Any], metrics: Dict[str, float]) -> Tuple[bool, Dict[str, Any]]:
    """GATE: Must pass all gate criteria to transition to next stage."""
    gates = stage_cfg.get('gate', {})
    details = {}
    all_passed = True

    if not gates:
        return True, {'note': 'no gate criteria'}

    for metric, threshold in gates.items():
        val = metrics.get(metric)
        if val is None:
            details[metric] = {'status': 'UNKNOWN', 'reason': 'metric not found in validation'}
            all_passed = False
            continue
        # Lower-is-better metrics (MAE)
        if 'MAE' in metric or 'mae' in metric:
            passed = val <= threshold
            details[metric] = {
                'status': 'PASS' if passed else 'FAIL',
                'value': val,
                'threshold': threshold,
                'gap': threshold - val if passed else val - threshold,
            }
        else:
            passed = val >= threshold
            details[metric] = {
                'status': 'PASS' if passed else 'FAIL',
                'value': val,
                'threshold': threshold,
                'gap': val - threshold if passed else threshold - val,
            }
        if not passed:
            all_passed = False

    return all_passed, details


def evaluate_health(stage_cfg: Dict[str, Any], snapshot: LogSnapshot,
                    liveness_summary: Dict[str, Any],
                    det_health_history: List[Dict[str, Any]]) -> Tuple[bool, Dict[str, Any]]:
    """HEALTH: Liveness and gradient sanity."""
    health_cfg = stage_cfg.get('health', {})
    details = {}
    all_passed = True

    # Check per-head liveness ratios — only for heads active in this stage
    active = get_active_heads(stage_cfg)
    for head in active:
        key = f'{head}_alive_ratio'
        ratio = liveness_summary.get(key, 1.0)
        passing = ratio >= 0.5  # at least 50% of steps alive
        if not passing:
            all_passed = False
        details[f'{head}_liveness'] = {
            'status': 'PASS' if passing else 'FAIL',
            'alive_ratio': ratio,
        }

    # Check loss spike factor (uses snapshot instead of re-parsing lines)
    spike_factor = snapshot.loss_trend.get('det_loss_spike_factor', 1.0)
    max_spike = health_cfg.get('max_loss_spike_factor', 10.0)
    if spike_factor > max_spike:
        all_passed = False
    details['loss_spike'] = {
        'status': 'PASS' if spike_factor <= max_spike else 'FAIL',
        'spike_factor': spike_factor,
        'threshold': max_spike,
    }

    # DET-HEALTH assessment
    det_health = assess_det_health(det_health_history)
    details['det_health'] = det_health
    if det_health['status'] == 'UNHEALTHY':
        all_passed = False
        details['det_health']['status'] = 'FAIL'
    elif det_health['status'] == 'WARN':
        details['det_health']['status'] = 'WARN'  # non-blocking

    # Near-zero ratio from latest DET-HEALTH snapshot
    if det_health_history:
        latest_nz = det_health_history[-1]['near_zero']
        if latest_nz > 0.5:
            all_passed = False
            details['near_zero_ratio'] = {
                'status': 'FAIL',
                'value': latest_nz,
                'reason': f'{latest_nz:.2%} logits near zero — degenerating',
            }
        elif latest_nz > 0.2:
            details['near_zero_ratio'] = {
                'status': 'WARN',
                'value': latest_nz,
            }

    return all_passed, details


def evaluate_convergence(stage_cfg: Dict[str, Any], state: StageState,
                         metrics: Dict[str, float]) -> Tuple[bool, Dict[str, Any]]:
    """CONVERGENCE: Rate of metric improvement with trend analysis.

    Uses metric_history for rolling-window improvement detection.
    Detects plateaus earlier than simple best-so-far comparison.
    """
    conv_cfg = stage_cfg.get('convergence', {})
    details = {}
    patience = conv_cfg.get('patience_epochs', 8)
    min_improvement = conv_cfg.get('min_improvement', 0.005)

    # No validation data available yet — too early to judge
    if not metrics:
        details['combined'] = {
            'status': 'UNKNOWN',
            'reason': 'No validation metrics in recent log — training may have just started or restarted',
        }
        return True, details

    # Primary metric to track: det_mAP50 or combined
    primary_metric = 'det_mAP50' if 'det_mAP50' in metrics else 'combined'
    current_val = metrics.get(primary_metric, metrics.get('combined', 0.0))
    epoch = state.epoch

    # Track best absolute
    best_combined = state.best_metric
    if current_val > best_combined:
        best_combined = current_val

    # ── Trend analysis from metric_history ──
    history = state.metric_history
    window_3 = [h.get(primary_metric, 0) for h in history[-3:]] if len(history) >= 3 else []
    window_5 = [h.get(primary_metric, 0) for h in history[-5:]] if len(history) >= 5 else []

    # Rolling improvement: delta over last 3 epochs
    rolling_improvement = (window_3[-1] - window_3[0]) / max(len(window_3), 1) if window_3 else 0.0
    # Per-epoch improvement (min_improvement = 0.005 means 0.5% per epoch)
    per_epoch_improvement = rolling_improvement / max(len(window_3), 1) if window_3 else 0.0

    # ── Early detection: metric oscillating (up-down-up-down) ──
    oscillation = False
    if len(window_5) >= 5:
        diffs = [window_5[i] - window_5[i - 1] for i in range(1, len(window_5))]
        sign_changes = sum(1 for i in range(1, len(diffs)) if diffs[i] * diffs[i - 1] < 0)
        if sign_changes >= 3:
            oscillation = True
            details['oscillation'] = {
                'status': 'WARN',
                'detail': f'{sign_changes} sign changes in last 5 epochs',
            }

    # ── Plateau detection ──
    epochs_no_improve = 0
    if len(history) >= 2:
        # Count epochs since last improvement > min_improvement
        for h in reversed(history[:-1]):
            if abs(h.get(primary_metric, 0) - current_val) >= min_improvement:
                break
            epochs_no_improve += 1

    plt = per_epoch_improvement < min_improvement / 2 and epoch > 3
    stalled = epochs_no_improve >= patience

    # Determine status
    if current_val > state.best_metric and per_epoch_improvement >= min_improvement:
        details[primary_metric] = {
            'status': 'IMPROVING',
            'current': current_val,
            'best': state.best_metric,
            'delta': current_val - state.best_metric,
            'epoch_trend': f'{per_epoch_improvement:.4f}/ep',
        }
        return True, details

    if epoch <= 3:
        # Too early to judge convergence
        details[primary_metric] = {
            'status': 'WARMING_UP',
            'current': current_val,
            'epoch': epoch,
        }
        return True, details

    if current_val == 0 and len(history) >= patience:
        details[primary_metric] = {
            'status': 'FAIL',
            'reason': f'{primary_metric} still 0 after {patience} epochs of recorded metrics',
        }
        return False, details

    if stalled:
        status = 'STALLED'
        all_pass = False
    elif plt:
        status = 'PLATEAUING'
        all_pass = True  # warn but don't fail yet
    else:
        status = 'FLAT'
        all_pass = True

    details[primary_metric] = {
        'status': status,
        'current': current_val,
        'best': best_combined,
        'epochs_no_improve': epochs_no_improve,
        'patience': patience,
        'rolling_improvement': rolling_improvement,
        'per_epoch_improvement': per_epoch_improvement,
    }

    if oscillation:
        details[primary_metric]['oscillation_detected'] = True

    return all_pass, details


def evaluate_validation(stage_cfg: Dict[str, Any], metrics: Dict[str, float]) -> Tuple[bool, Dict[str, Any]]:
    """VALIDATION: Metric floors — below these is FAIL."""
    val_cfg = stage_cfg.get('validation', {})
    details = {}
    all_passed = True

    if not val_cfg:
        return True, {'note': 'no validation floor criteria'}

    for metric, threshold in val_cfg.items():
        metric_name = metric.replace('_min', '').replace('_max', '')
        val = metrics.get(metric_name)
        if val is None:
            continue

        is_upper_bound = '_max' in metric or 'MAE' in metric
        if is_upper_bound:
            passed = val <= threshold
        else:
            passed = val >= threshold

        details[metric_name] = {
            'status': 'PASS' if passed else 'FAIL',
            'value': val,
            'threshold': threshold,
        }
        if not passed:
            all_passed = False

    return all_passed, details


def evaluate_stability(stage_cfg: Dict[str, Any], snapshot: LogSnapshot,
                       liveness_summary: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    """STABILITY: Training dynamics health."""
    stab_cfg = stage_cfg.get('stability', {})
    details = {}
    all_passed = True

    min_liveness = stab_cfg.get('min_liveness_ratio', 0.7)

    # Overall liveness ratio — only for heads active in this stage
    active = get_active_heads(stage_cfg)
    for head in active:
        key = f'{head}_alive_ratio'
        ratio = liveness_summary.get(key, 1.0)
        if ratio < min_liveness:
            details[f'{head}_liveness_stability'] = {
                'status': 'WARN',
                'alive_ratio': ratio,
                'min_required': min_liveness,
            }

    # Crash count from snapshot (parsed in single pass instead of re-iterating lines)
    details['crash_count'] = {
        'value': snapshot.crash_count,
        'status': 'OK' if snapshot.crash_count == 0 else 'WARN',
    }
    if snapshot.crash_count > 0:
        all_passed = False

    return all_passed, details


def evaluate_all_checklists(stage_cfg: Dict[str, Any], state: StageState,
                            metrics: Dict[str, float], snapshot: LogSnapshot,
                            liveness_summary: Dict[str, Any],
                            det_health_history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Run all 5 checklist evaluations.

    Accepts a pre-parsed LogSnapshot instead of raw log_lines so each
    checklist function doesn't re-iterate the same lines independently.
    """
    gate_pass, gate_details = evaluate_gate(stage_cfg, metrics)
    health_pass, health_details = evaluate_health(stage_cfg, snapshot, liveness_summary, det_health_history)
    conv_pass, conv_details = evaluate_convergence(stage_cfg, state, metrics)
    val_pass, val_details = evaluate_validation(stage_cfg, metrics)
    stab_pass, stab_details = evaluate_stability(stage_cfg, snapshot, liveness_summary)

    return {
        'gate': {'passed': gate_pass, 'details': gate_details},
        'health': {'passed': health_pass, 'details': health_details},
        'convergence': {'passed': conv_pass, 'details': conv_details},
        'validation': {'passed': val_pass, 'details': val_details},
        'stability': {'passed': stab_pass, 'details': stab_details},
    }


# =========================================================================
# 20-Why Root Cause Analysis
# =========================================================================

_WHY_RULES = [
    {
        'pattern': r'det.*DEAD.*act.*DEAD.*psr.*DEAD',
        'title': 'All heads DEAD — catastrophic collapse',
        'whys': [
            'Why 1: All task heads show zero gradient → no learning signal',
            'Why 2: Backbone or shared layers produce zero/constant gradient',
            'Why 3: Loss is NaN or zero for all tasks (check total loss line)',
            'Why 4: Feature extractor (ConvNeXt) may have collapsed to constant output',
            'Why 5: BatchNorm running stats diverged (check for NaN in norm layers)',
            'Why 6: Learning rate too high → optimizer stepped into bad region',
            'Why 7: Check AMP/non-AMP compatibility — FP32/AMP mixing may corrupt grads',
            'Why 8: Residual stream may have dead ReLU/GELU units (saturated activations)',
            'Why 9: Weight initialization may have collapsed (check conv kernel stats)',
            'Why 10: Gradient clipping may be masking true gradient norm',
        ],
        'fix': 'Restore from checkpoint before collapse, reduce LR 10x, enable grad clipping at 1.0',
    },
    {
        'pattern': r'det=nan|det.*nan',
        'title': 'Detection NaN loss',
        'whys': [
            'Why 1: Detection loss computation produced NaN',
            'Why 2: Focal Loss gamma * (1-pt)^gamma * log(pt) over/underflow',
            'Why 3: Classification logits diverged (check cls_preds mean)',
            'Why 4: Regression targets contain extreme values or Inf',
            'Why 5: Anchor assignment produced degenerate matches',
            'Why 6: Box targets outside image dimensions after augmentation',
            'Why 7: Gradient overflow during mixed-precision backward',
            'Why 8: RetinaNet head bias initialization shifted logits to extreme',
            'Why 9: FPN output features contained NaN (upstream conv issue)',
            'Why 10: Backbone feature map had corrupted spatial structure',
        ],
        'fix': 'Check DET-DEBUG output. Enable DET_HEALTH_WARMUP if not set. Reduce LR by 0.5x.',
    },
    {
        'pattern': r'det=DEAD',
        'title': 'Detection head DEAD (zero grad)',
        'whys': [
            'Why 1: Det head gradient is zero → no detection learning signal',
            'Why 2: If seq batch (seq_every=2): det loss is intentionally zeroed → expected behavior',
            'Why 3: If non-seq batch still DEAD: cls/reg losses are vanishing',
            'Why 4: Focal Loss (1-p_t)^gamma saturates at gamma=2 for easy negatives',
            'Why 5: Class imbalance: most anchors are background → cls loss near zero',
            'Why 6: cls_score bias init = pi=0.05 may be too low for sparse detections',
            'Why 7: RetinaNet head may need bias re-init (--reinit-heads)',
            'Why 8: FPN may be producing zero-activation features for small objects',
            'Why 9: DET_LR_MULTIPLIER=5.0 may be too low for collapsed head',
            'Why 10: Check if DET_WARMUP finished (250 steps) — before that grads are suppressed',
        ],
        'fix': 'If seq batch → ignore (expected). If non-seq: check DET-DEBUG, verify warmup completed.',
    },
    {
        'pattern': r'psr=DEAD|psr_grad.*0\.00',
        'title': 'PSR head DEAD',
        'whys': [
            'Why 1: PSR head receives zero gradient',
            'Why 2: PSR transition loss requires sequence batches (USE_PSR_SEQUENCE_MODE)',
            'Why 3: Without seq batches, only per-frame focal loss applies',
            'Why 4: Per-frame PSR has constant fill-forward labels → constant prediction is optimal',
            'Why 5: Constant prediction → zero gradient for bias term',
            'Why 6: TransformerEncoder in PSR head may have dead attention heads',
            'Why 7: Causal masking may be blocking gradient flow to early time steps',
            'Why 8: PSR sensitivity weight = 0.01 may be too low to affect total loss',
            'Why 9: PSR class imbalance (11 components × 36 steps, mostly zeros)',
            'Why 10: psr_order_prior loss term may dominate and suppress other PSR gradients',
        ],
        'fix': 'Verify USE_PSR_SEQUENCE_MODE=True. Check psr_sensitivity_weight > 0. Consider re-init.',
    },
    {
        'pattern': r'act=DEAD',
        'title': 'Activity head DEAD',
        'whys': [
            'Why 1: Activity head receives zero gradient',
            'Why 2: Temporal Bank features may be detached (FEATURE_BANK_DETACH=True)',
            'Why 3: If detach=True, only the activity classifier gets gradient (no backbone)',
            'Why 4: Activity loss is cross-entropy — needs at least some non-zero logit diff',
            'Why 5: With 74 classes, random init produces near-uniform logits → ~1/74 confidence',
            'Why 6: Label smoothing may be suppressing logit differences further',
            'Why 7: Temporal bank feature dimension mismatch → zeroed features',
            'Why 8: Activity head LR in head_params group (10x backbone) may be mismatched',
            'Why 9: If staged training: activity may be frozen in Stage 1/2',
            'Why 10: Check if TRAIN_ACT flag is actually False (ablation mode)',
        ],
        'fix': 'Verify class counts are loaded. Check Temporal Bank forward pass. May need more epochs.',
    },
    {
        'pattern': r'pose=DEAD|head_pose=DEAD',
        'title': 'Pose head DEAD',
        'whys': [
            'Why 1: Pose head receives zero gradient',
            'Why 2: Head pose targets may be missing or all-zeros in selected recs',
            'Why 3: Geo head pose (6D rotation) needs valid rotation matrices',
            'Why 4: Stop-gradient on PoseFiLM features blocks head-to-backbone gradient',
            'Why 5: Hand/body keypoints may be absent in some frames',
            'Why 6: Keypoint regression MSE may be too small to affect total loss',
            'Why 7: Other task losses dominate via Kendall weighting (exp(-s) * L)',
            'Why 8: HeadPoseFiLM conditioning may saturate (all gammas close to 0 or 1)',
            'Why 9: Body keypoint head output dimension 17×2 may have collapsed coordinates',
            'Why 10: 9-DoF MSE loss may be dwarfed by other losses in magnitude',
        ],
        'fix': 'Verify target presence in dataset. Check Kendall log_var_pose value (high→suppressed).',
    },
    {
        'pattern': r'cls_mean.*-[5-9]|cls_mean.*-1[0-9]|near_zero.*0\.[5-9]',
        'title': 'DET-HEALTH: Detection head logit collapse',
        'whys': [
            'Why 1: cls_preds logits have collapsed (very negative mean or high near-zero ratio)',
            'Why 2: All logits nearly equal → sigmoid produces uniform predictions → no gradient',
            'Why 3: Focal Loss on uniform predictions gives constant loss → det head learns nothing',
            'Why 4: Bias initialization (pi=0.05) may be too low for this dataset/class count',
            'Why 5: Re-initializing with --reinit-heads may not have reset bias correctly',
            'Why 6: LR too high → optimizer overshot good region immediately after reinit',
            'Why 7: Warmup too short → head LR ramped too fast into degenerate equilibrium',
            'Why 8: Check if DET_WARMUP completed — before that grads are suppressed',
            'Why 9: Try higher bias init (pi=0.1 or 0.2) to push initial logits toward positive',
            'Why 10: If repeating across retries, try reducing LR 5-10x with longer warmup',
        ],
        'fix': 'Restart with reduce_lr_5x or reduce_lr_10x_warmup_2x strategy. If persists, try higher bias init.',
    },
    {
        'pattern': r'CUDA out of memory|RuntimeError',
        'title': 'CUDA OOM / Runtime Error',
        'whys': [
            'Why 1: GPU VRAM exhausted (RTX 3060 = 12GB)',
            'Why 2: Batch_size=2 + grad_accum=16 creates large activation memory',
            'Why 3: Sequence batches (T=4) inflate memory 2-4x over frame batches',
            'Why 4: Gradient checkpointing may not cover all transformer layers',
            'Why 5: EMA shadow weights double model memory during EMA update',
            'Why 6: Validation batch may use different batch_size than training',
            'Why 7: DataLoader prefetch threads accumulate cached tensors',
            'Why 8: torch.cuda.empty_cache() not called between val/train transitions',
            'Why 9: Memory fragmentation from mixed-size tensors in multi-head model',
            'Why 10: Other GPU processes consuming VRAM (check nvidia-smi)',
        ],
        'fix': 'Kill other GPU processes. Reduce batch_size to 1. Disable EMA if needed.',
    },
]


def run_20_why_analysis(log_lines: List[str]) -> Dict[str, Any]:
    """Analyze failure patterns in log lines and return 20-why diagnosis."""
    log_text = '\n'.join(log_lines)
    findings = []

    for rule in _WHY_RULES:
        if re.search(rule['pattern'], log_text, re.IGNORECASE):
            findings.append({
                'title': rule['title'],
                'whys': rule['whys'],
                'fix': rule['fix'],
            })
            if len(findings) >= 3:
                break

    # Additional generic checks
    if not findings:
        # No specific pattern matched — do generic analysis
        generic_whys = [
            'Why 1: No specific failure pattern identified from log tail',
            'Why 2: Check if training process is still running',
            'Why 3: Verify train.log has recent entries (not stalled)',
            'Why 4: Check GPU utilization with nvidia-smi',
            'Why 5: Loss not decreasing may indicate LR too high or too low',
            'Why 6: Validation metrics flatlined → model may be in local optimum',
            'Why 7: Check if SUBSET_RATIO provides enough data diversity',
            'Why 8: Verify checkpoint is being saved (checkpoint file timestamp)',
            'Why 9: Kendall log_vars may have saturated (check log_var values)',
            'Why 10: No news is good news — training may be progressing normally',
        ]
        findings.append({
            'title': 'No specific pattern — generic health check',
            'whys': generic_whys,
            'fix': 'Continue monitoring. If metrics are improving, no action needed.',
        })

    return {
        'findings': findings,
        'timestamp': datetime.now(timezone.utc).isoformat(),
    }


# =========================================================================
# Decision Engine
# =========================================================================

def decide_action(state: StageState, checklist: Dict[str, Any]) -> str:
    """Core decision: what to do next based on checklists.

    Returns: 'continue' | 'kill_and_retry' | 'advance_stage' | 'wait' | 'escalate'
    """
    gate = checklist.get('gate', {}).get('passed', False)
    health = checklist.get('health', {}).get('passed', True)
    convergence = checklist.get('convergence', {}).get('passed', True)
    stability = checklist.get('stability', {}).get('passed', True)

    # If training is not running, advance or launch
    if state.training_pid is None or not is_pid_alive(state.training_pid):
        if gate and state.epoch > 0:
            return 'advance_stage'
        else:
            return 'continue'  # will launch it

    # If stability failed (crashes, OOM)
    if not stability:
        return 'kill_and_retry'

    # If health check failed (DEAD heads)
    if not health:
        return 'kill_and_retry'

    # If convergence stalled (no improvement)
    if not convergence:
        return 'kill_and_retry'

    # If all passes and gate is met → advance
    if gate:
        return 'advance_stage'

    # Gate not met but training healthy → continue
    return 'continue'


def should_escalate(state: StageState) -> bool:
    """Check if retries exhausted and we need human intervention."""
    return state.retry_count >= RETRY_ESCALATION_THRESHOLD


def get_retry_recommendation(state: StageState, why_result: Dict[str, Any]) -> str:
    """Generate human-readable recommendation based on retry state."""
    strategy = select_retry_strategy(state)
    msg = f'Retry #{state.retry_count + 1} — strategy: {strategy["name"]} ({strategy["description"]})'
    if state.retry_count > 0:
        msg += f'\n  Previous strategies tried: {", ".join(state.strategies_tried)}'
    msg += f'\n  LR mult={strategy["lr_mult"]}×, warmup mult={strategy["warmup_mult"]}×'
    if should_escalate(state):
        msg += '\n  ⚠️  RETRIES EXHAUSTED — need human intervention'
    for f in why_result.get('findings', []):
        msg += f'\n  Root: {f["title"]}'
        msg += f'\n  Fix:  {f["fix"]}'
    return msg


def get_stage_index(stage_name: str) -> int:
    for i, s in enumerate(RF_STAGES):
        if s['name'] == stage_name:
            return i
    return -1


# =========================================================================
# Main Check Cycle
# =========================================================================

def cmd_check() -> None:
    """Main cron entry point: evaluate current stage and act."""
    state = load_state()

    # ── Reconcile state PID with lock file PID ──
    lock_pid = read_lock_pid()
    if lock_pid is not None and is_pid_alive(lock_pid):
        if state.training_pid != lock_pid:
            logger.warning(f'State PID={state.training_pid} != lock PID={lock_pid} — reconciling')
            state.training_pid = lock_pid
            save_state(state)
    elif lock_pid is None and state.training_pid is not None and is_pid_alive(state.training_pid):
        # Lock file missing but training alive: re-create lock
        write_lock_pid(state.training_pid)
        logger.info(f'Re-created PID lock file for PID={state.training_pid}')

    logger.info('=' * 60)
    logger.info(f'Stage Manager Check @ {datetime.now().isoformat()}')
    logger.info(f'Current stage: {state.current_stage} [{state.status}]')
    if state.retry_count > 0:
        logger.info(f'Retry #{state.retry_count} | Strategy: {state.current_strategy} | '
                    f'Tried: {", ".join(state.strategies_tried) or "none"}')

    stage_idx = get_stage_index(state.current_stage)
    if stage_idx < 0:
        logger.error(f'Unknown stage: {state.current_stage}')
        return

    stage_cfg = _STAGE_BY_NAME[state.current_stage]

    # ── Incremental log reading ──
    # Read new lines since last cursor position. Falls back to tail 500
    # when cursor is 0 (first check or truncated log).
    log_lines, state.log_cursor = read_train_log_incr(state.log_cursor)

    # ── Single-pass log parsing (replaces 7 redundant line iterations) ──
    snapshot = parse_log_snapshot(log_lines)

    # Merge DET-HEALTH into state history (deduplicated by step)
    if snapshot.det_health_lines:
        existing_steps = {h['step'] for h in state.det_health_history}
        for hl in snapshot.det_health_lines:
            if hl['step'] not in existing_steps:
                state.det_health_history.append(hl)
        latest_dh = snapshot.det_health_lines[-1]
        logger.info(f'DET-HEALTH: cls_mean={latest_dh["cls_mean"]:.4f} '
                    f'std={latest_dh["cls_std"]:.4f} near_zero={latest_dh["near_zero"]:.2%}')

    # Log DET-DEBUG floor-fraction warning if many near-zero logits
    if snapshot.det_debug_lines:
        latest_dd = snapshot.det_debug_lines[-1]
        if latest_dd['floor_frac'] > 0.5:
            logger.warning(f'DET-DEBUG: floor_frac={latest_dd["floor_frac"]:.2%} — many near-zero logits')

    # FAST PATH: target-met signal from training subprocess
    if TARGET_MET_FILE.exists():
        logger.info(f'Target-met signal found — advancing from {state.current_stage}')
        try:
            signal_data = json.loads(TARGET_MET_FILE.read_text())
            logger.info(f'  Epoch: {signal_data.get("epoch", "?")}  Metrics: {signal_data.get("gate_details", {})}')
        except Exception:
            signal_data = {}
        TARGET_MET_FILE.unlink(missing_ok=True)

        state.stage_history.append({
            'stage': state.current_stage,
            'epochs_completed': signal_data.get('epoch', state.epoch),
            'best_metrics': signal_data.get('metrics', {}),
            'checklist': {'gate': {'passed': True, 'details': signal_data.get('gate_details', {})}},
            'timestamp': datetime.now(timezone.utc).isoformat(),
        })

        next_idx = stage_idx + 1
        if next_idx >= len(RF_STAGES):
            logger.info('ALL STAGES COMPLETE!')
            state.status = 'completed'
            save_state(state)
            _print_paper_results(state)
            return

        next_stage = RF_STAGES[next_idx]
        logger.info(f'Advancing to {next_stage["name"]}: {next_stage["description"]}')

        kill_training(state)
        # Reset retry state for new stage
        state.retry_count = 0
        state.current_strategy = 'default'
        state.strategies_tried = []
        state.metric_history = []
        state.det_health_history = []
        state.current_stage = next_stage['name']
        state.stage_index = next_idx
        state.status = 'idle'
        state.gate_passed = False
        state.epoch = 0
        save_state(state)
        _launch_current_stage(state, next_stage, retry=False)
        return

    # Check if training process is alive
    training_alive = state.training_pid is not None and is_pid_alive(state.training_pid)
    logger.info(f'Training PID: {state.training_pid} [{ "ALIVE" if training_alive else "DEAD" }]')

    if training_alive:
        # Use snapshot data (single-pass) instead of individual parsers
        state.epoch = snapshot.epoch
        metrics = snapshot.metrics

        logger.info(f'Epoch: {snapshot.epoch}')
        if metrics:
            logger.info(f'Val metrics: det_mAP50={metrics.get("det_mAP50", "N/A"):.4f}, '
                        f'act_top1={metrics.get("act_clip_accuracy", 0):.4f}, '
                        f'psr_f1={metrics.get("psr_f1_at_t", 0):.4f}')
        if snapshot.liveness.get('total_checks', 0) > 0:
            logger.info(f'Liveness: det={snapshot.liveness.get("det_alive_ratio", 0):.0%} '
                        f'act={snapshot.liveness.get("act_alive_ratio", 0):.0%} '
                        f'pose={snapshot.liveness.get("pose_alive_ratio", 0):.0%}')

        # Track metric history for convergence analysis
        if metrics and snapshot.epoch > 0:
            if not state.metric_history or state.metric_history[-1].get('epoch', -1) != snapshot.epoch:
                metrics['epoch'] = snapshot.epoch
                state.metric_history.append(metrics)
                if len(state.metric_history) > 20:
                    state.metric_history = state.metric_history[-20:]

        # Run checklist evaluations (passes snapshot instead of raw lines)
        checklist = evaluate_all_checklists(stage_cfg, state, metrics, snapshot,
                                            snapshot.liveness, state.det_health_history)
        state.checklist_results = checklist

        # Log checklist summary
        for cat in ['gate', 'health', 'convergence', 'validation', 'stability']:
            c = checklist.get(cat, {})
            status = 'PASS' if c.get('passed') else 'FAIL' if c.get('details') else 'N/A'
            logger.info(f'  [{cat:12s}] {status}')
            # Log details on failure
            if not c.get('passed', True) and c.get('details'):
                for detail_key, detail_val in c['details'].items():
                    if isinstance(detail_val, dict) and detail_val.get('status') in ('FAIL', 'WARN'):
                        logger.info(f'    ⚠ {detail_key}: {detail_val}')

        # Decision
        decision = decide_action(state, checklist)
        logger.info(f'Decision: {decision}')

        if decision == 'continue':
            logger.info('Training healthy — continuing to monitor.')
            state.last_check_time = datetime.now(timezone.utc).isoformat()
            save_state(state)

        elif decision == 'kill_and_retry':
            logger.warning('Checklist failure — initiating kill and retry.')
            why = run_20_why_analysis(log_lines)

            # Log the retry recommendation
            recommendation = get_retry_recommendation(state, why)
            logger.warning(f'Retry plan:\n{recommendation}')

            state.issues_log.append({
                'action': 'kill_and_retry',
                'stage': state.current_stage,
                'epoch': snapshot.epoch,
                'retry_count': state.retry_count,
                'strategy': select_retry_strategy(state)['name'],
                'why_analysis': why,
                'checklist': checklist,
                'timestamp': datetime.now(timezone.utc).isoformat(),
            })
            for f in why['findings']:
                logger.warning(f'  Root cause: {f["title"]}')
                logger.warning(f'  Fix: {f["fix"]}')

            # Track strategy and increment retry count
            current_strategy = select_retry_strategy(state)
            state.strategies_tried.append(current_strategy['name'])
            state.retry_count += 1
            # current_strategy must reflect the ACTIVE strategy (post-increment) for state file accuracy
            state.current_strategy = select_retry_strategy(state)['name']

            kill_training(state)
            state.status = 'retrying'

            # Check if we need human intervention
            if should_escalate(state):
                logger.error('⚠️  RETRIES EXHAUSTED — need human intervention!')
                logger.error(f'  Tried: {", ".join(state.strategies_tried)}')
                logger.error(f'  Last failure: {why["findings"][0]["title"] if why["findings"] else "unknown"}')
                state.status = 'failed'
                save_state(state)
                return

            _launch_current_stage(state, stage_cfg, retry=True)

        elif decision == 'advance_stage':
            logger.info(f'Gate PASSED for {state.current_stage}!')
            # Record stage completion
            state.stage_history.append({
                'stage': state.current_stage,
                'epochs_completed': snapshot.epoch,
                'best_metric': state.best_metric,
                'best_metrics': snapshot.metrics,
                'checklist': checklist,
                'retries': state.retry_count,
                'strategies_used': list(state.strategies_tried),
                'timestamp': datetime.now(timezone.utc).isoformat(),
            })

            # Advance to next stage
            next_idx = stage_idx + 1
            if next_idx >= len(RF_STAGES):
                logger.info('ALL STAGES COMPLETE! Final model ready for paper.')
                state.status = 'completed'
                save_state(state)
                _print_paper_results(state)
                return

            next_stage = RF_STAGES[next_idx]
            logger.info(f'Advancing to {next_stage["name"]}: {next_stage["description"]}')

            kill_training(state)
            # Reset retry state for new stage
            state.retry_count = 0
            state.current_strategy = 'default'
            state.strategies_tried = []
            state.metric_history = []
            state.det_health_history = []
            state.current_stage = next_stage['name']
            state.stage_index = next_idx
            state.status = 'idle'
            state.gate_passed = False
            state.epoch = 0
            save_state(state)

            # Launch next stage
            _launch_current_stage(state, next_stage, retry=False)

    else:
        # Training not running
        if state.status == 'completed':
            logger.info('All stages complete!')
            _print_paper_results(state)
            return

        if state.status == 'running':
            logger.warning('Training process died unexpectedly!')
            why = run_20_why_analysis(log_lines)
            state.issues_log.append({
                'action': 'unexpected_crash',
                'stage': state.current_stage,
                'why_analysis': why,
                'timestamp': datetime.now(timezone.utc).isoformat(),
            })
            for f in why['findings']:
                logger.warning(f'  Root cause: {f["title"]}')

        # If gate already passed for this stage and training died, advance
        if state.gate_passed and state.epoch > 0:
            logger.info('Gate was passed. Advancing to next stage.')
            next_idx = stage_idx + 1
            if next_idx < len(RF_STAGES):
                state.stage_history.append({
                    'stage': state.current_stage,
                    'epochs_completed': state.epoch,
                    'best_metric': state.best_metric,
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                })
                next_stage = RF_STAGES[next_idx]
                # Reset retry state for new stage
                state.retry_count = 0
                state.current_strategy = 'default'
                state.strategies_tried = []
                state.metric_history = []
                state.det_health_history = []
                state.current_stage = next_stage['name']
                state.stage_index = next_idx
                state.status = 'idle'
                state.epoch = 0
                state.gate_passed = False
                save_state(state)
                _launch_current_stage(state, next_stage, retry=False)
                return

        # Launch or re-launch current stage — escalate on repeated failure
        state.metric_history = []
        state.det_health_history = []
        had_previous_run = state.training_pid is not None or len(state.issues_log) > 0
        if had_previous_run:
            # Training previously ran and died — treat as retry
            strategy = select_retry_strategy(state)
            state.strategies_tried.append(strategy['name'])
            state.retry_count += 1
            # current_strategy must reflect the ACTIVE strategy (post-increment)
            state.current_strategy = select_retry_strategy(state)['name']
            logger.warning(f'Training had previous run — retry #{state.retry_count} '
                          f'(next: {select_retry_strategy(state)["name"]})')
            if should_escalate(state):
                logger.error('⚠️  RETRIES EXHAUSTED — need human intervention!')
                state.status = 'failed'
                save_state(state)
                return
            _launch_current_stage(state, stage_cfg, retry=True)
        else:
            _launch_current_stage(state, stage_cfg, retry=False)


def _launch_current_stage(state: StageState, stage_cfg: Dict[str, Any],
                          retry: bool = False) -> None:
    """Launch (or re-launch) the current stage training.

    Implements duplicate-prevention:
      1. Check PID lock file — if process is alive, skip launch.
      2. Scan for existing train.py processes for this preset — kill them.
      3. Write lock file after successful launch.
    """
    # ── Duplicate prevention: check lock file ──
    lock_pid = read_lock_pid()
    if lock_pid is not None and is_pid_alive(lock_pid):
        # Verify this PID is actually running our training
        import subprocess as sp
        try:
            result = sp.run(
                ['pgrep', '-f', f'train.py.*--preset {stage_cfg["preset"]}'],
                capture_output=True, text=True, timeout=5,
            )
            matching_pids = [int(p) for p in result.stdout.strip().split()] if result.stdout.strip() else []
            if lock_pid in matching_pids:
                logger.warning(f'Training already running (PID={lock_pid}) — skipping duplicate launch.')
                state.training_pid = lock_pid
                state.status = 'running'
                save_state(state)
                return
        except Exception:
            pass

    # ── Duplicate prevention: kill any existing training processes ──
    preset = stage_cfg.get('preset', 'stage_rf1')
    existing = get_existing_train_pids(preset)
    if existing:
        logger.warning(f'Found {len(existing)} existing training process(es) — killing before launch.')
        killed = kill_all_train_pids(preset)
        if killed > 0:
            logger.info(f'Killed {killed} stale training process(es).')
        time.sleep(2)  # Wait for cleanup

    # Select retry strategy
    strategy = select_retry_strategy(state) if retry else RETRY_STRATEGIES[0]
    if retry:
        logger.info(f'Retry #{state.retry_count + 1} — strategy: {strategy["name"]} ({strategy["description"]})')

    # Determine resume source
    resume_from = None

    if state.stage_index > 0 and state.current_stage != RF_STAGES[0]['name']:
        # Resume from previous stage's best checkpoint
        prev_stage = RF_STAGES[state.stage_index - 1]
        prev_ckpt = CKPT_DIR / prev_stage['name'] / 'best.pth'
        if prev_ckpt.exists():
            resume_from = prev_ckpt
            logger.info(f'Resuming from {prev_stage["name"]} best: {prev_ckpt}')
        elif BEST_CKPT.exists():
            resume_from = BEST_CKPT
            logger.info(f'Resuming from shared best: {BEST_CKPT}')
    elif state.stage_index == 0 and not retry:
        # RF1: prefer run-specific latest (crash recovery or resumed training)
        if LATEST_CKPT.exists():
            resume_from = LATEST_CKPT
            logger.info(f'RF1 resuming from run checkpoint: {LATEST_CKPT}')
        else:
            orig_latest = RUNS_DIR / 'full_multi_task_tma_tbank' / 'checkpoints' / 'latest.pth'
            if orig_latest.exists():
                resume_from = orig_latest
                logger.info(f'RF1 resuming from original checkpoint: {orig_latest}')
    elif retry:
        # Retry: resume from latest within current stage
        stage_ckpt = CKPT_DIR / stage_cfg['name'] / 'latest.pth'
        if stage_ckpt.exists():
            resume_from = stage_ckpt
            logger.info(f'Retry resuming from stage checkpoint: {stage_ckpt}')
        elif LATEST_CKPT.exists():
            resume_from = LATEST_CKPT

    # Apply strategy to stage_cfg (override reinit_heads from strategy)
    launch_cfg = dict(stage_cfg)
    if strategy.get('reinit_heads'):
        launch_cfg['reinit_heads'] = True

    pid = launch_training(launch_cfg, resume_from, strategy=strategy)
    if pid:
        state.training_pid = pid
        state.status = 'running'
        state.run_start_time = datetime.now(timezone.utc).isoformat()
        write_lock_pid(pid)  # Write PID lock file to prevent duplicates
        save_state(state)
        logger.info(f'Launched {stage_cfg["name"]} (PID={pid}) [lock={PID_LOCK_FILE}]')
    else:
        logger.error(f'Failed to launch {stage_cfg["name"]}')
        state.status = 'failed'
        save_state(state)


def _print_paper_results(state: StageState) -> None:
    """Print final paper results in a copy-ready format."""
    # Get the last stage's best metrics from history
    final_metrics = {}
    for entry in reversed(state.stage_history):
        if entry.get('best_metrics'):
            final_metrics = entry['best_metrics']
            break

    if not final_metrics:
        # Try reading the latest val from train.log
        log_lines = read_train_log(500)
        final_metrics = parse_val_metrics(log_lines) or {}

    logger.info('')
    logger.info('=' * 60)
    logger.info('  PAPER RESULTS — Copy-ready')
    logger.info('=' * 60)

    det_map50 = final_metrics.get('det_mAP50', 'N/A')
    det_map50_95 = final_metrics.get('det_mAP50:0.95', final_metrics.get('det_mAP50_95', 'N/A'))
    act_top1 = final_metrics.get('act_clip_accuracy', final_metrics.get('act_top1', 'N/A'))
    act_frame = final_metrics.get('act_frame_accuracy', 'N/A')
    psr_f1 = final_metrics.get('psr_f1_at_t', 'N/A')
    psr_edit = final_metrics.get('psr_edit_score', 'N/A')
    pose_mae = final_metrics.get('forward_angular_MAE_deg', 'N/A')

    logger.info(f'  Detection mAP@0.50    : {det_map50}')
    logger.info(f'  Detection mAP@0.50:0.95: {det_map50_95}')
    logger.info(f'  Activity Top-1 (clip) : {act_top1}')
    logger.info(f'  Activity Top-1 (frame): {act_frame}')
    logger.info(f'  PSR F1@T              : {psr_f1}')
    logger.info(f'  PSR Edit Score        : {psr_edit}')
    logger.info(f'  Pose MAE (deg)        : {pose_mae}')
    logger.info('')
    logger.info('  Baselines:')
    logger.info(f'  YOLOv8m mAP@0.50      : {PAPER_BASELINES["det_mAP50"]}')
    logger.info(f'  MViTv2 Top-1          : {PAPER_BASELINES["act_top1"]}')
    logger.info(f'  B2 PSR F1             : {PAPER_BASELINES["psr_f1_at_t"]}')

    # Beats?
    beats_det = isinstance(det_map50, float) and det_map50 >= PAPER_BASELINES['det_mAP50']
    beats_act = isinstance(act_top1, float) and act_top1 >= PAPER_BASELINES['act_top1']
    beats_psr = isinstance(psr_f1, float) and psr_f1 >= PAPER_BASELINES['psr_f1_at_t']
    logger.info('')
    logger.info(f'  Beats YOLOv8m : {"YES" if beats_det else "no"}')
    logger.info(f'  Beats MViTv2  : {"YES" if beats_act else "no"}')
    logger.info(f'  Beats B2      : {"YES" if beats_psr else "no"}')
    logger.info('=' * 60)


# =========================================================================
# CLI Commands
# =========================================================================

def cmd_status() -> None:
    """Print current state."""
    state = load_state()
    logger.info(f'Current stage  : {state.current_stage} [{state.status}]')
    logger.info(f'Stage index    : {state.stage_index + 1}/{len(RF_STAGES)}')
    logger.info(f'Training PID   : {state.training_pid} {"(alive)" if is_pid_alive(state.training_pid) else "(dead)"}')
    logger.info(f'Epoch          : {state.epoch}')
    logger.info(f'Best combined  : {state.best_metric:.4f}')
    logger.info(f'Gate passed    : {state.gate_passed}')
    logger.info(f'History        : {len(state.stage_history)} stages completed')
    logger.info(f'Issues         : {len(state.issues_log)} logged')
    if state.retry_count > 0:
        logger.info(f'Retries        : {state.retry_count} (strategy: {state.current_strategy})')
        logger.info(f'Strategies tried: {", ".join(state.strategies_tried)}')
    if state.det_health_history:
        latest_dh = state.det_health_history[-1]
        logger.info(f'DET-HEALTH     : cls_mean={latest_dh["cls_mean"]:.4f} near_zero={latest_dh["near_zero"]:.2%}')
    if state.metric_history:
        logger.info(f'Metric history  : {len(state.metric_history)} epochs tracked')

    if state.stage_history:
        logger.info('\nStage history:')
        for h in state.stage_history:
            logger.info(f'  {h["stage"]}: {h.get("epochs_completed", "?")} epochs, '
                        f'best={h.get("best_metric", 0):.4f}')

    # Print current stage info
    stage_cfg = _STAGE_BY_NAME.get(state.current_stage)
    if stage_cfg:
        logger.info(f'\nCurrent stage config:')
        logger.info(f'  Description : {stage_cfg["description"]}')
        logger.info(f'  Subset ratio: {stage_cfg["subset_ratio"]}')
        logger.info(f'  Max epochs  : {stage_cfg["max_epochs"]}')
        logger.info(f'  Active heads: {stage_cfg["active_heads"]}')
        logger.info(f'  Preset      : {stage_cfg["preset"]}')


def cmd_abort() -> None:
    """Abort current training and mark state as failed."""
    state = load_state()
    if state.training_pid and is_pid_alive(state.training_pid):
        logger.warning(f'Killing training PID {state.training_pid}')
        kill_training(state, force=True)
        logger.info('Training killed.')
    else:
        logger.info('No training process to kill.')
    state.status = 'aborted'
    save_state(state)
    logger.info('State set to "aborted". Use --reset to start fresh.')


def cmd_launch(stage_name: str) -> None:
    """Force-launch a specific stage."""
    stage_name = stage_name.lower()
    if stage_name not in _STAGE_BY_NAME:
        logger.error(f'Unknown stage: {stage_name}. Available: {list(_STAGE_BY_NAME.keys())}')
        return

    state = load_state()
    # Kill any existing training
    if state.training_pid and is_pid_alive(state.training_pid):
        logger.warning(f'Killing existing training PID {state.training_pid}')
        kill_training(state, force=True)

    stage_idx = get_stage_index(stage_name)
    stage_cfg = _STAGE_BY_NAME[stage_name]
    state.current_stage = stage_name
    state.stage_index = stage_idx
    state.status = 'idle'
    state.epoch = 0
    state.gate_passed = False
    save_state(state)
    _launch_current_stage(state, stage_cfg, retry=False)


def cmd_reset() -> None:
    """Reset all state (fresh start)."""
    state = load_state()
    if state.training_pid and is_pid_alive(state.training_pid):
        logger.warning(f'Killing training PID {state.training_pid}')
        kill_training(state, force=True)
    STATE_FILE.unlink(missing_ok=True)
    logger.info('State reset. Next --check will start RF1 fresh.')


# =========================================================================
# Entry Point
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description='RF1-RF10 Stage Manager for IndustReal training.',
    )
    parser.add_argument('--check', action='store_true', help='Evaluate current stage and act (cron mode)')
    parser.add_argument('--status', action='store_true', help='Show current state')
    parser.add_argument('--abort', action='store_true', help='Kill current training')
    parser.add_argument('--launch', type=str, default=None, help='Force-launch a stage (e.g., RF1)')
    parser.add_argument('--reset', action='store_true', help='Reset all state')

    args = parser.parse_args()

    if args.check:
        cmd_check()
    elif args.status:
        cmd_status()
    elif args.abort:
        cmd_abort()
    elif args.launch:
        cmd_launch(args.launch)
    elif args.reset:
        cmd_reset()
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
