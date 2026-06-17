#!/usr/bin/env python3
"""
RF1-RF10 Training Supervisor — Comprehensive autonomous training orchestration.

Extends stage_manager.py with:
  - Deep logit-level collapse detection (DET-DEBUG cls_mean)
  - PSR diagnostics and auto-tuning
  - GPU monitoring (memory, utilization)
  - Automatic config.py edits for parameter tuning
  - Multi-layer intervention with 5-why root cause analysis
  - Epoch-level orchestration (wait, checkpoint, advance)
  - Stage transition with config backpropagation

Usage (cron):
  python -m src.training.training_supervisor

This script is designed to be called every 3-5 minutes from cron.
It is stateless — all state lives in rf_stage_state.json.
"""

import json
import logging
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── Paths ──────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_FILE = PROJECT_ROOT / 'src' / 'config.py'
TRAIN_SCRIPT = PROJECT_ROOT / 'src' / 'training' / 'train.py'
STAGE_MANAGER = PROJECT_ROOT / 'src' / 'training' / 'stage_manager.py'
RUNS_DIR = PROJECT_ROOT / 'src' / 'runs'
STATE_FILE = RUNS_DIR / 'rf_stage_state.json'
RF_RUN_DIR = RUNS_DIR / 'rf_stages'
SUBPROCESS_LOG = RF_RUN_DIR / 'logs' / 'subprocess.log'
TRAIN_LOG_DIR = RF_RUN_DIR / 'logs'
CKPT_DIR = RF_RUN_DIR / 'checkpoints'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger('training_supervisor')

# ── GPU Monitoring ─────────────────────────────────────────────────────────

def get_gpu_info() -> Dict[str, Any]:
    """Query nvidia-smi for GPU utilization and memory."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode == 0 and result.stdout.strip():
            parts = result.stdout.strip().split(',')
            if len(parts) >= 3:
                return {
                    'gpu_util_pct': float(parts[0].strip()),
                    'mem_used_mib': float(parts[1].strip()),
                    'mem_total_mib': float(parts[2].strip()),
                    'mem_util_pct': float(parts[1].strip()) / float(parts[2].strip()) * 100 if float(parts[2].strip()) > 0 else 0,
                    'temp_c': float(parts[3].strip()) if len(parts) > 3 else 0,
                }
    except Exception as e:
        logger.warning(f'GPU query failed: {e}')
    return {'gpu_util_pct': 0, 'mem_used_mib': 0, 'mem_total_mib': 0, 'mem_util_pct': 0, 'temp_c': 0}


# ── Config Editing ─────────────────────────────────────────────────────────

def read_config_value(key: str) -> Optional[str]:
    """Read a config value as raw text from config.py."""
    if not CONFIG_FILE.exists():
        return None
    try:
        content = CONFIG_FILE.read_text()
        m = re.search(rf'^{re.escape(key)}\s*=\s*(.+)', content, re.MULTILINE)
        if m:
            return m.group(1).strip()
    except Exception:
        pass
    return None


def set_config_value(key: str, value: str, comment: str = '') -> bool:
    """Change a config value in config.py. Returns True if changed."""
    if not CONFIG_FILE.exists():
        logger.error(f'Config file not found: {CONFIG_FILE}')
        return False
    try:
        content = CONFIG_FILE.read_text()
        # Match the assignment line (possibly with trailing comment)
        pattern = rf'^({re.escape(key)}\s*=\s*).+$'
        replacement = f'\\g<1>{value}'
        if comment:
            replacement += f'  # {comment}'
        new_content, count = re.subn(pattern, replacement, content, count=1, flags=re.MULTILINE)
        if count == 0:
            # Key not found — append
            line = f'{key} = {value}'
            if comment:
                line += f'  # {comment}'
            new_content = content.rstrip() + '\n' + line + '\n'
        if new_content != content:
            CONFIG_FILE.write_text(new_content)
            logger.info(f'Config change: {key} = {value}  ({comment})')
            return True
        return False
    except Exception as e:
        logger.error(f'Failed to set config {key}: {e}')
        return False


# ── Log Parsing ────────────────────────────────────────────────────────────

def tail_log(path: Path, n_lines: int = 200) -> List[str]:
    """Read last N lines of a file."""
    if not path.exists():
        return []
    try:
        with open(path) as f:
            lines = f.readlines()
        return lines[-n_lines:]
    except Exception as e:
        logger.warning(f'Failed to read {path}: {e}')
        return []


def parse_det_debug(lines: List[str]) -> List[Dict[str, Any]]:
    """Extract latest DET-DEBUG cls_preds: mean=... lines."""
    results = []
    pat = re.compile(
        r'\[DET-DEBUG step=(\d+)\]\s+cls_preds:\s+'
        r'sum=[\d.-]+\s+min=[\d.-]+\s+max=[\d.-]+\s+'
        r'mean=([\d.-]+)\s+std=([\d.-]+)\s+'
        r'med_abs=[\d.-]+\s+near_zero=([\d.]+)'
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


def parse_det_health(lines: List[str]) -> List[Dict[str, Any]]:
    """Extract DET-HEALTH lines."""
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


def parse_psr_diag(lines: List[str]) -> List[Dict[str, Any]]:
    """Extract PSR_DIAG lines for PSR head health."""
    results = []
    pat = re.compile(
        r'\[PSR_DIAG\]\s+loss=([\d.e+-]+)\s+'
        r'finite=\w+\s+\|\s+'
        r'shape=\([^)]+\)\s+'
        r'total=(\d+)\s+'
        r'valid=(\d+).*?'
        r'logits\[min/max/mean\]=([\d.-]+)/([\d.-]+)/([\d.-]+).*?'
        r'zeros=(\d+)\s+ones=(\d+)'
    )
    for line in lines:
        m = pat.search(line)
        if m:
            results.append({
                'loss': float(m.group(1)),
                'total': int(m.group(2)),
                'valid': int(m.group(3)),
                'logits_min': float(m.group(4)),
                'logits_max': float(m.group(5)),
                'logits_mean': float(m.group(6)),
                'zeros': int(m.group(7)),
                'ones': int(m.group(8)),
            })
    return results


def parse_epoch_progress(lines: List[str]) -> Tuple[int, int, int]:
    """Extract current epoch and step from progress bar lines."""
    epoch = 0
    step = 0
    total_steps = 0
    for line in reversed(lines):
        # Epoch 57 [no-staging]:  19%|█▉  | 472/2482
        m = re.search(r'Epoch (\d+).*?(\d+)/(\d+)\s+\[', line)
        if m:
            epoch = int(m.group(1))
            step = int(m.group(2))
            total_steps = int(m.group(3))
            break
    return epoch, step, total_steps


def parse_det_loss(lines: List[str]) -> List[float]:
    """Extract detection loss values from log lines."""
    losses = []
    pat = re.compile(r'det=([\d.e+-]+)\(c=([\d.e+-]+)')
    for line in lines:
        m = pat.search(line)
        if m:
            losses.append(float(m.group(1)))
    return losses


def parse_step_speed(lines: List[str]) -> Optional[float]:
    """Extract most recent it/s from progress bar."""
    for line in reversed(lines):
        m = re.search(r'([\d.]+)it/s', line)
        if m:
            return float(m.group(1))
    return None


# ── State Management ───────────────────────────────────────────────────────

def load_state() -> Dict[str, Any]:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    return {}


def save_state(state: Dict[str, Any]) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    # Strip ephemeral fields that stage_manager doesn't understand
    clean = {k: v for k, v in state.items() if not k.startswith('_')}
    STATE_FILE.write_text(json.dumps(clean, indent=2, default=str))


def is_pid_alive(pid: Optional[int]) -> bool:
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
        return True
    except (OSError, PermissionError):
        return False


def is_any_training_running() -> bool:
    """Check if ANY train.py process is running (beyond just the tracked PID)."""
    try:
        result = subprocess.run(
            ['pgrep', '-af', 'train.py'],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split('\n'):
                if 'train.py' in line and 'training_supervisor' not in line and 'stage_manager' not in line:
                    return True
    except Exception:
        pass
    return False


def kill_any_training() -> bool:
    """Kill all train.py processes. Returns True if any were killed."""
    killed = False
    try:
        result = subprocess.run(
            ['pgrep', '-af', 'train.py'],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split('\n'):
                if 'train.py' in line and 'training_supervisor' not in line and 'stage_manager' not in line:
                    parts = line.strip().split(None, 1)
                    if parts:
                        try:
                            pid = int(parts[0])
                            os.kill(pid, signal.SIGTERM)
                            killed = True
                        except (ValueError, OSError):
                            pass
        if killed:
            time.sleep(2)
    except Exception:
        pass
    return killed


MIN_INTERVAL_MINUTES = 30


def check_cooldown(state: Dict) -> bool:
    """Return True if we should skip (cooldown active). Updates last run time."""
    last_run = state.get('_last_supervisor_run', 0)
    now = time.time()
    elapsed_min = (now - last_run) / 60
    if elapsed_min < MIN_INTERVAL_MINUTES and last_run > 0:
        logger.info(f'[COOLDOWN] {elapsed_min:.0f}m < {MIN_INTERVAL_MINUTES}m — skipping (next check at '
                    f'{datetime.fromtimestamp(last_run + MIN_INTERVAL_MINUTES * 60).strftime("%H:%M")})')
        return True
    return False


def kill_training(pid: Optional[int]) -> bool:
    if pid is None or not is_pid_alive(pid):
        return False
    try:
        os.kill(pid, signal.SIGTERM)
        for _ in range(10):
            time.sleep(0.5)
            if not is_pid_alive(pid):
                return True
        os.kill(pid, signal.SIGKILL)
        return True
    except Exception as e:
        logger.error(f'Kill failed: {e}')
        return False


# ── Deep Analysis ──────────────────────────────────────────────────────────

SYMPTOM_DATABASE = [
    {
        'id': 'det_collapse_cls_mean',
        'description': 'Detection head logit collapse — cls_mean very negative',
        'detect': lambda dd: any(d.get('cls_mean', 0) < CLS_MEAN_CRITICAL for d in dd if d.get('cls_mean') is not None),
        'severity': 'CRITICAL',
        'whys': [
            'Why 1: cls_preds logits collapsed to very negative values → all sigmoid outputs near 0',
            'Why 2: Sigmoid near 0 → Focal Loss applies high (1-pt)^gamma modulation → loss saturates',
            'Why 3: Saturated loss → zero gradient → detection head stops learning',
            'Why 4: With seq-every-2, det head only trains on every other batch → slower recovery',
            'Why 5: AMP GradScaler may have underflowed detection gradients from PSR spike on seq step',
        ],
        'fixes': [
            {'type': 'config', 'key': 'MIXED_PRECISION', 'value': 'False', 'comment': 'Disable AMP — PSR seq loss spikes corrupt GradScaler'},
            {'type': 'config', 'key': 'GRAD_CLIP_NORM', 'value': '1.0', 'comment': 'Ensure gradient clipping is active'},
            {'type': 'config', 'key': 'PSR_WEIGHT', 'value': '10.0', 'comment': 'Reduce PSR weight to prevent backbone disruption'},
            {'type': 'restart', 'retry_strategy': 'reduce_lr_10x_warmup_2x'},
        ],
    },
    {
        'id': 'det_collapse_near_zero',
        'description': 'Detection head degenerate — near_zero ratio > 50%',
        'detect': lambda dd: any(d.get('near_zero', 0) > 0.5 for d in dd),
        'severity': 'CRITICAL',
        'whys': [
            'Why 1: Majority of logits are near zero → network produces uniform predictions',
            'Why 2: Uniform predictions → very low gradients → no learning signal',
            'Why 3: Bias initialization may be too low for this dataset',
            'Why 4: Re-init heads may not have reset parameters correctly',
        ],
        'fixes': [
            {'type': 'config', 'key': 'DET_LR_MULTIPLIER', 'value': '10.0', 'comment': 'Double det LR to escape degenerate region'},
            {'type': 'restart', 'retry_strategy': 'reduce_lr_5x'},
        ],
    },
    {
        'id': 'psr_dead',
        'description': 'PSR head producing zero loss (stuck at ~1.5e-08)',
        'detect': lambda psr: any(p['loss'] < 1e-7 for p in psr[-10:]) if len(psr) >= 5 else False,
        'severity': 'WARN',
        'whys': [
            'Why 1: PSR loss at ~1.5e-08 → essentially zero → PSR head not learning',
            'Why 2: PSR head predicts mostly zeros (20/22) → fill-forward labels are trivial',
            'Why 3: PSR logits diverge (min=-30, max=+30) → transformer produces extreme values',
            'Why 4: Causal attention masking may prevent gradient flow to early time steps',
            'Why 5: PSR warmup (4000 steps with 2x multiplier) not yet complete at current step',
        ],
        'fixes': [
            {'type': 'config', 'key': 'PSR_SEQ_LOSS_SCALE', 'value': '3.0', 'comment': 'Double seq loss scale to push PSR out of zero region'},
            {'type': 'config', 'key': 'PSR_WARMUP_STEPS', 'value': '2000', 'comment': 'Half warmup to activate PSR earlier'},
            {'type': 'restart', 'retry_strategy': 'default'},
        ],
    },
    {
        'id': 'psr_logits_diverging',
        'description': 'PSR logits mean becoming more negative over time',
        'detect': lambda psr: (
            len(psr) >= 5
            and all(p['logits_mean'] < -5 for p in psr[-3:])
            and psr[-1]['logits_mean'] < psr[0]['logits_mean'] - 5
        ),
        'severity': 'WARN',
        'whys': [
            'Why 1: PSR logits mean trending negative → bias shifting toward -inf',
            'Why 2: Most targets are zeros (background) → optimal bias predicts all zeros',
            'Why 3: Transformer outputs diverge without meaningful temporal signal',
            'Why 4: PSR sequence length (T=2) may be too short for temporal pattern learning',
        ],
        'fixes': [
            {'type': 'config', 'key': 'PSR_WEIGHT', 'value': '50.0', 'comment': 'Increase PSR weight to amplify signal'},
            {'type': 'config', 'key': 'PSR_SEQ_LOSS_SCALE', 'value': '3.0', 'comment': 'Amplify seq loss to overcome zero-loss equilibrium'},
            {'type': 'restart', 'retry_strategy': 'default'},
        ],
    },
    {
        'id': 'gpu_oom',
        'description': 'GPU out of memory',
        'detect': lambda _: any(
            'CUDA out of memory' in l for l in tail_log(SUBPROCESS_LOG, 100)
        ) if SUBPROCESS_LOG.exists() else False,
        'severity': 'CRITICAL',
        'whys': [
            'Why 1: RTX 3060 12GB VRAM exhausted',
            'Why 2: Seq batches (T=2) + grad_accum=16 create large activation memory',
            'Why 3: Gradient checkpointing may not cover all layers',
            'Why 4: EMA shadow weights double model memory',
        ],
        'fixes': [
            {'type': 'config', 'key': 'USE_BACKBONE_CHECKPOINT', 'value': 'True', 'comment': 'Enable gradient checkpointing to reduce memory'},
            {'type': 'restart', 'retry_strategy': 'reduce_lr_5x'},
        ],
    },
    {
        'id': 'loss_spike',
        'description': 'Detection loss spike > 10x rolling mean',
        'detect': lambda _: False,  # checked via stage_manager
        'severity': 'HIGH',
        'whys': [
            'Why 1: Loss spike indicates optimizer stepped into bad region',
            'Why 2: PSR seq loss spike (~1077) on seq batch corrupts shared backbone',
            'Why 3: Zero backbone grads on seq steps prevents PSR from disrupting det features',
            'Why 4: If spike persists despite zero-backbone-grad fix, PSR_WEIGHT still too high',
        ],
        'fixes': [
            {'type': 'config', 'key': 'GRAD_CLIP_NORM', 'value': '0.5', 'comment': 'Tighter gradient clipping to dampen spikes'},
            {'type': 'config', 'key': 'PSR_WEIGHT', 'value': '15.0', 'comment': 'Reduce PSR weight for stability'},
            {'type': 'restart', 'retry_strategy': 'reduce_lr_5x'},
        ],
    },
    {
        'id': 'no_progress',
        'description': 'Validation metrics not improving over patience window',
        'detect': lambda _: False,  # checked via stage_manager --check
        'severity': 'HIGH',
        'whys': [
            'Why 1: Training is not improving validation metrics',
            'Why 2: Learning rate may be too low (slow convergence) or too high (oscillation)',
            'Why 3: Dataset subset ratio may be too small for generalization',
            'Why 4: Kendall weighting may suppress important task gradients',
        ],
        'fixes': [
            {'type': 'config', 'key': 'WARMUP_EPOCHS', 'value': '10', 'comment': 'Extended warmup for better convergence'},
            {'type': 'restart', 'retry_strategy': 'reduce_lr_2x_warmup_2x'},
        ],
    },
]


def _stage_trains_psr(stage_name: str) -> bool:
    """Check if the given stage preset trains PSR.

    Reads the preset dict from config.py to determine whether train_psr is
    True for this stage. Returns True by default (safe — won't skip checks
    for unknown stages).
    """
    try:
        from src import config as C
        preset_key = f'stage_{stage_name}'
        presets = getattr(C, 'PRESETS', {})
        if not presets:
            # Try loading presets from config
            presets = getattr(C, 'APPLY_PRESET_MAP', {}) or {}
        if not presets:
            # Fallback: check if apply_preset sets train_psr directly
            return True
        stage_cfg = presets.get(preset_key, {})
        return bool(stage_cfg.get('train_psr', True))
    except Exception:
        return True  # safe default — don't skip if we can't determine


def diagnose(det_debug: List[Dict], det_health: List[Dict],
             psr_diag: List[Dict], gpu: Dict[str, Any],
             stage_name: str = '') -> List[Dict]:
    """Run symptom detection on all available data.

    Args:
        stage_name: Current RF stage name (e.g. 'rf1'). Used to skip
            PSR checks when train_psr=False for the stage.
    """
    triggered = []
    for symptom in SYMPTOM_DATABASE:
        if symptom['detect'](det_debug):
            triggered.append(symptom)

    # PSR-specific (needs psr_diag) — skip if stage doesn't train PSR
    trains_psr = _stage_trains_psr(stage_name) if stage_name else True
    if not trains_psr:
        logger.info(f'[STAGE-AWARE] Skipping PSR checks — train_psr=False for {stage_name}')
    for symptom in SYMPTOM_DATABASE:
        if symptom['id'] in ('psr_dead', 'psr_logits_diverging'):
            if not trains_psr:
                continue  # PSR not trained in this stage — expected behavior
            if symptom['detect'](psr_diag):
                if symptom not in triggered:
                    triggered.append(symptom)
    # OOM check
    for symptom in SYMPTOM_DATABASE:
        if symptom['id'] == 'gpu_oom' and symptom['detect'](None):
            if symptom not in triggered:
                triggered.append(symptom)
    return triggered


def apply_fixes(symptoms: List[Dict]) -> bool:
    """Apply fixes for all triggered symptoms. Returns True if restart needed."""
    needs_restart = False
    applied = set()

    for symptom in symptoms:
        if symptom['id'] in applied:
            continue
        applied.add(symptom['id'])
        logger.warning(f'[INTERVENTION] Applying fixes for: {symptom["description"]}')

        for fix in symptom['fixes']:
            if fix['type'] == 'config':
                set_config_value(fix['key'], fix['value'], fix['comment'])
            elif fix['type'] == 'restart':
                needs_restart = True

    return needs_restart


# ── Stage Manager Integration ──────────────────────────────────────────────

def run_stage_manager_check() -> Tuple[int, str]:
    """Run stage_manager --check and return (returncode, stdout)."""
    try:
        result = subprocess.run(
            [sys.executable, str(STAGE_MANAGER), '--check'],
            capture_output=True, text=True, timeout=120,
            cwd=str(PROJECT_ROOT),
        )
        return result.returncode, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        logger.error('stage_manager --check timed out (120s)')
        return -1, 'timeout'
    except Exception as e:
        logger.error(f'stage_manager --check failed: {e}')
        return -1, str(e)


def launch_training_via_stage_manager(stage: str) -> bool:
    """Launch training via stage_manager --launch."""
    # Safety check: refuse if any training process is already running
    if is_any_training_running():
        logger.warning('[CONFLICT] Refusing to launch — training process already running')
        return False
    try:
        result = subprocess.run(
            [sys.executable, str(STAGE_MANAGER), '--launch', stage.upper()],
            capture_output=True, text=True, timeout=30,
            cwd=str(PROJECT_ROOT),
        )
        if result.returncode != 0:
            logger.error(f'Launch failed: {result.stderr[:500]}')
            return False
        logger.info(f'Launched {stage}')
        return True
    except Exception as e:
        logger.error(f'Launch exception: {e}')
        return False


# ── Epoch-Level Orchestration ─────────────────────────────────────────────

def epoch_sync(state: Dict[str, Any], log_lines: List[str]) -> Dict[str, Any]:
    """Track epoch transitions and trigger epoch-level actions.
    Uses in-memory-only keys (_last_epoch, _steps_in_epoch) stripped before save.
    """
    epoch, step, total_steps = parse_epoch_progress(log_lines)
    if epoch > 0:
        prev_epoch = state.get('_last_epoch', 0)
        if epoch != prev_epoch:
            logger.info(f'[EPOCH] {epoch} (was {prev_epoch}) — step {step}/{total_steps}')
            state['_last_epoch'] = epoch
            state['epoch'] = epoch
            state['_steps_in_epoch'] = step
    return state


# ── DET-DEBUG Collapse Monitor ─────────────────────────────────────────────

CLS_MEAN_HEALTHY_MIN = -2.8  # ideal: -2.2 to -2.5
CLS_MEAN_WARN = -15.0        # warning threshold (pi=0.01 normal range is -11 to -12)
CLS_MEAN_CRITICAL = -18.0    # collapse threshold (verified non-collapse at -12 with 237 alive anchors)


def assess_det_head(det_debug: List[Dict]) -> Tuple[str, Optional[float], str]:
    """Assess detection head health from DET-DEBUG cls_mean.

    Returns: (status, cls_mean, reason)
    """
    if not det_debug:
        return 'UNKNOWN', None, 'No DET-DEBUG data'

    latest = det_debug[-1]
    cls_mean = latest.get('cls_mean')
    if cls_mean is None:
        return 'UNKNOWN', None, 'No cls_mean in latest DET-DEBUG'

    near_zero = latest.get('near_zero', 0)
    step = latest.get('step', 0)

    if cls_mean < CLS_MEAN_CRITICAL:
        return 'COLLAPSED', cls_mean, f'cls_mean={cls_mean:.3f} < {CLS_MEAN_CRITICAL} at step {step}'
    elif cls_mean < CLS_MEAN_WARN:
        return 'WARN', cls_mean, f'cls_mean={cls_mean:.3f} in warning zone at step {step}'
    elif near_zero > 0.5:
        return 'WARN', cls_mean, f'near_zero={near_zero:.2%} — degenerate logits at step {step}'
    elif CLS_MEAN_HEALTHY_MIN <= cls_mean <= -1.5:
        return 'HEALTHY', cls_mean, f'cls_mean={cls_mean:.3f} in ideal range at step {step}'
    else:
        return 'OK', cls_mean, f'cls_mean={cls_mean:.3f} at step {step}'


# ── PSR Head Monitor ───────────────────────────────────────────────────────

PSR_LOSS_ZERO_THRESHOLD = 1e-6


def assess_psr_head(psr_diag: List[Dict]) -> Tuple[str, Optional[float], str]:
    """Assess PSR head health from PSR_DIAG entries.

    Returns: (status, loss, reason)
    """
    if not psr_diag:
        return 'UNKNOWN', None, 'No PSR_DIAG data'

    latest = psr_diag[-1]
    loss = latest.get('loss', 0)
    logits_mean = latest.get('logits_mean', 0)

    if loss < PSR_LOSS_ZERO_THRESHOLD:
        return 'DEAD', loss, f'PSR loss={loss:.3e} — stuck at zero'
    elif logits_mean < -15:
        return 'DIVERGING', loss, f'PSR logits mean={logits_mean:.2f} — trending negative'
    elif logits_mean > -5:
        return 'ACTIVE', loss, f'PSR logits mean={logits_mean:.2f} — healthy range'
    else:
        return 'STABLE', loss, f'PSR loss={loss:.3e} logits mean={logits_mean:.2f}'


# ── Checkpoint Monitor ─────────────────────────────────────────────────────

def check_checkpoints(state: Dict) -> Dict[str, Any]:
    """Check checkpoint freshness and validity."""
    result = {}
    for name, path in [('latest', CKPT_DIR / 'latest.pth'),
                       ('best', CKPT_DIR / 'best.pth'),
                       ('stage_latest', CKPT_DIR / state.get('current_stage', 'rf1') / 'latest.pth')]:
        if path.exists():
            mtime = path.stat().st_mtime
            age_min = (time.time() - mtime) / 60
            size_mb = path.stat().st_size / (1024 * 1024)
            result[name] = {
                'exists': True,
                'age_min': round(age_min, 1),
                'size_mb': round(size_mb, 1),
            }
            if age_min > 60:
                result[name]['stale'] = True
        else:
            result[name] = {'exists': False}
    return result


# ── Main Supervisor Loop ───────────────────────────────────────────────────

def main():
    logger.info('=' * 60)
    logger.info(f'Training Supervisor @ {datetime.now().isoformat()}')
    logger.info('=' * 60)

    # ── 1. Load state & check cooldown ──
    state = load_state()
    if not state:
        logger.error('No state file — run stage_manager --launch RF1 first')
        return

    if check_cooldown(state):
        return

    # ── 2. Check PID / conflicting processes ──
    pid = state.get('training_pid')
    stage = state.get('current_stage', '?')
    status = state.get('status', '?')
    pid_alive = is_pid_alive(pid)

    # Detect orphaned/zombie training processes the PID file doesn't track
    other_running = is_any_training_running()
    if not pid_alive and other_running:
        logger.warning(f'[CONFLICT] Stale PID {pid} but other train.py process(es) found — cleaning up')
        kill_any_training()
    elif pid_alive and not is_pid_alive(pid):
        # PID died between checks
        pid_alive = False

    logger.info(f'Stage: {stage} [{status}]  PID: {pid} [{"ALIVE" if pid_alive else "DEAD"}]'
                f'  other_train={"YES" if other_running else "no"}')

    # ── 3. Read logs ──
    sub_lines = tail_log(SUBPROCESS_LOG, 500) if SUBPROCESS_LOG.exists() else []
    epoch, step, total_steps = parse_epoch_progress(sub_lines) if sub_lines else (0, 0, 0)

    # ── 4. GPU check ──
    gpu = get_gpu_info()
    logger.info(f'GPU: {gpu["gpu_util_pct"]:.0f}% util  {gpu["mem_used_mib"]:.0f}/{gpu["mem_total_mib"]:.0f} MiB  '
                f'{gpu["temp_c"]:.0f}°C')

    # OOM guard
    if gpu['mem_util_pct'] > 95 and pid_alive:
        logger.warning(f'[OOM GUARD] GPU memory at {gpu["mem_util_pct"]:.0f}% — potential OOM risk')

    # ── 5. Parse DET-DEBUG ──
    det_debug = parse_det_debug(sub_lines)
    det_health = parse_det_health(sub_lines)
    psr_diag = parse_psr_diag(sub_lines)

    if det_debug:
        latest_dd = det_debug[-1]
        logger.info(f'DET-DEBUG: step={latest_dd["step"]} cls_mean={latest_dd["cls_mean"]:.3f} '
                    f'std={latest_dd["cls_std"]:.3f} near_zero={latest_dd["near_zero"]:.2%}')

    if det_health:
        latest_dh = det_health[-1]
        logger.info(f'DET-HEALTH: step={latest_dh["step"]} cls_mean={latest_dh["cls_mean"]:.3f} '
                    f'near_zero={latest_dh["near_zero"]:.2%}')

    if psr_diag:
        latest_psr = psr_diag[-1]
        logger.info(f'PSR: loss={latest_psr["loss"]:.3e} logits_mean={latest_psr["logits_mean"]:.1f} '
                    f'zeros={latest_psr["zeros"]}/{latest_psr["total"]}')

    # ── 6. Detection head assessment ──
    det_status, det_cls_mean, det_reason = assess_det_head(det_debug)
    if det_status in ('COLLAPSED', 'WARN'):
        logger.warning(f'[DET-{det_status}] {det_reason}')

    # ── 7. PSR head assessment (stage-aware) ──
    trains_psr = _stage_trains_psr(stage) if stage else True
    psr_status, psr_loss, psr_reason = assess_psr_head(psr_diag)
    if psr_status in ('DEAD', 'DIVERGING') and not trains_psr:
        logger.info(f'[PSR-{psr_status}] {psr_reason} — EXPECTED (train_psr=False for {stage})')
    elif psr_status in ('DEAD', 'DIVERGING'):
        logger.info(f'[PSR-{psr_status}] {psr_reason}')

    # ── 8. Checkpoints ──
    ckpt_info = check_checkpoints(state)
    for name, info in ckpt_info.items():
        if info.get('stale'):
            logger.warning(f'[CKPT] {name} stale — {info["age_min"]} min old')
        elif info.get('exists'):
            logger.info(f'[CKPT] {name}: {info["age_min"]} min old, {info["size_mb"]} MB')

    # ── 9. Run stage_manager --check (handles 5-category checklist + decision) ──
    logger.info('─' * 40)
    logger.info('Running stage_manager --check...')
    retcode, sm_output = run_stage_manager_check()

    # Reload state after stage_manager modified it
    state = load_state()
    pid_alive = is_pid_alive(state.get('training_pid'))

    # Parse stage_manager output for decision
    decision = 'continue'
    for line in sm_output.split('\n'):
        if 'Decision:' in line:
            decision = line.split('Decision:')[-1].strip()
            logger.info(f'Stage Manager decision: {decision}')

    # ── 10. Deep intervention (only if training alive AND we detected issues) ──
    if pid_alive:
        symptoms = diagnose(det_debug, det_health, psr_diag, gpu, stage_name=stage)
        if symptoms:
            severities = [s['severity'] for s in symptoms]
            critical = [s for s in symptoms if s['severity'] == 'CRITICAL']
            warnings = [s for s in symptoms if s['severity'] in ('HIGH', 'WARN')]

            # Log all symptoms
            for s in symptoms:
                logger.warning(f'[SYMPTOM] {s["id"]}: {s["description"]} (severity={s["severity"]})')
                for why in s['whys'][:3]:  # top 3 whys
                    logger.warning(f'  {why}')

            # CRITICAL symptoms → immediate kill + fix + restart
            if critical:
                logger.error(f'[CRITICAL] {len(critical)} critical symptom(s) detected!')
                logger.error('Intervening: killing training, applying fixes, restarting.')
                kill_training(pid)
                time.sleep(2)

                # Apply config fixes
                apply_fixes(critical + warnings)

                # Wait for stage_manager to detect death and relaunch
                logger.info('Waiting for stage_manager to detect death and handle retry...')
                time.sleep(5)

                # If stage_manager didn't restart, force it
                state = load_state()
                if not is_pid_alive(state.get('training_pid')) and state.get('status') != 'running':
                    logger.info('Stage did not restart — forcing launch via stage_manager')
                    launch_training_via_stage_manager(stage)

            # WARN symptoms → log and let stage_manager handle (it may retry)
            elif warnings:
                logger.info(f'[WARN] {len(warnings)} warning(s) — monitoring')

    # ── 11. Epoch sync ──
    if sub_lines:
        state = epoch_sync(state, sub_lines)

    # ── 12. Summary ──
    logger.info('─' * 40)
    if pid_alive:
        logger.info(f'Status: RUNNING  Epoch {epoch} step {step}/{total_steps}  '
                    f'det_cls_mean={det_cls_mean or "N/A"}  PSR={psr_status}')
    elif state.get('status') == 'completed':
        logger.info('Status: ALL STAGES COMPLETE!')
    elif state.get('status') == 'retrying':
        logger.info(f'Status: RETRYING (attempt {state.get("retry_count", 0)})')
    elif state.get('status') == 'failed':
        logger.info('Status: FAILED — human intervention needed')
    else:
        logger.info(f'Status: {state.get("status", "?")} — waiting for stage_manager')

    # ── 13. Progress estimate ──
    if pid_alive and det_cls_mean is not None:
        progress_pct = (step / total_steps * 100) if total_steps else 0
        logger.info(f'Progress: epoch {epoch} step {step}/{total_steps} ({progress_pct:.0f}%)  '
                    f'det_cls_mean={det_cls_mean:.3f}')

    # ── Save state + cooldown timestamp ──
    state['_last_supervisor_run'] = time.time()
    save_state(state)
    logger.info(f'[COOLDOWN] Next check allowed after {datetime.fromtimestamp(time.time() + MIN_INTERVAL_MINUTES * 60).strftime("%H:%M")}')


if __name__ == '__main__':
    main()
