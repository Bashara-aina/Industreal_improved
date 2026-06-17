import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

"""
 IndustReal Configuration — POPW Adaptation
 ===========================================
 Single unified configuration for IndustReal dataset:
   - Action Recognition (AR): 74 atomic action classes
   - Assembly State Detection (ASD): 24 assembly states
    - Procedure Step Recognition (PSR): 36 procedure steps, 11 components
   - Single egocentric RGB camera

 Dataset location: /home/newadmin/swarm-bot/project/popw/working/data/dataset/industreal/
 Code location:    /home/newadmin/swarm-bot/project/popw/working/code/industreal_improved/
 """

import logging
from pathlib import Path

_cfg_logger = logging.getLogger(__name__)

# =========================================================================
# Debug / profiling flags
# =========================================================================
BENCHMARK_MODE = True
DEBUG_MODE         = False
DEBUG_MAX_VIDEOS   = 2  # smoke test: 2 recordings only
DEBUG_FRAME_STRIDE = 10

SUBSET_RATIO = 0.05   # 5% subset for quick training

TRAIN_FRAME_STRIDE = 3  # A.2: stride 3 → T=16 covers 1.6s at 30FPS (median action)
EVAL_FRAME_STRIDE  = 1
USE_SPATIAL_AUG = True           # Enable spatial augmentation (flip, crop)

# Ablation flags
# =========================================================================
TRAIN_DET       = True
TRAIN_HEAD_POSE = True    # Train 9-DoF head pose head with Kendall uncertainty (was False — headpose_film was untrained)
TRAIN_ACT       = True
TRAIN_PSR       = True
USE_KENDALL     = True   # Kendall weighting active for 4 tasks (det, act, psr, head_pose 9-DoF MSE)

# Hand-FiLM conditioning (hand keypoints → FiLM modulation on activity features)
USE_HAND_FILM   = True
HAND_FILM_CHANNELS = 768   # ConvNeXt C5 channel count

# Backbone configuration — ResNet-50 or ConvNeXt-Tiny (Doc 01 D.1)
# =========================================================================
BACKBONE = 'convnext_tiny'

# ConvNeXt channel counts (used if BACKBONE='convnext_tiny')
CONVNEXT_CHANNELS = {
    'c2': 96,
    'c3': 192,
    'c4': 384,
    'c5': 768,
}

# HeadPoseFiLM — second-stage FiLM from 9-DoF head pose (Doc 01 E)
USE_HEADPOSE_FILM = True

# VideoMAE V2 stream — 2-stream activity recognition (Doc 01 §B.1)
# ============================================================================
# Doc 01 §B.1: This is the single biggest unlock for Activity Top-1.
# Setting this to True gives +5 to +7% Top-1 improvement.
# Cost: +22M frozen params, ~600 MB VRAM. FPS drops ~25%.
# NOTE: When switching from False to True, classifier head reinitializes
# because act_logits dim doubles. Train from scratch or use strict=False.
USE_VIDEOMAE = True
VIDEOMAE_CKPT = 'MCG-NJU/videomae-small-finetuned-kinetics'
VIDEOMAE_NUM_FRAMES = 16   # temporal window size for VideoMAE clip
VIDEOMAE_SAMPLE_STRIDE = 1  # sample every N frames from the clip window

# VideoMAE fine-tuning unlock (Doc 01 §B.1): freeze for first N epochs, then unfreeze
# to let the temporal stream adapt to IndustReal kinematics. Unfreezing too early
# (before backbone is warmed up) causes interference; too late wastes the stream's
# Kinetics pretraining. Epoch 10 is after Stage 1 (5) + Stage 2 start (6-10).
VIDEOMAE_UNFREEZE_EPOCH = 10  # unfreeze VideoMAE at epoch 10 (after Stage 1/2 warmup)
VIDEOMAE_UNFREEZE_LR = 1e-5

# Temporal modeling options
USE_TMA_CELL = True       # GRU-based Temporal Masked Attention Cell
USE_TEMPORAL_BANK = True  # Feature Bank (Doc 01 A.2: T=16)
FEATURE_BANK_WINDOW = 16    # T=16 — 1.6s context at 30FPS (median action)
EMA_SMOOTHING = False      # Not in diagram — removed

# =========================================================================
# Paths
# =========================================================================
POPW_ROOT = Path('/media/newadmin/master/POPW/datasets/industreal')

# Output root for runs (relative to this config file's directory)
OUTPUT_ROOT = Path(__file__).parent / 'runs'

# Recordings root (train/val/test splits)
RECORDINGS_ROOT = POPW_ROOT / 'recordings'

# Official train/val/test CSVs for AR task splits
TRAIN_CSV = POPW_ROOT / 'splits' / 'train.csv'
VAL_CSV   = POPW_ROOT / 'splits' / 'val.csv'
TEST_CSV  = POPW_ROOT / 'splits' / 'test.csv'

# =========================================================================
# Camera configuration (single egocentric RGB)
# =========================================================================
CAMERAS = ['rgb']
NUM_CAMERAS = len(CAMERAS)
CAMERA = 'rgb'

# =========================================================================
# Dataset constants
# =========================================================================

# --- Assembly State Detection (ASD) ---
NUM_DET_CLASSES = 24   # background + 22 assembly states + error_state

DET_CLASS_NAMES = {
    1:  'background',
    2:  '10000000000',
    3:  '10010010000',
    4:  '10010100000',
    5:  '10010110000',
    6:  '11100000000',
    7:  '11110010000',
    8:  '11110100000',
    9:  '11110110000',
    10: '11110111100',
    11: '11110111110',
    12: '11110110001',
    13: '11110111101',
    14: '11110111111',
    15: '11110101111',
    16: '11110011111',
    17: '11110011110',
    18: '11110101110',
    19: '11100001110',
    20: '11101101110',
    21: '11101011110',
    22: '11101111110',
    23: '11101111111',
    24: 'error_state',
}

# --- Action Recognition (AR) ---
# [ROOT-CAUSE FIX — activity class count] -----------------------------------
# industreal_dataset.py:_parse_ar_labels writes the *raw* action_id straight
# into the per-frame label (`labels[start:end+1] = action_id`) with NO
# remapping. action_id 0 = NA/background; real action IDs run 1..74. So the
# label index space is exactly 0..74 → the classifier MUST have 75 channels.
# IDs 37 and 64 are absent in stock IndustReal, leaving two permanently-cold
# channels — harmless (no GT lands there, so CE/LDAM never push them).
#
# The previous code computed NUM_CLASSES_ACT by scanning AR_labels.csv on disk
# and PRUNING the missing-ID names, which produced 74 when 37/64 were absent
# and 75 when present. With a 74-wide head, a label of 74 indexes out of range
# → CUDA device-side assert that kills the full-dataset run. We therefore pin
# the count to a fixed constant equal to (max raw action_id) + 1 = 75.
NUM_ACT_RAW_IDS = 74          # action IDs 1..74 (0 is NA); IndustReal spec
NUM_CLASSES_ACT = NUM_ACT_RAW_IDS + 1   # 75 = NA(0) + IDs 1..74; FIXED, not data-derived


def _load_act_class_names() -> list:
    """
    Build a 75-entry name list indexed BY RAW action_id, so name[i] describes
    class index i exactly as the dataset emits it. No pruning, no shifting:
    index 0 = 'NA', indices 1..74 = the action name from AR_labels.csv (or an
    'unknown_i' placeholder for the cold IDs 37/64). Display/reporting only —
    NUM_CLASSES_ACT above is the source of truth for tensor shapes.
    """
    import os

    recordings_root = RECORDINGS_ROOT
    id_to_name = {}
    for split in ['train', 'val', 'test']:
        split_root = recordings_root / split
        if not split_root.exists():
            continue
        for rec in sorted(os.listdir(split_root)):
            ar_file = split_root / rec / 'AR_labels.csv'
            if not ar_file.exists():
                continue
            try:
                with open(ar_file, encoding='utf-8') as f:
                    for row in f:
                        parts = row.strip().split(',')
                        if len(parts) >= 3:
                            aid = int(parts[1])
                            if aid not in id_to_name:
                                id_to_name[aid] = parts[2]
            except OSError:
                continue

    names = []
    for i in range(NUM_CLASSES_ACT):           # 0..74, raw-id-aligned
        if i == 0:
            names.append('NA')
        else:
            names.append(id_to_name.get(i, f'unknown_{i}'))
    return names


ACT_CLASS_NAMES = _load_act_class_names()
assert len(ACT_CLASS_NAMES) == NUM_CLASSES_ACT == 75, (
    f'Activity class space must be a fixed 75 (NA + raw action IDs 1..74), '
    f'got NUM_CLASSES_ACT={NUM_CLASSES_ACT}, len(names)={len(ACT_CLASS_NAMES)}. '
    f'The dataset uses raw action_id as the class index, so the head width is '
    f'invarianT to which IDs happen to appear on disk.'
)

# --- Head pose ---
NUM_KEYPOINTS = 17
KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
]
KEYPOINT_FLIP_PAIRS = [
    (1, 2), (3, 4), (5, 6), (7, 8),
    (9, 10), (11, 12), (13, 14), (15, 16),
]

# 9-DoF head pose: forward_vector[3] + position[3] + up_vector[3]
NUM_HEAD_POSE_DOF = 9

# Hand joints: 26 MediaPipe-style joints per hand, (x, y) each = 52-D per hand
NUM_HAND_JOINTS = 26

# --- Procedure Step Recognition (PSR) ---
NUM_PSR_STEPS = 36       # number of distinct procedure step types (from procedure_info.json)
NUM_PSR_COMPONENTS = 11  # number of assembly components (comp0-comp19 in PSR_labels_raw.csv)

# =========================================================================
# Image and model
# =========================================================================
# Anchor sizes for RetinaNet detection head (Doc 01 B.3)
# These are calibrated via k-means on GT boxes: python calibrate_anchors.py
# Calibrated via k-means on 923 GT boxes (2 train recordings on disk):
#   w=146-594px, h=89-463px, w/h ratio=0.82-2.95 (wider than tall)
#   sqrt(area)=188-436px, k-means centers: 164, 269, 333, 338, 404
# Paper spec (matches RetinaNet P3-P7): (24, 48, 96, 192, 384)
# =========================================================================
ANCHOR_SIZES = (24, 48, 96, 192, 384)
IMG_WIDTH       = 1280
IMG_HEIGHT      = 720
IMG_SIZE        = (IMG_WIDTH, IMG_HEIGHT)
ORIGINAL_WIDTH  = 1280
ORIGINAL_HEIGHT = 720

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# =========================================================================
# Training (RTX 3060 12 GB) — Optimized for VideoMAE + ConvNeXt + 5 heads + TMA + TemporalBank
# NOTE: BATCH_SIZE=1 is REQUIRED. VideoMAE alone uses +600MB VRAM; batch=2 causes OOM.
# GRAD_ACCUM=32 maintains effective batch=32 (same as old BATCH_SIZE=8, accum=4).
BATCH_SIZE = 6        # Original — BATCH_SIZE=8 caused OOM even with more memory cap
GRAD_ACCUM_STEPS = 6  # Keeps EFFECTIVE_BATCH=32
EFFECTIVE_BATCH      = BATCH_SIZE * GRAD_ACCUM_STEPS  # 32

VAL_BATCH_SIZE = 16   # Was 8→32; bumped again (VRAM headroom: 1.30GB/12.5GB used at batch_size=8)
VAL_NUM_WORKERS = 4
VAL_PREFETCH_FACTOR  = 4

EPOCHS        = 100 
BASE_LR       = 5e-4   # Per paper: "5e-4"
WEIGHT_DECAY  = 1e-4
WARMUP_EPOCHS = 5
USE_COSINE_ANNEALING = True
T_0 = 10
T_mult = 2
PATIENCE      = 10
GRAD_CLIP_NORM = 1.0
VAL_EVERY = 1    # [BENCHMARK] Evaluate every 1 epoch (BENCHMARK_MODE override)
EVAL_MAX_BATCHES = -1

NUM_WORKERS = 8
PIN_MEMORY = True
MIXED_PRECISION = True   # [FIX] Was False — FP16 for RTX 3060 Ampere tensor cores
SEED            = 42

# EMA (Exponential Moving Average) — [FIX #4 HIGH] Enabled per paper §Training: EMA=0.999 in Stage 3
USE_EMA        = True
EMA_DECAY      = 0.999  # standard decay for image models

# Mixup augmentation for activity head
USE_MIXUP      = True
MIXUP_ALPHA    = 0.4

# Desktop stability knobs
CUDA_MEMORY_FRACTION = 0.98
TRAIN_NICE = 10
TORCH_NUM_THREADS = 12

# Validation memory-safety knobs
VAL_PREFETCH_FACTOR = 2   # 2 workers × 2 prefetch (was 4 — overkill for VAL_NUM_WORKERS=0)
TRAIN_PREFETCH_FACTOR = 4  # 2 workers × 2 prefetch = 4 batches queued (was 4)
# FIX: Lower DET_EVAL_SCORE_THRESH from 0.5 to 0.0 to avoid zero predictions when
# model sigmoid scores cluster near 0.5 (e.g., [0.47, 0.53]) — all filtered out at 0.5.
# With 0.0, at least the top-scoring prediction per location is kept for mAP calculation.
DET_EVAL_SCORE_THRESH = 0.0
DET_EVAL_MAX_PER_IMAGE = 300
DET_EVAL_NMS_IOU_THRESH = 0.5  # NMS IoU threshold for detection evaluation
SAVE_VAL_CONFUSION_MATRIX = False
COMPUTE_VAL_TOP5 = True

# Auto-fallback for Linux systems with small /dev/shm
DATALOADER_AUTO_FALLBACK = True

# Performance flags
USE_UINT8_DATA_PIPELINE = True
CUDNN_DETERMINISTIC = True   # Full CUDA determinism (warn_only if unsupported)
CUDNN_BENCHMARK = False     # Required for reproducibility — was True

# Ampere (RTX 3060) speedups
ALLOW_TF32 = True
MATMUL_PRECISION = 'high'

# =========================================================================
# Loss hyperparameters
# =========================================================================
FOCAL_ALPHA   = 0.25
FOCAL_GAMMA   = 2.0
GIOU_WEIGHT   = 2.0  # Doc 01 B.2: GIoU regression weight vs cls weight=1.0

WING_OMEGA   = 0.05
WING_EPSILON = 0.005

CB_BETA  = 0.999
CB_GAMMA = 2.0
CB_LABEL_SMOOTHING = 0.1  # label smoothing for 74-class activity recognition

# PSR temporal smoothing weight (transition-aware loss)
PSR_TEMPORAL_SMOOTH_WEIGHT = 0.05  # encourages predicted transitions to match label transitions

# =========================================================================
# Synthetic pretraining (Doc 01 B.1)
# =========================================================================
PRETRAIN_DET_ON_SYNTH = True   # Pretrain backbone+FPN+det head on synthetic data
PRETRAIN_DET_EPOCHS = 20  # Doc 01 §B: 20 epochs for detection pretraining on synthetic data
PRETRAIN_DET_LR = 5e-4
PRETRAIN_DET_FRAME_STRIDE = 10  # 1-in-10 frames → ~728K train frames; enough diversity for backbone pretraining

# Doc 01 B.3: ASD-only augmentation during pretrain
PRETRAIN_MOSAIC_PROB = 0.3   # probability of mosaic (4-frame collage)
PRETRAIN_MIXUP_PROB  = 0.2   # probability of mixup (2-frame alpha blend)
PRETRAIN_HFLIP_PROB  = 0.5   # probability of random horizontal flip

# =========================================================================
# Staged training (Doc 2 B.1)
# =========================================================================
STAGED_TRAINING = False
STAGE1_EPOCHS = 5    # Detection-only warmup
STAGE2_EPOCHS = 10   # Add pose + head pose
STAGE3_EPOCHS = 85   # Full multi-task with EMA — 5+10+85=100 total — was 35
ACT_RAMP_EPOCHS = 5  # Activity loss ramp-up
ACTIVITY_LOSS_CAP = 80.0  # Cap activity loss to prevent NaN cascade at Stage 3 entry (epoch 16). 80 allows LDAM losses (~55) to pass without capping while still protecting against extreme spikes. Gradient = cap/loss at saturation: 80/400=0.2 (vs 40/400=0.1 with old cap).
# Smooth loss caps: x if x<=cap, cap*(1+log(x/cap)) if x>cap. Gradient=1 below cap, cap/x above cap (never zero).
DET_LOSS_CAP = 50.0      # Detection: GIoU + Focal cls loss cap
POSE_LOSS_CAP = 30.0     # Body keypoint Wing Loss cap
PSR_LOSS_CAP = 20.0      # PSR focal loss + temporal smooth cap
HEAD_POSE_LOSS_CAP = 30.0  # Head pose 9-DoF MSE cap
STAGE3_WARMUP_EPOCHS = 3  # LR warmup epochs at Stage 3 entry to stabilize new head activation

# =========================================================================
# LDAM-DRW for activity (Doc 01 §B.2 + Doc 2 C.2)
# =========================================================================
USE_LDAM_DRW = True   # Use LDAM+DRW instead of CB-Focal for activity
LDAM_MAX_M = 0.5
LDAM_S = 30
LDAM_DRW_EPOCH = 0    # Switch to CB weights at this epoch (DRW deferred re-weighting)
# DRW activates immediately (epoch 0) when activity training begins, applying class-balanced
# weights from the start. This corrects the prior misconfiguration where DRW was delayed
# to epoch 60, resulting in 60 epochs of unweighted LDAM margin loss before CB re-weighting.
# Features being "stable" at epoch 60 was not supported by IndustReal experimental evidence.

# =========================================================================
# PSR focal loss (Doc 01 §D + Doc 2 C.3)
# =========================================================================
PSR_FOCAL_ALPHA = 0.25
PSR_FOCAL_GAMMA = 1.0  # was 2.0 — gamma=2.0 collapses PSR to trivial solution
                       # (predicting 0 always gives near-zero gradient on easy negatives).
                       # gamma=1.0 gives 40% gradient ratio (active vs inactive) vs
                       # 2.6% with gamma=2.0, enabling the head to escape local minimum.

# PSR sequence-mode training (Doc 01 §D.1) — THE biggest PSR unlock
# =========================================================================
# Doc 01 §D.1: Current training samples one frame per step, making the
# temporal Transformer dormant. Sequence-mode trains on contiguous T-frame
# windows where the Transformer is properly engaged.
USE_PSR_SEQUENCE_MODE = True   # Doc 01 §D.2: PSR sequence-mode — THE biggest PSR unlock
PSR_SEQUENCE_LENGTH = 4        # T=4 keeps memory bounded on 12GB GPU
PSR_SEQ_EVERY_N_BATCHES = 10  # Draw one sequence batch every N normal batches

# =========================================================================
# Augmentation (Doc 2 D)
# =========================================================================
USE_RANDAUGMENT = True   # Photometric augmentation for backbone
MIXUP_ALPHA = 0.4
CUTMIX_ALPHA = 1.0       # Alternate Mixup/CutMix each epoch
RANDOM_TEMPORAL_STRIDE = True  # Random frame stride {1,2,3} per clip (dataset.py line 875)

# =========================================================================
# Optimizer (Doc 2 E)
# =========================================================================
USE_LION = False        # [PAPER-ALIGN] Use AdamW (paper Table 3 specifies AdamW, not Lion)
ONE_CYCLE_LR = False     # Use OneCycleLR instead of CosineAnnealingWarmRestarts
USE_SWA = False          # Stochastic Weight Averaging at end of training
SWA_LR = 1e-5
SWA_EPOCHS = 10

# =========================================================================
# TTA (Doc 2 F)
# =========================================================================
USE_TTA = False          # Test-time augmentation (flip + multi-crop)
TTA_FLIP = True
TTA_CROPS = 5           # 5-crop TTA for activity

POSE_CONF_THRESHOLD = 0.1

# =========================================================================
# Monitoring / Visualization (TrainingMonitor integration)
# =========================================================================
MONITOR_ENABLED = True
SAVE_VIZ_EPOCHS = 5
NUM_VIZ_SAMPLES = 8
MONITOR_LOG_INTERVAL = 10
LOG_EFFICIENCY_EVERY = 10  # log GFLOPs/FPS every N epochs (0=disable)
# Detection metrics: compute_det_metrics_extended does 11×(24 classes × 35084 frames) nested
# Python loops = ~87 min/epoch. Set to True to enable, False to skip (epoch快了~87min).
SKIP_DET_METRICS_EVAL = False  # True = skip detection mAP computation each epoch
# Efficiency metrics: compute_efficiency_metrics does 35 forward passes each epoch.
# Set to True to skip except when (epoch % LOG_EFFICIENCY_EVERY == 0).
SKIP_EFFICIENCY_METRICS = True  # True = only compute every LOG_EFFICIENCY_EVERY epochs

# Kendall gradient sentinel logging (Doc 2 §B.1)
# Log gradient norms of Kendall log_var params every N steps to detect
# silent Kendall failures (Bug #9 reincarnation)
LOG_KENDALL_GRAD_EVERY = 100  # Set to 0 to disable

# Stage transition logging (Doc 2 §B.2)
LOG_STAGE_TRANSITION = True  # Log trainable param counts at each stage start

# Per-component PSR prevalence sanity check (Doc 2 §B.5)
LOG_PSR_PREVALENCE_EVERY = 10  # epochs; set to 0 to disable

# =========================================================================
# Outputs — dynamic, recomputed after preset application
# =========================================================================
CHECKPOINT_DIR = OUTPUT_ROOT / 'checkpoints'
LOG_DIR        = OUTPUT_ROOT / 'logs'
EVAL_SAVE_DIR  = OUTPUT_ROOT / 'eval_outputs'

COCO_CACHE_SIZE = 30
NUM_ACT_CLASSES = NUM_CLASSES_ACT

# Alias for backward compatibility / other modules expecting NUM_CLASSES
NUM_CLASSES = NUM_CLASSES_ACT

# =========================================================================
# PRESET SYSTEM
# =========================================================================
PRESETS = {
    'benchmark_full': {
        'description': (
            'Full benchmark comparison run. ConvNeXt-Tiny, single RGB camera, '
            'USE_TMA_CELL=True, USE_TEMPORAL_BANK=True, temporal ordering + Kendall. '
            'Hand-FiLM conditioning on activity head. '
            'VAL_EVERY=1, all metrics every epoch. '
            'BATCH_SIZE=1 due to VideoMAE VRAM cost (+600MB, ~25% FPS drop). '
            'Effective batch 32 via GRAD_ACCUM=32. '
            'Comparable against: MViTv2 Kinetics, YOLOv8m COCO+synth+real, B3 rule-based PSR, STORM-PSR.'
        ),
        'dataset_mode':       'manual_only',
        'backbone':           'convnext_tiny',
        'use_tma_cell':      True,
        'use_temporal_bank':  True,
        'use_hand_film':     True,
        'benchmark_mode':     True,
        'batch_size':        1,   # VideoMAE + ConvNeXt + TMA + TemporalBank exceeds 12GB at batch=2
        'grad_accum_steps':  32,  # Keep effective batch 32 (same as old 8×4)
    },
    'benchmark_quick': {
        'description': (
            'Quick baseline — no temporal, no Hand-FiLM. Faster iteration. '
            'Use when testing ASD detection + activity + PSR without temporal overhead.'
        ),
        'dataset_mode':       'manual_only',
        'backbone':           'convnext_tiny',
        'use_tma_cell':       True,
        'use_temporal_bank':  True,
        'use_hand_film':      True,
        'benchmark_mode':     True,
        'batch_size':         4,
    },
}


def apply_preset(preset_name: str) -> None:
    """Apply a preset configuration by name. Updates global flags."""
    global BENCHMARK_MODE, VAL_EVERY, BATCH_SIZE
    global USE_TMA_CELL, USE_TEMPORAL_BANK, USE_HAND_FILM
    global GRAD_ACCUM_STEPS

    if preset_name not in PRESETS:
        raise ValueError(f'Unknown preset: {preset_name}. Available: {list(PRESETS.keys())}')

    preset = PRESETS[preset_name]
    BENCHMARK_MODE = preset.get('benchmark_mode', False)
    USE_TMA_CELL = preset.get('use_tma_cell', False)
    USE_TEMPORAL_BANK = preset.get('use_temporal_bank', False)
    USE_HAND_FILM = preset.get('use_hand_film', USE_HAND_FILM)
    BATCH_SIZE = preset.get('batch_size', BATCH_SIZE)
    GRAD_ACCUM_STEPS = preset.get('grad_accum_steps', GRAD_ACCUM_STEPS)

    update_dynamic_paths()

    if 'description' in preset:
        _cfg_logger.info(f'[config] Applied preset "{preset_name}": {preset["description"]}')
    else:
        _cfg_logger.info(f'[config] Applied preset "{preset_name}"')


def update_dynamic_paths():
    """Recompute all dynamic paths after config changes."""
    global OUTPUT_ROOT, CHECKPOINT_DIR, LOG_DIR, EVAL_SAVE_DIR

    parts = ['full_multi_task']
    if USE_TMA_CELL:
        parts.append('tma')
    if USE_TEMPORAL_BANK:
        parts.append('tbank')
    if BENCHMARK_MODE:
        parts.append('benchmark')

    OUTPUT_ROOT = Path(__file__).parent / 'runs' / '_'.join(parts)
    CHECKPOINT_DIR = OUTPUT_ROOT / 'checkpoints'
    LOG_DIR        = OUTPUT_ROOT / 'logs'
    EVAL_SAVE_DIR  = OUTPUT_ROOT / 'eval_outputs'


# =========================================================================
# CONFIG VALIDATION
# =========================================================================
def _validate_paths():
    """Check that critical paths exist."""
    issues = []

    if not POPW_ROOT.exists():
        issues.append(f'POPW_ROOT does not exist: {POPW_ROOT}')

    if not RECORDINGS_ROOT.exists():
        issues.append(f'recordings_root does not exist: {RECORDINGS_ROOT}')

    for name, path in [('train.csv', TRAIN_CSV), ('val.csv', VAL_CSV), ('test.csv', TEST_CSV)]:
        if not path.exists():
            issues.append(f'{name} does not exist: {path}')

    # Check at least one recording exists
    train_rec_dir = RECORDINGS_ROOT / 'train'
    if train_rec_dir.exists():
        recordings = list(train_rec_dir.iterdir())
        if recordings:
            sample_rec = recordings[0]
            rgb_dir = sample_rec / 'rgb'
            ar_file = sample_rec / 'AR_labels.csv'
            od_file = sample_rec / 'OD_labels.json'
            if not rgb_dir.exists():
                issues.append(f'Sample recording {sample_rec.name} has no rgb/ directory')
            if not ar_file.exists():
                issues.append(f'Sample recording {sample_rec.name} has no AR_labels.csv')
            if not od_file.exists():
                issues.append(f'Sample recording {sample_rec.name} has no OD_labels.json')

    if issues:
        for issue in issues:
            _cfg_logger.warning(f'[config] {issue}')
    else:
        _cfg_logger.info('[config] All critical paths validated.')
        sample_rec = list((RECORDINGS_ROOT / 'train').iterdir())[0].name
        _cfg_logger.info(f'[config] Sample recording: {sample_rec}')
        _cfg_logger.info(f'[config] NUM_CLASSES_ACT = {NUM_CLASSES_ACT}')
        _cfg_logger.info(f'[config] NUM_DET_CLASSES = {NUM_DET_CLASSES}')
        _cfg_logger.info(f'[config] CAMERAS = {CAMERAS}')


# =========================================================================
# INITIALIZATION
# =========================================================================
_validate_paths()
update_dynamic_paths()

_cfg_logger.info('[config] Initialized IndustReal config')
_cfg_logger.info(f'[config] POPW_ROOT: {POPW_ROOT}')
_cfg_logger.info(f'[config] OUTPUT_ROOT: {OUTPUT_ROOT}')
_cfg_logger.info(f'[config] BENCHMARK_MODE={BENCHMARK_MODE}, VAL_EVERY={VAL_EVERY}')
_cfg_logger.info(f'[config] USE_TMA_CELL={USE_TMA_CELL}, USE_TEMPORAL_BANK={USE_TEMPORAL_BANK}')
