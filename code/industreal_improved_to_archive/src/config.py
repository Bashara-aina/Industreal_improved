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
BENCHMARK_MODE = True  # When True: VAL_EVERY=1, full metrics every epoch
DEBUG_MODE         = False
DEBUG_MAX_VIDEOS   = 2  # smoke test: 2 recordings only
DEBUG_FRAME_STRIDE = 10

SUBSET_RATIO = 1.0  # [FIX] Was 0.05 — full dataset for full benchmark

TRAIN_FRAME_STRIDE = 3  # A.2: stride 3 → T=16 covers 1.6s at 30FPS (median action)
EVAL_FRAME_STRIDE  = 1
USE_SPATIAL_AUG = True           # Enable spatial augmentation (flip, crop)

# Ablation flags
# =========================================================================
TRAIN_DET       = True
TRAIN_HEAD_POSE = False  # No keypoint-based pose loss in IndustReal (dataset has 9-DoF gaze, not COCO keypoints)
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
VIDEOMAE_UNFREEZE_EPOCH = -1
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
# 75 action classes (IDs 0-74, all populated in AR_labels.csv)
# Class index 0 = 'NA' (prepended), real IDs shifted by +1 (ID 0 -> index 1)
_NUM_ACT_CLASSES_FALLBACK = 75

def _load_act_class_names() -> list:
    """
    Load AR action class names from the dataset.
    AR_labels.csv uses action IDs 0-74 (excluding 37, 64) mapped to action names.
    We build a full 75-slot list (indices 0-74) so the ID directly indexes the list.
    Unknown IDs (37, 64) are filled with placeholder names.
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
                            name = parts[2]
                            if aid not in id_to_name:
                                id_to_name[aid] = name
            except OSError:
                continue

    # Build full 75-entry list (IDs 0-74), then prune unknowns to get 74 entries.
    full_names = []
    for i in range(75):
        if i in id_to_name:
            full_names.append(id_to_name[i])
        else:
            full_names.append(f'unknown_{i}')

    # Prune unknown entries so len == 74
    pruned = [n for n in full_names if not n.startswith('unknown_')]
    return pruned


ACT_CLASS_NAMES = _load_act_class_names()
# The AR_labels.csv does not have an explicit NA/background row.
# The 73 unique action IDs (0-74 excluding 37, 64) cover all real actions.
# For classification, prepend 'NA' as class 0 so total = 75 classes (74 AR + NA).
# AR_labels.csv uses raw IDs 0-73; dataset maps these directly as class indices 0-73.
# NA placeholder fills index 0 (no raw ID 0 in labels); no ID shift needed.
if ACT_CLASS_NAMES and ACT_CLASS_NAMES[0] != 'NA':
    ACT_CLASS_NAMES.insert(0, 'NA')
NUM_CLASSES_ACT = len(ACT_CLASS_NAMES)  # 75

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
NUM_PSR_COMPONENTS = 11  # number of assembly components (comp0-comp10 in PSR_labels_raw.csv)

# =========================================================================
# Image and model
# =========================================================================
# Anchor sizes for RetinaNet detection head (Doc 01 B.3)
# These are calibrated via k-means on GT boxes: python calibrate_anchors.py
# Calibrated on 923 GT boxes (2 train recordings on disk):
#   w=146-594px, h=89-463px, w/h ratio=0.82-2.95 (wider than tall)
#   sqrt(area)=188-436px, k-means centers: 164, 269, 333, 338, 404
# Coverage with (128,192,256,384,512): 100% IoU>=0.5, 82.6% IoU>=0.75
# =========================================================================
ANCHOR_SIZES = (128, 192, 256, 384, 512)
IMG_WIDTH       = 1280
IMG_HEIGHT      = 720
IMG_SIZE        = (IMG_WIDTH, IMG_HEIGHT)
ORIGINAL_WIDTH  = 1280
ORIGINAL_HEIGHT = 720

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# =========================================================================
# Training (RTX 3060 12 GB)
# =========================================================================
BATCH_SIZE       = 8     # [FIX] Was 4 — RTX 3060 at BS=8 uses ~0.87GB in Stage 3 forward+backward
GRAD_ACCUM_STEPS = 4     # [FIX] Effective batch 32 (8×4). 4-step accum keeps per-step memory low while maximizing GPU utilization
EFFECTIVE_BATCH  = BATCH_SIZE * GRAD_ACCUM_STEPS  # 32

VAL_BATCH_SIZE  = 4     # [FIX] Increased from 2 for faster eval (more samples/batch)
VAL_NUM_WORKERS = 4     # [FIX] Increased from 0 for parallel data loading
VAL_PREFETCH_FACTOR = 4 # [FIX] Increased from 1 for pipeline prefetch

EPOCHS        = 50 
BASE_LR       = 5e-4   # Per paper: "5e-4"
WEIGHT_DECAY  = 1e-4
WARMUP_EPOCHS = 5
USE_COSINE_ANNEALING = True
T_0 = 10
T_mult = 2
PATIENCE      = 10
GRAD_CLIP_NORM = 1.0
VAL_EVERY = 1    # [BENCHMARK] Evaluate every 1 epoch (BENCHMARK_MODE override)
EVAL_MAX_BATCHES = 15   # [FIX] Reduced from 30 to prevent validation OOM on subset runs

NUM_WORKERS     = 4     # [OPT] 4 workers optimal for parallel data loading
PIN_MEMORY      = True
MIXED_PRECISION = True   # [FIX] Was False — FP16 for RTX 3060 Ampere tensor cores
SEED            = 42

# EMA (Exponential Moving Average) — [FIX #4 HIGH] Enabled per paper §Training: EMA=0.999 in Stage 3
USE_EMA        = True
EMA_DECAY      = 0.999  # standard decay for image models

# Mixup augmentation for activity head
USE_MIXUP      = True
MIXUP_ALPHA    = 0.4

# Desktop stability knobs
CUDA_MEMORY_FRACTION = 0.88
TRAIN_NICE = 10
TORCH_NUM_THREADS = 4   # [CONVOY FIX] was 8 — 4 threads eliminates lock convoy on this HW

# Validation memory-safety knobs
VAL_PREFETCH_FACTOR = 4   # [FIX] Match NUM_WORKERS=4 for pipeline prefetch
TRAIN_PREFETCH_FACTOR = 4
DET_EVAL_SCORE_THRESH = 0.5
DET_EVAL_MAX_PER_IMAGE = 300
DET_EVAL_NMS_IOU_THRESH = 0.5  # NMS IoU threshold for detection evaluation
SAVE_VAL_CONFUSION_MATRIX = False
COMPUTE_VAL_TOP5 = True

# Auto-fallback for Linux systems with small /dev/shm
DATALOADER_AUTO_FALLBACK = True

# Performance flags
USE_UINT8_DATA_PIPELINE = True
CUDNN_DETERMINISTIC = False
CUDNN_BENCHMARK = True

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
# Staged training (Doc 02 B.1)
# =========================================================================
STAGED_TRAINING = True
STAGE1_EPOCHS = 5    # Detection-only warmup
STAGE2_EPOCHS = 10   # Add pose + head pose
STAGE3_EPOCHS = 85   # Full multi-task with EMA
ACT_RAMP_EPOCHS = 5  # Activity loss ramp-up
ACTIVITY_LOSS_CAP = 40.0  # Cap activity loss to prevent NaN cascade at Stage 3 entry (epoch 16) — loss spiked to 40.8 in prior runs
STAGE3_WARMUP_EPOCHS = 3  # LR warmup epochs at Stage 3 entry to stabilize new head activation

# =========================================================================
# LDAM-DRW for activity (Doc 01 §B.2 + Doc 02 C.2)
# =========================================================================
USE_LDAM_DRW = True   # Use LDAM+DRW instead of CB-Focal for activity
LDAM_MAX_M = 0.5
LDAM_S = 30
LDAM_DRW_EPOCH = 60   # Switch to CB weights at this epoch (DRW deferred re-weighting)
# Doc 01 §B.2: DRW activates after epoch 60 when features are stable.
# Recipe default of 60 confirmed correct for IndustReal long-tail classes.

# =========================================================================
# PSR focal loss (Doc 01 §D + Doc 02 C.3)
# =========================================================================
PSR_FOCAL_ALPHA = 0.25
PSR_FOCAL_GAMMA = 2.0

# PSR sequence-mode training (Doc 01 §D.1) — THE biggest PSR unlock
# =========================================================================
# Doc 01 §D.1: Current training samples one frame per step, making the
# temporal Transformer dormant. Sequence-mode trains on contiguous T-frame
# windows where the Transformer is properly engaged.
USE_PSR_SEQUENCE_MODE = True   # Doc 01 §D.2: PSR sequence-mode — THE biggest PSR unlock
PSR_SEQUENCE_LENGTH = 4        # T=4 keeps memory bounded on 12GB GPU
PSR_SEQ_EVERY_N_BATCHES = 10  # Draw one sequence batch every N normal batches

# =========================================================================
# Augmentation (Doc 02 D)
# =========================================================================
USE_RANDAUGMENT = True   # Photometric augmentation for backbone
MIXUP_ALPHA = 0.4
CUTMIX_ALPHA = 1.0       # Alternate Mixup/CutMix each epoch
RANDOM_TEMPORAL_STRIDE = True  # Random frame stride {2,3,4,5} per clip

# =========================================================================
# Optimizer (Doc 02 E)
# =========================================================================
USE_LION = True         # Use Lion optimizer instead of AdamW (Doc 02 E.1)
ONE_CYCLE_LR = False     # Use OneCycleLR instead of CosineAnnealingWarmRestarts
USE_SWA = False          # Stochastic Weight Averaging at end of training
SWA_LR = 1e-5
SWA_EPOCHS = 10

# =========================================================================
# TTA (Doc 02 F)
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

# Kendall gradient sentinel logging (Doc 02 §B.1)
# Log gradient norms of Kendall log_var params every N steps to detect
# silent Kendall failures (Bug #9 reincarnation)
LOG_KENDALL_GRAD_EVERY = 100  # Set to 0 to disable

# Stage transition logging (Doc 02 §B.2)
LOG_STAGE_TRANSITION = True  # Log trainable param counts at each stage start

# Per-component PSR prevalence sanity check (Doc 02 §B.5)
LOG_PSR_PREVALENCE_EVERY = 10  # epochs; set to 0 to disable

# =========================================================================
# Outputs — dynamic, recomputed after preset application
# =========================================================================
CHECKPOINT_DIR = OUTPUT_ROOT / 'checkpoints'
LOG_DIR        = OUTPUT_ROOT / 'logs'
EVAL_SAVE_DIR  = OUTPUT_ROOT / 'eval_outputs'

COCO_CACHE_SIZE = 30
NUM_ACT_CLASSES = NUM_CLASSES_ACT

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
            'Comparable against: MViTv2 Kinetics, YOLOv8m COCO+synth+real, B3 rule-based PSR, STORM-PSR.'
        ),
        'dataset_mode':       'manual_only',
        'backbone':           'convnext_tiny',
        'use_tma_cell':      True,
        'use_temporal_bank':  True,
        'use_hand_film':     True,
        'benchmark_mode':     True,
        'batch_size':        1,   # Hand-FiLM + TMACell + TemporalBank exceeds 12GB at batch=2
        'grad_accum_steps':  32,  # Keep effective batch ~32
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
