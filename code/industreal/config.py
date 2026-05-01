import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

"""
IndustReal Configuration — POPW Adaptation
============================================
Single unified configuration for IndustReal dataset:
  - Action Recognition (AR): 74 atomic action classes
  - Assembly State Detection (ASD): 24 assembly states
  - Procedure Step Recognition (PSR): 36 procedure steps, 11 components
  - Single egocentric RGB camera

Dataset location: /home/newadmin/swarm-bot/project/popw/working/data/dataset/industreal/
Code location:    /home/newadmin/swarm-bot/project/popw/working/code/industreal/
"""

import logging
from pathlib import Path

_cfg_logger = logging.getLogger(__name__)

# =========================================================================
# Debug / profiling flags
# =========================================================================
DEBUG_MODE         = False
DEBUG_MAX_VIDEOS   = 20
DEBUG_FRAME_STRIDE = 10

TRAIN_FRAME_STRIDE = 5
EVAL_FRAME_STRIDE  = 1

VAL_EVERY = 3
EVAL_MAX_BATCHES = 4000

# =========================================================================
# Ablation flags
# =========================================================================
TRAIN_DET       = True
TRAIN_HEAD_POSE = True
TRAIN_ACT       = True
TRAIN_PSR       = True
USE_KENDALL     = True

# =========================================================================
# Paths
# =========================================================================
POPW_ROOT = Path('/home/newadmin/swarm-bot/project/popw/working/data/dataset/industreal')

# Output root for runs
OUTPUT_ROOT = Path('/home/newadmin/swarm-bot/project/popw/working/code/industreal/runs')

# Recordings root (train/val/test splits)
RECORDINGS_ROOT = POPW_ROOT / 'recordings'

# Official train/val/test CSVs for AR task splits
TRAIN_CSV = POPW_ROOT / 'train.csv'
VAL_CSV   = POPW_ROOT / 'val.csv'
TEST_CSV  = POPW_ROOT / 'test.csv'

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
# 74 action classes (IDs 0-74 non-consecutive; IDs 37 and 64 are unused)
# Class index 0 = ID 0 = 'take_short_brace'
_NUM_ACT_CLASSES_FALLBACK = 74

def _load_act_class_names() -> list:
    """
    Load AR action class names from the dataset.
    AR_labels.csv uses action IDs 0-74 (excluding 37, 64) mapped to action names.
    We build a full 75-slot list (indices 0-74) so the ID directly indexes the list.
    Unknown IDs (37, 64) are filled with placeholder names.
    """
    import csv
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
# For classification, prepend 'NA' as class 0 so total = 74 classes.
# When loading AR_labels, add 1 to action_id to shift: data_id -> class_index
# (data_id 0 -> index 1, data_id 1 -> index 2, etc.)
if ACT_CLASS_NAMES and ACT_CLASS_NAMES[0] != 'NA':
    ACT_CLASS_NAMES.insert(0, 'NA')
NUM_CLASSES_ACT = len(ACT_CLASS_NAMES)  # 74

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

# --- Procedure Step Recognition (PSR) ---
NUM_PSR_STEPS = 36       # number of distinct procedure step types (from procedure_info.json)
NUM_PSR_COMPONENTS = 11  # number of assembly components (comp0-comp10 in PSR_labels_raw.csv)

# =========================================================================
# Image and model
# =========================================================================
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
BATCH_SIZE       = 4     # 1280x720 images — reduced from IKEA's 8x640x480
GRAD_ACCUM_STEPS = 8
EFFECTIVE_BATCH  = BATCH_SIZE * GRAD_ACCUM_STEPS  # 32

VAL_BATCH_SIZE  = 4
VAL_NUM_WORKERS = 1

EPOCHS        = 100
BASE_LR       = 1e-4
WEIGHT_DECAY  = 1e-4
WARMUP_EPOCHS = 5
PATIENCE      = 10
GRAD_CLIP_NORM = 1.0

NUM_WORKERS     = 4
PIN_MEMORY      = True
MIXED_PRECISION = True
SEED            = 42

# Desktop stability knobs
CUDA_MEMORY_FRACTION = 0.88
TRAIN_NICE = 10
TORCH_NUM_THREADS = 8

# Validation memory-safety knobs
VAL_PREFETCH_FACTOR = 1
DET_EVAL_SCORE_THRESH = 0.5
DET_EVAL_MAX_PER_IMAGE = 300
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

WING_OMEGA   = 0.05
WING_EPSILON = 0.005

CB_BETA  = 0.9999
CB_GAMMA = 2.0

POSE_CONF_THRESHOLD = 0.1

# =========================================================================
# Outputs
# =========================================================================
CHECKPOINT_DIR = OUTPUT_ROOT / 'checkpoints'
LOG_DIR        = OUTPUT_ROOT / 'logs'
EVAL_SAVE_DIR  = OUTPUT_ROOT / 'eval_outputs'

COCO_CACHE_SIZE = 30
NUM_ACT_CLASSES = NUM_CLASSES_ACT

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
        _cfg_logger.info(f'[config] All critical paths validated.')
        sample_rec = list((RECORDINGS_ROOT / 'train').iterdir())[0].name
        _cfg_logger.info(f'[config] Sample recording: {sample_rec}')
        _cfg_logger.info(f'[config] NUM_CLASSES_ACT = {NUM_CLASSES_ACT}')
        _cfg_logger.info(f'[config] NUM_DET_CLASSES = {NUM_DET_CLASSES}')
        _cfg_logger.info(f'[config] CAMERAS = {CAMERAS}')


# =========================================================================
# INITIALIZATION
# =========================================================================
_validate_paths()

_cfg_logger.info(f'[config] Initialized IndustReal config')
_cfg_logger.info(f'[config] POPW_ROOT: {POPW_ROOT}')
_cfg_logger.info(f'[config] OUTPUT_ROOT: {OUTPUT_ROOT}')
