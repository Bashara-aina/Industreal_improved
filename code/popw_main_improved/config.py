import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

"""
POPW Main Improved — Benchmark Configuration
============================================

Configures POPW to produce results directly comparable against all
IKEA ASM benchmark papers.

Architecture matches IndustReal exactly:
  - ConvNeXt-Tiny backbone (ImageNet pretrained)
  - FPN neck (P3-P7, 256ch)
  - Detection: RetinaNet-style (7 furniture part classes)
  - Pose: Heatmap + soft-argmax (17 COCO keypoints)
  - PoseFiLM: keypoint-conditioned FiLM modulation
  - Activity: Feature Bank + ViT Temporal Block (33 classes)

Usage:
  # Benchmark comparison run (all metrics, full evaluation)
  python train.py --preset benchmark_full

  # Quick baseline (no temporal, faster)
  python train.py --preset benchmark_quick

  # Resume from checkpoint
  python train.py --resume runs/.../checkpoints/best.pth
"""

import logging
from pathlib import Path

_cfg_logger = logging.getLogger(__name__)

# =========================================================================
# BENCHMARK MODE
# =========================================================================
BENCHMARK_MODE = True

# =========================================================================
# Runtime parameters (set via CLI flags or presets)
# =========================================================================
DATASET_MODE = 'manual_only'
USE_FILM = False
USE_GCN_SKELETON = True
DETECTION_MODE = 'dev3_only'

# =========================================================================
# Backbone configuration
# =========================================================================
BACKBONE = 'convnext_tiny'
USE_OKS_LOSS = False  # Use Wing Loss instead of OKS Loss

# =========================================================================
# Debug / profiling flags
# =========================================================================
DEBUG_MODE         = False
DEBUG_MAX_VIDEOS   = 20
DEBUG_FRAME_STRIDE  = 10
TRAIN_FRAME_STRIDE_RANGE = (2, 8)
EVAL_FRAME_STRIDE         = 1
USE_SPATIAL_AUG           = True

# Ablation flags
TRAIN_DET   = True
TRAIN_POSE  = True
TRAIN_ACT   = True
USE_KENDALL = True

# =========================================================================
# Temporal modeling flags
# =========================================================================
USE_TEMPORAL = True
USE_VIT_TEMPORAL = True
TEMPORAL_SEQUENCE_LEN = 16
TEMPORAL_BANK_SHORT_T = 8
TEMPORAL_BANK_LONG_T = 32
TEMPORAL_BANK_UPDATE_INTERVAL = 8

# =========================================================================
# Multi-view fusion flags
# =========================================================================
USE_MULTIVIEW_ACTIVITY = True
USE_MULTIVIEW_DETECTION = True

# =========================================================================
# Monitoring / Visualization (TrainingMonitor integration)
# =========================================================================
MONITOR_ENABLED = True
SAVE_VIZ_EPOCHS = 5
NUM_VIZ_SAMPLES = 8
MONITOR_LOG_INTERVAL = 10

# =========================================================================
# Paths
# =========================================================================
POPW_ROOT = Path('/media/newadmin/master/POPW')
IKEA_RAW_ROOT = POPW_ROOT / 'IKEA_RAW'
GITHUB_ROOT = POPW_ROOT / 'github' / 'IKEA_ASM_Dataset-master'

DATASET_ROOT = POPW_ROOT / 'datasets' / DATASET_MODE

OUTPUT_ROOT = (
    POPW_ROOT / 'popw_main_improved' / 'runs'
    / f'{DATASET_MODE}_{"film" if USE_FILM else "no_film"}'
)

IMAGES_ROOT = DATASET_ROOT / 'images'
ANNOTATIONS_ROOT = DATASET_ROOT / 'annotations'

COCO_RAW_ROOT = (
    IKEA_RAW_ROOT / 'annotations'
    / 'Final_Annotations_Segmentation_Tracking'
)
ANNO_RAW_ROOT = IKEA_RAW_ROOT / 'annotations'

CAMERA = 'dev3'
CAMERAS = ['dev1', 'dev2', 'dev3']
NUM_CAMERAS = len(CAMERAS)

SPLIT_FILES_ROOT = Path(
    '/media/newadmin/master/POPW/github/'
    'IKEA_ASM_Dataset-master/toolbox/dataset_indexing_files'
)
TRAIN_SPLIT_FILE = SPLIT_FILES_ROOT / 'train_cross_env.txt'
TEST_SPLIT_FILE  = SPLIT_FILES_ROOT / 'test_cross_env.txt'

VAL_RATIO = 0.15

ACTION_LOOKUP_FILE = POPW_ROOT / 'ikea_workernet_FULL' / 'action_lookup.json'

_ACT_CLASS_FILE = Path(
    '/media/newadmin/master/POPW/IKEA_RAW/misc/ANU_ikea_dataset'
    '/indexing_files/atomic_action_list.txt'
)

# =========================================================================
# Dataset constants
# =========================================================================
FURNITURE_TYPES = [
    'Kallax_Shelf_Drawer',
    'Lack_Coffee_Table',
    'Lack_Side_Table',
    'Lack_TV_Bench',
]

NUM_DET_CLASSES = 7
DET_CLASS_NAMES = {
    1: 'table_top', 2: 'leg',         3: 'shelf',
    4: 'side_panel', 5: 'front_panel', 6: 'bottom_panel', 7: 'rear_panel',
}

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

_NUM_ACT_CLASSES_FALLBACK = 33

def _load_act_class_names() -> list:
    path = _ACT_CLASS_FILE
    if not path.exists():
        _cfg_logger.warning(
            f'[config] atomic_action_list.txt not found at:\n'
            f'  {path}\n'
            f'  Falling back to {_NUM_ACT_CLASSES_FALLBACK} generic class names.'
        )
        fallback = ['NA'] + [f'action_{i:02d}' for i in range(1, _NUM_ACT_CLASSES_FALLBACK)]
        return fallback
    try:
        with open(path, 'r', encoding='utf-8') as f:
            names = [ln.strip() for ln in f if ln.strip()]
    except OSError as exc:
        raise FileNotFoundError(
            f'[config] Cannot read atomic_action_list.txt at {path}: {exc}'
        ) from exc
    names.sort()
    names.insert(0, 'NA')
    return names


ACT_CLASS_NAMES = _load_act_class_names()
NUM_CLASSES_ACT = len(ACT_CLASS_NAMES)

# =========================================================================
# Image and model dimensions
# =========================================================================
IMG_WIDTH       = 640
IMG_HEIGHT      = 480
IMG_SIZE        = (IMG_WIDTH, IMG_HEIGHT)
ORIGINAL_WIDTH  = 1920
ORIGINAL_HEIGHT = 1080

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Anchor sizes for RetinaNet detection head
ANCHOR_SIZES = (32, 64, 128, 256, 512)

# =========================================================================
# Training hyperparameters
# =========================================================================
BATCH_SIZE       = 2
GRAD_ACCUM_STEPS = 8
EFFECTIVE_BATCH  = BATCH_SIZE * GRAD_ACCUM_STEPS   # 16

VAL_BATCH_SIZE  = 4
VAL_NUM_WORKERS = 1

EPOCHS        = 150
BASE_LR       = 1e-4
WEIGHT_DECAY  = 1e-4
WARMUP_EPOCHS = 5
USE_COSINE_ANNEALING = True
T_0 = 10
T_mult = 2
PATIENCE      = 10
GRAD_CLIP_NORM = 1.0

NUM_WORKERS     = 1
PIN_MEMORY      = True
MIXED_PRECISION = False
SEED            = 42

# Desktop stability knobs
CUDA_MEMORY_FRACTION = 0.88
TRAIN_NICE = 10
TORCH_NUM_THREADS = 8

# Validation frequency
VAL_EVERY = 1 if BENCHMARK_MODE else 3
EVAL_MAX_BATCHES = 4000

# Validation memory-safety knobs
VAL_PREFETCH_FACTOR = 1
DET_EVAL_SCORE_THRESH = 0.5
DET_EVAL_MAX_PER_IMAGE = 300
SAVE_VAL_CONFUSION_MATRIX = True
COMPUTE_VAL_TOP5 = True

# Performance flags
DATALOADER_AUTO_FALLBACK = True
USE_UINT8_DATA_PIPELINE = True
USE_TTA = False
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

LABEL_SMOOTHING = 0.1

POSE_CONF_THRESHOLD = 0.1

# Activity warmup epochs
ACT_WARMUP_EPOCHS = 5

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

    # ── Benchmark comparison presets ──────────────────────────────────────

    'benchmark_full': {
        'description': (
            'Full benchmark comparison run. ConvNeXt-Tiny, dev3 only, manual_only dataset, '
            'PoseFiLM + ViT Temporal Block activity head. '
            'VAL_EVERY=1, all metrics every epoch, visualizations saved every 5 epochs. '
            'Comparable against: P3D, I3D RGB+pose, ResNeXt-101-FPN, MaskRCNN-ft, STEPs, PTMA.'
        ),
        'dataset_mode':         'manual_only',
        'use_film':            True,
        'detection_mode':       'dev3_only',
        'backbone':             'convnext_tiny',
        'use_oks_loss':        False,
        'use_temporal':        True,
        'use_vit_temporal':    True,
        'use_multiview_activity': False,
        'benchmark_mode':      True,
        'batch_size':         2,
    },

    'benchmark_quick': {
        'description': (
            'Quick baseline — single-frame only (USE_TEMPORAL=False). '
            'Faster iteration, tests detection + pose + activity without temporal overhead. '
            'Beat: P3D (60.4%), I3D RGB+pose (64.15%), ResNeXt-101-FPN (85.3% AP@0.5), MaskRCNN-ft PCK.'
        ),
        'dataset_mode':         'manual_only',
        'use_film':            True,
        'detection_mode':       'dev3_only',
        'backbone':             'convnext_tiny',
        'use_oks_loss':        False,
        'use_temporal':        False,
        'use_vit_temporal':    False,
        'use_multiview_activity': False,
        'benchmark_mode':      True,
        'batch_size':         4,
    },

    'benchmark_vit_temporal': {
        'description': (
            'Full diagram-accurate architecture: ViT Temporal Block (MHSA 4-head + FFN 512→2048→512). '
            'USE_TEMPORAL=True + USE_VIT_TEMPORAL=True. '
            'Higher params but matches architecture diagram exactly. '
            'Use when ViT path outperforms GRU path.'
        ),
        'dataset_mode':         'manual_only',
        'use_film':            True,
        'detection_mode':       'dev3_only',
        'backbone':             'convnext_tiny',
        'use_oks_loss':        False,
        'use_temporal':        True,
        'use_vit_temporal':    True,
        'use_multiview_activity': False,
        'benchmark_mode':      True,
        'batch_size':         2,
    },

    'benchmark_multiview': {
        'description': (
            'Multi-view fusion: dev1+dev2+dev3 for activity recognition. '
            'Target: Beat Aganian et al. Top-1 80.2% (all views, most relevant). '
            'USE_MULTIVIEW_ACTIVITY=True. Still uses dev3 only for detection/pose (benchmark requirement).'
        ),
        'dataset_mode':         'manual_only',
        'use_film':            True,
        'detection_mode':       'dev3_only',
        'backbone':             'convnext_tiny',
        'use_oks_loss':        False,
        'use_temporal':        True,
        'use_vit_temporal':    False,
        'use_multiview_activity': True,
        'benchmark_mode':      True,
        'batch_size':         2,
    },

    # ── Legacy / development presets ──────────────────────────────────────

    'improved3': {
        'dataset_mode': 'manual_only',
        'use_film': True,
        'detection_mode': 'dev3_only',
        'backbone': 'convnext_tiny',
        'use_oks_loss': False,
        'description': '1% annotated frames only, ConvNeXt-Tiny, Wing Loss, PoseFiLM; dev3 only'
    },
    'improved4': {
        'dataset_mode': 'manual_pseudo',
        'use_film': True,
        'detection_mode': 'dev3_only',
        'backbone': 'convnext_tiny',
        'use_oks_loss': False,
        'description': '1% annotated + 99% pseudo-GT, ConvNeXt-Tiny, Wing Loss, PoseFiLM; dev3 only'
    },
    'improved3_temporal': {
        'dataset_mode': 'manual_only',
        'use_film': True,
        'detection_mode': 'dev3_only',
        'backbone': 'convnext_tiny',
        'use_oks_loss': False,
        'use_temporal': True,
        'description': 'improved3 + ViT Temporal Block activity head'
    },
    'improved4_temporal': {
        'dataset_mode': 'manual_pseudo',
        'use_film': True,
        'detection_mode': 'dev3_only',
        'backbone': 'convnext_tiny',
        'use_oks_loss': False,
        'use_temporal': True,
        'description': 'improved4 + ViT Temporal Block activity head'
    },
    'improved3_multiview': {
        'dataset_mode': 'manual_only',
        'use_film': True,
        'detection_mode': 'all_cameras',
        'backbone': 'convnext_tiny',
        'use_oks_loss': False,
        'use_multiview_activity': True,
        'description': 'improved3 with dev1+dev2+dev3 multi-camera training + multi-view activity fusion'
    },
}


# =========================================================================
# Config validation & auto-fallback
# =========================================================================
def _validate_and_fallback():
    global DETECTION_MODE
    if DETECTION_MODE != 'all_cameras':
        return
    sample_furniture = 'Kallax_Shelf_Drawer'
    sample_video = None
    for video_dir in sorted((COCO_RAW_ROOT / 'train' / sample_furniture).iterdir()):
        if video_dir.is_dir():
            sample_video = video_dir.name
            break
    if not sample_video:
        return
    coco_file_dev1 = COCO_RAW_ROOT / 'train' / sample_furniture / sample_video / 'dev1' / 'manual_coco_format.json'
    coco_file_dev2 = COCO_RAW_ROOT / 'train' / sample_furniture / sample_video / 'dev2' / 'manual_coco_format.json'
    if not (coco_file_dev1.exists() and coco_file_dev2.exists()):
        _cfg_logger.warning(
            f'[config] dev1/dev2 COCO annotations not found.\n'
            f'  Falling back to DETECTION_MODE="dev3_only".\n'
            f'  To enable all_cameras, run: python scripts/generate_dev12_annotations.py'
        )
        DETECTION_MODE = 'dev3_only'


def apply_preset(preset_name: str) -> None:
    """Apply a preset configuration by name. Updates all global flags + dynamic paths."""
    global DATASET_MODE, USE_FILM, DETECTION_MODE, BACKBONE, USE_OKS_LOSS
    global USE_TEMPORAL, USE_VIT_TEMPORAL
    global USE_MULTIVIEW_ACTIVITY, USE_MULTIVIEW_DETECTION
    global BENCHMARK_MODE
    global VAL_EVERY, BATCH_SIZE

    if preset_name not in PRESETS:
        raise ValueError(f'Unknown preset: {preset_name}. Available: {list(PRESETS.keys())}')

    preset = PRESETS[preset_name]
    DATASET_MODE = preset['dataset_mode']
    USE_FILM = preset.get('use_film', True)
    DETECTION_MODE = preset.get('detection_mode', 'dev3_only')
    BACKBONE = preset.get('backbone', BACKBONE)
    USE_OKS_LOSS = preset.get('use_oks_loss', False)
    USE_TEMPORAL = preset.get('use_temporal', False)
    USE_VIT_TEMPORAL = preset.get('use_vit_temporal', False)
    USE_MULTIVIEW_ACTIVITY = preset.get('use_multiview_activity', False)
    USE_MULTIVIEW_DETECTION = preset.get('use_multiview_detection', False)
    BENCHMARK_MODE = preset.get('benchmark_mode', False)

    if 'description' in preset:
        _cfg_logger.info(f'[config] Applied preset "{preset_name}": {preset["description"]}')
    else:
        _cfg_logger.info(f'[config] Applied preset "{preset_name}"')

    BATCH_SIZE = preset.get('batch_size', BATCH_SIZE)

    if BENCHMARK_MODE:
        VAL_EVERY = 1
    else:
        VAL_EVERY = 3

    update_dynamic_paths()


def update_dynamic_paths():
    """Recompute all dynamic paths after config changes."""
    global DATASET_ROOT, OUTPUT_ROOT, IMAGES_ROOT, ANNOTATIONS_ROOT
    global CHECKPOINT_DIR, LOG_DIR, EVAL_SAVE_DIR

    DATASET_ROOT = POPW_ROOT / 'datasets' / DATASET_MODE

    parts = [DATASET_MODE]
    if USE_FILM:
        parts.append('film')
    else:
        parts.append('no_film')
    if USE_TEMPORAL:
        parts.append('temporal')
    if USE_VIT_TEMPORAL:
        parts.append('vit')
    if USE_MULTIVIEW_ACTIVITY:
        parts.append('multiview')

    OUTPUT_ROOT = POPW_ROOT / 'popw_main_improved' / 'runs' / '_'.join(parts)

    IMAGES_ROOT = DATASET_ROOT / 'images'
    ANNOTATIONS_ROOT = DATASET_ROOT / 'annotations'
    CHECKPOINT_DIR = OUTPUT_ROOT / 'checkpoints'
    LOG_DIR = OUTPUT_ROOT / 'logs'
    EVAL_SAVE_DIR = OUTPUT_ROOT / 'eval_outputs'


# =========================================================================
# Initialization
# =========================================================================
_validate_and_fallback()
update_dynamic_paths()

_cfg_logger.info(
    f'[config] DATASET_MODE={DATASET_MODE}, USE_FILM={USE_FILM}, '
    f'DETECTION_MODE={DETECTION_MODE}, BACKBONE={BACKBONE}'
)
_cfg_logger.info(f'[config] USE_TEMPORAL={USE_TEMPORAL}, USE_VIT_TEMPORAL={USE_VIT_TEMPORAL}, '
    f'USE_MULTIVIEW_ACTIVITY={USE_MULTIVIEW_ACTIVITY}')
_cfg_logger.info(f'[config] Dataset root: {DATASET_ROOT}')
_cfg_logger.info(f'[config] Output root: {OUTPUT_ROOT}')
_cfg_logger.info(f'[config] Train split: {TRAIN_SPLIT_FILE.name}')
_cfg_logger.info(f'[config] VAL_EVERY={VAL_EVERY}, BENCHMARK_MODE={BENCHMARK_MODE}')