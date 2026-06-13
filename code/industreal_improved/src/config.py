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

SUBSET_RATIO = 1.0    # Full dataset — production training for paper results

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
TRAIN_MAX_STEPS = int(os.environ.get('TRAIN_MAX_STEPS', 0))  # 0=disabled; set >0 to stop after N batches

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
USE_VIDEOMAE = False  # [FIX 2026-06-12] Disabled for FP32 12GB fitting — ConvNeXt-only fallback for activity
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
# Allow override via environment variable for experiment naming
if 'OUTPUT_ROOT_OVERRIDE' in os.environ:
    OUTPUT_ROOT = Path(os.environ['OUTPUT_ROOT_OVERRIDE'])

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
NUM_ACT_RAW_IDS = 74          # max raw action_id in dataset is 74 (IDs 0..74, ID 37 absent)
NUM_CLASSES_ACT = NUM_ACT_RAW_IDS + 1   # 75 = IDs 0..74; FIXED, not data-derived


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
DET_POS_IOU_THRESH = 0.5       # RetinaNet anchor matching: positive IoU threshold (standard: 0.5)
DET_NEG_IOU_THRESH = 0.4       # RetinaNet anchor matching: negative IoU threshold (standard: 0.4)
# RC-25 recovery: zero det_conf input to activity head during recovery.
# [AUDIT FIX 2026-06-11] env override added: a checkpoint TRAINED with
# det_conf zeroed must also be EVALUATED with det_conf zeroed (otherwise the
# activity head sees an input distribution it never trained on — the same
# train/eval mismatch class as RC-17). Eval scripts don't apply presets, so:
#     ZERO_DET_CONF=1 python eval_... .py
ZERO_DET_CONF_FOR_RECOVERY = bool(int(os.environ.get('ZERO_DET_CONF', '0')))
IMG_WIDTH       = 1280
IMG_HEIGHT      = 720
IMG_SIZE        = (IMG_WIDTH, IMG_HEIGHT)
ORIGINAL_WIDTH  = 1280
ORIGINAL_HEIGHT = 720

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# =========================================================================
# Training (RTX 3060 12 GB) — Optimized for VideoMAE + ConvNeXt + 5 heads + TMA + TemporalBank
# NOTE: BATCH_SIZE=1 is REQUIRED for RTX 3060 12GB with VideoMAE+ConvNeXt+TMA+TemporalBank+5 heads.
# Previous BATCH_SIZE=6 OOM'd at ConvNeXt stage2 with batch=2 (only 64 MiB free).
# BATCH_SIZE=1 with GRAD_ACCUM_STEPS=32 gives effective batch=32.
# OOM fallback: train.py automatically halves batch if CUDA OOM is detected.
BATCH_SIZE = 1        # FP32 (AMP broken) + ConvNeXt + VideoMAE + 5 heads = 12GB at batch=1
GRAD_ACCUM_STEPS = 32 # Effective batch = 32 (paper target)
EFFECTIVE_BATCH      = BATCH_SIZE * GRAD_ACCUM_STEPS  # 32

VAL_BATCH_SIZE = 16   # Was 8→32; bumped again (VRAM headroom: 1.30GB/12.5GB used at batch_size=8)
VAL_NUM_WORKERS = 1   # Reduced from 4 to prevent CPU RAM OOM
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
EVAL_MAX_BATCHES = -1  # Full validation set every epoch (no cap)

NUM_WORKERS = 4        # Was 2 — increased for full-data training; 64GB RAM available
PIN_MEMORY = True
MIXED_PRECISION = False  # [AUDIT FIX 2026-06-11] fp16 AMP is a CONFIRMED failure mode on this
                         # model (R2: first NaN at backbone.0.conv1.weight; diag_amp_nan.py /
                         # diag_amp_2step.py — AMP fails at the first optimizer step, FP32
                         # succeeds). The earlier flip back to True regressed that finding.
                         # Keep False until AMP is re-validated with a dedicated smoke run.
SEED            = 42

# EMA (Exponential Moving Average) — [FIX #4 HIGH] Enabled per paper §Training: EMA=0.999 in Stage 3
USE_EMA        = True
EMA_DECAY      = 0.999  # standard decay for image models

# Mixup augmentation for activity head
# [AUDIT FIX 2026-06-11 / RC-15] DISABLED. The implementations in train.py
# (mixup_activity / cutmix_activity) mix the OUTPUT LOGITS after the forward
# pass instead of the input images (cutmix even builds images_mixed and never
# feeds it to the model), and LDAMLoss argmaxes the soft label — so for
# lam < 0.5 the loss supervises frame i's logits with frame j's label. That is
# pure label corruption (~50% wrong labels at CutMix alpha=1.0) and a direct
# driver of the 1-class activity collapse. Do NOT re-enable until the
# implementation mixes IMAGES BEFORE the forward pass.
USE_MIXUP      = False
MIXUP_ALPHA    = 0.4

# Desktop stability knobs
CUDA_MEMORY_FRACTION = 0.95  # allow near-full VRAM (RTX 3060 12GB); rustdesk uses ~142 MiB
TRAIN_NICE = 10
TORCH_NUM_THREADS = 12

# Validation memory-safety knobs
VAL_PREFETCH_FACTOR = 2   # 2 workers × 2 prefetch (was 4 — overkill for VAL_NUM_WORKERS=0)
TRAIN_PREFETCH_FACTOR = 4  # 2 workers × 2 prefetch = 4 batches queued (was 4)
# FIX: Lower DET_EVAL_SCORE_THRESH from 0.5 to 0.0 to avoid zero predictions when
# model sigmoid scores cluster near 0.5 (e.g., [0.47, 0.53]) — all filtered out at 0.5.
# With 0.0, at least the top-scoring prediction per location is kept for mAP calculation.
# [FIX] Changed from 0.0 to 0.05: with threshold 0.0, ALL anchors (1.3M+) pass the filter,
# top-300 are kept after NMS, and the false-positive flood drives AP to 0.0 by construction.
# mAP needs a low-but-nonzero floor so the PR curve integrates correctly.
# [FIX] Changed from 0.05 to 0.03: bias=-3.4 init produces scores ~0.033; 0.05 filters all
# predictions even when model shows good localization (bestIoU_max=0.923, 554 preds at IoU>0.5).
# 0.03 is high enough to filter the early-training false-positive flood but low enough to
# capture predictions when the model is learning and scores are ~0.033-0.05.
# [FIX 2026-06-04] Bumped to 0.1: with collapsed det head (flat scores ≈ 0.03 across 1.66M
# predictions), 0.03 passes every anchor through NMS, drowning AP. 0.1 filters noise when
# the head is untrained and still permits real detections once scores separate from the floor.
# [FIX 2026-06-04 #2] Lowered to 0.02: at epoch=3, the actual crash_recovery.pth produces
# score_max=0.076 and score_p99=0.022. With 0.1, EVERY prediction was rejected by the
# keep_mask filter, leaving dp_boxes empty for all 64 images, so compute_ap_multi_thresh
# had no positives and returned mAP=0. 0.02 is high enough to filter the random-init noise
# floor (probe shows only 3107 anchors above 0.05 across 1.66M) but lets through the real
# localizations that the probe confirms (151 preds at IoU>0.5 in batch 0 alone).
DET_EVAL_SCORE_THRESH = 0.02
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
FOCAL_ALPHA   = 0.75  # was 0.25 — α=0.25 starves positives of gradient (99.8% bg). α=0.75 gives positives 3× weight vs background.
FOCAL_GAMMA   = 2.0
GIOU_WEIGHT   = 2.0  # Doc 01 B.2: GIoU regression weight vs cls weight=1.0

WING_OMEGA   = 0.05
WING_EPSILON = 0.005

CB_BETA  = 0.99
CB_GAMMA = 1.0  # was 2.0 — gamma=2.0 collapses activity to trivial solution
                # (focuses predictions on 1-4 of 75 classes; dominant class reaches
                # 100% by epoch 6 in v4 log). Mirrors PSR_FOCAL_GAMMA=1.0 fix at L418.
                # gamma=1.0 gives ~10x more gradient on hard classes, allowing the
                # head to escape the focal-γ=2.0 attractor and explore more classes.
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
STAGED_TRAINING = False  # Full production: all 5 heads active from epoch 0
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
HEAD_POSE_POS_SCALE = 100.0  # Standardizes raw position (~110 in CSV) to O(1); also fixes mm/cm unit ambiguity
STAGE3_WARMUP_EPOCHS = 3  # LR warmup epochs at Stage 3 entry to stabilize new head activation
# PSR_WARMUP_EPOCHS disabled: STAGE3_WARMUP_EPOCHS already ramps psr_head via the
# dedicated param group LR at train.py:2511-2526. Combining both ramps multiplied
# gradient suppression (1/5 loss-side × 1/3 LR-side = 1/15 at epoch 16) — too
# aggressive when used together. STAGE3_WARMUP alone is sufficient.
PSR_WARMUP_EPOCHS = 0  # loss-side ramp disabled; STAGE3_WARMUP_EPOCHS handles psr_head
CLEAR_FRAME_CACHE_EPOCH_END = True  # free ~5-7GB FRAME_CACHE between epochs

# =========================================================================
# LDAM-DRW for activity (Doc 01 §B.2 + Doc 2 C.2)
# =========================================================================
# NOTE: USE_LDAM_DRW is set to False below for A/B testing (CB-Focal vs LDAM).
# Set back to True for the full 100-epoch run after confirming activity loss moves.
USE_LDAM_DRW = True   # [OPUS FIX #2 + USER-AUTH] LDAM+DRW (long-tail fix, +~2% Top-1)
LDAM_MAX_M = 0.5
LDAM_S = 30
LDAM_DRW_EPOCH = 0    # Switch to CB weights at this epoch (DRW deferred re-weighting)
# DRW activates immediately (epoch 0) when activity training begins, applying class-balanced
# weights from the start. This corrects the prior misconfiguration where DRW was delayed
# to epoch 60, resulting in 60 epochs of unweighted LDAM margin loss before CB re-weighting.
# Features being "stable" at epoch 60 was not supported by IndustReal experimental evidence.

# [OPUS FIX] LDAM_USE_DRW flag: when True, LDAMLoss.set_class_counts wires cb_weights
# so that DRW applies class-balanced re-weighting at epoch >= LDAM_DRW_EPOCH.
# Gate behind this flag for easy A/B testing vs LDAM margins only (no CB re-weighting).
# NOTE: Set to True for full run after confirming CB-Focal moves the loss.
LDAM_USE_DRW = True

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
PSR_SEQUENCE_LENGTH = 4        # stayed at 4 — T=8 caused CUDA OOM with SEQ_EVERY_N_BATCHES=2
                              # (sequence-mode memory doubled and fires 5x more often). T=4 is
                              # the memory-bounded choice; the bigger unlock is below.
PSR_SEQ_EVERY_N_BATCHES = 10  # Draw one sequence batch every N normal batches

# Fix 1 (2026-06-06): penalize constant per-frame PSR predictions in T=1 mode.
# Stage 3 epoch 16 collapsed logit std to 0.12%; this penalty keeps it > 1e-3.
PSR_SENSITIVITY_WEIGHT = 0.01  # 5% of typical binary-focal magnitude — gentle nudge

# =========================================================================
# Augmentation (Doc 2 D)
# =========================================================================
USE_RANDAUGMENT = True   # Photometric augmentation for backbone
MIXUP_ALPHA = 0.4
CUTMIX_ALPHA = 0.0       # [AUDIT FIX 2026-06-11 / RC-15] was 1.0 — cutmix_activity mixes
                         # LOGITS, not images (label corruption); the train.py gate
                         # `CUTMIX_ALPHA > 0 and epoch % 2 == 1` made it fire on every odd
                         # epoch at stage 3 with NO lam range gating. Keep 0.0 until the
                         # implementation mixes images before the forward pass.
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
SKIP_DET_METRICS_EVAL = False  # True = skip detection mAP (~87 min/epoch) — saves ~90 min per epoch
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
    'recovery': {
        'description': (
            'RC-25/RC-28 recovery preset (joint heads). FP32 ONLY, no staged '
            'training, det_conf live (sigmoid-bounded). Run recovery_det_only '
            'FIRST to bootstrap detection, then this for joint training.'
        ),
        'dataset_mode':       'manual_only',
        'backbone':           'convnext_tiny',
        'use_tma_cell':       True,
        'use_temporal_bank':  True,
        'use_hand_film':      True,
        'benchmark_mode':     False,
        'batch_size':         1,
        'grad_accum_steps':   8,      # [RC-28] effective batch 8: with empty frames now
                                      # skipped in det loss, ~1 GT frame lands per update;
                                      # 32 starved update frequency, 4 (as hand-edited on
                                      # the training box) was too noisy for LDAM/Kendall.
        'zero_det_conf':      False,  # [RC-28 FIX 2026-06-12] was True. The zeroing
                                      # guarded against SATURATED raw det_conf (O(10-100));
                                      # that is fixed at the source (sigmoid bound + healthy
                                      # logits, step-0 median |z|=2.95). Zeroing now only
                                      # starves the activity head of a legitimate signal.
        'staged_training':    False,  # All heads active from epoch 0
        'mixed_precision':    False,  # [RC-29 — DO NOT FLIP THIS TO True] fp16 GradScaler
                                      # SILENTLY skips optimizer.step() on inf/NaN grads;
                                      # the June 11-12 recovery runs (preset hand-edited to
                                      # mixed_precision=True on the training box) produced
                                      # 4 validation cycles IDENTICAL to 4 decimals — the
                                      # signature of zero committed steps. Watch the
                                      # "[RC-29] optimizer windows" epoch summary line.
        'use_mixup':          False,  # RC-15: logit-mixing corrupts activity labels
        'use_ema':            False,  # short recovery runs: best.pth must hold the RAW
                                      # trained weights, not an EMA blend lagging at init
        # Explicit task flags so this preset is self-contained even if another
        # preset (e.g. recovery_det_only) was applied earlier in the process.
        'train_det':          True,
        'train_act':          True,
        'train_psr':          True,
        'train_head_pose':    True,
    },
    'recovery_det_only': {
        'description': (
            'Stage R1 detection bootstrap: det + head_pose ONLY (activity/PSR '
            'losses off). With empty frames skipped in det loss, all det '
            'gradient comes from GT-bearing frames. Gate: det_mAP50 >= 0.05 '
            'before switching to the joint recovery preset.'
        ),
        'dataset_mode':       'manual_only',
        'backbone':           'convnext_tiny',
        'use_tma_cell':       True,
        'use_temporal_bank':  True,
        'use_hand_film':      True,
        'benchmark_mode':     False,
        'batch_size':         1,
        'grad_accum_steps':   8,
        'zero_det_conf':      False,
        'staged_training':    False,
        'mixed_precision':    False,  # [RC-29] FP32 only — see 'recovery' preset comment
        'use_mixup':          False,
        'use_ema':            False,
        'train_det':          True,
        'train_act':          False,  # activity loss OFF — pure detection bootstrap
        'train_psr':          False,  # PSR loss OFF
        'train_head_pose':    True,   # cheap, healthy, gives backbone a 2nd stable signal
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
    global GRAD_ACCUM_STEPS, EFFECTIVE_BATCH
    global ZERO_DET_CONF_FOR_RECOVERY, STAGED_TRAINING, MIXED_PRECISION
    global USE_MIXUP, USE_EMA
    global TRAIN_DET, TRAIN_ACT, TRAIN_PSR, TRAIN_HEAD_POSE

    if preset_name not in PRESETS:
        raise ValueError(f'Unknown preset: {preset_name}. Available: {list(PRESETS.keys())}')

    preset = PRESETS[preset_name]
    BENCHMARK_MODE = preset.get('benchmark_mode', False)
    USE_TMA_CELL = preset.get('use_tma_cell', False)
    USE_TEMPORAL_BANK = preset.get('use_temporal_bank', False)
    USE_HAND_FILM = preset.get('use_hand_film', USE_HAND_FILM)
    BATCH_SIZE = preset.get('batch_size', BATCH_SIZE)
    GRAD_ACCUM_STEPS = preset.get('grad_accum_steps', GRAD_ACCUM_STEPS)
    ZERO_DET_CONF_FOR_RECOVERY = preset.get('zero_det_conf', ZERO_DET_CONF_FOR_RECOVERY)
    STAGED_TRAINING = preset.get('staged_training', STAGED_TRAINING)
    MIXED_PRECISION = preset.get('mixed_precision', MIXED_PRECISION)
    USE_MIXUP = preset.get('use_mixup', USE_MIXUP)
    USE_EMA = preset.get('use_ema', USE_EMA)
    # [RC-28] per-task ablation flags (recovery_det_only). train.py caches
    # these into CFG_TRAIN_* via _refresh_runtime_cfg() right after the preset
    # is applied, and passes them into MultiTaskLoss / model construction.
    TRAIN_DET = preset.get('train_det', TRAIN_DET)
    TRAIN_ACT = preset.get('train_act', TRAIN_ACT)
    TRAIN_PSR = preset.get('train_psr', TRAIN_PSR)
    TRAIN_HEAD_POSE = preset.get('train_head_pose', TRAIN_HEAD_POSE)
    # [AUDIT FIX 2026-06-11] EFFECTIVE_BATCH was computed once at import and
    # went stale when a preset changed BATCH_SIZE / GRAD_ACCUM_STEPS.
    EFFECTIVE_BATCH = BATCH_SIZE * GRAD_ACCUM_STEPS

    update_dynamic_paths()

    if 'description' in preset:
        _cfg_logger.info(f'[config] Applied preset "{preset_name}": {preset["description"]}')
    else:
        _cfg_logger.info(f'[config] Applied preset "{preset_name}"')


def update_dynamic_paths():
    """Recompute all dynamic paths after config changes."""
    global OUTPUT_ROOT, CHECKPOINT_DIR, LOG_DIR, EVAL_SAVE_DIR

    # Don't override if set by environment variable
    if 'OUTPUT_ROOT_OVERRIDE' in os.environ:
        OUTPUT_ROOT = Path(os.environ['OUTPUT_ROOT_OVERRIDE'])
    else:
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
