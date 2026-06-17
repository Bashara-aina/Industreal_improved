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

# [OPUS v5 AUDIT] Bring-up mode: flip guards from silent-fallback to assert-and-crash
# so bugs surface in 200 steps, not 8 GPU-hours. Disable for production.
ASSERT_AND_CRASH = int(os.environ.get('ASSERT_AND_CRASH', '0')) == 1
LIVENESS_EVERY = 100  # [FIX 2026-06-15] 2x frequency — every 100 steps instead of 200 for tighter monitoring
LIVENESS_GRAD_EVERY = 200  # [FIX 2026-06-15] Separate grad-norm liveness (kept at 200 while output liveness is at 100)
DET_DEBUG_EVERY = 50  # [FIX4] Detection head debug diagnostic frequency (--reinit-heads only)
# NOTE: Detection head warmup is HARDCODED in train.py (50 zero-grad + 200 linear ramp, 250 total).
# DET_WARMUP_STEPS was considered as a config variable but was never wired up — see train.py for the actual logic.
DET_LR_MULTIPLIER = 1.0  # WD is now scaled proportionally with LR in train.py, so WD/LR stays constant when _stage_lr_mult reduces LR. Was 5.0 (collapsed full-head), then 1.0 (stagnant because WD wasn't scaled).

# [OPUS v5 AUDIT] Simplify loss assembly during bring-up (#49).
# Disables per-task ramps and smooth-caps so gradient attribution is clean.
# Enable for production, disable for diagnosis (with ASSERT_AND_CRASH).
SIMPLIFY_LOSS = int(os.environ.get('SIMPLIFY_LOSS', '0')) == 1  # 1 = no ramps/caps (diagnosis)

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

# [FIX 2026-06-15] Gradient checkpointing for ConvNeXt backbone
# Uses torch.utils.checkpoint on each of the 4 backbone stages, trading
# ~20% compute for ~50% activation memory reduction during backprop.
# Required when USE_PSR_SEQUENCE_MODE=True to prevent OOM on RTX 3060 12GB.
USE_BACKBONE_CHECKPOINT = True

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
# [OPUS v5 AUDIT #13] Anchor sizes. Original (24,48,96,192,384) covered only 1.6% of GT.
# K-means on 14,122 boxes gave (195,335,375,445,578) but these were too large — missed
# small GT (h p10=156px). Keep the guess anchors which empirically gave 0.0172 mAP.
ANCHOR_SIZES = (96, 160, 256, 384, 512)
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

# =========================================================================
# !!! WARNING — Anchor sizes are ABSOLUTE PIXEL values, not normalized !!!
# =========================================================================
# ANCHOR_SIZES (96, 160, 256, 384, 512) are in absolute pixels, calibrated
# for 1280x720 input via k-means on GT boxes (Doc 01 B.3). They are NOT
# normalized by IMG_WIDTH/IMG_HEIGHT.
#
# If IMG_SIZE is reduced (e.g. to 320x240 via a "fast" preset):
#   - ANCHOR_SIZES do NOT scale — a 96px anchor at 320-wide covers 30%
#     of image width (vs 7.5% at 1280-wide). The 512px anchor exceeds the
#     image entirely.
#   - GT boxes ARE properly rescaled in _extract_boxes_from_coco() via
#     _sx/_sy factors, so GT locations are correct at any IMG_SIZE.
#   - The anchor-to-box IoU matching changes because anchors stay at
#     original pixel sizes while GT boxes shrink. Recalibrate anchors
#     via calibrate_anchors.py before switching IMG_SIZE.
#   - At small sizes, P6 (stride 64) and P7 (stride 128) feature maps
#     become tiny (e.g. 5x4 and 3x2 at 320x240) — the top two FPN levels
#     contribute minimal spatial signal.
#
# Presets can override IMG_WIDTH/IMG_HEIGHT via apply_preset() after
# import, so this assertion only guards against direct value edits:
assert IMG_SIZE[0] == IMG_WIDTH and IMG_SIZE[1] == IMG_HEIGHT, (
    f'IMG_SIZE={IMG_SIZE} must equal (IMG_WIDTH={IMG_WIDTH}, IMG_HEIGHT={IMG_HEIGHT}). '
    f'Direct edits to IMG_SIZE without corresponding IMG_WIDTH/HEIGHT changes are invalid.'
)
# Original (native) resolution — used for anchor calibration reference
ORIGINAL_WIDTH  = 1280
ORIGINAL_HEIGHT = 720

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# =========================================================================
# Training (RTX 3060 12 GB) — ConvNeXt + 5 heads + TMA + TemporalBank
# BATCH_SIZE=2 is safe: VRAM at 3.78GB/11.4GB (33%) with batch=1, VideoMAE is OFF.
# The BATCH_SIZE=1 constraint was written for VideoMAE+ConvNeXt+TMA+TemporalBank+5 heads.
# With VideoMAE disabled, batch=2 uses ~7.6GB — well within 12GB (proven by R2.5 probe).
# OOM fallback: train.py automatically halves batch if CUDA OOM is detected.
BATCH_SIZE = 2        # Was 1 (VideoMAE-era constraint). VRAM 33% at batch=1 → 2× safe.
GRAD_ACCUM_STEPS = 16 # Effective batch = 32 (paper target: batch=2 × accum=16)
EFFECTIVE_BATCH      = BATCH_SIZE * GRAD_ACCUM_STEPS  # 32

VAL_BATCH_SIZE = 4   # Was 16 (8x train batch with FP32 → OOM on RTX 3060). 4× is safe with no_grad.
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
GRAD_CLIP_NORM = 5.0  # Ensure gradient clipping is active; raised from 1.0 for 5-head multi-task model
VAL_EVERY = 1    # [BENCHMARK] Evaluate every 1 epoch (BENCHMARK_MODE override)
VAL_EVERY_N_STEPS = 1000  # [FIX 2026-06-15] Intra-epoch validation every 1K global steps. 200-batch gated eval (~3 min) for early divergence detection without full epoch overhead.
EVAL_MAX_BATCHES = -1  # Full validation set every epoch (no cap)

NUM_WORKERS = 8        # Was 4 — 64GB RAM + 32GB /dev/shm handles 8 workers easily
PIN_MEMORY = True
USE_AMP = True           # Mixed precision training flag. All AMP infrastructure in train.py
                         # (GradScaler, autocast, NaN guards, RC-29 telemetry) gates on this.
MIXED_PRECISION = False  # Disable AMP — PSR seq loss spikes corrupt GradScaler
                         # losses over 1100+ steps). AMP with GradScaler gives ~2x training
                         # speedup. Keep the existing NaN/isfinite guards and RC-29 telemetry
                         # for detection of any future AMP-related gradient overflow.
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
DET_EVAL_SCORE_THRESH = 0.001  # [OPUS v5 AUDIT] Lowered from 0.02 → 0.001 for YOLOv8 comparability. YOLOv8 reports at ~0.001; 0.02 understates our mAP.
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
FOCAL_ALPHA   = 0.90  # RF: 0.90 (was 0.75) — α=0.75 still collapses at 173K:1 neg/pos ratio. α=0.90 gives positives 9× weight vs background, ensuring net-positive gradient even in no-positive batches (OHEM keeps 64 worst negs).
FOCAL_GAMMA   = 2.0
GIOU_WEIGHT   = 2.0  # Doc 01 B.2: GIoU regression weight vs cls weight=1.0

# Hard negative mining for detection FocalLoss (RF stage collapse fix)
# With ~0.01% positive anchors (~20/173K per image), the cumulative gradient
# from negatives dominates → cls logits drift to -16 over ~850 steps → collapse.
# OHEM keeps only the top-K hardest negatives per image (K = num_pos * RATIO),
# breaking the cumulative negative gradient cycle. Ref: RetinaNet OHEM ablation.
DET_OHEM_ENABLED = True
DET_OHEM_RATIO = 1.0     # 1:1 — 3:1 still allows neg-dominant gradient when n_pos<22
DET_OHEM_MIN_NEG = 16    # lower floor to prevent neg gradient overwhelm when n_pos≈0

# Asymmetric focal gamma: prevent cls_mean collapse by keeping positive gradient alive
# With gamma=2 on both pos/neg, well-classified positives (p≈0.9) have near-zero gradient:
#   (1-p)^2 * CE = 0.01 * 0.105 = 0.001 per anchor → neg gradient dominates → collapse.
# With gamma_pos=0 (no suppression):
#   positives: 1 * CE = 0.105 per anchor (75× more gradient)
#   negatives: p^2 * CE (gamma=2, standard) — only OHEM-kept hard negatives contribute
DET_ASYMMETRIC_GAMMA = True     # enable per-class gamma (pos vs neg)
DET_GAMMA_POS = 0.0             # no gamma suppression for positives
DET_GAMMA_NEG = 2.0             # standard FL gamma for negatives

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

# [INTERVENTION 2026-06-14] PSR and pose weight multipliers
# Audit verdict: PSR and hand pose heads produce DEAD backbone gradients.
# PSR_WEIGHT amplifies the PSR loss BEFORE Kendall weighting (applied at Kendall
# assembly in losses.py). POSE_LOSS_WEIGHT replaces the old *0.001 fixed multiplier.
PSR_WEIGHT = 10.0       # Reduce PSR weight to prevent backbone disruption
POSE_LOSS_WEIGHT = 0.01    # Reduced from 0.02 — pose output at 4.46 (strongest head), less weight needed

# [FIX 2026-06-15] Detection empty-frame background loss
# With 99.3% of frames having no GT boxes (activity-balanced sampler doesn't
# prioritize object frames), the detection head receives gradient on <1% of
# batches. To prevent weight drift between positive frames, we subsample
# anchor locations on empty frames and compute a small background focal loss.
# The subsampling + scale bounds the gradient to ~0.005-0.9 per empty image
# (vs 130-200 for full 173K anchors), preventing the RC-28 collapse while
# keeping detection head weights active.
DET_EMPTY_SAMPLE = 2048     # [TUNE 2026-06-15] Increased from 512 — detection head grad norm was decaying to 0.0049 (DEAD) between GT-bearing batches. More bg samples + higher scale keeps gradient alive.
DET_EMPTY_BG_SCALE = 0.05   # [TUNE 2026-06-15] Increased from 0.01 — detection head was dying (grad norm 0.005, DEAD). Empty-frame bg loss at 0.01 scale produced ~0.003-0.005 loss, insufficient to maintain weights over ~2200 steps without GT.

# [FIX 2026-06-15] Task-aware sampling: upweight frames with GT boxes
# The purely activity-balanced sampler yields ~24% GT-bearing batches.
# With ~1:5000 positive:negative anchor ratio, the classifier gradient
# is dominated by negatives even with focal loss gamma=2.
# Upweighting GT frames by 2x shifts per-batch GT density toward 40-50%,
# giving the classifier enough positive gradient to learn discriminative features.
USE_TASK_AWARE_SAMPLING = True
TASK_AWARE_DET_BOOST = 2.0      # 2x weight for frames with GT boxes
TASK_AWARE_PSR_BOOST = 1.5      # 1.5x weight for frames with PSR labels

# [DET GT-FRAME SAMPLING 2026-06-16] Root fix for the detection class-imbalance
# "death spiral". IndustReal OD labels are SPARSE (only a small fraction of RGB
# frames carry boxes), but the sample index includes every stride-th frame for
# the dense tasks (activity/pose/PSR). The activity-balanced sampler therefore
# draws GT-bearing frames only rarely, so the detector sees a positive box in
# <1% of steps and its positive logits decay (death spiral). The legacy
# TASK_AWARE_DET_BOOST (a *constant* 2x multiplier) cannot fix this because the
# resulting GT density still scales with the base OD density: 2x of 0.7% is
# still ~1.4%.
#
# DET_GT_FRAME_FRACTION instead targets an ABSOLUTE per-batch GT fraction. When
# > 0, get_sampler() redistributes the sampling mass so that — in expectation —
# this fraction of every batch is GT-bearing, INDEPENDENT of the base OD density.
# This guarantees the detector positive gradient on (nearly) every step.
#   0.0  = disabled (legacy behaviour — constant-boost only)
#   0.9  = detection-dominant stages (RF1/RF2, recovery_det_only)
#   0.4  = detection + other heads (RF3-RF10): leave room for activity balance
# The actual value is derived per-stage in apply_preset() from the active heads,
# and can be overridden by adding 'det_gt_frame_fraction' to a preset.
# WARNING: 0.0 means NO positive-frame reweighting — ~99.3% of training steps
# see zero GT boxes, causing the detection classifier to decay (death spiral).
# Set to 0.90 for detection-dominant stages (RF1/RF2), 0.40 for mixed stages.
# The apply_preset() function overrides this per-stage; this default matters
# only when running WITHOUT a preset.
DET_GT_FRAME_FRACTION = float(os.environ.get('DET_GT_FRAME_FRACTION', '0.90'))

# [FIX 2026-06-15] Activity dominance control
# The 3-layer collapse cascade showed activity dominating all other heads via:
# 1. Larger loss magnitude → Kendall precision advantage → backbone overfits to activity
# 2. PSR/pose gradients die → no multi-task signal → activity collapse to 2/75 classes
# Fix: lower ACTIVITY_HEAD_GRAD_CLIP + weigh activity loss down before Kendall
ACTIVITY_HEAD_GRAD_CLIP = 0.1  # Reduced from 0.5 — prevent activity dominating backbone
ACTIVITY_LOSS_WEIGHT = 0.2     # Down-weight activity loss 80% before Kendall weighting (was 0.3 — activity still dominating at 0.4-11.0 vs PSR 0.006-0.13)

# [FIX 2026-06-15] Per-task Kendall log_var bounds to prevent multi-task collapse cascade
# Activity log_var FLOOR (min) prevents precision-boosting above exp(0)=1.0
# PSR/pose log_var CEILING (max) prevents suppression below exp(0)=1.0
# Together these guarantee no task can zero out another through Kendall
KENDALL_LOG_VAR_MIN_ACT = -0.5    # Allow moderate activity precision boost (was 0.0 — activity losing, needs room)
KENDALL_LOG_VAR_MAX_PSR = 0.0     # PSR can't be suppressed (min prec=1.0) — keep while PSR recovers
KENDALL_LOG_VAR_MAX_POSE = 3.0    # Allow pose suppression (was 0.0 — pose dominating at 4.46 vs PSR 0.05)

# [FIX 2026-06-15] Step-based PSR warmup (works when STAGED_TRAINING=False)
# Ramps PSR precision multiplier from PSR_WARMUP_INIT_MULT → 1.0 over first N steps
# Gives PSR a gradient head start before activity loss dominates
PSR_WARMUP_INIT_MULT = 2.0        # Reduced from 3.0 — gentler warmup start since sequence mode provides signal

# [FIX 2026-06-16] Regression gradient warmup for --reinit-heads
# Ramps det_reg loss multiplier from REINIT_REG_WARMUP_INIT_MULT → 1.0 over the first
# N steps after --reinit-heads. Prevents gradient shock when the freshly reinitialized
# regression head encounters its first GT boxes (~step 751), which otherwise corrupts
# shared FPN features and collapses both regression and classification.
REINIT_REG_WARMUP_STEPS = 1000     # Steps to ramp regression loss; covers the window before first GT boxes arrive
REINIT_REG_WARMUP_INIT_MULT = 0.01 # Initial reg_loss multiplier (1% → full at step 1000)

# [FIX 2026-06-16] Detach FPN features for regression subnet — gradients-only fix
# When True, reg_subnet receives feat.detach() so regression gradients don't flow into
# shared FPN features. This is the root fix for detection head collapse after --reinit-heads:
# regression gradient shock corrupts classification through FPN even with loss warmup.
# Set True for RF1 (--reinit-heads), False for converged stages.
DETACH_REG_FPN = True

# [FIX 2026-06-16] Detach FPN features for PSR head — gradient isolation
# When True, PSR receives feat.detach() so PSR gradients don't flow into
# shared FPN features. Prevents PSR loss spikes from corrupting detection
# features through the backbone. Set True for RF1, False for full training.
DETACH_PSR_FPN = True
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
USE_LDAM_DRW = False  # [OPUS v5] Disabled — s=30 amplifies 30× on top of CB sampling + LS → 1-class collapse. Use plain CE + label smoothing for first joint runs.
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
PSR_FOCAL_GAMMA = 1.0  # [TUNE 2026-06-15] Reduced from 1.5 — per-frame focal loss was saturated at ~9e-05 with gamma=1.5. The re-warmed head needs gentler focusing; gamma=1.0 recovers gradient on hard examples.

# Per-component PSR focal loss weights (11 components)
# Inverse prevalence weighting: rarer steps get higher weight
# From validation set prevalence: [1.0, 0.823, 0.831, 0.505, 0.199, 0.62, 0.604, 0.454, 0.454, 0.363, 0.217]
PSR_COMP_WEIGHTS = [1.0, 1.21, 1.20, 1.98, 5.03, 1.61, 1.66, 2.20, 2.20, 2.75, 4.61]

# PSR sequence-mode training (Doc 01 §D.1) — THE biggest PSR unlock
# =========================================================================
# Doc 01 §D.1: Current training samples one frame per step, making the
# temporal Transformer dormant. Sequence-mode trains on contiguous T-frame
# windows where the Transformer is properly engaged.
USE_PSR_SEQUENCE_MODE = True   # [FIX 2026-06-15] Enabled — PSR head outputs DEAD bias gradient without temporal context.
PSR_SEQUENCE_LENGTH = 8        # [FIX E1] Increased from 2 to 8 — gives Transformer meaningful temporal context. Verified safe with gradient checkpointing.
                              # (sequence-mode memory doubled and fires 5x more often). T=4 is
                              # the memory-bounded choice; the bigger unlock is below.
PSR_SEQ_EVERY_N_BATCHES = 2  # [FIX 2026-06-16] Restored to 2. At 1, every batch is a seq batch → detection NEVER trains. At 2, alternating freeze works but PSR gradients on seq steps pull backbone features away from detection. Fix: zero backbone grads on seq steps in train.py.
PSR_SEQ_LOSS_SCALE = 1.5     # [TUNE 2026-06-15] Reduced from 3.0 — PSR seq loss shows spike-decay cycles (period ~200-250 batches). At 3.0x, spikes reached 45-60, causing weight disruption. 1.5x retains gradient amplification while damping spike magnitude.

# [OPUS v5] PSR transition objective — use Gaussian-smeared transition targets
# + MonotonicDecoder instead of per-frame BCE/focal on fill-forward labels.
# Per-frame focal on 95%-static labels makes constant output near-optimal.
# psr_transition.py already implements build_transition_targets + MonotonicDecoder.
USE_PSR_TRANSITION = True     # [FIX E1] Enable transition targets — prevents constant-output collapse on ~95% static per-frame labels
PSR_TRANSITION_SIGMA = 3.0   # Gaussian sigma for transition target smearing (frames)
PSR_LOSS_WEIGHT = 5.0        # [FIX E1] Gradient scaling multiplier for PSR loss (applied before Kendall weighting). PSR loss is ~0.01 vs activity ~1-5, so this prevents Kendall from suppressing PSR.

# [OPUS v5 AUDIT #83] Procedure-order prior: penalize invalid assembly step transitions.
# B2 baseline (F1=0.731) beats STORM-PSR largely because of order constraints.
# Each component must be monotonic (0→1 only) in assembly — no disassembly.
# Set True for R2.5 PSR training; implemented in psr_transition.py MonotonicDecoder.
USE_PSR_ORDER_PRIOR = False

# [OPUS v5 AUDIT] Geometry-aware head pose: replace 9-raw-number MSE MLP with
# 6D continuous rotation (Zhou et al. CVPR 2019) → rotation matrix → orthonormal vectors.
# Loss is IDENTICAL to non-geo head (head_pose_loss_split: position MSE + direction MSE).
# The comment blaming this for NaN (R2.5) was WRONG — root cause was PSR sensitivity penalty
# (Bessel-corrected std with N-1=0). Verified: 0 NaN events in 1100+ steps after PSR fix.
USE_GEO_HEAD_POSE = True     # Was False — NaN blame disproven. Geometry-aware is strictly better.

# [OPUS v5 AUDIT] FeatureBank gradient control — the bank stores temporal features
# but .detach() prevents gradient flow through bank entries. Slot -1 overwrite
# further limits learning to the current frame only (#14-16).
# Set False for R2 to allow temporal gradient through the bank.
FEATURE_BANK_DETACH = True   # True = legacy behavior (no gradient through bank)
                              # False = gradient flows through bank (enables temporal learning)
FEATURE_BANK_DETACH_GRAD_ENTRIES_ONLY = False  # [E2] True = only detach stored bank entries (not current frame).
                                                  #       Current frame keeps gradient; historical entries are detached.
                                                  #       Intermediate between FEATURE_BANK_DETACH=True and False.
FEATURE_BANK_SLOT_OVERWRITE = True  # True = legacy: live frame overwrites slot -1
                                     # False = bank accumulates without overwrite (#16)

# Fix 1 (2026-06-06): penalize constant per-frame PSR predictions in T=1 mode.
# Stage 3 epoch 16 collapsed logit std to 0.12%; this penalty keeps it > 1e-3.
# [R2.5 FIX 2026-06-14] Re-enabled with correction=0 on std() — NaN root cause fixed.
# With correction=0, batch_size=1 gives std=0 → -log(0+1e-3)=6.9 (finite). The
# torch.isfinite() guard catches any edge case. Verified: 0 NaN events in 1100+ steps.
PSR_SENSITIVITY_WEIGHT = 0.01  # Was 0.0 — re-enabled after std(correction=0) fix. 0.01 gives ~0.07 loss contribution at std=0.

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
# Knowledge Distillation (E6)
# =========================================================================
# Distillation from specialist baselines (YOLOv8m for det, MViTv2 for act).
# When enabled, DistillationLoss is added to the total loss.
# Teacher predictions must be pre-generated (see distillation.py --generate).
USE_DISTILLATION = False

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

# [OPUS v5] Eval cadence: compute full detection mAP every N epochs; fast gate-only eval
# (EVAL_MAX_BATCHES capped) on other epochs. 0 = eval every epoch (no skip).
DET_METRICS_EVERY_N = int(os.environ.get('DET_METRICS_EVERY_N', '5'))  # Full mAP eval every N epochs; 0=every epoch
GATE_EVAL_MAX_BATCHES = int(os.environ.get('GATE_EVAL_MAX_BATCHES', '200'))  # Max val batches on non-full-eval epochs

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
    # [OPUS v5] Paper-run preset: enables ALL winnable-task fixes for the final run.
    'paper_run': {
        'description': 'Final paper-run preset — PSR transition, geo head pose, bank gradient.',
        'dataset_mode':       'manual_only',
        'backbone':           'convnext_tiny',
        'use_tma_cell':       True,
        'use_temporal_bank':  True,
        'use_hand_film':      True,
        'benchmark_mode':     False,
        'batch_size':         2,   # Was 1 — VRAM 33% at batch=1, VideoMAE OFF. Batch=2 is safe (proven by R2.5 probe).
        'grad_accum_steps':   16,  # Was 8 — with batch=2, effective batch = 32 (paper target)
        'zero_det_conf':      False,
        'staged_training':    False,
        'mixed_precision':    False,  # [FIX 2026-06-15] Reverted to False: model has AMP-incompatible ops (crash: "Found dtype Float but expected Half" in backward). Train in FP32 until AMP compatibility is fixed.
        'use_mixup':          False,
        'use_ema':            True,
        'train_det':          True,
        'train_act':          True,
        'train_psr':          True,
        'train_head_pose':    True,
        # [BLOCKER-C] Winnable-task flags — actually set by apply_preset now
        'use_psr_transition':       True,
        'use_geo_head_pose':        True,    # [R2.5] Was False — NaN blame disproven. Root cause was PSR sensitivity (std correction=0). Geo head is strictly better (6D rotation → orthonormal).
        'feature_bank_detach':      True,   # keep detached — gradient through bank causes double-backward crash (#3092789)
        'feature_bank_slot_overwrite': False,
        'use_psr_order_prior':      True,
        'psr_sensitivity_weight':   0.01,  # Was 0.0 — re-enabled after std(correction=0) NaN fix
        'use_ldam_drw':             False,
        # [FIX D7] Detach FPN gradients to prevent regression/PSR gradient shock
        # through shared FPN features. Required when running outside stage_manager
        # (which otherwise only appends --detach-reg-fpn / --detach-psr-fpn for
        # stages with reinit_heads=True).
        'detach_reg_fpn':           True,
        'detach_psr_fpn':           True,
    },
    # =====================================================================
    # [RF STAGES] RF1–RF10 progressive multi-task training presets
    # =====================================================================
    # Progressive unlocking: each stage adds a task head and/or increases
    # data ratio. Managed externally by stage_manager.py.
    #
    # RF1 : Detection only                                   20%    20 ep
    # RF2 : + Body + Head Pose                               35%    15 ep
    # RF3 : + Activity (no PSR)                              35%    15 ep
    # RF4 : + PSR (all heads, transition enabled)            50%    20 ep
    # RF5 : Consolidate all heads                            50%    10 ep
    # RF6 : Scale data                                       65%    10 ep
    # RF7 : Scale data                                       65%    10 ep
    # RF8 : Scale data                                       80%    10 ep
    # RF9 : Scale data                                       90%    10 ep
    # RF10: Final full-data push                            100%    15 ep
    # =====================================================================
    'stage_rf1': {
        'description': 'RF1: Detection only \u2014 stabilize det head after reinit (20% data, 20 ep).',
        'dataset_mode':       'manual_only',
        'backbone':           'convnext_tiny',
        'use_tma_cell':       True,
        'use_temporal_bank':  True,
        'use_hand_film':      True,
        'benchmark_mode':     False,
        'batch_size':         4,
        'grad_accum_steps':   8,
        'zero_det_conf':      False,
        'staged_training':    False,
        'mixed_precision':    False,
        'use_mixup':          False,
        'use_ema':            True,
        'train_det':          True,
        'train_act':          False,
        'train_psr':          False,
        # [RF1 FIX 2026-06-17] Was False. Restored to True to match the
        # Opus-prescribed recovery_det_only recipe. Head pose is cheap and gives
        # the backbone a dense, feature-diverse per-frame signal alongside
        # detection, so the shared trunk keeps learning even when detection's
        # positive anchors are sparse. This is insurance on top of the primary
        # fix (detach_reg_fpn=False in stage_manager); to ablate and confirm the
        # regression-gradient fix alone is sufficient, set this back to False.
        'train_head_pose':    True,
        'use_psr_transition':       False,
        'use_geo_head_pose':        True,
        'feature_bank_detach':      True,
        'feature_bank_slot_overwrite': False,
        'use_psr_order_prior':      False,
        'psr_sensitivity_weight':   0.0,
        'use_ldam_drw':             False,
        # [RF1 2026-06-17] Regression gradient MUST reach FPN/backbone for RF1.
        # Despite FIX D7's recommendation, detaching regression in the bootstrap
        # stage leaves only the sparse classification path to train shared features
        # — the model never escapes π=0.01 background equilibrium ("localizes but
        # won't fire"). REINIT_REG_WARMUP_STEPS=1000 at 1% initial multiplier is
        # the correct gradient shock guard (ramps reg loss 1%→100% over 1000 steps).
        # The stage_manager.py RF1 stage already sets detach_reg_fpn: False and
        # does NOT pass --detach-reg-fpn; this preset override ensures the model
        # actually receives regression gradient even when apply_preset() runs first.
        # Other stages retain detach_reg_fpn: True for multi-head gradient isolation.
        'detach_reg_fpn':           False,
        'detach_psr_fpn':           True,
    },
    'stage_rf2': {
        'description': 'RF2: Detection + Body+Head Pose (35% data, 15 ep).',
        'dataset_mode':       'manual_only',
        'backbone':           'convnext_tiny',
        'use_tma_cell':       True,
        'use_temporal_bank':  True,
        'use_hand_film':      True,
        'benchmark_mode':     False,
        'batch_size':         4,
        'grad_accum_steps':   8,
        'zero_det_conf':      False,
        'staged_training':    False,
        'mixed_precision':    False,
        'use_mixup':          False,
        'use_ema':            True,
        'train_det':          True,
        'train_act':          False,
        'train_psr':          False,
        'train_head_pose':    True,
        'use_psr_transition':       False,
        'use_geo_head_pose':        True,
        'feature_bank_detach':      True,
        'feature_bank_slot_overwrite': False,
        'use_psr_order_prior':      False,
        'psr_sensitivity_weight':   0.0,
        'use_ldam_drw':             False,
        # [FIX D7] Detach FPN gradients to prevent regression/PSR gradient shock
        # through shared FPN features. Required when running outside stage_manager
        # (which otherwise only appends --detach-reg-fpn / --detach-psr-fpn for
        # stages with reinit_heads=True).
        'detach_reg_fpn':           True,
        'detach_psr_fpn':           True,
    },
    'stage_rf3': {
        'description': 'RF3: Detection + Pose + Activity (35% data, 15 ep).',
        'dataset_mode':       'manual_only',
        'backbone':           'convnext_tiny',
        'use_tma_cell':       True,
        'use_temporal_bank':  True,
        'use_hand_film':      True,
        'benchmark_mode':     False,
        'batch_size':         4,
        'grad_accum_steps':   8,
        'zero_det_conf':      False,
        'staged_training':    False,
        'mixed_precision':    False,
        'use_mixup':          False,
        'use_ema':            True,
        'train_det':          True,
        'train_act':          True,
        'train_psr':          False,
        'train_head_pose':    True,
        'use_psr_transition':       False,
        'use_geo_head_pose':        True,
        'feature_bank_detach':      True,
        'feature_bank_slot_overwrite': False,
        'use_psr_order_prior':      False,
        'psr_sensitivity_weight':   0.0,
        'use_ldam_drw':             False,
        # [FIX D7] Detach FPN gradients to prevent regression/PSR gradient shock
        # through shared FPN features. Required when running outside stage_manager
        # (which otherwise only appends --detach-reg-fpn / --detach-psr-fpn for
        # stages with reinit_heads=True).
        'detach_reg_fpn':           True,
        'detach_psr_fpn':           True,
    },
    'stage_rf4': {
        'description': 'RF4: All heads + PSR transition (50% data, 20 ep).',
        'dataset_mode':       'manual_only',
        'backbone':           'convnext_tiny',
        'use_tma_cell':       True,
        'use_temporal_bank':  True,
        'use_hand_film':      True,
        'benchmark_mode':     False,
        'batch_size':         4,
        'grad_accum_steps':   8,
        'zero_det_conf':      False,
        'staged_training':    False,
        'mixed_precision':    False,
        'use_mixup':          False,
        'use_ema':            True,
        'train_det':          True,
        'train_act':          True,
        'train_psr':          True,
        'train_head_pose':    True,
        'use_psr_transition':       True,
        'use_geo_head_pose':        True,
        'feature_bank_detach':      True,
        'feature_bank_slot_overwrite': False,
        'use_psr_order_prior':      True,
        'psr_sensitivity_weight':   0.01,
        'use_ldam_drw':             False,
        # [FIX D7] Detach FPN gradients to prevent regression/PSR gradient shock
        # through shared FPN features. Required when running outside stage_manager
        # (which otherwise only appends --detach-reg-fpn / --detach-psr-fpn for
        # stages with reinit_heads=True).
        'detach_reg_fpn':           True,
        'detach_psr_fpn':           True,
    },
    'stage_rf5': {
        'description': 'RF5: Consolidate all heads (50% data, 10 ep).',
        'dataset_mode':       'manual_only',
        'backbone':           'convnext_tiny',
        'use_tma_cell':       True,
        'use_temporal_bank':  True,
        'use_hand_film':      True,
        'benchmark_mode':     False,
        'batch_size':         4,
        'grad_accum_steps':   8,
        'zero_det_conf':      False,
        'staged_training':    False,
        'mixed_precision':    False,
        'use_mixup':          False,
        'use_ema':            True,
        'train_det':          True,
        'train_act':          True,
        'train_psr':          True,
        'train_head_pose':    True,
        'use_psr_transition':       True,
        'use_geo_head_pose':        True,
        'feature_bank_detach':      True,
        'feature_bank_slot_overwrite': False,
        'use_psr_order_prior':      True,
        'psr_sensitivity_weight':   0.01,
        'use_ldam_drw':             False,
        # [FIX D7] Detach FPN gradients to prevent regression/PSR gradient shock
        # through shared FPN features. Required when running outside stage_manager
        # (which otherwise only appends --detach-reg-fpn / --detach-psr-fpn for
        # stages with reinit_heads=True).
        'detach_reg_fpn':           True,
        'detach_psr_fpn':           True,
    },
    'stage_rf6': {
        'description': 'RF6: Scale data to 65% (10 ep).',
        'dataset_mode':       'manual_only',
        'backbone':           'convnext_tiny',
        'use_tma_cell':       True,
        'use_temporal_bank':  True,
        'use_hand_film':      True,
        'benchmark_mode':     False,
        'batch_size':         4,
        'grad_accum_steps':   8,
        'zero_det_conf':      False,
        'staged_training':    False,
        'mixed_precision':    False,
        'use_mixup':          False,
        'use_ema':            True,
        'train_det':          True,
        'train_act':          True,
        'train_psr':          True,
        'train_head_pose':    True,
        'use_psr_transition':       True,
        'use_geo_head_pose':        True,
        'feature_bank_detach':      True,
        'feature_bank_slot_overwrite': False,
        'use_psr_order_prior':      True,
        'psr_sensitivity_weight':   0.01,
        'use_ldam_drw':             False,
        # [FIX D7] Detach FPN gradients to prevent regression/PSR gradient shock
        # through shared FPN features. Required when running outside stage_manager
        # (which otherwise only appends --detach-reg-fpn / --detach-psr-fpn for
        # stages with reinit_heads=True).
        'detach_reg_fpn':           True,
        'detach_psr_fpn':           True,
    },
    'stage_rf7': {
        'description': 'RF7: Continue at 65% data (10 ep).',
        'dataset_mode':       'manual_only',
        'backbone':           'convnext_tiny',
        'use_tma_cell':       True,
        'use_temporal_bank':  True,
        'use_hand_film':      True,
        'benchmark_mode':     False,
        'batch_size':         4,
        'grad_accum_steps':   8,
        'zero_det_conf':      False,
        'staged_training':    False,
        'mixed_precision':    False,
        'use_mixup':          False,
        'use_ema':            True,
        'train_det':          True,
        'train_act':          True,
        'train_psr':          True,
        'train_head_pose':    True,
        'use_psr_transition':       True,
        'use_geo_head_pose':        True,
        'feature_bank_detach':      True,
        'feature_bank_slot_overwrite': False,
        'use_psr_order_prior':      True,
        'psr_sensitivity_weight':   0.01,
        'use_ldam_drw':             False,
        # [FIX D7] Detach FPN gradients to prevent regression/PSR gradient shock
        # through shared FPN features. Required when running outside stage_manager
        # (which otherwise only appends --detach-reg-fpn / --detach-psr-fpn for
        # stages with reinit_heads=True).
        'detach_reg_fpn':           True,
        'detach_psr_fpn':           True,
    },
    'stage_rf8': {
        'description': 'RF8: Scale data to 80% (10 ep).',
        'dataset_mode':       'manual_only',
        'backbone':           'convnext_tiny',
        'use_tma_cell':       True,
        'use_temporal_bank':  True,
        'use_hand_film':      True,
        'benchmark_mode':     False,
        'batch_size':         4,
        'grad_accum_steps':   8,
        'zero_det_conf':      False,
        'staged_training':    False,
        'mixed_precision':    False,
        'use_mixup':          False,
        'use_ema':            True,
        'train_det':          True,
        'train_act':          True,
        'train_psr':          True,
        'train_head_pose':    True,
        'use_psr_transition':       True,
        'use_geo_head_pose':        True,
        'feature_bank_detach':      True,
        'feature_bank_slot_overwrite': False,
        'use_psr_order_prior':      True,
        'psr_sensitivity_weight':   0.01,
        'use_ldam_drw':             False,
        # [FIX D7] Detach FPN gradients to prevent regression/PSR gradient shock
        # through shared FPN features. Required when running outside stage_manager
        # (which otherwise only appends --detach-reg-fpn / --detach-psr-fpn for
        # stages with reinit_heads=True).
        'detach_reg_fpn':           True,
        'detach_psr_fpn':           True,
    },
    'stage_rf9': {
        'description': 'RF9: Scale data to 90% (10 ep).',
        'dataset_mode':       'manual_only',
        'backbone':           'convnext_tiny',
        'use_tma_cell':       True,
        'use_temporal_bank':  True,
        'use_hand_film':      True,
        'benchmark_mode':     False,
        'batch_size':         4,
        'grad_accum_steps':   8,
        'zero_det_conf':      False,
        'staged_training':    False,
        'mixed_precision':    False,
        'use_mixup':          False,
        'use_ema':            True,
        'train_det':          True,
        'train_act':          True,
        'train_psr':          True,
        'train_head_pose':    True,
        'use_psr_transition':       True,
        'use_geo_head_pose':        True,
        'feature_bank_detach':      True,
        'feature_bank_slot_overwrite': False,
        'use_psr_order_prior':      True,
        'psr_sensitivity_weight':   0.01,
        'use_ldam_drw':             False,
        # [FIX D7] Detach FPN gradients to prevent regression/PSR gradient shock
        # through shared FPN features. Required when running outside stage_manager
        # (which otherwise only appends --detach-reg-fpn / --detach-psr-fpn for
        # stages with reinit_heads=True).
        'detach_reg_fpn':           True,
        'detach_psr_fpn':           True,
    },
    'stage_rf10': {
        'description': 'RF10: Final full-data push (100% data, 15 ep).',
        'dataset_mode':       'manual_only',
        'backbone':           'convnext_tiny',
        'use_tma_cell':       True,
        'use_temporal_bank':  True,
        'use_hand_film':      True,
        'benchmark_mode':     False,
        'batch_size':         4,
        'grad_accum_steps':   8,
        'zero_det_conf':      False,
        'staged_training':    False,
        'mixed_precision':    False,
        'use_mixup':          False,
        'use_ema':            True,
        'train_det':          True,
        'train_act':          True,
        'train_psr':          True,
        'train_head_pose':    True,
        'use_psr_transition':       True,
        'use_geo_head_pose':        True,
        'feature_bank_detach':      True,
        'feature_bank_slot_overwrite': False,
        'use_psr_order_prior':      True,
        'psr_sensitivity_weight':   0.01,
        'use_ldam_drw':             False,
    },
    # =====================================================================
    # [FIX 12] Image size presets — reduce IMG_SIZE for faster training
    # =====================================================================
    # WARNING: ANCHOR_SIZES are ABSOLUTE PIXEL values (see IMG_SIZE section).
    # These presets do NOT recalibrate anchors. Detection AP may drop at smaller
    # sizes because 512px anchors exceed the image at 320x240.
    #
    # Expected speedup vs 1280x720 (batch_size=2, RTX 3060):
    #   480x360: 1.78x fewer pixels (~173K vs ~307K) → ~1.3-1.5x speedup
    #   320x240: 4x fewer pixels (~77K vs ~307K) → ~2-3x speedup
    # =====================================================================
    'img_size_fast': {
        'description': (
            'Fast image size: 320x240 (4x fewer pixels than 1280x720). '
            'Expected ~2-3x training speedup. Detection AP may drop because '
            'ANCHOR_SIZES=(96,160,256,384,512) are absolute pixel values — '
            'the 512px anchor exceeds the 320-wide image. Recalibrate anchors '
            'via calibrate_anchors.py before production use.'
        ),
        'img_width':   320,
        'img_height':  240,
    },
    'img_size_balanced': {
        'description': (
            'Balanced image size: 480x360 (1.78x fewer pixels than 1280x720). '
            'Expected ~1.3-1.5x speedup with minimal mAP impact. Anchors remain '
            'at absolute pixel values; 512px anchor covers the full 480-wide '
            'image, so P6-P7 contribute limited spatial signal.'
        ),
        'img_width':   480,
        'img_height':  360,
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
    # [OPUS v5 BLOCKER-C FIX] Winnable-task flags — must be in global list for preset to set them
    global USE_PSR_TRANSITION, USE_GEO_HEAD_POSE, FEATURE_BANK_DETACH, FEATURE_BANK_SLOT_OVERWRITE
    global USE_LDAM_DRW, PSR_SENSITIVITY_WEIGHT, USE_PSR_ORDER_PRIOR
    global USE_VIDEOMAE
    global SUBSET_RATIO
    global DET_GT_FRAME_FRACTION
    global IMG_WIDTH, IMG_HEIGHT, IMG_SIZE
    # [FIX D7] FPN gradient detach flags — must be global for preset to set them
    global DETACH_REG_FPN, DETACH_PSR_FPN

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
    # [FIX 12] Image size override — update IMG_WIDTH/IMG_HEIGHT/IMG_SIZE
    if 'img_width' in preset:
        IMG_WIDTH = preset['img_width']
        IMG_HEIGHT = preset.get('img_height', IMG_HEIGHT)
        IMG_SIZE = (IMG_WIDTH, IMG_HEIGHT)
    # [RC-28] per-task ablation flags (recovery_det_only). train.py caches
    # these into CFG_TRAIN_* via _refresh_runtime_cfg() right after the preset
    # is applied, and passes them into MultiTaskLoss / model construction.
    TRAIN_DET = preset.get('train_det', TRAIN_DET)
    TRAIN_ACT = preset.get('train_act', TRAIN_ACT)
    TRAIN_PSR = preset.get('train_psr', TRAIN_PSR)
    TRAIN_HEAD_POSE = preset.get('train_head_pose', TRAIN_HEAD_POSE)
    SUBSET_RATIO = preset.get('subset_ratio', SUBSET_RATIO)
    # [FIX D7] FPN gradient detach — apply from preset so running outside
    # stage_manager still gets gradient isolation for shared FPN features.
    DETACH_REG_FPN = bool(preset.get('detach_reg_fpn', DETACH_REG_FPN))
    DETACH_PSR_FPN = bool(preset.get('detach_psr_fpn', DETACH_PSR_FPN))
    # [DET GT-FRAME SAMPLING 2026-06-16] Derive the absolute per-batch GT-frame
    # target from the active heads (unless the preset overrides it explicitly).
    # Detection-dominant stages (det on, activity/PSR off) need aggressive GT
    # oversampling because OD labels are sparse; once activity/PSR are active we
    # relax toward the activity-balanced sampler so those dense tasks still see a
    # representative frame distribution. See DET_GT_FRAME_FRACTION docs above.
    if 'det_gt_frame_fraction' in preset:
        DET_GT_FRAME_FRACTION = float(preset['det_gt_frame_fraction'])
    elif TRAIN_DET and not (TRAIN_ACT or TRAIN_PSR):
        DET_GT_FRAME_FRACTION = 0.9   # RF1, RF2, recovery_det_only
    elif TRAIN_DET:
        DET_GT_FRAME_FRACTION = 0.4   # RF3-RF10 (detection + activity/PSR)
    else:
        DET_GT_FRAME_FRACTION = 0.0   # no detection head active
    # Non-silent: this fix is the whole point of the run — log it every time.
    _cfg_logger.info(
        f'[config] DET_GT_FRAME_FRACTION = {DET_GT_FRAME_FRACTION:.2f} '
        f'(train_det={TRAIN_DET}, train_act={TRAIN_ACT}, train_psr={TRAIN_PSR})'
    )
    # [OPUS v5 BLOCKER-C FIX] Winnable-task flags — actually set them from preset.
    # These were declared but never assigned → paper_run was a no-op for its key flags.
    USE_PSR_TRANSITION = bool(preset.get('use_psr_transition', USE_PSR_TRANSITION))
    # [R3 FIX 2026-06-14] PSR transition objective requires sequence batches (dim==3,
    # i.e. TMA cell with batch_size >= 2 or USE_PSR_SEQUENCE_MODE=True). Without
    # sequential context every batch is per-frame → transition loss is always zeroed
    # → PSR head gets zero gradient → DEAD. Fall back to per-frame focal loss.
    if not USE_PSR_SEQUENCE_MODE and USE_PSR_TRANSITION:
        import logging as _lg
        _lg.getLogger('industreal').warning(
            f'[R3] USE_PSR_SEQUENCE_MODE={USE_PSR_SEQUENCE_MODE} → forcing '
            f'USE_PSR_TRANSITION=False (transition objective requires sequence batches)'
        )
        USE_PSR_TRANSITION = False
    USE_GEO_HEAD_POSE = bool(preset.get('use_geo_head_pose', USE_GEO_HEAD_POSE))
    FEATURE_BANK_DETACH = bool(preset.get('feature_bank_detach', FEATURE_BANK_DETACH))
    FEATURE_BANK_SLOT_OVERWRITE = bool(preset.get('feature_bank_slot_overwrite', FEATURE_BANK_SLOT_OVERWRITE))
    USE_LDAM_DRW = bool(preset.get('use_ldam_drw', USE_LDAM_DRW))
    PSR_SENSITIVITY_WEIGHT = float(preset.get('psr_sensitivity_weight', PSR_SENSITIVITY_WEIGHT))
    USE_PSR_ORDER_PRIOR = bool(preset.get('use_psr_order_prior', USE_PSR_ORDER_PRIOR))
    USE_VIDEOMAE = bool(preset.get('use_videomae', False))
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
# [FIX 2026-06-17] PSR_WEIGHT moved from line 1362 into the loss-weights section.
# PSR_WEIGHT = 10.0 was previously declared after the INITIALIZATION block,
# creating a module-level constant that could silently override config values.
