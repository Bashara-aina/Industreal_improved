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

 Dataset location: /media/newadmin/master/POPW/datasets/industreal/
 Code location:    /home/newadmin/swarm-bot/master/POPW/working/code/industreal_improved/
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

SUBSET_RATIO = 1.0    # Full dataset — use all training data

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
LIVENESS_EVERY = 500  # Reduced from 100 — gates relaxed, less overhead needed
LIVENESS_GRAD_EVERY = 200  # [FIX 2026-06-15] Separate grad-norm liveness (kept at 200 while output liveness is at 100)
DET_DEBUG_EVERY = 50  # [FIX4] Detection head debug diagnostic frequency (--reinit-heads only)
# NOTE: Detection head warmup is HARDCODED in train.py (50 zero-grad + 200 linear ramp, 250 total).
# DET_WARMUP_STEPS was considered as a config variable but was never wired up — see train.py for the actual logic.
DET_LR_MULTIPLIER = 1.0  # [REVERT 2026-06-21 (Opus v11 Q2)] Back to 1.0. The detach_reg_fpn=False
                         # restart must change ONE training-dynamics variable. 2.0 was our own
                         # untested addition bundled with the detach flip — a confound. Restore the
                         # v8 baseline; if the det head looks gradient-starved AFTER detach stabilizes
                         # (≥2 epochs of upward trend), raise this ALONE to 1.5–2.0 and observe.
DET_BIAS_LR_FACTOR = 1.0  # [REVERT 2026-06-21 (Opus v11 Q2)] Back to 1.0. v8 found bias acceleration
                         # toward the all-background equilibrium IS the collapse mechanism (called an
                         # "own-goal" 5×). IOU_FLOOR=0.2 reduces false-positive labels but does NOT
                         # reverse the dominant negative gradient direction under 173K:1 imbalance — a
                         # 4× bias LR just reaches that equilibrium faster. The RetinaNet prior is set
                         # once via reinit_pi, not driven by a high bias LR.

# [OPUS v8 FIX 2026-06-20] Kendall multi-task fixes for RF2 detection collapse.
# Three independent mechanisms were letting head_pose dominate the shared backbone:
#
# (1) KENDALL_HP_PREC_CAP — head_pose precision can never exceed detection precision.
#     Without this, head_pose (loss ≈ 0.01) gets Kendall-optimal precision ~54.6×,
#     vs detection (loss ≈ 0.5) getting ~1.4×. The shared backbone is then optimized
#     for head_pose, losing object-discriminative features. See Opus v8 §1.1.
KENDALL_HP_PREC_CAP = True
#     Alternative: use fixed weights instead of learned Kendall for det-bootstrap.
#     Detection drives backbone at λ=1.0, head_pose just stabilizes at λ=0.1-0.3.
#     Set True for RF1-RF2; re-enable Kendall at RF3+ once detection is real.
KENDALL_FIXED_WEIGHTS = False  # default off; toggled per-stage by stage_manager

# (2) KENDALL_STAGED_TRAINING — kill the double curriculum (Opus v8 §3 Fix 3).
#     The RF stage manager already controls which heads train. The epoch-indexed
#     Kendall staging in losses.py (STAGE1_EPOCHS=5, STAGE2_EPOCHS=10) duplicates
#     this and silently triggers head_pose takeover at epoch 6. Setting False
#     makes staged_training in the loss a no-op; the RF stage manager is the
#     sole curriculum.
KENDALL_STAGED_TRAINING = False  # [FIX] was True — see Opus v8 §1.2 timing

# Head-pose fixed-loss weight when KENDALL_FIXED_WEIGHTS=True.
# Matches the λ range recommended in Opus v8 §3 Fix 1: 0.1–0.3
KENDALL_HP_FIXED_LAMBDA = 0.2

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
DET_POS_IOU_THRESH = 0.4       # [FIX 2026-06-20 (Opus v8 §3)] Was 0.5 — lowered to 0.4 so small assembly parts
                               # light up more positive anchors. For typical IndustReal GT (h≈156px at 720p),
                               # only ~1 anchor/GT clears IoU≥0.5 with these anchor sizes. 0.4 gives ~3–5× more
                               # positives, fixing the supply-side root cause of gradient starvation at source.
DET_NEG_IOU_THRESH = 0.4       # RetinaNet anchor matching: negative IoU threshold (standard: 0.4)
DET_POS_IOU_TOP_K = 9          # [FIX 2026-06-20 (Opus v8 §3)] Top-k force-match per GT (was single argmax).
DET_POS_IOU_IOU_FLOOR = 0.2    # [FIX 2026-06-21 (Opus v9 §R2)] Minimum IoU for top-k force-match anchors.
                               # Without this floor, top-k can label near-zero-IoU anchors as "positive",
                               # injecting label noise into the cls head. At 0.2, only anchors with
                               # ≥20% box overlap get positive labels. The argmax best anchor is always
                               # assigned regardless of floor (standard RetinaNet behavior).
DET_POS_ANCHOR_PROBE_EVERY = 1000  # Reduced from 200 — gates relaxed, less overhead
                                  # every N images. Logs mean/median/max/min of p_t on pos_mask anchors.
                                  # Set 0 to disable.
                               # Standard RetinaNet only force-matches the single best anchor per GT (~1 pos/GT).
                               # For small objects, this starves the classifier. Top-9 via IoU assigns ~6-10
                               # positive anchors per GT, giving the cls_subnet enough positive gradient to
                               # maintain discriminative features. Verify with MATCH_PROBE. RetinaNet-on-COCO
                               # doesn't need this because COCO objects cover enough anchor area; IndustReal
                               # assembly parts are systematically smaller than their anchor cells.
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
GRAD_ACCUM_STEPS = 8  # Per paper §Implementation: batch=2 × accum=8 = 16 effective
EFFECTIVE_BATCH      = BATCH_SIZE * GRAD_ACCUM_STEPS  # 16

VAL_BATCH_SIZE = 4   # Was 16 (8x train batch with FP32 → OOM on RTX 3060). 4× is safe with no_grad.
VAL_NUM_WORKERS = 0      # [FIX 2026-06-30] 0 workers — match NUM_WORKERS to avoid CUDA hangs


EPOCHS        = 100 
BASE_LR       = 5e-4   # Per paper: "5e-4"
WEIGHT_DECAY  = 5e-2   # Per paper §Implementation: "5 × 10⁻² (bias/norm excluded)"
WARMUP_EPOCHS = 2      # Per paper §Implementation: "Warmup (2 ep) → OneCycleLR"
USE_COSINE_ANNEALING = False  # Per paper: uses OneCycleLR instead
T_0 = 10
T_mult = 2
PATIENCE      = 10
GRAD_CLIP_NORM = 1.0  # Per paper §Implementation: "ℓ₂-norm = 1.0"
VAL_EVERY = 1    # [BENCHMARK] Evaluate every 1 epoch (BENCHMARK_MODE override)
VAL_EVERY_N_STEPS = 2500  # Reduced from 5000 — halves crash exposure window with mid-epoch checkpoint safety
EVAL_MAX_BATCHES = 500    # Cap validation to 500 batches (~2 min) per epoch
                          # Full 38K-frame eval takes 5+ hours. Fast val every epoch,
                          # then one full eval at the very end before the paper deadline.

NUM_WORKERS = 0          # [FIX 2026-06-30] Set from 4 to 0 — eliminate DataLoader worker deadlocks
                        # that hung training 3 times in 12 hours. No workers = single-process
                        # loading, ~25% slower but stable. CUDA + multiprocessing deadlocks
                        # on Python 3.13 + PyTorch 2.12 have no other reliable fix.
RAM_CACHE_MAX_IMAGES = 5000  # Cap RAM image cache to ~1.8 GB (JPEG bytes) — prevents OOM
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
EMA_DECAY      = 0.995  # Per paper §Implementation: "EMA decay 0.995 (from stage rf3)"

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
CUDNN_DETERMINISTIC = False    # Max speed — reproducibility not critical for RF2-RF10
CUDNN_BENCHMARK = True         # Auto-tune conv algorithms for +15-20% throughput

# Ampere (RTX 3060) speedups
ALLOW_TF32 = True
MATMUL_PRECISION = 'high'

# =========================================================================
# Loss hyperparameters
# =========================================================================
FOCAL_ALPHA   = 0.25  # Per paper: standard RetinaNet α=0.25, γ=2 (was 0.90 during debugging)
FOCAL_GAMMA   = 2.0

# Per-class alpha for detection focal loss.
# Mechanism: higher α = stronger positive gradient when class IS target,
# WEAKER negative gradient when class is NOT target. Dominant classes need
# LOW α to learn suppression on non-target anchors (1-α penalty).
# Format: {model_class_idx: alpha}. Classes not listed default to FOCAL_ALPHA (0.90).
DET_CLASS_ALPHAS = {
        # [FIX 2026-06-20 v4] Corrected model-index mapping.
        # v3 had critical errors: idx 7,12,17 (AP=0.7/0.7/0.4) given HIGH alpha
        # as "stuck"; idx 20 (AP=0.0, 709 train GT) given LOW alpha as "already
        # perfect"; idx 9 (AP=0.004) labeled as "dominant" with LOW alpha.
        #
        # HIGH alpha = stronger pos gradient, weaker neg. Use for stuck classes.
        # LOW alpha = moderate pos, meaningful neg. Use for dominant classes.
        # Default FOCAL_ALPHA=0.25 (per paper §Implementation).
        #
        # === Truly stuck (AP=0.0, significant train GT) --- HIGH alpha ===
        20: 0.96,  # cat 21 '11101011110', train=709, val=91,  AP=0.000
        18: 0.95,  # cat 19 '11100001110', train=340, val=47,  AP=0.000
        8:  0.94,  # cat 9  '11110110000', train=142, val=20,  AP=0.000
        9:  0.94,  # cat 10 '11110111100', train=427, val=88,  AP=0.004
        # === Stuck but low train count --- moderate alpha ===
        6:  0.92,  # cat 7  '11110010000', train=65,  val=91,  AP=0.000
        16: 0.90,  # cat 17 '11110011110', train=26,  val=27,  AP=0.000
        # === Marginal (0.05 < AP < 0.20) --- moderate boost ===
        22: 0.92,  # cat 23 '11101111111', train=2000, val=249, AP=0.159
        11: 0.92,  # cat 12 '11110110001', train=226,  val=68,  AP=0.095
        # === Dominant (AP > 0.4) --- LOW alpha to improve suppression ===
        10: 0.78,  # cat 11 '11110111110', train=1913, val=156, AP=0.807
        7:  0.80,  # cat 8  '11110100000', train=1852, val=223, AP=0.704
        12: 0.80,  # cat 13 '11110111101', train=1136, val=430, AP=0.676
        17: 0.85,  # cat 18 '11110101110', train=1067, val=263, AP=0.433
        # === Moderate performer --- slight low alpha ===
        4:  0.88,  # cat 5  '10010110000', train=590,  val=246, AP=0.270
        # === AP=1.0 artifact (no GT in 0.50 subset val) ===
        21: 0.85,  # cat 22 '11101111110', train=561,  val=175, AP=1.000
        # === Zero val GT (can't measure) --- default-adjacent ===
        2:  0.88,  # cat 3  '10010010000', train=349, val=0
        14: 0.88,  # cat 15 '11110101111', train=126, val=0
        # Zero train GT (can't learn): idx 13/19/23 --- default
        # Zero val GT (can't measure): idx 1/3/15 --- default
    }
GIOU_WEIGHT   = 2.0  # Doc 01 B.2: GIoU regression weight vs cls weight=1.0

# Hard negative mining for detection FocalLoss (RF stage collapse fix)
# With ~0.01% positive anchors (~20/173K per image), the cumulative gradient
# from negatives dominates → cls logits drift to -16 over ~850 steps → collapse.
# OHEM keeps only the top-K hardest negatives per image (K = num_pos * RATIO),
# breaking the cumulative negative gradient cycle. Ref: RetinaNet OHEM ablation.
DET_OHEM_ENABLED = True
DET_OHEM_RATIO = 2.0     # [FIX 2026-06-19 v2] was 5.0 — 5:1 was too aggressive, suppressed all predictions.
                         # 2:1 + gamma_neg=1.5 gives ~3.5× more negative gradient — enough to break stale
                         # equilibrium without overwhelming positives.
DET_OHEM_MIN_NEG = 32    # [FIX 2026-06-19 v2] was 128 — 128 negatives dominated the gradient in low-pos batches.
                         # 32 provides diverse negatives without overwhelming the positive signal.

# Asymmetric focal gamma: prevent cls_mean collapse by keeping positive gradient alive
# With gamma=2 on both pos/neg, well-classified positives (p≈0.9) have near-zero gradient:
#   (1-p)^2 * CE = 0.01 * 0.105 = 0.001 per anchor → neg gradient dominates → collapse.
# With gamma_pos=0 (no suppression):
#   positives: 1 * CE = 0.105 per anchor (75× more gradient)
#   negatives: p^2 * CE (gamma=2, standard) — only OHEM-kept hard negatives contribute
DET_ASYMMETRIC_GAMMA = True     # enable per-class gamma (pos vs neg)
DET_GAMMA_POS = 0.0             # no gamma suppression for positives
DET_GAMMA_NEG = 1.5             # [FIX 2026-06-19 v2] was 1.0 — v1 fix at 1.0 was too aggressive.
                                 # At p=0.074, gamma=1.0 gives 0.074 effective weight per negative (13.5× increase),
                                 # which was excessive with RATIO=5/MIN_NEG=128 (67.5× total gradient) → suppressed
                                 # all predictions. gamma=1.5 gives p^0.5=0.27 → 3.5× moderate increase, enough
                                 # to break equilibrium without overwhelming positives. Paired with RATIO=2/MIN_NEG=32.

WING_OMEGA   = 0.05
WING_EPSILON = 0.005

# Per paper §3.7.1: activity uses CrossEntropyLoss with label smoothing (not CB-Focal).
# CB_BETA and CB_GAMMA are unused in the CE path — kept for LDAM-DRW ablation compat.
CB_BETA  = 0.99
CB_GAMMA = 1.0
CB_LABEL_SMOOTHING = 0.1  # label smoothing for 74-class activity CE loss (paper §3.7.1)

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
REINIT_PI = 0.01  # cls_score bias prior for reinit (RF1 uses 0.05)
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
POSE_LOSS_WEIGHT = 5.0      # [FIX 2026-06-18] Increased from 0.01 to compensate for coordinate
                            # normalization. Raw Wing loss dropped from ~267 (pixel-space mismatch)
                            # to ~0.05-0.2 ([0,1] normalized space). At 5.0, contribution ≈ 0.25-1.0,
                            # comparable to pre-fix contribution of ~2.67. Tune based on training.

# [FIX 2026-06-18] Soft-argmax temperature for gradient flow
# Low temperature (0.07-0.1) gives precise coordinates but kills gradient through
# the softmax — weights become one-hot, so d(coords)/d(logits) ≈ 0 everywhere.
# Higher temperature (1.0) distributes softmax probability across more pixels,
# allowing Wing loss gradients to flow back through soft-argmax into heatmap_head
# conv layers. Lower eval temperature preserves coordinate precision at inference.
SOFT_ARGMAX_TEMPERATURE = 0.07    # Per paper §5.2: "Soft-argmax (T=0.07)" (was 0.1 during debugging)
SOFT_ARGMAX_TEMP_TRAIN = 1.0      # Training temperature for gradient flow

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
ACTIVITY_HEAD_GRAD_CLIP = 1.0  # [FIX 2026-06-30] Raised from 0.3 — gradient norm at 0.012 is well below even 0.3;
                                # clip was not constraining anything. Raising to 1.0 removes the ceiling so
                                # when activity does escape the degenerate equilibrium, it isn't capped.
ACTIVITY_LR_MULTIPLIER = 20.0   # [FIX 2026-06-30 v3] Raised from 10.0 — even 10x with gradient
                                # centralization didn't break the 1/75 class degenerate equilibrium
                                # because the gradient norm is too small (0.01). At 20x, 1.0e-2 lr ×
                                # 0.01 = 1.0e-4 update/step. Combined with reinitialized classifier
                                # weights and gradient centralization, this should be sufficient to
                                # push the head out of the degenerate attractor.
ACTIVITY_LOSS_WEIGHT = 0.8     # Per paper: "CE (label_smooth=0.1) × 0.8" (was 0.2 during debugging)

# [FIX 2026-06-30 Opus consult] ACTIVITY_HEAD_SIMPLE — bypass the TCN+2xViT
# temporal stack and classify the projected per-frame feature directly with a
# small MLP. Root cause of the activity collapse: training uses a class-balanced
# WeightedRandomSampler (per-frame, shuffled), so the FeatureBank ring buffer is
# fed NON-CONSECUTIVE frames from random videos — there is no real temporal
# signal. The 8.2M-param ViT+TCN head therefore (a) learns from noise, (b) adds
# huge overfitting capacity on a 3.7k-frame dataset, and (c) makes the live frame
# 1-of-17 tokens behind two self-attention blocks, attenuating its gradient. A
# direct MLP gives a strong, short gradient path and ~150K params. Re-enable the
# temporal stack only when training on true sequence batches (sequence_mode).
ACTIVITY_HEAD_SIMPLE = True
ACTIVITY_HEAD_SIMPLE_HIDDEN = 256  # hidden width of the simple MLP classifier

# Per paper: "ℒ_hp = MSE × 5.0" — explicit multiplier for head pose loss
HEAD_POSE_LOSS_WEIGHT = 5.0

# Per paper §5.4: gradient scaling blend for activity projection.
# Blend ratio: 0.05·C5_mod2 + 0.95·detach(C5_mod2) lets 5% gradient through.
# Set to 0.0 for full detach (debugging), 1.0 for full gradient flow (ablations).
# [FIX 2026-06-29 v2] Raised from 0.10 to 0.30 for RF4 activity collapse fix.
# Activity collapse to 3/75 classes diagnosed as insufficient backbone gradient.
# At 0.30, 3x more activity signal reaches backbone through c5_mod_blend
# to shape discriminative features, while 70% detach still protects FPN.
# Tradeoff: backbone drift risk mitigated by ACTIVITY_HEAD_GRAD_CLIP=1.0.
# [FIX 2026-06-29 v3] Raised from 0.50 to 0.70 — 3 epochs of training showed
# minimal activity recovery (4/75 classes). At 0.70, ~parity with detection gradient.
# [FIX 2026-06-30] Raised from 0.70 to 1.0 — 3 epochs of RF4 showed activity
# gradient still 0.012 (30x below detection). At 1.0, full gradient flows through
# c5_mod_blend into the backbone, giving activity the strongest possible signal
# to shape discriminative features. Risk of backbone drift is managed by
# ACTIVITY_HEAD_GRAD_CLIP=1.0 and ACTIVITY_LR_MULTIPLIER=3.0 (head-level, not backbone).
ACTIVITY_GRAD_BLEND_RATIO = 1.00

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
# [FIX 2026-06-29 v2] Reduced from 2.0 to 1.0 — PSR head shows NO_GRAD across all
# epochs because gamma=2.0 kills gradients for near-0.5 predictions (all PSR logits
# are in [-0.7, 0.7]). Lower gamma = stronger gradient signal to break the all-ones equilibrium.
# [FIX 2026-06-30 v3] Reduced from 1.0 to 0.5 — PSR head still shows zero F1.
# Gamma=1.0 was insufficient to break the all-ones/zero-gradient equilibrium.
# At gamma=0.5, gradient magnitude roughly doubles for near-0.5 predictions.
PSR_FOCAL_GAMMA = 0.5  # Per paper §3.6: "Binary Focal Loss (α=0.25, γ=2.0)"

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
# [FIX 2026-06-30 v4] Use DETACH_GRAD_ENTRIES_ONLY instead of FEATURE_BANK_DETACH=False.
# False causes "backward through graph a second time" crash (#3092789) because bank
# entries are shared across frames. The intermediate option detaches stored entries
# but keeps gradient on the current frame, giving activity some temporal gradient path
# without crashing.
FEATURE_BANK_DETACH = True  # True = legacy behavior (no gradient through bank)
FEATURE_BANK_DETACH_GRAD_ENTRIES_ONLY = True  # [FIX 2026-06-30] detach stored entries, keep current frame grad.
                                                # E2: intermediate between FEATURE_BANK_DETACH=True and False.
                                                # Current frame keeps gradient; historical entries are detached.
FEATURE_BANK_SLOT_OVERWRITE = True  # True = legacy: live frame overwrites slot -1
                                     # False = bank accumulates without overwrite (#16)

# Fix 1 (2026-06-06): penalize constant per-frame PSR predictions in T=1 mode.
# Stage 3 epoch 16 collapsed logit std to 0.12%; this penalty keeps it > 1e-3.
# [R2.5 FIX 2026-06-14] Re-enabled with correction=0 on std() — NaN root cause fixed.
# With correction=0, batch_size=1 gives std=0 → -log(0+1e-3)=6.9 (finite). The
# torch.isfinite() guard catches any edge case. Verified: 0 NaN events in 1100+ steps.
# [FIX 2026-06-29 v2] Raised from 0.01 to 0.10 — PSR head produces identical logits
# (near 0.5) for all components. Sensitivity penalty pushes per-component means apart
# so sigmoid threshold (0.5) actually separates placed/unplaced components.
# At 0.01: ~0.046 loss contribution at std=0 → too weak. At 0.10: ~0.46 → meaningful gradient.
# [FIX 2026-06-30 v3] Raised from 0.10 to 0.50 — still zero PSR F1 at 0.10.
# Need stronger penalty to push per-component logits apart from the near-0.5 equilibrium.
# At 0.50, the sensitivity penalty (~0.46 standard dev loss) becomes a significant
# gradient signal that actively drives component logits toward opposite sigmoid extremes.
PSR_SENSITIVITY_WEIGHT = 0.50

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
ONE_CYCLE_LR = True      # Per paper §Implementation: "Warmup (2 ep) → OneCycleLR"
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
DET_METRICS_EVERY_N = int(os.environ.get('DET_METRICS_EVERY_N', '1'))  # Full mAP eval every N epochs; 0=every epoch
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
        'use_psr_transition': False,  # Must match train_psr — MonotonicDecoder crashes on empty PSR data
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
        'feature_bank_slot_overwrite': True,
        'use_psr_order_prior':      True,
        'psr_sensitivity_weight':   0.01,  # Was 0.0 — re-enabled after std(correction=0) NaN fix
        'use_ldam_drw':             False,
        'use_videomae':             False,  # KEEP OFF — RTX 3060 12GB FP32 OOM risk
        # [FIX 2026-06-21 Opus v11 Q5] detach_reg_fpn=False for ALL non-reinit stages.
        # These RF3–RF10 / paper_run stages are continuations, NOT reinit bootstraps:
        # the regression head already carries good GIoU signal that SHOULD shape the
        # shared FPN. detach=True severs the densest detection gradient from the trunk
        # → features stay non-discriminative → cls sticks at the background equilibrium
        # (the exact RF2 6-epoch plateau). The reg-loss warmup (REINIT_REG_WARMUP_STEPS)
        # is the correct guard against any reinit gradient shock; detach is redundant
        # overkill — matches stage_rf1 + the stage_manager recovery strategy (108–121).
        # detach_psr_fpn is left True (PSR is a separate head, out of scope for v11).
        'detach_reg_fpn':           False,
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
        # [RF1 FIX 2026-06-18] Disable RandAugment for RF1 bootstrap stage.
        # With only 7 recordings and ~16 positive anchors per batch, every
        # positive training example is RandAugment-distorted (num_ops=2,
        # magnitude=9). The detection head learns features specific to the
        # augmented training images that don't fire on clean validation images,
        # causing det_mAP50 ≈ 0.01 even after 10 epochs while training
        # DET-HEALTH shows cls_max=15.6. Disable for RF1; re-evaluate for RF2+.
        'use_randaugment':          False,
        # [RF1 FIX 2026-06-18] Disable spatial augmentation (horizontal flip +
        # random crop+resize) for RF1 bootstrap stage. Training applies
        # apply_spatial_aug() via industreal_dataset.py:886 when augment=True,
        # but validation has augment=False. Same root cause as RandAugment:
        # the detection head learns features on cropped/flipped views that
        # don't fire on clean validation images. Disable for RF1 so training
        # and validation see pixel-identical images; re-evaluate for RF2+.
        'use_spatial_aug':          False,
        # [RF1 FIX 2026-06-18] cls_score bias init pi prior for reinit.
        # Default is 0.01 (bias=-4.60), but RF1 bootstrap benefits from a
        # higher pi=0.05 (bias=-2.94) so the model starts closer to reasonable
        # confidence, accelerating the escape from background equilibrium.
        'reinit_pi':                0.05,
    },
    'stage_rf2': {
        'description': 'RF2: Detection + Body+Head Pose (35% data, 15 ep).',
        'dataset_mode':       'manual_only',
        'backbone':           'convnext_tiny',
        'use_tma_cell':       True,
        'use_temporal_bank':  True,
        'use_hand_film':      True,
        'benchmark_mode':     False,
        'batch_size':         8,       # Sweet spot — 16GB VRAM
        'grad_accum_steps':   8,
        'zero_det_conf':      False,
        'staged_training':    False,
        'mixed_precision':    False,   # [FIX 2026-06-19] PSR forward pass still runs (train_psr=False doesn't skip it)
                                        # and PSR ops produce FP32 tensors that crash autocast backward.
                                        # Stay FP32 for now; debug PSR dtype compatibility separately for perf gain.
        'use_mixup':          False,
        'use_randaugment':    True,    # Explicit: photometric aug for backbone generalization
        'use_spatial_aug':    True,    # Explicit: horizontal flip + random crop
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
        # [FIX 2026-06-21] Was True (detached). RF2 is NOT a reinit stage — the
        # regression head already has decent GIoU signal that SHOULD flow into FPN.
        # detach=True cuts reg gradients (the strongest detection signal) from FPN,
        # leaving only tiny cls gradient (norm=0.01) and head_pose to update features.
        # False lets regression GIoU signal help FPN → cls escapes equilibrium.
        'detach_reg_fpn':           False,
        'detach_psr_fpn':           True,
        # [FIX 2026-06-19 v2] RF2 reinit also needs pi=0.05 for same reason as RF1 —
        # default 0.01 (bias=-4.60) starts too cold; 0.05 (bias=-2.94) accelerates
        # escape from background equilibrium after head reinit.
        'reinit_pi':                0.05,
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
        # [FIX 2026-06-28 20-agent] Enable slot overwrite so current-frame
        # proj_feat gradient (5% through c5_mod_blend → backbone) reaches
        # TCN+ViT+classifier. Without this, activity can never adapt features.
        'feature_bank_slot_overwrite': True,
        'use_psr_order_prior':      False,
        'psr_sensitivity_weight':   0.0,
        'use_ldam_drw':             False,
        # [FIX 2026-06-21 Opus v11 Q5] detach_reg_fpn=False for ALL non-reinit stages.
        # These RF3–RF10 / paper_run stages are continuations, NOT reinit bootstraps:
        # the regression head already carries good GIoU signal that SHOULD shape the
        # shared FPN. detach=True severs the densest detection gradient from the trunk
        # → features stay non-discriminative → cls sticks at the background equilibrium
        # (the exact RF2 6-epoch plateau). The reg-loss warmup (REINIT_REG_WARMUP_STEPS)
        # is the correct guard against any reinit gradient shock; detach is redundant
        # overkill — matches stage_rf1 + the stage_manager recovery strategy (108–121).
        # detach_psr_fpn is left True (PSR is a separate head, out of scope for v11).
        'detach_reg_fpn':           False,
        'detach_psr_fpn':           True,
    },
    'ablation_single_task': {
        'description': 'Ablation A: Single-task baseline matching RF3 except act/PSR off.',
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
        'train_act':          False,  # ← ONLY DIFFERENCE from stage_rf3
        'train_psr':          False,  # ← ONLY DIFFERENCE from stage_rf3
        'train_head_pose':    True,
        'use_psr_transition':       False,
        'use_geo_head_pose':        True,
        'feature_bank_detach':      True,
        'feature_bank_slot_overwrite': True,
        'use_psr_order_prior':      False,
        'psr_sensitivity_weight':   0.0,
        'use_ldam_drw':             False,
        'detach_reg_fpn':           False,
        'detach_psr_fpn':           True,
        # Explicitly match RF3's GT frame fraction (0.4), not the detection-only
        # default (0.9) — we want a clean ablation, not recovery bootstrapping.
        'det_gt_frame_fraction':    0.4,
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
        'feature_bank_detach_grad_entries_only': True,
        'feature_bank_slot_overwrite': True,
        'use_psr_order_prior':      True,
        'psr_sensitivity_weight':   0.50,
        'use_ldam_drw':             False,
        # [FIX 2026-06-21 Opus v11 Q5] detach_reg_fpn=False for ALL non-reinit stages.
        # These RF3–RF10 / paper_run stages are continuations, NOT reinit bootstraps:
        # the regression head already carries good GIoU signal that SHOULD shape the
        # shared FPN. detach=True severs the densest detection gradient from the trunk
        # → features stay non-discriminative → cls sticks at the background equilibrium
        # (the exact RF2 6-epoch plateau). The reg-loss warmup (REINIT_REG_WARMUP_STEPS)
        # is the correct guard against any reinit gradient shock; detach is redundant
        # overkill — matches stage_rf1 + the stage_manager recovery strategy (108–121).
        # detach_psr_fpn is left True (PSR is a separate head, out of scope for v11).
        'detach_reg_fpn':           False,
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
        # [FIX 2026-06-28 RF4 audit] Re-enable slot_overwrite so activity gradient
        # reaches the backbone through proj_feat → c5_mod_blend (10% blend ratio).
        # RF4 turned this off (fear of temporal bias from always-overwriting slot -1),
        # but the consequence was zero backbone gradient for activity → accuracy cap.
        # The TCN kernel_size=5 learns temporal patterns regardless of which slot
        # holds the current frame — the other 7 positions carry temporal history.
        'feature_bank_slot_overwrite': True,
        'use_psr_order_prior':      True,
        'psr_sensitivity_weight':   0.01,
        'use_ldam_drw':             False,
        # [FIX 2026-06-21 Opus v11 Q5] detach_reg_fpn=False for ALL non-reinit stages.
        # These RF3–RF10 / paper_run stages are continuations, NOT reinit bootstraps:
        # the regression head already carries good GIoU signal that SHOULD shape the
        # shared FPN. detach=True severs the densest detection gradient from the trunk
        # → features stay non-discriminative → cls sticks at the background equilibrium
        # (the exact RF2 6-epoch plateau). The reg-loss warmup (REINIT_REG_WARMUP_STEPS)
        # is the correct guard against any reinit gradient shock; detach is redundant
        # overkill — matches stage_rf1 + the stage_manager recovery strategy (108–121).
        # detach_psr_fpn is left True (PSR is a separate head, out of scope for v11).
        'detach_reg_fpn':           False,
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
        # [FIX 2026-06-28 RF4 audit] Re-enable slot_overwrite so activity gradient
        # reaches the backbone through proj_feat → c5_mod_blend (10% blend ratio).
        'feature_bank_slot_overwrite': True,
        'use_psr_order_prior':      True,
        'psr_sensitivity_weight':   0.01,
        'use_ldam_drw':             False,
        # [FIX 2026-06-21 Opus v11 Q5] detach_reg_fpn=False for ALL non-reinit stages.
        # These RF3–RF10 / paper_run stages are continuations, NOT reinit bootstraps:
        # the regression head already carries good GIoU signal that SHOULD shape the
        # shared FPN. detach=True severs the densest detection gradient from the trunk
        # → features stay non-discriminative → cls sticks at the background equilibrium
        # (the exact RF2 6-epoch plateau). The reg-loss warmup (REINIT_REG_WARMUP_STEPS)
        # is the correct guard against any reinit gradient shock; detach is redundant
        # overkill — matches stage_rf1 + the stage_manager recovery strategy (108–121).
        # detach_psr_fpn is left True (PSR is a separate head, out of scope for v11).
        'detach_reg_fpn':           False,
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
        'feature_bank_slot_overwrite': True,
        'use_psr_order_prior':      True,
        'psr_sensitivity_weight':   0.01,
        'use_ldam_drw':             False,
        # [FIX 2026-06-21 Opus v11 Q5] detach_reg_fpn=False for ALL non-reinit stages.
        # These RF3–RF10 / paper_run stages are continuations, NOT reinit bootstraps:
        # the regression head already carries good GIoU signal that SHOULD shape the
        # shared FPN. detach=True severs the densest detection gradient from the trunk
        # → features stay non-discriminative → cls sticks at the background equilibrium
        # (the exact RF2 6-epoch plateau). The reg-loss warmup (REINIT_REG_WARMUP_STEPS)
        # is the correct guard against any reinit gradient shock; detach is redundant
        # overkill — matches stage_rf1 + the stage_manager recovery strategy (108–121).
        # detach_psr_fpn is left True (PSR is a separate head, out of scope for v11).
        'detach_reg_fpn':           False,
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
        'feature_bank_slot_overwrite': True,
        'use_psr_order_prior':      True,
        'psr_sensitivity_weight':   0.01,
        'use_ldam_drw':             False,
        # [FIX 2026-06-21 Opus v11 Q5] detach_reg_fpn=False for ALL non-reinit stages.
        # These RF3–RF10 / paper_run stages are continuations, NOT reinit bootstraps:
        # the regression head already carries good GIoU signal that SHOULD shape the
        # shared FPN. detach=True severs the densest detection gradient from the trunk
        # → features stay non-discriminative → cls sticks at the background equilibrium
        # (the exact RF2 6-epoch plateau). The reg-loss warmup (REINIT_REG_WARMUP_STEPS)
        # is the correct guard against any reinit gradient shock; detach is redundant
        # overkill — matches stage_rf1 + the stage_manager recovery strategy (108–121).
        # detach_psr_fpn is left True (PSR is a separate head, out of scope for v11).
        'detach_reg_fpn':           False,
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
        'feature_bank_slot_overwrite': True,
        'use_psr_order_prior':      True,
        'psr_sensitivity_weight':   0.01,
        'use_ldam_drw':             False,
        # [FIX 2026-06-21 Opus v11 Q5] detach_reg_fpn=False for ALL non-reinit stages.
        # These RF3–RF10 / paper_run stages are continuations, NOT reinit bootstraps:
        # the regression head already carries good GIoU signal that SHOULD shape the
        # shared FPN. detach=True severs the densest detection gradient from the trunk
        # → features stay non-discriminative → cls sticks at the background equilibrium
        # (the exact RF2 6-epoch plateau). The reg-loss warmup (REINIT_REG_WARMUP_STEPS)
        # is the correct guard against any reinit gradient shock; detach is redundant
        # overkill — matches stage_rf1 + the stage_manager recovery strategy (108–121).
        # detach_psr_fpn is left True (PSR is a separate head, out of scope for v11).
        'detach_reg_fpn':           False,
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
        'feature_bank_slot_overwrite': True,
        'use_psr_order_prior':      True,
        'psr_sensitivity_weight':   0.01,
        'use_ldam_drw':             False,
        # [FIX 2026-06-21 Opus v11 Q5] Was implicit (no key → module default). Make it
        # explicit: non-reinit final stage → reg gradient SHOULD shape the FPN. See stage_rf2.
        'detach_reg_fpn':           False,
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
    global USE_RANDAUGMENT, USE_SPATIAL_AUG
    global DETACH_REG_FPN, DETACH_PSR_FPN
    global REINIT_PI

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
    USE_RANDAUGMENT = preset.get('use_randaugment', USE_RANDAUGMENT)
    USE_SPATIAL_AUG = preset.get('use_spatial_aug', USE_SPATIAL_AUG)
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
    REINIT_PI = float(preset.get('reinit_pi', REINIT_PI))
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
