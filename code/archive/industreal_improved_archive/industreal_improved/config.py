"""
Configuration for POPW multi-task model on IndustReal dataset.
Hyperparameters derived from popw_paper.tex architecture specification.
"""

from dataclasses import dataclass, field
from typing import List, Tuple
import os


@dataclass
class Config:
    # ==================== DATASET ====================
    DATA_ROOT: str = "/home/newadmin/swarm-bot/project/popw/working/data/datasets/industreal"
    SPLITS_DIR: str = os.path.join(DATA_ROOT, "splits")
    RECORDINGS_DIR: str = os.path.join(DATA_ROOT, "recordings")

    # Image settings
    IMG_HEIGHT: int = 720
    IMG_WIDTH: int = 1280

    # Dataset splits
    TRAIN_SPLIT: str = os.path.join(SPLITS_DIR, "train.csv")
    VAL_SPLIT: str = os.path.join(SPLITS_DIR, "val.csv")
    TEST_SPLIT: str = os.path.join(SPLITS_DIR, "test.csv")

    # ==================== TASKS ====================
    NUM_CLASSES_DET: int = 24  # ASD classes
    NUM_CLASSES_ACT: int = 74  # Activity classes
    NUM_CLASSES_PSR: int = 11  # PSR components
    NUM_JOINTS: int = 17  # Body keypoints (for IKEA ASM, not IndustReal)
    HEAD_POSE_DIM: int = 9  # 9-DoF head pose (forward, position, up)

    # Task flags
    TRAIN_HEAD_POSE: bool = False  # IndustReal has no body keypoints, only head pose
    TRAIN_BODY_POSE: bool = False  # Body pose not in IndustReal
    USE_VIDEOMAE: bool = True  # Enable VideoMAE stream for activity

    # ==================== BACKBONE ====================
    BACKBONE: str = "convnext_tiny"  # ConvNeXt-Tiny pretrained on ImageNet
    BACKBONEpretrained: bool = True
    BACKBONE_channels: Tuple[int, int, int, int] = (96, 192, 384, 768)

    # FPN
    FPN_CHANNELS: int = 256
    FPN_P6_STRIDE: int = 2
    FPN_P7_STRIDE: int = 2

    # ==================== ANCHORS (calibrated from k-means) ====================
    ANCHOR_SIZES: List[int] = field(default_factory=lambda: [24, 48, 96, 192, 384])
    ANCHOR_RATIOS: List[float] = field(default_factory=lambda: [0.5, 1.0, 2.0])
    ANCHOR_SCALES: List[int] = field(default_factory=lambda: [24, 48, 96, 192, 384])

    # ==================== DETECTION HEAD ====================
    DET_NUM_CONV: int = 4
    DET_CONV_DIM: int = 256
    DET_IOU_THRESH: float = 0.5
    DET_CONF_THRESH: float = 0.05
    DET_NMS_IOU_THRESH: float = 0.5
    DET_MAX_DETECTIONS: int = 100

    # Detection loss
    FOCAL_ALPHA: float = 0.25
    FOCAL_GAMMA: float = 2.0
    GIOU_WEIGHT: float = 1.0

    # ==================== POSE HEAD (body keypoints) ====================
    HEATMAP_SIZE: Tuple[int, int] = (180, 320)  # H/4, W/4
    WING_LOSS_W: float = 0.05
    WING_LOSS_EPS: float = 0.005
    SOFTMAX_TEMP: float = 0.1

    # ==================== HEAD POSE HEAD ====================
    HEAD_POSE_MLP_DIMS: List[int] = field(default_factory=lambda: [1152, 512, 256, 9])
    HEAD_POSE_LOSS_SCALE: float = 0.001

    # ==================== ACTIVITY HEAD ====================
    ACT_DET_CONTEXT_DIM: int = 24
    ACT_SPATIAL_DIM: int = 768 + 256  # C5_mod2 GAP + P4 GAP
    ACT_JOINT_DIM: int = ACT_DET_CONTEXT_DIM + ACT_SPATIAL_DIM  # 1048
    ACT_PROJECTION_DIM: int = 512
    ACT_FEATURE_BANK_LEN: int = 16
    ACT_TCN_KERNEL: int = 5
    ACT_TCN_DILATION: int = 1
    ACT_VIT_NUM_LAYERS: int = 2
    ACT_VIT_NUM_HEADS: int = 8
    ACT_VIT_D_K: int = 64
    ACT_VIT_FFN_DIM: int = 2048
    ACT_VIT_DROPOUT: float = 0.1
    ACT_DROPPATH: List[float] = field(default_factory=lambda: [0.1, 0.15])
    ACT_NUM_CLASSES: int = 74

    # LDAM-DRW
    LDAM_DRW_EPOCH: int = 35  # Activate DRW at epoch 35 (recommended from Doc 01)
    LDAM_LABEL_SMOOTH: float = 0.1

    # Temporal augmentation
    TEMPORAL_STRIDE_MIN: int = 2
    TEMPORAL_STRIDE_MAX: int = 5

    # ==================== PSR HEAD ====================
    PSR_INPUT_DIM: int = 768  # Multi-scale concat: GAP(P3)+GAP(P4)+GAP(P5)
    PSR_MLP_DIMS: List[int] = field(default_factory=lambda: [768, 256, 128])
    PSR_TRANSFORMER_LAYERS: int = 3
    PSR_TRANSFORMER_HEADS: int = 4
    PSR_TRANSFORMER_DMODEL: int = 128
    PSR_NUM_COMPONENTS: int = 11
    PSR_MAX_CACHE_LEN: int = 32

    # PSR loss
    PSR_FOCAL_ALPHA: float = 0.25
    PSR_FOCAL_GAMMA: float = 2.0
    PSR_SMOOTHNESS_WEIGHT: float = 0.05

    # ==================== FILM CONDITIONING ====================
    POSE_FILM_INPUT_DIM: int = 51  # 34 keypoints + 17 confidence
    POSE_FILM_HIDDEN_DIM: int = 512
    POSE_FILM_OUTPUT_DIM: int = 768  # Match C5 channels

    HEAD_POSE_FILM_INPUT_DIM: int = 9
    HEAD_POSE_FILM_HIDDEN_DIM: int = 256
    HEAD_POSE_FILM_OUTPUT_DIM: int = 768

    # ==================== TRAINING ====================
    MAX_EPOCHS: int = 100
    BATCH_SIZE: int = 2
    GRAD_ACCUM_STEPS: int = 16
    NUM_WORKERS: int = 0

    # Optimizer
    OPTIMIZER: str = "lion"
    BASE_LR: float = 1e-4
    WEIGHT_DECAY: float = 0.05

    # LR schedule
    LR_SCHEDULE: str = "cosine"
    WARMUP_EPOCHS: int = 3

    # EMA
    USE_EMA: bool = True
    EMA_DECAY: float = 0.999

    # Staged training
    STAGED_TRAINING: bool = True
    STAGE1_EPOCHS: int = 5  # Detection only
    STAGE2_EPOCHS: int = 10  # + Pose + Head Pose

    # Kendall uncertainty
    KENDALL_INIT_DET: float = 0.0
    KENDALL_INIT_POSE: float = -1.0
    KENDALL_INIT_ACT: float = 0.0
    KENDALL_INIT_PSR: float = 0.0
    KENDALL_S_MIN: float = -4.0
    KENDALL_S_MAX: float = 2.0

    # Activity ramp (starts at epoch/5, max 1.0)
    ACTIVITY_RAMP_START: int = 1
    ACTIVITY_RAMP_END: int = 5

    # Precursor training
    PRETRAIN_DET_ON_SYNTH: bool = True
    PRETRAIN_DET_EPOCHS: int = 20

    # SWA
    USE_SWA: bool = False
    SWA_EPOCHS: int = 8
    SWA_LR: float = 1e-5

    # Mixed precision
    MIXED_PRECISION: bool = False

    # Early stopping
    PATIENCE: int = 10

    # ==================== AUGMENTATION ====================
    RANDOM_FLIP: bool = True
    RANDOM_CROP: bool = False
    COLOR_JITTER: bool = True
    NORMALIZE_IMAGENET: bool = True

    # ==================== EVALUATION ====================
    EVAL_TTA: bool = False
    EVAL_FLIP_TTA: bool = False
    EVAL_CROP_TTA: bool = False

    # PSR tolerances
    PSR_TOLERANCE_FRAMES: List[int] = field(default_factory=lambda: [3, 5])

    # ==================== PATHS ====================
    RUN_DIR: str = "/home/newadmin/swarm-bot/project/popw/working/code/industreal_improved/runs"
    CHECKPOINT_DIR: str = os.path.join(RUN_DIR, "checkpoints")
    LOG_DIR: str = os.path.join(RUN_DIR, "logs")

    # ==================== RANDOM ====================
    SEED: int = 42

    # ==================== VIDEO ====================
    VIDEO_FPS: int = 10
    CLIP_FRAMES: int = 16

    def __post_init__(self):
        os.makedirs(self.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(self.LOG_DIR, exist_ok=True)


# Global config instance
C = Config()