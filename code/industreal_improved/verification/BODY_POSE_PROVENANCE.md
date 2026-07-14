# Body Pose Provenance (Q38)

## Pseudo-Keypoint Generation

| Step | File | Line | Detail |
|------|------|------|--------|
| Raw data source | `src/data/industreal_dataset.py` | 640--666 | `_parse_hands()` loads `hands.csv` (52-D per frame: 26 left-hand joints x 2 coords, MediaPipe landmarks) |
| Per-frame load | `src/data/industreal_dataset.py` | 944--945 | `hand_joints = torch.from_numpy(cache.hands[frame_num]).float()` |
| Collate: reshape | `src/data/industreal_dataset.py` | 1877--1884 | 52-D flat -> [26, 2], select first 17 indices |
| Collate: normalize | `src/data/industreal_dataset.py` | 1893 | Normalize from pixel coords to [0,1] via `C.IMG_WIDTH` and `C.IMG_HEIGHT` |
| Collate: confidence | `src/data/industreal_dataset.py` | 1894 | `pose_confidence = torch.ones(...)` (all joints treated as observed) |
| Target dict | `src/data/industreal_dataset.py` | 1900--1911 | `"keypoints"` and `"pose_confidence"` set in `targets` dict |
| Null check | `src/data/industreal_dataset.py` | 955 | `torch.zeros((17, 2))` with comment `# No keypoints in IndustReal` |

**Key finding:** The "pseudo-keypoints" are MediaPipe left-hand landmarks from `hands.csv`, down-sampled from 26 to 17 joints and normalized to [0,1]. They are NOT derived from detection bounding boxes (contradicts the paper text at `paper/main.tex:99`). The dataset contains no ground-truth COCO-style body keypoints.

## Body Pose Loss

| Component | File | Line | Detail |
|-----------|------|------|--------|
| `WingLoss` class | `src/training/losses.py` | 516--542 | omega=0.05, epsilon=0.005 |
| `PoseLoss` class | `src/training/losses.py` | 545--577 | WingLoss + 0.1 * confidence L2 |
| Pose loss block | `src/training/losses.py` | 1492--1508 | Condition: `self.train_pose and "keypoints" in targets` |
| Loss weight | `src/training/losses.py` | 1504--1506 | `C.POSE_LOSS_WEIGHT` (default 5.0 per `src/config.py:1059`) |
| Loss cap | `src/training/losses.py` | 1511--1512 | `C.POSE_LOSS_CAP` (default 30.0) |
| Frozen guard | `src/training/losses.py` | 1516--1517 | zeros loss when `FREEZE_BODY_POSE_BRANCH=True` |
| Kendall total | `src/training/losses.py` | 2042--2057 | `pose_contribution` includes `prec_hp * loss_pose` |

## FREEZE_BODY_POSE_BRANCH Semantics

| Aspect | File | Line | Value |
|--------|------|------|-------|
| Default | `src/config.py` | 80--83 | `False` |
| Model freeze | `src/models/model.py` | 2085--2089 | Sets `pose_head.parameters()` `requires_grad=False` |
| Loss zeroing | `src/training/losses.py` | 1516--1517 | Zeros `loss_pose` tensor |

**Default is `False`**: the body pose branch is active by default.

## Loss Contribution to Total

**Yes**, body pose loss contributes to the total loss when the branch is active (`FREEZE_BODY_POSE_BRANCH=False`). See `src/training/losses.py:2026-2060`: `pose_contribution = prec_hp * loss_pose + prec_hp * loss_head_pose + lv_hp` is added to `total` when `self.train_pose=True`.

However, since IndustReal has no real keypoint annotations, `loss_pose` operates on pseudo-keypoints derived from MediaPipe hand landmarks. The loss signal is real (WingLoss on valid target values) but the supervision targets are synthetic.

## Proposed Limitations Paragraph

```latex
\item \textbf{Body pose provenance.} The body pose head predicts 17 COCO-style keypoints,
but IndustReal contains no ground-truth body keypoint annotations.
Pseudo-labels are derived from MediaPipe left-hand landmarks (26 joints, down-sampled
to 17, normalized to $[0,1]$ pixel coordinates) rather than from detection bounding
boxes as previously stated. This head is trained with Wing Loss ($\omega=0.05$,
$\varepsilon=0.005$) at weight 5.0, contributing to the Kendall total alongside body pose
log-variance regularization. The pseudo-label quality and distribution mismatch
(hand vs.\ full-body keypoints) limit the head's independent value; we include it
for architectural completeness and leave genuine body pose annotation to future work.
```
