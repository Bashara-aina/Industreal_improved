# Doc 218: Data & Augmentation Strategy

**Generated:** 2026-07-11
**Scope:** IndustReal multi-task dataset (detection, activity, PSR, head pose)
**Codebase:** `industreal_improved/src/data/` + `industreal_improved/src/training/pretrain_synthetic.py`

---

## 1. Dataset Statistics

### 1.1 Splits and Frame Counts

The IndustReal dataset is split into train/val/test by distinct assembly recordings (different operators, trials, and assembly variants). Each recording is a single egocentric RGB video at 1280x720 resolution, captured at 10 FPS via HoloLens 2.

| Split | Recordings | Raw Frames | Stride=3 Frames | OD-Labeled Frames | AR-Labeled Frames |
|-------|-----------|-----------|----------------|-------------------|-------------------|
| Train | 36 | 78,931 | 26,322 | 14,122 (17.9%) | 69,189 (87.7%) |
| Val | 16 | 38,036 | 12,683 | 3,102 (8.2%) | 37,280 (98.0%) |
| Test | 32 | 90,105 | 30,047 | 9,701 (10.8%) | 81,642 (90.6%) |
| **Total** | **84** | **207,072** | **69,052** | **26,925** | **188,111** |

**Key observations:**

- **Training uses stride=3** (`TRAIN_FRAME_STRIDE = 3`), sampling every third frame. This yields ~26K training samples -- a modest dataset for multi-task learning with four heads and a ConvNeXt-Tiny backbone.
- **Validation and test use stride=1** (`EVAL_FRAME_STRIDE = 1`), evaluating on all frames for metric stability.
- **Detection annotations are sparse**: only 17.9% of train frames carry OD labels (roughly 7% after stride-3 subsampling in expectation). This ~7% density is the single largest challenge for the detection head: ~93% of training steps see zero GT boxes.
- **Activity labels are dense**: 87.7% of train frames are covered by AR_labels.csv spans. After stride-3 subsampling, activity supervision is available for nearly every training sample.
- **Pose and hands are dense**: pose.csv and hands.csv exist for essentially all frames (99.95% coverage).

### 1.2 Recording Structure

Each recording contains:

```
{recording_id}/
  rgb/000000.jpg ... 0000N.jpg   (1280x720, 10 FPS JPEG frames)
  AR_labels.csv                   (sparse: per-action spans with start/end frames)
  OD_labels.json                  (COCO format: per-frame bounding boxes with category_id)
  PSR_labels_raw.csv              (sparse: per-component state change events, fill-forward)
  pose.csv                        (dense: 9-DoF head pose per frame)
  hands.csv                       (dense: 52-D MediaPipe-style hand joints per frame)
```

The recordings are named by operator and assembly variant, e.g., `01_assy_0_1` = operator 01, assembly trial 0, take 1. Main recordings (`01_main_0_1`) cover the full assembly procedure; assy recordings cover specific sub-assemblies. Each recording is ~2,000-3,000 frames at 10 FPS (3-5 minutes).

---

## 2. Annotation Density Per Head

### 2.1 Detection (ASD): ~17.9% of Frames Have GT, 1 Box Each, 24 Classes

Detection annotations follow COCO format: each annotation has a bounding box (`[x, y, w, h]`) and a category_id (1-24). Critically, each annotated frame has **exactly one box** -- there is no multi-object per-frame annotation. The 24 classes correspond to assembly state codes (e.g., `10000000000`, `11110111111`) plus `background` (class 1) and `error_state` (class 24).

**Detection class distribution (train, 14,122 annotations):**

Class distribution is highly imbalanced. The most frequent classes:

- Class 23 (`11101111111`): 2,000 instances (14.2%)
- Class 11 (`11110111110`): 1,913 instances (13.5%)
- Class 8 (`11110100000`): 1,852 instances (13.1%)
- Class 1 (`background`): 1,639 instances (11.6%)
- Class 13 (`11110111101`): 1,136 instances (8.0%)

The tail classes:

- Class 17: 26 instances (0.18%)
- Class 16: 34 instances (0.24%)
- Class 7: 65 instances (0.46%)
- Class 2: 80 instances (0.57%)

**Two classes are entirely absent in the train split**: class 14 and class 20 have zero instances. The model can never learn to predict these from training data -- they are zero-shot classes that will always be wrong if they appear in val/test.

Each GT box covers a significant portion of the frame: median sqrt-area ~402px (at 1280x720), median height ~303px. Despite the large apparent size, the effective positive anchor count is only ~1-3 anchors per GT with standard IoU=0.5 matching, contributing to gradient starvation (addressed via `DET_POS_IOU_THRESH=0.4`, `DET_POS_IOU_TOP_K=9`, and `IOU_FLOOR=0.2`).

### 2.2 Activity (AR): 75 Classes, Power-Law Distribution, ~20K Labeled Frames (at Stride=3)

The activity task uses raw action_id from AR_labels.csv as class labels (0 = "take_short_brace", 1-74 = atomic assembly actions). Two IDs are absent (37, 64), leaving 72 active classes but a 75-wide classifier head.

**Class distribution (train, 69,189 labeled frames):**

The distribution follows a power law: 5 classes dominate, while 48 classes have fewer than 100 frames.

| Count Range | Number of Classes | Example Classes |
|-------------|------------------|-----------------|
| >5,000 | 4 | 7 (8,485), 6 (5,878), 1 (5,654), ... |
| 100-5,000 | 20 | 3 (2,897), 5 (2,351), 71 (2,362), ... |
| 10-99 | 32 | 14 (42), 73 (33), 12 (33), ... |
| <10 | 16 | 64 (11), 65 (21), 67 (19), 70 (22), 11 (24), ... |
| Absent | 2 | ID 37, ID 66 |

This extreme tail (16 classes with fewer than 10 frames, 7 of which are singletons within a stride-3 training epoch) is the root cause of the activity collapse observed historically: the classifier learns to predict the 5 head classes and ignores the tail entirely, yielding macro-F1 near zero.

The config addresses this via two mechanisms:
- **ACT_CLASS_GROUPING='hybrid'**: collapses classes with fewer than `ACT_HYBRID_THRESHOLD=100` frames into verb-based groups (~13-18 output classes instead of 75). Well-supported classes (>100 frames) retain their identity.
- **ACT_SAMPLER_MODE='balanced'**: gives every class with >=15 frames equal sampling weight, preventing the head classes from dominating the data loader.

### 2.3 PSR: 11 Binary Components, Sparse Transition Events, Temporal Structure

PSR labels track the assembly state of 11 physical components over time. Each component is binary (0 = not yet assembled, 1 = assembled). The raw labels are stored sparsely in `PSR_labels_raw.csv`: only frames where a component changes state are recorded, and the dataset fills forward between changes.

**Transition statistics (train, 36 recordings):**
- **Total transition events**: 301 across all 36 recordings
- **Average per recording**: 8.4 transitions (range: 5-9)
- **Positive component occupation**: 54.9% of all component-frames are 1 (assembled); 45.1% are 0 (not yet assembled)
- **Frames with any component = 1**: ~99.96% -- because components accumulate monotonically, most frames have at least one assembled component

The temporal structure is critical: PSR components transition monotonically (0 -> 1, never 1 -> 0 under normal operation) and the order is causal (you cannot assemble component 7 before component 3). This temporal dependency is exploited by the `PSRTransitionPredictor` (causal transformer decoder with transition head), which uses sequence-mode training (T=32 frame windows) rather than per-frame classification.

The extreme sparsity of transitions (<10 events per recording, vs ~2,000 frames) means the model sees "no transition" on 99.5% of frames. The recent PSR_TRANSITION_THRESHOLD_HI=0.5 hysteresis patch prevents the model from firing transitions at noise-level probability.

### 2.4 Pose: Continuous 6D (plus Position), Most Frames Labeled

Head pose is the only task with essentially complete annotation coverage: pose.csv contains 9-DoF entries for ~78,895 of 78,931 train frames (99.95%). The 9-DoF consists of:

- **forward_vector** (3-D): unit gaze direction
- **position** (3-D): head position in world coordinates (~m scale, divided by 100 via `HEAD_POSE_POS_SCALE=100`)
- **up_vector** (3-D): unit head-up direction

Both forward and up vectors are normalized to unit length on load. Position is divided by 100 to bring it to O(1) scale, balancing gradient contributions across vector (loss ~0.01-0.05) and position (loss ~0.1-0.5) channels.

The head pose head uses a simple 3-layer MLP with MSE loss. Because the labels are dense (every frame) and the task is comparatively easy (continuous regression with smooth temporal variation), head pose produces the lowest loss of all tasks. This caused the Kendall multi-task weighting to assign head pose extremely high precision (up to 54.6x), dominating the shared backbone -- the root cause of the detection collapse fixed by `KENDALL_HP_PREC_CAP`.

---

## 3. Current Augmentation

### 3.1 Detection Augmentation (`det_augment.py`)

The `DetectionAugment` class applies three batch-level augmentations to images and detection boxes. Key constraint: it must preserve batch shape because the shared data loader serves all four tasks.

**Flip (p=0.5):** Horizontal flip with corresponding box coordinate inversion. Symmetric for assembly parts (left/right symmetric).

**Color jitter (p=0.5):** Mild brightness, contrast, and saturation adjustments. Applied uniformly across the batch to preserve temporal consistency. Each adjusted by ±0.2 range.

**Random crop + pad (p=0.3):** Crops 85-95% of the image region, then pads back to original size with value=0.5. Boxes are offset, clipped, and filtered (min area 100 sq px).

### 3.2 Spatial Augmentation (`industreal_dataset.py`)

The `apply_spatial_aug` function in the dataset applies:
- Horizontal flip (p=0.5) with COCO keypoint swapping
- Random crop (80-100% scale, 0.9-1.1 aspect ratio) with bilinear resize back to original size

This runs on the per-frame RGB tensor before the data loader yields it. It operates on all tasks simultaneously because it transforms the actual image tensor.

### 3.3 Synthetic Pretrain Augmentation (`pretrain_synthetic.py`)

During detection-only pretraining (20 epochs), YOLO-style augmentations are active:
- **Mosaic (p=0.3)**: 4-image collage at 2x resolution, then resize back. Exposes the detector to multiple assembly states in one image, improving generalization despite having only ~7% OD-labeled frames.
- **MixUp (p=0.2)**: 2-image alpha blend with Beta(0.5, 0.5) mixing coefficient. Both sets of boxes are retained as ground truth.
- **Horizontal flip (p=0.5)**: Standard flip with box remapping.

### 3.4 What is NOT Currently Augmented

- **No temporal augmentation**: Activity recognition uses individual frames, not clips, with no frame dropout, speed perturbation, or temporal jitter.
- **No PSR-specific augmentation**: Sequence-mode PSR training passes raw frame windows through the backbone without temporal augmentation.
- **No geometric augmentation with label transforms for pose**: The head pose task receives augmented images but the 9-DoF labels are not transformed (rotation would change gaze direction).

---

## 4. Per-Task Augmentation Opportunities

Each task has different sensitivity to augmentation. A task-aware augmentation policy -- applying the right transforms to the right branches -- is the recommended path forward.

### 4.1 Detection: MixUp, CutMix, AutoAugment, RandAugment, YOLO-Style Mosaic

Detection is the most data-limited task (~7% GT density at stride=3, ~1 box per GT frame) and benefits most from aggressive augmentation. Current mosaic/mixup are confined to pretrain; bringing them to full multi-task training is the largest upside opportunity.

**Recommendations:**

1. **YOLO-style mosaic (p=0.5)**: 4-image collage creates synthetic scenes with multiple assembly parts. Each image in the collage is from a random training recording, cross-pollinating appearance variation. The challenge is integrating with the shared data loader -- mosaic changes effective batch composition and requires per-image random recording selection. Implement as a dataset wrapper (not a batch transform) to avoid batch-shape issues.

2. **MixUp (p=0.3)**: Alpha-blend two images and concatenate their boxes. Cheap to implement as a batch transform after indexing. Label assignment is straightforward: both sets of boxes are valid.

3. **RandAugment (N=2, M=9)**: Randomly apply 2 of 14 available image transforms (shear, translate, rotate, contrast, brightness, sharpness, solarize, posterize, etc.) at magnitude 9. Stronger than current color jitter, known to improve detection on small datasets.

4. **CutMix**: Replace a rectangular region of one image with patches from another. Combines with MixUp for complementary augmentation effects.

5. **Copy-paste augmentation**: Paste GT objects from one frame onto another with random scaling. Particularly effective for detecting rare assembly states (classes with <100 instances). The background frame provides realistic context; the pasted object provides the rare class example.

**Implementation pathway** for multi-task compatibility: apply detection augmentations ONLY in the `train_step` before the detection loss computation (as `DetectionAugment` already does). The shared backbone processes the unaugmented image; the detection branch gets its own augmented copy. This preserves PSR temporal order, activity class identity, and pose geometry.

### 4.2 Activity: Temporal Jittering, Speed Perturbation, Frame Dropout, RandAugment

Activity recognition currently uses single-frame classification with a temporal feature bank (T=16). Augmentation opportunities exist at both spatial and temporal levels:

**Spatial (per-frame):**
- **RandAugment (N=1, M=5)**: Milder than detection's RandAugment. Apply to activity frames only. Strong color shifts help invariance to lighting conditions across different assembly workstations.
- **Random grayscale (p=0.1)**: Forces the model to use texture and shape rather than color cues. Cheap regularization.

**Temporal (clip-level):**
- **Frame dropout (p=0.1)**: Randomly drop individual frames from the T=16 clip, replacing with the nearest retained frame. Simulates occlusion from hand movement across the egocentric camera.
- **Speed perturbation**: Randomly skip or repeat frames to simulate varying assembly speed. Applied to the clip before feature extraction.
- **Temporal jitter**: Randomly shift the clip window by 1-3 frames forward or backward, sampling different substeps of the same action.

**Important constraint**: Activity is the highest-weighted metric (combined weight 0.35). Augmentation that changes the action class (e.g., dropping critical frames that distinguish "screw" from "tighten") must be excluded. Verb-preserving augmentation (speed perturbation, mild jitter) is safe; random crop that removes the object being manipulated is not.

### 4.3 PSR: Temporal Consistency Augmentation (No Augmentation That Breaks Transitions)

PSR is uniquely sensitive to augmentation because its labels encode temporal state transitions. Any augmentation that breaks temporal causality or changes component ordering will corrupt the label.

**Safe augmentations:**
- **Minimal spatial augmentation only**: The PSR head operates on high-level features from the shared backbone, which already receives augmented images. No additional spatial augmentation needed.
- **Temporal masking (p=0.15)**: Randomly mask out intermediate frames in the T=32 window, replacing with linear interpolation of features. Forces the PSR transformer to reason about the remaining context, improving robustness to feature noise.

**Dangerous augmentations (DO NOT use for PSR):**
- Frame reordering
- Speed perturbation that drops transition frames
- Clip reversal
- Any augment that changes the relative timing of transitions

The monotonic decoder (`MonotonicDecoder` with hysteresis) already handles temporal consistency by requiring sustained probability above threshold_lo for N consecutive frames. This is a decoding-time augmentation, not a training-time one.

### 4.4 Pose: Geometric Augmentation with Corresponding Label Transforms

Head pose has dense labels and benefits from geometric augmentation that transforms both the image and the 9-DoF labels consistently.

**Safe augmentations:**
- **Rotation (0-30 degrees)**: Rotate the image and apply the inverse rotation to the forward_vector. Formulas for 3D rotation of the gaze vector under 2D image rotation are well-defined.
- **Horizontal flip**: Already applied by `apply_spatial_aug`. Requires swapping left/right components of the up_vector (negate x-component of gaze direction).
- **Random crop + resize**: Changes apparent head position. The position labels in pose.csv are in world coordinates, NOT image coordinates, so position labels remain unchanged. However, the image crop changes the visual context for head pose estimation.

**Low-value augmentations:**
- **Color jitter**: Head pose relies on edge and gradient features, not color. Color augmentation adds minimal value.
- **CutOut / random erasing**: Potentially removes important facial features from the egocentric view (the operator's hands occlude the camera, but removing the assembly context may confuse the pose regressor).

**Recommended approach**: Apply geometric augmentations without label transforms at first (the MSE loss can tolerate small misalignment). If pose performance degrades, implement full label-transform augmentation using the known camera intrinsics and rotation matrix.

---

## 5. Data Sampling Strategies

### 5.1 Class-Balanced Sampling for Activity

The dataset's `get_sampler()` returns a `WeightedRandomSampler` with class-balanced weights computed via one of two modes:

**'balanced' mode (current default):**
- Each class with >= `ACT_SAMPLER_COUNT_FLOOR=15` frames gets equal sampling weight.
- Classes below 15 frames get mass proportional to their count (avoids memorizing 1-7 frame singletons by repeating them 50x per epoch).
- Effective per-class mass ratio: max/min ~1.0 for well-supported classes, tail classes appear at their natural frequency.

**'cb' mode (effective number):**
- Uses `CB_BETA=0.99` effective number weighting: weight = 1/((1-beta^n)/(1-beta)).
- Only partially balances: head classes still get ~4-5x the tail's mass.
- Still active when `USE_CB_FOCAL_ACT=True` (the loss itself rebalances).

**Task-aware weighting extension:**
When `USE_TASK_AWARE_SAMPLING=True`, the base activity-balanced weights are multiplied by:
- `TASK_AWARE_DET_BOOST=2.0` for frames with GT boxes
- `TASK_AWARE_PSR_BOOST=1.5` for frames with PSR-positive labels

### 5.2 Detection GT-Frame Oversampling

The single most impactful sampling change for detection was `DET_GT_FRAME_FRACTION`. This mechanism redistributes the total sampling mass so that in expectation, a target fraction of every batch carries detection GT frames.

**Per-stage values** (set by `apply_preset()`):
- RF1/RF2 detection-dominant stages: `det_gt_frame_fraction=0.9`
- RF3-RF10 multi-task stages: `det_gt_frame_fraction=0.4`
- Detection-absent stages: `det_gt_frame_fraction=0.0`

The `GuaranteedGTBatchSampler` wraps this further: it takes the weighted sampler's output and enforces that each batch contains AT LEAST one GT-bearing frame. If a batch has none, it replaces the last index with a random GT frame. This is a hard guarantee that the detection head receives a positive gradient on every step.

**Why two mechanisms?** `DET_GT_FRAME_FRACTION` works at the sampler level (probabilistic: ~40% of batches have GT). `GuaranteedGTBatchSampler` works at the batch level (deterministic: 100% of batches have GT). Both are needed because at 7% base OD density, even 40% target GT fraction can leave batches GT-free during early training when the weighted sampler's distribution hasn't converged.

### 5.3 PSR Transition-Aware Sampling

PSR transitions are rare (~301 events across 36 recordings, ~8 per recording). The task-aware weighting applies `TASK_AWARE_PSR_BOOST=1.5` to frames with any positive PSR component, but this is a weak signal because 99.96% of frames have some component=1.

A more targeted approach would upsample frames NEAR transitions (within a temporal window around each transition event). This would give the PSR head more exposure to the critical frames where component states change, at the cost of biasing away from steady-state frames. Implementation is straightforward: build a transition mask from `PSR_labels_raw.csv` that labels each frame within `w` frames of any transition as "transition-adjacent", and apply a boost (e.g., 3x) to those frames.

---

## 6. Missing Data Handling

Each frame in the dataset may have labels for some tasks but not others. The dataset handles missing labels with sentinel values:

| Task | Missing Sentinel | Impact on Loss |
|------|-----------------|----------------|
| Activity (AR) | `-1` (int64) | Frames with `-1` are excluded from CE/Focal loss computation via masking |
| Detection (ASD) | Empty `gt_boxes`/`gt_classes` tensors | Empty frames compute a subsampled background focal loss (`DET_EMPTY_SAMPLE=2048`), preventing gradient starvation |
| PSR | N/A (fill-forward always produces valid labels) | No missing-data path needed |
| Head Pose | `zeros` array (no pose.csv) | Zero loss contributed -- frames without pose are extremely rare (<0.05%) |
| Hand Joints | `zeros` array (no hands.csv) | Hands are used only as FiLM conditioning input, not a training loss |

**Activity sentinel handling:** The `-1` sentinel is explicitly masked in `losses.py`: frames with `action_label=-1` contribute zero gradient. This is the only task where missing data is both common (~12% of train frames) and handled via masking rather than zero-fill.

**Detection empty-frame loss:** Empty frames (no GT boxes) are NOT masked out. Instead, they contribute a subsampled background loss (`DET_EMPTY_SAMPLE=2048` anchors out of ~173K) scaled by `DET_EMPTY_BG_SCALE=0.05`. This small but non-zero loss keeps the detection head weights from decaying to zero between GT-bearing batches (a phenomenon observed in RC-28 where head grad norm dropped to 0.0049 and stayed dead).

**PSR fill-forward philosophy:** Missing PSR data does not exist per se -- the fill-forward from sparse transition events creates dense labels. The design choice of filling forward (rather than masking unlabeled frames) assumes that component states are monotonic and persistent. This is correct for normal assembly but would fail for disassembly sequences or error recovery.

---

## 7. Synthetic Data Generation

### 7.1 Current: Detection-Only Synthetic Pretraining

The pretrain pipeline (`pretrain_synthetic.py`) uses a synthetic data generator to create unlimited detection training data. The current implementation feeds real frames through the same dataset but with aggressive augmentation (mosaic + mixup + flip) at stride=10 for 20 epochs.

**True synthetic data generation** (not yet implemented but architecture-ready): The `synthetic_data_gen/` module can render random assembly states with realistic materials and lighting. Key capabilities:
- **Random state assembly**: Places 11 components in random assembly configurations, rendering the resulting visual appearance.
- **Appearance randomization**: Randomizes material properties (metallic roughness, color), lighting (position, intensity, color temperature), and camera pose (within a workspace-constrained frustum).
- **Automatic annotation**: Since the renderer knows the component positions, it produces perfect OD_labels.json annotations with zero labeling cost.
- **Distribution matching**: The synthetic generator can be biased to produce rare classes (classes 7, 16, 17 with <100 instances) at higher frequency, rebalancing the detection class distribution.

### 7.2 Future: Multi-Task Synthetic Data

Extending synthetic data generation to support all four tasks would be transformative:

**Activity synthesis**: Render short action clips (2-3 seconds at 10 FPS) with known action labels. An animated hand model performs the assembly action while the camera follows a pre-recorded head trajectory. This gives ground-truth activity labels that are perfectly aligned, enabling pretraining on tasks that currently have limited supervision.

**PSR synthesis**: Render the full assembly sequence with all 11 component transitions, providing perfect PSR transition labels. The temporal ordering is deterministic (component order is known from procedure_info.json), so transitions occur at known frame boundaries.

**Pose synthesis**: The rendering engine knows the exact camera position in world coordinates (the inverse of the head pose), providing perfect 9-DoF pose labels.

**Risk**: Synthetic-to-real domain gap. The backbone fine-tuned on synthetic data may learn rendering artifacts rather than real assembly features. Mitigated by:
- Progressive unfreezing (synthetic pretrain -> real data fine-tune)
- Domain randomization (vary render parameters during synthetic generation)
- Feature-level domain adaptation (GAN-based alignment of synthetic/real feature distributions)

---

## 8. Self-Supervised Pretraining on IndustReal Data

### 8.1 Motivation

The current pipeline uses ImageNet-pretrained ConvNeXt-Tiny as the backbone. While ImageNet provides general visual features, it is domain-shifted from egocentric industrial assembly: no hands, no metallic parts, no assembly fixtures. Self-supervised pretraining on the IndustReal frames themselves (before task-specific fine-tuning) could bridge this gap by learning the visual distribution of assembly scenes.

### 8.2 Masked Image Modeling (MIM) for Backbone

IndustReal has 207K frames across 84 recordings -- sufficient for MIM pretraining (MAE-style). The approach:

1. **Mask random patches** (75% masking ratio) from each 1280x720 frame.
2. **Reconstruct the masked patches** using a lightweight decoder attached to the ConvNeXt encoder.
3. **Pretrain for 50 epochs** on all train + val + test frames (no labels needed).
4. **Discard the decoder** and initialize multi-task training with the MIM-pretrained encoder.

**Expected benefit**: The encoder learns assembly-specific features (hand shapes, tool appearances, component geometries) that ImageNet initialization misses. This could reduce the total fine-tuning data requirement by 30-50% for the detection and activity heads.

### 8.3 Cross-Frame Temporal Consistency (CPC / SimCLR-Temporal)

Beyond single-frame MIM, temporal self-supervision is uniquely suited to IndustReal's sequential assembly structure:

- **Temporal proximity contrastive learning**: Frames close in time (within 5 frames) are positives; frames far apart (50+ frames apart or from different recordings) are negatives. A projector head learns representations invariant to small temporal shifts.
- **Temporal order verification**: Sample 3 frames from a recording, shuffle them, and train a classifier to predict the correct order. This forces the model to learn the monotonic progression of assembly states.
- **Future frame prediction**: Given a window of T=8 frames, predict the feature representation of the next 4 frames. The prediction error acts as a self-supervised loss.

The temporal consistency approaches require careful batch construction (multiple frames from the same recording per batch) and may conflict with the class-balanced activity sampler.

### 8.4 Incorporating Synthetic Data

The synthetic data generator can produce unlimited frames with known annotations. The self-supervised pretraining can use a combined real + synthetic corpus:

- **Real frames (207K)**: Provide realistic texture, lighting, and hand appearance.
- **Synthetic frames (500K+)**: Provide diversity in assembly state configurations and component appearances.

The combined pretraining exposes the backbone to a broader distribution than either source alone, with the synthetic data filling the long-tail states that are rare in real recordings.

### 8.5 Practical Implementation

For the current pipeline, the highest-value self-supervised addition would be:

1. **Single-frame MAE (50 epochs)**: Train on all 207K frames with no label access. Add to the training workflow as an optional `--pretrain-mim` flag that runs before the main multi-task training.
2. **Convergence**: After 50 epochs of MIM, the backbone should have learned assembly-specific features. Multi-task fine-tuning should converge faster (fewer epochs to match current metrics) and achieve higher peak macro-F1 for activity and detection.

The cost is ~2-3 GPU-days for 50 epochs of MAE on 207K 720p frames (with significant downsampling to 224x224 for MAE efficiency). For most deployment scenarios this is justified by the downstream improvement.

---

## Summary of Recommendations

| Priority | Augmentation | Task(s) | Effort | Impact |
|----------|-------------|---------|--------|--------|
| P0 | Detection mosaic + mixup in full training | Detection | Medium | High |
| P1 | Transition-adjacent PSR upsampling | PSR | Low | Medium |
| P1 | RandAugment for detection branch | Detection | Low | Medium |
| P2 | Temporal jitter for activity clips | Activity | Medium | Medium |
| P2 | MAE self-supervised pretraining | All tasks | High | High |
| P3 | Geometric pose augmentation with label transforms | Head pose | High | Low-Med |
| P3 | Multi-task synthetic data generation | All tasks | Very High | High |

The single highest-value change is bringing mosaic/mixup augmentation into the full multi-task training loop (currently restricted to detection pretrain). Combined with the existing `GuaranteedGTBatchSampler` and `DET_GT_FRAME_FRACTION=0.4`, this would give the detection head both sufficient GT exposure and sufficient image-level variation to learn discriminative features for all 24 classes.
