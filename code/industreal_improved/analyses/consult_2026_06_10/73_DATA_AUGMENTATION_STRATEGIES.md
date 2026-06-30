# 73: Data Augmentation & Preprocessing Strategies [2026-07-01]

## Current Data Pipeline

```
Source: 3,667 train frames from IndustReal (36 recordings)
Loading: RAM cache (8,000 JPEGs, ~2.7 GB RAM)
Sampler: WeightedRandomSampler (class-balanced weights)
Batching: 4 frames/batch, 8 gradient accumulation steps = 32 effective
GT frame mix: DET_GT_FRAME_FRACTION=0.4 (40% batches have OD labels)
Augmentation: NONE (USE_MIXUP=False, USE_RANDAUGMENT=False)
```

## Strategy 1: Mixup (Q10)

**What:** Mixup creates synthetic training example by linearly interpolating
between two frames and their labels: `x' = λx₁ + (1-λ)x₂, y' = λy₁ + (1-λ)y₂`
where λ ~ Beta(α, α).

**Why it might help:** Mixup encourages the network to produce linear
interpolations between class predictions. For classes with few exemplars, each
training frame is "paired" with other classes' frames, creating synthetic
training signals.

**Why it might not help:** Mixup requires paired frames. With WeightedRandomSampler,
consecutive frames are from different recordings. Mixup pairs a minority-class
frame with a majority-class frame, producing a soft label where the majority class
dominates. Net effect: minority class gets diluted, not amplified.

**Implementation:** Already in data pipeline (USE_MIXUP), just need to set it True.
```
config.py: USE_MIXUP = True
# Only for activity? Mixup affects all tasks equally.
```

**Cost:** Config change only.

## Strategy 2: RandAugment (Q10)

**What:** Random augmentation pipeline (color jitter, contrast, brightness,
rotation, translation, shear, posterize, solarize). N=2 transforms, M=9 magnitude.

**Why it might help:** Each minority-class frame gets 2 random augmentations per
epoch, creating diverse training examples without collecting new data.

**Why it might not help:** Egocentric assembly frames have limited variance.
Brightness/contrast changes don't add new semantic information about which
action is being performed.

**Implementation:** Already in data pipeline (USE_RANDAUGMENT). Set True.

**Cost:** Config change only.

## Strategy 3: CutOut / Random Erasing

**What:** Randomly mask a square region of the input image (e.g. 50×50 pixels).
Forces the network to use non-local features for classification.

**Implementation:** Add to data transforms in dataset:
```python
from torchvision.transforms import RandomErasing
# p=0.5, scale=(0.02, 0.1), ratio=(0.3, 3.3)
```

**Cost:** 30 min to implement and test.

## Strategy 4: Oversample Minority Classes (Q9)

**What:** Every class with <10 frames gets duplicated 10× in the dataset load.
Not saved to disk — just appends to the frame index list at dataset init time.

**Implementation:**
```python
_counts = np.bincount(frame_activity_labels)
_oversample_threshold = 10
_extra_indices = []
for cls in range(num_classes):
    if _counts[cls] < _oversample_threshold:
        cls_indices = np.where(frame_activity_labels == cls)[0]
        _extra_indices.extend(np.random.choice(cls_indices, 10, replace=True).tolist())
all_indices.extend(_extra_indices)  # 48 classes × 10 = 480 extra frames
```

**Why it might help:** More gradient updates for minority classes. The
WeightedRandomSampler already up-weights rare classes, but the total number
of frames seen per epoch for a class with 3 frames is still just 3. Oversampling
to 30 gives 10× more updates.

**Why it might not help:** The same 3 frames get repeated 10×. The network
memorizes them rather than learning generalizable features.

**Cost:** 30 min.

## Strategy 5: Class Merging / Hierarchy (Q8)

**What:** Group semantically similar actions into super-classes.

```
Current 74 classes → Proposed ~15 super-classes:
1. "take" actions (take_pin_short, take_pin_long, take_small_screw_pin, ...)
2. "plug" actions (plug_short_pin, plug_long_pin, plug_small_screw_pin, ...)
3. "align" actions (align_objects, align_short_pin, ...)
4. "tighten" actions (tighten_tooth_washer, tighten_short_pin, ...)
5. "loosen" actions (loosen_tooth_washer, loosen_short_pin, ...)
6. "pull" actions (pull_small_screw_pin, pull_long_pin, ...)
7. "insert" actions (insert_short_pin, insert_long_pin, ...)
8. "remove" actions (remove_short_pin, remove_long_pin, ...)
9. Other (miscellaneous actions with <5 frames)
10-15. Combinations of above
```

**Why it might help:** Each super-class would have 50-200+ frames instead of
3-50. The network can learn meaningful features for each super-class. Fine-grained
classification within each super-class can be a second stage if needed.

**Why it might not help:** Loss of fine-grained action recognition — the paper
claims "74-class activity recognition" which becomes "15-class action group
recognition." May impact AAIML reviewer perception.

**Cost:** Requires re-mapping label file. ~1 hour.

## Strategy 6: External Data / Pseudo-Labeling

**What:** Use the trained model from epoch 0 to generate pseudo-labels on the
1,928 validation frames (which have ground-truth activity labels but could be
augmented with model predictions for semi-supervised learning).

**Not practical:** We have 3.7k train frames + 1.9k val frames. The val frames
already have labels. There is no unlabeled data pool.

## Strategy 7: Class-Balanced Sampler + Sequential Batch Sampling

**What:** Combine class-balanced weighting with sequential frame ordering.

The current WeightedRandomSampler shuffles frames for class balance but destroys
temporal ordering. A two-stage sampler:
1. Group frames by recording_id (maintains temporal ordering within recording)
2. Sample recordings with class-balanced probability
3. Within each sampled recording, yield consecutive frames

This preserves class balance AND temporal ordering. Activity head gets real
temporal sequences; detection/pose/PSR can continue with shuffled batches.

**Implementation:** This is the "per-recording sequential sampler" from file 72
(Alt 5). Requires dual DataLoader architecture.

**Cost:** ~2 hours.

## Priority Matrix for Opus

| # | Strategy | Impact | Cost | Code Change |
|:-|----------|--------|------|-------------|
| 1 | Mixup | Medium | Zero | Config toggle |
| 2 | RandAugment | Low-Medium | Zero | Config toggle |
| 3 | CB-Focal with γ=5.0 | Medium | Zero | Config change |
| 4 | Oversample minority 10× | Medium-High | 30 min | Dataset code |
| 5 | Class merging (74→15) | High | 1 hour | Label remap |
| 6 | Sequential sampler | High | 2 hours | Dual DataLoader |
| 7 | Drop activity | N/A | Zero | Paper scope change |
