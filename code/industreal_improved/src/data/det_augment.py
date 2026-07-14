"""Detection-specific data augmentation (Opus 192 §5.5).

[OPUS 192 Q6] "for detection which is data-limited, YOLO-style mosaic/mixup
genuinely helps. So apply augmentation to the detection branch, not to
activity. Do not temporally jitter PSR (temporal order IS its label)."

Full YOLOv8-style mosaic (4-image compose) and mixup (alpha-blend) would
require changing batch shape, which the shared data loader can't do. This
module implements the lighter, batch-shape-preserving augmentations:

  - Random horizontal flip (p=0.5) — safe for assembly (left/right symmetric)
  - Random color jitter (brightness/contrast/saturation, mild)
  - Random crop + pad (small translation)

All three are applied to images AND bboxes for detection only. Other heads
see the original images. This is implemented by calling the augmentation
in train_step before computing the detection loss.

Usage:
    from src.data.det_augment import DetectionAugment
    aug = DetectionAugment(p_flip=0.5, p_color=0.5, p_crop=0.3)
    aug_images, aug_targets = aug(images, targets)
"""

import random
from typing import Tuple

import torch
import torch.nn.functional as F


class DetectionAugment:
    """Batch-preserving detection augmentation (light)."""

    def __init__(
        self,
        p_flip: float = 0.5,
        p_color: float = 0.5,
        p_crop: float = 0.3,
        brightness_range: float = 0.2,
        contrast_range: float = 0.2,
        saturation_range: float = 0.2,
    ):
        self.p_flip = p_flip
        self.p_color = p_color
        self.p_crop = p_crop
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.saturation_range = saturation_range

    def __call__(
        self,
        images: torch.Tensor,  # [B, 3, T, H, W] (after permute)
        targets: dict,  # contains 'detection' list of {boxes, labels}
    ) -> Tuple[torch.Tensor, dict]:
        """Apply augmentations to images and detection boxes (other targets unchanged).

        Args:
            images: [B, 3, T, H, W] normalized image tensor
            targets: dict with 'detection' key (list of dicts per image)

        Returns:
            (augmented_images, augmented_targets)
        """
        B, C, T, H, W = images.shape
        device = images.device
        aug_images = images.clone()
        aug_targets = {k: v for k, v in targets.items()}

        # 1) Random horizontal flip (per-batch, p=p_flip)
        if random.random() < self.p_flip:
            aug_images = aug_images.flip(-1)  # flip W
            # Flip detection boxes: x1' = W - x2, x2' = W - x1
            new_det = []
            for det in aug_targets.get("detection", []):
                boxes = det.get("boxes")
                if boxes is None or boxes.numel() == 0:
                    new_det.append(det)
                    continue
                boxes = boxes.clone()
                if boxes.dim() == 1:
                    boxes = boxes.unsqueeze(0)
                new_boxes = boxes.clone()
                new_boxes[:, 0] = W - boxes[:, 2]
                new_boxes[:, 2] = W - boxes[:, 0]
                new_det.append({**det, "boxes": new_boxes})
            aug_targets["detection"] = new_det

        # 2) Random color jitter (per-batch, p=p_color)
        if random.random() < self.p_color:
            # Apply same jitter to all frames in the batch (so temporal structure preserved)
            brightness = 1.0 + (random.random() * 2 - 1) * self.brightness_range
            contrast = 1.0 + (random.random() * 2 - 1) * self.contrast_range
            saturation = 1.0 + (random.random() * 2 - 1) * self.saturation_range
            # Brightness: scale pixels
            aug_images = aug_images * brightness
            # Contrast: scale deviation from mean
            mean = aug_images.mean(dim=(2, 3, 4), keepdim=True)  # per-sample, per-channel
            aug_images = (aug_images - mean) * contrast + mean
            # Saturation: scale toward grayscale (approximate)
            gray = aug_images.mean(dim=1, keepdim=True)  # [B, 1, T, H, W]
            aug_images = gray + (aug_images - gray) * saturation
            # [FIX Claude Science V2 Agent 9] Do NOT clamp to [0,1] — images
            # arrive pre-normalized (mean=0.45, std=0.225) and clamping destroys
            # the normalization, mapping all negative values to 0 and all bright
            # values to 1. The color jitter parameters (±0.2 range) are mild
            # enough that values stay within ±3σ after normalization.

        # 3) Random crop + pad (per-batch, p=p_crop)
        if random.random() < self.p_crop:
            # Crop a smaller region, then pad back to original size
            crop_h = int(H * random.uniform(0.85, 0.95))
            crop_w = int(W * random.uniform(0.85, 0.95))
            y_offset = random.randint(0, H - crop_h)
            x_offset = random.randint(0, W - crop_w)
            # Crop
            cropped = aug_images[
                :, :, :, y_offset : y_offset + crop_h, x_offset : x_offset + crop_w
            ]
            # Pad back
            aug_images = F.pad(
                cropped,
                [x_offset, W - x_offset - crop_w, y_offset, H - y_offset - crop_h],
                value=0.5,
            )
            # Adjust detection boxes: subtract offset, clip to crop
            new_det = []
            for det in aug_targets.get("detection", []):
                boxes = det.get("boxes")
                if boxes is None or boxes.numel() == 0:
                    new_det.append(det)
                    continue
                boxes = boxes.clone()
                if boxes.dim() == 1:
                    boxes = boxes.unsqueeze(0)
                # Subtract offset
                new_boxes = boxes.clone()
                new_boxes[:, 0] = (boxes[:, 0] - x_offset).clamp(0, W)
                new_boxes[:, 1] = (boxes[:, 1] - y_offset).clamp(0, H)
                new_boxes[:, 2] = (boxes[:, 2] - x_offset).clamp(0, W)
                new_boxes[:, 3] = (boxes[:, 3] - y_offset).clamp(0, H)
                # Filter boxes with area < some threshold (degenerate after crop)
                areas = (new_boxes[:, 2] - new_boxes[:, 0]).clamp(min=0) * (
                    new_boxes[:, 3] - new_boxes[:, 1]
                ).clamp(min=0)
                valid = areas > 100  # min 100 sq pixels
                new_boxes = new_boxes[valid]
                if new_boxes.numel() == 0:
                    new_det.append({**det, "boxes": torch.zeros(0, 4, device=device)})
                else:
                    new_det.append({**det, "boxes": new_boxes})
            aug_targets["detection"] = new_det

        return aug_images, aug_targets
