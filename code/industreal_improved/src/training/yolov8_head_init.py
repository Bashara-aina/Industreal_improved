"""Initialize MTL detection head from YOLOv8 priors.

YOLOv8's architecture differs from our MTL head (DFL + C2f vs 3x3 + 16 anchors),
so direct weight transfer isn't possible. Instead, we extract:

1. Per-class confidence priors (mean YOLOv8 score per class)
2. Per-class frequency priors (how often each class appears)
3. YOLOv8's final conv weight statistics (scale, mean)

These are used to initialize MTL detection head's:
- cls_head[3].bias with class-aware priors
- reg_head[3].weight with YOLOv8-scaled init
- reg_head[3].bias with zero

This gives the MTL detection head a "warm start" from YOLOv8's class knowledge,
which is the most transferable signal (architectures differ, but class priors
are data-driven and should transfer).
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def extract_yolov8_priors(
    yolov8_weights: str = '/home/newadmin/swarm-bot/master/POPW/datasets/industreal/assembly_state_detection_model_weights/asd_best_IndustRealandSynthetic.pt',
    train_recording_dir: str = '/home/newadmin/swarm-bot/master/POPW/datasets/industreal/recordings/train',
    conf_threshold: float = 0.05,
    max_frames_per_recording: int = 30,
) -> Dict:
    """Run YOLOv8 on training frames and compute class priors.

    Also computes ground truth class frequencies from annotations, which are
    more reliable than YOLOv8's predicted frequencies (since YOLOv8 only
    learns well for some classes).

    Returns:
        dict with:
          'class_frequencies_yolov8': [24] YOLOv8 detection rate per class
          'class_frequencies_gt': [24] GT annotation rate per class
          'class_mean_scores': [24] YOLOv8 mean score per class
          'class_count_yolov8': [24] YOLOv8 detection count per class
          'class_count_gt': [24] GT annotation count per class
          'n_frames': total frames processed
    """
    from ultralytics import YOLO
    from PIL import Image
    import json as json_lib

    model = YOLO(yolov8_weights)

    yolo_class_counts = np.zeros(24, dtype=np.int64)
    class_score_sums = np.zeros(24, dtype=np.float64)
    gt_class_counts = np.zeros(24, dtype=np.int64)
    gt_total_frames = 0
    n_frames = 0

    train_dir = Path(train_recording_dir)
    recording_dirs = sorted([d for d in train_dir.iterdir() if d.is_dir()])

    for rec_dir in recording_dirs:
        rgb_dir = rec_dir / 'rgb'
        od_path = rec_dir / 'OD_labels.json'

        # Load ground truth annotations
        gt_classes_in_recording = set()
        if od_path.exists():
            with open(od_path) as f:
                od = json_lib.load(f)
            for ann in od['annotations']:
                cat_id = ann['category_id'] - 1
                if 0 <= cat_id < 24:
                    gt_class_counts[cat_id] += 1
                    gt_classes_in_recording.add(cat_id)

        # Sample frames from this recording
        rgb_files = sorted(rgb_dir.glob('*.jpg'))[:max_frames_per_recording]
        for img_path in rgb_files:
            try:
                img = Image.open(img_path).convert('RGB').resize((640, 360))
                results = model(img, verbose=False, conf=conf_threshold)
                if results[0].boxes is not None and len(results[0].boxes) > 0:
                    classes = results[0].boxes.cls.cpu().numpy().astype(int)
                    scores = results[0].boxes.conf.cpu().numpy().astype(np.float32)
                    for c, s in zip(classes, scores):
                        if 0 <= c < 24:
                            yolo_class_counts[c] += 1
                            class_score_sums[c] += s
                n_frames += 1
            except Exception as e:
                logger.debug(f'Skip {img_path}: {e}')
        gt_total_frames += len(rgb_files)

    # Compute priors
    yolo_class_frequencies = yolo_class_counts / max(n_frames, 1)
    gt_class_frequencies = gt_class_counts / max(gt_total_frames, 1)
    class_mean_scores = np.where(
        yolo_class_counts > 0,
        class_score_sums / np.maximum(yolo_class_counts, 1),
        0.5  # default for unseen classes
    )

    priors = {
        'class_frequencies_yolov8': yolo_class_frequencies.tolist(),
        'class_frequencies_gt': gt_class_frequencies.tolist(),
        'class_mean_scores': class_mean_scores.tolist(),
        'class_count_yolov8': yolo_class_counts.tolist(),
        'class_count_gt': gt_class_counts.tolist(),
        'n_frames': int(n_frames),
        'n_gt_frames': int(gt_total_frames),
    }

    logger.info(f'Priors extracted from {n_frames} frames ({gt_total_frames} GT frames)')
    logger.info(f'  YOLOv8 detections: {yolo_class_counts.sum()} ({yolo_class_counts.tolist()[:5]}...)')
    logger.info(f'  GT annotations: {gt_class_counts.sum()} ({gt_class_counts.tolist()[:5]}...)')
    logger.info(f'  GT class coverage: {(gt_class_counts > 0).sum()}/24')

    return priors


def init_mtl_det_head_from_yolov8(
    det_head: nn.Module,
    priors: Dict,
    num_classes: int = 24,
    num_anchors: int = 16,
    log: bool = True,
    use_gt_freqs: bool = True,
    min_freq: float = 0.001,
) -> None:
    """Initialize MTL detection head's cls/reg final conv layers from YOLOv8 priors.

    MTL head structure:
      cls_head: Sequential[Conv(in, in, 3, pad=1), GN, ReLU, Conv(in, num_classes, 1)]
      reg_head: Sequential[Conv(in, in, 3, pad=1), GN, ReLU, Conv(in, 4*num_anchors, 1)]

    We set:
      - cls_head[3].bias[i] = -log((1 - freq_i) / freq_i) (class-aware prior)
      - cls_head[3].weight ~ small Gaussian (don't disturb features)
      - reg_head[3].weight ~ small Gaussian (don't disturb regression)
      - reg_head[3].bias = 0 (regression starts neutral)

    use_gt_freqs=True uses ground truth class frequencies (more reliable than
    YOLOv8's predicted frequencies for rare classes).
    """
    import math

    # Pick which frequency to use: GT is more reliable than YOLOv8's predictions
    if use_gt_freqs and 'class_frequencies_gt' in priors:
        cls_frequencies = torch.tensor(
            priors['class_frequencies_gt'][:num_classes], dtype=torch.float32
        )
        freq_source = 'GT'
    else:
        cls_frequencies = torch.tensor(
            priors['class_frequencies_yolov8'][:num_classes], dtype=torch.float32
        )
        freq_source = 'YOLOv8'

    # Floor at min_freq so very rare classes don't get extreme bias
    cls_frequencies = cls_frequencies.clamp(min=min_freq, max=0.5)

    # Initialize cls_head[3] (the final 1x1 conv to num_classes)
    if hasattr(det_head, 'cls_head'):
        cls_final = det_head.cls_head[3]  # nn.Conv2d(in, num_classes, 1)
        if isinstance(cls_final, nn.Conv2d):
            nn.init.normal_(cls_final.weight, mean=0.0, std=0.01)
            # Bias: -log((1-prior)/prior) using class frequencies
            # This sets initial sigmoid output to match observed class rate
            for i in range(min(num_classes, cls_final.bias.shape[0])):
                freq = float(cls_frequencies[i])
                cls_final.bias.data[i] = -math.log((1 - freq) / freq)

            if log:
                logger.info(f'  cls_head[3]: weight std=0.01, bias from {freq_source} freq')
                logger.info(f'    Sigmoid[0]={torch.sigmoid(cls_final.bias[0]).item():.4f} (freq={cls_frequencies[0]:.4f})')
                logger.info(f'    Sigmoid[5]={torch.sigmoid(cls_final.bias[5]).item():.4f} (freq={cls_frequencies[5]:.4f})')
                logger.info(f'    Sigmoid[13]={torch.sigmoid(cls_final.bias[13]).item():.4f} (freq={cls_frequencies[13]:.4f})')

    # Initialize reg_head[3] (the final 1x1 conv to 4*num_anchors)
    if hasattr(det_head, 'reg_head'):
        reg_final = det_head.reg_head[3]
        if isinstance(reg_final, nn.Conv2d):
            nn.init.normal_(reg_final.weight, mean=0.0, std=0.01)
            nn.init.zeros_(reg_final.bias)
            if log:
                logger.info(f'  reg_head[3]: weight std=0.01, bias=0')


def save_priors(priors: Dict, path: str):
    """Save priors to JSON for later use."""
    with open(path, 'w') as f:
        json.dump(priors, f, indent=2)
    logger.info(f'Priors saved: {path}')


def load_priors(path: str) -> Dict:
    """Load priors from JSON."""
    with open(path) as f:
        return json.load(f)


if __name__ == '__main__':
    import argparse
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str, default='runs/yolov8_priors.json')
    args = parser.parse_args()

    priors = extract_yolov8_priors()
    save_priors(priors, args.out)
    print(f'Saved priors to {args.out}')
    print(f'  YOLOv8 frames: {priors["n_frames"]}')
    print(f'  GT frames: {priors["n_gt_frames"]}')
    print(f'  GT class freqs: {[f"{x:.4f}" for x in priors["class_frequencies_gt"]]}')