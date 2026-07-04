"""
Soft-NMS: Improved NMS with Gaussian penalty function
=====================================================
Instead of hard-suppressing overlapping boxes (standard NMS), Soft-NMS decays
their scores via a Gaussian function of the IoU with the highest-scoring box.
This retains nearby true positives that would otherwise be lost under a hard IoU
threshold, improving recall for overlapping/occluded objects.

Reference:
    Bodla, N., Singh, B., Chellappa, R., Davis, L. S. (2017).
    "Soft-NMS — Improving Object Detection With One Line of Code."
    ICCV 2017. https://arxiv.org/abs/1704.04503
"""

import numpy as np


def soft_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    sigma: float = 0.5,
    score_thresh: float = 0.001,
) -> np.ndarray:
    """
    Soft-NMS with Gaussian penalty function.

    Iteratively selects the highest-scoring box, then decays the scores of all
    remaining boxes according to a Gaussian weight::

        weight_i = exp(-iou(highest, box_i)^2 / sigma)

    Boxes whose score falls below ``score_thresh`` after decay are pruned.

    Args:
        boxes: [N, 4] array in (x1, y1, x2, y2) format.
        scores: [N] array of detection scores (any non-negative range).
        sigma: Gaussian spread parameter. Smaller values = stronger suppression.
               Typical range: 0.3-0.7. Default 0.5 per Bodla et al.
        score_thresh: minimum score threshold for keeping a detection after
                      decay. Default 0.001.

    Returns:
        keep: [M] int64 array of indices into the original ``boxes`` / ``scores``
              arrays that survived Soft-NMS.

    Reference:
        Algorithm 1 (Gaussian) in Bodla et al. ICCV 2017.
    """
    N = boxes.shape[0]
    if N == 0:
        return np.array([], dtype=np.int64)

    # Work on copies so we don't mutate the caller's arrays.
    boxes = np.asarray(boxes, dtype=np.float64).copy()
    scores = np.asarray(scores, dtype=np.float64).copy()

    # Compute areas once.
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    # `order` maps current array index -> original input index.
    order = np.arange(N)
    keep = []
    while scores.size > 0:
        # Pick the current max-score box.
        max_idx = scores.argmax()
        keep.append(order[max_idx])

        if scores.size == 1:
            break

        # IoU between the max-score box and all remaining boxes.
        max_box = boxes[max_idx]
        xx1 = np.maximum(max_box[0], x1)
        yy1 = np.maximum(max_box[1], y1)
        xx2 = np.minimum(max_box[2], x2)
        yy2 = np.minimum(max_box[3], y2)

        inter_w = np.maximum(0.0, xx2 - xx1)
        inter_h = np.maximum(0.0, yy2 - yy1)
        inter = inter_w * inter_h

        union = areas[max_idx] + areas - inter
        iou = inter / np.maximum(union, 1e-9)

        # Gaussian decay: weight = exp(-iou^2 / sigma).
        # Boxes with high IoU with the max box get heavily decayed;
        # boxes with low IoU are barely affected.
        weights = np.exp(-(iou * iou) / sigma)

        # Remove the selected box from all arrays.
        scores = np.delete(scores, max_idx)
        boxes = np.delete(boxes, max_idx, axis=0)
        x1 = np.delete(x1, max_idx)
        y1 = np.delete(y1, max_idx)
        x2 = np.delete(x2, max_idx)
        y2 = np.delete(y2, max_idx)
        areas = np.delete(areas, max_idx)
        weights = np.delete(weights, max_idx)
        order = np.delete(order, max_idx)

        # Apply the Gaussian penalty.
        scores = scores * weights

        # Prune boxes below threshold after decay.
        surviving = scores > score_thresh
        scores = scores[surviving]
        boxes = boxes[surviving]
        x1 = x1[surviving]
        y1 = y1[surviving]
        x2 = x2[surviving]
        y2 = y2[surviving]
        areas = areas[surviving]
        order = order[surviving]

    return np.array(keep, dtype=np.int64)
