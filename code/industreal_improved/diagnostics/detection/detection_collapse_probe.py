"""
detection_collapse_probe.py
===========================

Purpose
-------
The detector reports `total_preds=0` at the live eval threshold (0.05) and the
training-time GIoU term is pinned at its 1e-4 floor (predicted boxes never
overlap GT). That points to a collapsed head, but we need to know HOW collapsed:

  * Total collapse  -> even at a 0.01 threshold, boxes appear but best-IoU vs GT
                       is ~0 everywhere (head emits boxes nowhere near objects).
  * Partial         -> some predictions land on objects (IoU > 0.3/0.5); the
                       problem is only confidence calibration / score scale.

This file gives a drop-in probe for the eval loop plus a `num_pos` probe for the
training matcher, in the same spirit as the PSR diagnostic: exact decode logic
copied from the codebase, no behaviour change, just measurement.

Two drop-ins (bottom of file shows where they go):
  1. `probe_detection_batch(...)`  -> call once per eval batch (gate to first N
     batches). Lowers the score threshold to 0.01, decodes boxes, and reports
     score percentiles, prediction counts at several thresholds, and the
     best-IoU-vs-GT distribution. Logs via logging AND a flushed print.
  2. `probe_anchor_matching(...)`  -> call inside FocalLoss.forward per image to
     log num_pos / num_gt / max_iou so you can see positive-anchor starvation.

The `__main__` self-test fabricates (a) a collapsed head at the pi=0.03 prior and
(b) a head that correctly localizes one GT box, and prints what the probe sees in
each case, so you can confirm the metric is correct before trusting it on real data.

Run:  python detection_collapse_probe.py
"""

from __future__ import annotations

import logging

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("det_probe")

# Mirror config.py
IMG_WIDTH = 1280
IMG_HEIGHT = 720
NUM_DET_CLASSES = 24
DET_EVAL_MAX_PER_IMAGE = 300


# ===========================================================================
# Exact decode + IoU (decode_boxes is verbatim from evaluate.py:965)
# ===========================================================================
def decode_boxes(anchors: np.ndarray, deltas: np.ndarray) -> np.ndarray:
    a_cx = (anchors[:, 0] + anchors[:, 2]) / 2
    a_cy = (anchors[:, 1] + anchors[:, 3]) / 2
    a_w = anchors[:, 2] - anchors[:, 0]
    a_h = anchors[:, 3] - anchors[:, 1]
    dx, dy = deltas[:, 0], deltas[:, 1]
    dw = np.clip(deltas[:, 2], -4, 4)
    dh = np.clip(deltas[:, 3], -4, 4)
    pw, ph = np.exp(dw) * a_w, np.exp(dh) * a_h
    cx, cy = dx * a_w + a_cx, dy * a_h + a_cy
    return np.stack([cx - pw / 2, cy - ph / 2, cx + pw / 2, cy + ph / 2], axis=1)


def box_iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between [M,4] and [N,4] pixel xyxy boxes -> [M,N]."""
    if a.shape[0] == 0 or b.shape[0] == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    area_a = (a[:, 2] - a[:, 0]).clip(0) * (a[:, 3] - a[:, 1]).clip(0)
    area_b = (b[:, 2] - b[:, 0]).clip(0) * (b[:, 3] - b[:, 1]).clip(0)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clip(0)
    inter = wh[..., 0] * wh[..., 1]
    union = area_a[:, None] + area_b[None, :] - inter + 1e-9
    return inter / union


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


# ===========================================================================
# 1. Eval-loop drop-in
# ===========================================================================
def probe_detection_batch(
    cls_preds: np.ndarray,      # [B, N, NUM_DET_CLASSES] raw logits (pre-sigmoid)
    reg_preds: np.ndarray,      # [B, N, 4] anchor deltas
    anchors: np.ndarray,        # [N, 4] xyxy pixel anchors
    gt_boxes_per_img: list,     # list of [G_i, 4] xyxy pixel GT boxes
    probe_thresh: float = 0.01,
    iou_match: float = 0.5,
    tag: str = "",
    max_batches: int = 5,       # self-throttle: only probe the first N calls
    _state={"n": 0},
) -> dict:
    """
    Returns a summary dict and logs it. Designed to be called once per eval
    batch with numpy arrays already on CPU. Uses a LOW threshold (0.01) so a
    collapsed head still surfaces its best guesses for IoU inspection.

    Self-throttles to the first `max_batches` calls, so the drop-in is a single
    unconditional line in the eval loop -- no `if bi < 5` needed at the call site.
    Set max_batches<=0 to probe every batch.
    """
    _state["n"] += 1
    if max_batches > 0 and _state["n"] > max_batches:
        return {}
    B = cls_preds.shape[0]
    all_max_scores = []
    n_pred_001 = n_pred_005 = n_pred_030 = n_pred_050 = 0
    best_ious_all = []        # best IoU vs GT for each kept prediction
    n_gt_total = 0
    imgs_with_gt = 0

    for i in range(B):
        sig = _sigmoid(cls_preds[i])            # [N, 24]
        max_scores = sig.max(axis=1)            # [N]
        all_max_scores.append(max_scores)

        gt = np.asarray(gt_boxes_per_img[i], dtype=np.float32).reshape(-1, 4)
        n_gt_total += gt.shape[0]
        imgs_with_gt += int(gt.shape[0] > 0)

        keep = max_scores > probe_thresh
        if DET_EVAL_MAX_PER_IMAGE > 0 and keep.sum() > DET_EVAL_MAX_PER_IMAGE:
            topk = np.argpartition(max_scores, -DET_EVAL_MAX_PER_IMAGE)[-DET_EVAL_MAX_PER_IMAGE:]
            m = np.zeros_like(keep)
            m[topk] = True
            keep &= m

        ks = max_scores[keep]
        n_pred_001 += int((ks > 0.01).sum())
        n_pred_005 += int((ks > 0.05).sum())
        n_pred_030 += int((ks > 0.30).sum())
        n_pred_050 += int((ks > 0.50).sum())

        if keep.sum() > 0 and gt.shape[0] > 0:
            boxes = decode_boxes(anchors[keep], reg_preds[i][keep])
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, IMG_WIDTH)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, IMG_HEIGHT)
            ious = box_iou_xyxy(boxes, gt)      # [K, G]
            best_ious_all.append(ious.max(axis=1))   # best IoU per prediction

    max_scores_cat = np.concatenate(all_max_scores) if all_max_scores else np.zeros(0)
    best_ious = np.concatenate(best_ious_all) if best_ious_all else np.zeros(0)

    def _pct(a, q):
        return float(np.percentile(a, q)) if a.size else 0.0

    summary = {
        "tag": tag,
        "imgs": B,
        "imgs_with_gt": imgs_with_gt,
        "n_gt": n_gt_total,
        "score_p50": _pct(max_scores_cat, 50),
        "score_p99": _pct(max_scores_cat, 99),
        "score_max": float(max_scores_cat.max()) if max_scores_cat.size else 0.0,
        "preds>0.01": n_pred_001,
        "preds>0.05": n_pred_005,
        "preds>0.30": n_pred_030,
        "preds>0.50": n_pred_050,
        "bestIoU>0": int((best_ious > 1e-6).sum()),
        "bestIoU>0.1": int((best_ious > 0.1).sum()),
        "bestIoU>0.3": int((best_ious > 0.3).sum()),
        f"bestIoU>{iou_match}": int((best_ious > iou_match).sum()),
        "bestIoU_max": float(best_ious.max()) if best_ious.size else 0.0,
        "bestIoU_mean": float(best_ious.mean()) if best_ious.size else 0.0,
    }

    # Verdict uses the mAP@0.5 matching criterion, NOT IoU>0.1: with a dense
    # anchor grid a collapsed head still gets chance low-IoU overlaps, so only
    # IoU>0.5 (or a high bestIoU_max) distinguishes real localization.
    n_matched = summary[f"bestIoU>{iou_match}"]
    if n_matched == 0 and summary["bestIoU_max"] < iou_match:
        verdict = (f"TOTAL COLLAPSE (0 preds at IoU>{iou_match}, max={summary['bestIoU_max']:.2f}; "
                   "low-IoU hits are chance anchor overlap, not localization)")
    elif n_matched == 0:
        verdict = (f"NEAR-COLLAPSE (no match at {iou_match} but max IoU "
                   f"{summary['bestIoU_max']:.2f} -- head aims roughly at objects, mis-sized)")
    else:
        verdict = f"LOCALIZING ({n_matched} preds at IoU>{iou_match}) -> calibration issue, not collapse"
    msg = f"[DET_PROBE {tag}] {summary} | verdict: {verdict}"
    logger.info(msg)
    print(msg, flush=True)
    return summary


# ===========================================================================
# 2. Training matcher drop-in (paste into FocalLoss.forward, per-image loop)
# ===========================================================================
def probe_anchor_matching(
    matched_labels,             # torch tensor [N] from _match_anchors
    max_iou=None,               # optional torch tensor [N] of anchor->GT max IoU
    num_gt: int = 0,
    img_idx: int = 0,
    every: int = 200,
    _state={"n": 0},
):
    """
    Drop inside FocalLoss.forward right after `_match_anchors(...)`:

        matched_labels, matched_boxes = self._match_anchors(anchors, gt_boxes, gt_labels)
        probe_anchor_matching(matched_labels, num_gt=gt_boxes.shape[0], img_idx=i)

    Logs how many anchors are positive vs negative vs ignored, and num_gt, so
    positive-anchor starvation (the usual cause of background collapse on sparse
    GT) is visible.
    """
    _state["n"] += 1
    if every > 0 and _state["n"] % every != 0:
        return
    pos = int((matched_labels >= 0).sum())
    neg = int((matched_labels == -2).sum())
    ign = int((matched_labels == -1).sum())
    extra = ""
    if max_iou is not None:
        extra = f" max_iou[p50/p99/max]={float(max_iou.median()):.3f}/" \
                f"{float(max_iou.float().quantile(0.99)):.3f}/{float(max_iou.max()):.3f}"
    msg = (f"[MATCH_PROBE call={_state['n']} img={img_idx}] "
           f"num_gt={num_gt} pos={pos} neg={neg} ignore={ign}{extra}")
    logger.info(msg)
    print(msg, flush=True)


# ===========================================================================
# Self-test: collapsed head vs correctly-localizing head
# ===========================================================================
def _make_anchors(n_per_side: int = 40):
    """Grid of anchors tiling the image, ~constant size."""
    xs = np.linspace(40, IMG_WIDTH - 40, n_per_side)
    ys = np.linspace(40, IMG_HEIGHT - 40, n_per_side)
    cx, cy = np.meshgrid(xs, ys)
    cx, cy = cx.ravel(), cy.ravel()
    w = h = 120.0
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)


if __name__ == "__main__":
    print("=" * 90)
    print("detection_collapse_probe self-test")
    print("=" * 90)
    rng = np.random.default_rng(0)
    anchors = _make_anchors(40)            # [1600, 4]
    N = anchors.shape[0]
    B = 4

    # One GT box per image, placed near image centre.
    gt = [np.array([[560, 280, 760, 480]], dtype=np.float32) for _ in range(B)]

    # --- Case A: collapsed head at the pi=0.03 prior, random tiny deltas ---
    cls_collapsed = np.full((B, N, NUM_DET_CLASSES), -3.476, dtype=np.float32)  # sigmoid~0.03
    cls_collapsed += rng.normal(0, 0.05, cls_collapsed.shape)
    reg_collapsed = rng.normal(0, 0.02, (B, N, 4)).astype(np.float32)
    print("\n--- Case A: collapsed head (prior 0.03, no localization) ---")
    probe_detection_batch(cls_collapsed, reg_collapsed, anchors, gt, tag="collapsed", max_batches=0)

    # --- Case B: a few anchors near the GT fire high AND regress onto it ---
    cls_good = np.full((B, N, NUM_DET_CLASSES), -3.476, dtype=np.float32)
    reg_good = rng.normal(0, 0.02, (B, N, 4)).astype(np.float32)
    # find anchors whose centre is near GT centre and make them confident class 5
    a_cx = (anchors[:, 0] + anchors[:, 2]) / 2
    a_cy = (anchors[:, 1] + anchors[:, 3]) / 2
    near = (np.abs(a_cx - 660) < 80) & (np.abs(a_cy - 380) < 80)
    for b in range(B):
        cls_good[b, near, 5] = 3.0                      # sigmoid ~0.95
        # regress those anchors onto the GT box (encode exact deltas)
        an = anchors[near]
        acx = (an[:, 0] + an[:, 2]) / 2; acy = (an[:, 1] + an[:, 3]) / 2
        aw = an[:, 2] - an[:, 0]; ah = an[:, 3] - an[:, 1]
        gcx, gcy, gw, gh = 660, 380, 200, 200
        reg_good[b, near, 0] = (gcx - acx) / aw
        reg_good[b, near, 1] = (gcy - acy) / ah
        reg_good[b, near, 2] = np.log(gw / aw)
        reg_good[b, near, 3] = np.log(gh / ah)
    print("\n--- Case B: head correctly localizes the GT box ---")
    probe_detection_batch(cls_good, reg_good, anchors, gt, tag="localizing", max_batches=0)

    print("\nInterpretation:")
    print("  Case A should read TOTAL COLLAPSE: preds appear at 0.01 but bestIoU>0.1 == 0.")
    print("  Case B should read PARTIAL/healthy: bestIoU>0.5 > 0 and bestIoU_max ~ 1.0.")
    print("  On your real eval, run probe at the FIRST few batches; if it looks like")
    print("  Case A, the head emits boxes nowhere near objects -> recovery must restore")
    print("  positive-anchor learning (see MATCH_PROBE num_pos), not just confidence.")
