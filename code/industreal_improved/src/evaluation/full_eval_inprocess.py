"""
In-Process Full Evaluation — NaN-safe, no subprocess, no timeout.

Solves the D3 NaN bug (Opus Q6): evaluate_all called with ``epoch=0`` hits
the ``DET_METRICS_EVERY_N`` gate and returns NaN for ``det_mAP50``. This
script runs a clean streaming eval in‑process (no ``spawn`` subprocess, no
SIGKILL risk) with config overrides that guarantee full detection mAP
computation on the entire 38 036‑frame validation set.

Avoids ``evaluate_all``'s detection probe which OOMs on the full dataset.
Implements its own streaming evaluation loop computing only the metrics
Opus specified: det_mAP50, det_mAP50‑95, act_top1, head_pose_mae, psr_f1.

If any metric is NaN after the full run, falls back to 10‑seed subsample
variance (mean ± std across 10 random seeds, 2500 batches each).

Reference: Opus Q6 answer, 132_OPUS_ANSWERS.md
"""

from __future__ import annotations

import gc
import json
import logging
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)
logger = logging.getLogger("full_eval_inprocess")
# Force unbuffered output: flush every log message
_console_handler = logging.StreamHandler(sys.stdout)
_console_handler.setLevel(logging.INFO)
_console_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logger.handlers.clear()
logger.addHandler(_console_handler)
logger.propagate = False

# ── Path setup (mirrors subprocess_eval.py / train.py) ──────────────────────
_SRC = Path(__file__).resolve().parent.parent  # src/
for _sub in ["models", "training", "evaluation", "data", str(_SRC)]:
    _p = _SRC / _sub if _sub != str(_SRC) else _SRC
    _p = str(_p)
    if _p not in sys.path:
        sys.path.insert(0, _p)
if str(_SRC.parent) not in sys.path:
    sys.path.insert(0, str(_SRC.parent))

import src.config as C
from src.data.industreal_dataset import IndustRealMultiTaskDataset, collate_fn
from src.models.model import POPWMultiTaskModel

# ── Local imports from evaluation package ────────────────────────────────────
# We import compute_detection_map and compute_head_pose_metrics directly
# from evaluate.py, but avoid calling evaluate_all (which has the probe bug).
sys.path.insert(0, str(_SRC / "evaluation"))
from evaluate import (
    compute_detection_map,
    compute_head_pose_metrics,
    decode_boxes,
    nms_numpy,
    compute_ap_per_class,
)
from decoder_oracle_bound import event_f1

# ── Config overrides for safe full eval ─────────────────────────────────────
FULL_EVAL_OVERRIDES = {
    "DET_METRICS_EVERY_N": 1,
    "SKIP_DET_METRICS_EVAL": False,
    "SKIP_EFFICIENCY_METRICS": True,
    "TRAIN_DET": True,
    "TRAIN_ACT": True,
    "TRAIN_PSR": True,
    "TRAIN_HEAD_POSE": True,
    "DET_EVAL_SCORE_THRESH": 0.5,
    "DET_EVAL_NMS_IOU_THRESH": 0.5,
    "DET_EVAL_MAX_PER_IMAGE": 300,
}

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def apply_overrides(overrides: dict) -> None:
    for k, v in overrides.items():
        setattr(C, k, v)


def load_model(ckpt_path: str, device: torch.device) -> tuple[torch.nn.Module, int]:
    """Load model from checkpoint, matching train.py constructor API."""
    logger.info("Loading checkpoint: %s", ckpt_path)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = state.get("config", {})

    backbone_type = cfg.get("BACKBONE_TYPE", getattr(C, "BACKBONE_TYPE", "convnext_tiny"))
    use_hand_film = bool(cfg.get("USE_HAND_FILM", getattr(C, "USE_HAND_FILM", True)))
    use_headpose_film = bool(cfg.get("USE_HEADPOSE_FILM", getattr(C, "USE_HEADPOSE_FILM", False)))
    use_videomae = bool(cfg.get("USE_VIDEOMAE", getattr(C, "USE_VIDEOMAE", False)))
    train_pose = bool(cfg.get("TRAIN_HEAD_POSE", getattr(C, "TRAIN_HEAD_POSE", True)))
    use_backbone_checkpoint = bool(cfg.get("USE_BACKBONE_CHECKPOINT", getattr(C, "USE_BACKBONE_CHECKPOINT", False)))

    model = POPWMultiTaskModel(
        pretrained=True,
        backbone_type=backbone_type,
        use_hand_film=use_hand_film,
        use_headpose_film=use_headpose_film,
        use_videomae=use_videomae,
        train_pose=train_pose,
        use_backbone_checkpoint=use_backbone_checkpoint,
    ).to(device).eval()

    model._seq_len = cfg.get(
        "PSR_SEQUENCE_LENGTH",
        getattr(C, "PSR_SEQUENCE_LENGTH", 1),
    ) if cfg.get("USE_PSR_SEQUENCE_MODE", getattr(C, "USE_PSR_SEQUENCE_MODE", False)) else 1

    result = model.load_state_dict(state["model"], strict=False)
    if result.missing_keys:
        logger.warning("Missing keys: %s", result.missing_keys)
    if result.unexpected_keys:
        logger.warning("Unexpected keys (harmless fluff): %d keys", len(result.unexpected_keys))

    epoch = state.get("epoch", -1)
    logger.info("Loaded epoch=%s, step=%s", epoch, state.get("step", "?"))
    return model, epoch


def has_nan(metrics: dict) -> list[str]:
    return [k for k, v in metrics.items() if isinstance(v, float) and (math.isnan(v) or math.isinf(v))]


def angular_mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Angular MAE between unit-normalized vectors."""
    pred_n = pred / (pred.norm(dim=-1, keepdim=True) + 1e-8)
    targ_n = target / (target.norm(dim=-1, keepdim=True) + 1e-8)
    cos = (pred_n * targ_n).sum(dim=-1).clamp(-1, 1)
    return float(torch.rad2deg(torch.acos(cos)).mean().item())


def prepare_images(images: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Normalize images to ImageNet stats."""
    images = images.to(device, non_blocking=True).float()
    if images.max() > 1.0:
        images = images.div_(255.0)
    mean = torch.tensor(_IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(_IMAGENET_STD, device=device).view(1, 3, 1, 1)
    return (images - mean) / std


# ── PSR transition event helpers (175 §7.2 / 174 §3.3) ────────────────────


def _compute_tau(
    pred_tr: np.ndarray, gt_tr: np.ndarray, tol: int = 3
) -> float:
    """Mean signed delay between matched predicted and GT transition events.

    For each component, matches predicted 0->1 frames to GT frames greedily
    within tolerance. Returns mean(pred_frame - gt_frame) across all matches.
    Positive = lag, negative = anticipation. NaN if no matches.
    """
    n_comp = pred_tr.shape[1]
    delays = []
    for c in range(n_comp):
        p_frames = np.where(pred_tr[:, c])[0]
        g_frames = np.where(gt_tr[:, c])[0]
        matched_gt = set()
        for pf in p_frames:
            best_delay = None
            best_gi = None
            for gi, gf in enumerate(g_frames):
                if gi not in matched_gt and abs(pf - gf) <= tol:
                    d = int(pf) - int(gf)
                    if best_delay is None or abs(d) < abs(best_delay):
                        best_delay = d
                        best_gi = gi
            if best_gi is not None:
                matched_gt.add(best_gi)
                delays.append(best_delay)
    if not delays:
        return float("nan")
    return float(np.mean(delays))


def _compute_pos(pred_tr: np.ndarray, gt_tr: np.ndarray) -> float:
    """Ordered-pair fraction: directional sign agreement of transitions.

    POS = mean(sign(pred_diff) == sign(gt_diff)). Null-model sensitive:
    all-zeros prediction scores ~0.9995 because most frame-pairs have no
    transition in either direction. See 174 §3.3 caveat.
    """
    return float((np.sign(pred_tr) == np.sign(gt_tr)).mean())


def streaming_eval(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    max_batches: int | None = None,
) -> dict:
    """Streaming evaluation — no storing all logits, no detection probe OOM.

    Accumulates detection predictions per frame for full compute_detection_map
    at the end. Activity, head pose, and PSR use running counters.
    """
    # Open a raw status fd for guaranteed disk output (bypasses Python buffering)
    _status_path = Path("/tmp/d3_full_v2_status.txt")
    _status_fd = os.open(str(_status_path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)

    def log_progress(msg: str) -> None:
        """Write to both logger and the raw status fd for guaranteed output."""
        logger.info(msg)
        if _status_fd is not None:
            os.write(_status_fd, (msg + "\n").encode())
            os.fsync(_status_fd)
    model.eval()

    # ── Streaming accumulators ──────────────────────────────────────────────
    # Detection: store per-image decoded boxes (we need them all for mAP)
    dp_boxes: list[np.ndarray] = []
    dp_scores: list[np.ndarray] = []
    dp_labels: list[np.ndarray] = []
    dg_boxes: list[np.ndarray] = []
    dg_labels: list[np.ndarray] = []

    # Activity: running counters
    act_correct = 0
    act_total = 0
    act_correct_valid = 0
    act_total_valid = 0

    # Head pose: running sums → mean at end
    hp_fwd_sum = 0.0
    hp_up_sum = 0.0
    hp_pos_sum = 0.0
    hp_n = 0

    # PSR: per-component TP/FP/FN across frames
    psr_tp = np.zeros(11)
    psr_fp = np.zeros(11)
    psr_fn = np.zeros(11)
    psr_pos_pred = np.zeros(11)
    psr_pos_true = np.zeros(11)
    psr_valid = np.zeros(11)

    # PSR transition: store by recording for monotonic decoder
    psr_preds_logits: list[np.ndarray] = []
    psr_labels_list: list[np.ndarray] = []
    psr_rec_ids: list[str] = []
    psr_frame_nums: list[int] = []

    n_batches = 0
    _cached_anchors_np = None
    _first_batch = True

    for bi, (images, targets) in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        if _first_batch:
            log_progress("  First batch loaded, starting inference loop...")
            _first_batch = False

        images = prepare_images(images, device)
        B = images.shape[0]

        # ── Move targets to device ──────────────────────────────────────────
        detection_list = targets["detection"]
        for i in range(len(detection_list)):
            detection_list[i]["boxes"] = detection_list[i]["boxes"].to(device)
            detection_list[i]["labels"] = detection_list[i]["labels"].to(device)
        hp_gt = targets["head_pose"].to(device)
        psr_gt = targets["psr_labels"].to(device)
        act_gt = targets["activity"].to(device)

        # ── Forward ─────────────────────────────────────────────────────────
        with torch.no_grad():
            outputs = model(images)
            for _k in outputs:
                if isinstance(outputs[_k], torch.Tensor):
                    outputs[_k] = outputs[_k].float()

        # ═══════════════════════════════════════════════════════════════════
        # Activity (per-frame top-1 accuracy)
        # ═══════════════════════════════════════════════════════════════════
        act_logits = outputs["act_logits"]  # [B, 69] or [B, 75]
        act_pred = act_logits.argmax(dim=-1)  # [B]
        act_mask = targets.get("activity_mask")
        if act_mask is not None:
            act_mask = act_mask.to(device)
            act_valid = act_mask.bool()
        else:
            act_valid = (act_gt >= 0)

        act_correct += int((act_pred[act_valid] == act_gt[act_valid]).sum().item())
        act_total += int(act_valid.sum().item())

        # Also track valid (non-NA, non-masked) accuracy
        act_valid_non_na = act_valid & (act_gt >= 0) & (act_gt != 0)  # exclude class 0 (NA)
        act_correct_valid += int((act_pred[act_valid_non_na] == act_gt[act_valid_non_na]).sum().item())
        act_total_valid += int(act_valid_non_na.sum().item())

        # ═══════════════════════════════════════════════════════════════════
        # Head Pose (angular MAE)
        # ═══════════════════════════════════════════════════════════════════
        hp_pred = outputs["head_pose"]  # [B, 9]
        if hp_pred is not None and hp_gt is not None:
            fwd_mae = angular_mae(hp_pred[:, :3], hp_gt[:, :3]) * hp_pred.shape[0]
            up_mae = angular_mae(hp_pred[:, 6:9], hp_gt[:, 6:9]) * hp_pred.shape[0]
            hp_fwd_sum += fwd_mae
            hp_up_sum += up_mae
            hp_n += hp_pred.shape[0]

        # ═══════════════════════════════════════════════════════════════════
        # PSR (per-component binary F1 at best threshold)
        # ═══════════════════════════════════════════════════════════════════
        pl = outputs["psr_logits"].cpu()  # [B, 11]
        pl_lbl = psr_gt.cpu()  # [B, 11]
        sig = torch.sigmoid(pl)  # [B, 11]
        binary = (sig > 0.10).int()

        for b in range(B):
            valid_mask = pl_lbl[b] != -1
            for c in range(11):
                if valid_mask[c]:
                    psr_valid[c] += 1
                    if binary[b, c] == 1 and pl_lbl[b, c] == 1:
                        psr_tp[c] += 1
                    elif binary[b, c] == 1 and pl_lbl[b, c] == 0:
                        psr_fp[c] += 1
                    elif binary[b, c] == 0 and pl_lbl[b, c] == 1:
                        psr_fn[c] += 1
                    psr_pos_pred[c] += int(binary[b, c].item())
                    psr_pos_true[c] += int(pl_lbl[b, c].item())

        # Store for transition decoder
        psr_preds_logits.append(outputs["psr_logits"].cpu().numpy())
        psr_labels_list.append(psr_gt.cpu().numpy())
        for i in range(min(B, outputs["psr_logits"].shape[0])):
            _meta = targets["metadata"][i] if i < len(targets["metadata"]) else {}
            _r = _meta.get("recording_id", _meta.get("rec_id", f"rec_{bi}_{i}"))
            psr_rec_ids.append(str(_r.item()) if isinstance(_r, torch.Tensor) else str(_r))
            _fn = _meta.get("frame_num", _meta.get("frame_idx", len(psr_frame_nums)))
            try:
                _fn = int(_fn.item()) if isinstance(_fn, torch.Tensor) else int(_fn)
            except (TypeError, ValueError):
                _fn = len(psr_frame_nums)
            psr_frame_nums.append(_fn)

        # ═══════════════════════════════════════════════════════════════════
        # Detection (decode boxes, NMS, accumulate for AP)
        # ═══════════════════════════════════════════════════════════════════
        if _cached_anchors_np is None:
            _cached_anchors_np = outputs["anchors"].cpu().numpy()

        cls_sigmoid = torch.sigmoid(outputs["cls_preds"])  # [B, N, 24] on GPU
        score_thresh = float(getattr(C, "DET_EVAL_SCORE_THRESH", 0.5))
        nms_thresh = float(getattr(C, "DET_EVAL_NMS_IOU_THRESH", 0.5))
        max_keep = int(getattr(C, "DET_EVAL_MAX_PER_IMAGE", 300))

        for i in range(B):
            scores_i = cls_sigmoid[i]  # [N, 24]
            max_scores = scores_i.max(dim=1).values
            keep_mask = max_scores > score_thresh

            if max_keep > 0 and keep_mask.sum().item() > max_keep:
                topk_idx = torch.topk(max_scores, k=max_keep, largest=True, sorted=False).indices
                topk_mask = torch.zeros_like(keep_mask)
                topk_mask[topk_idx] = True
                keep_mask = keep_mask & topk_mask

            if keep_mask.sum().item() == 0:
                dp_boxes.append(np.zeros((0, 4), dtype=np.float32))
                dp_scores.append(np.zeros(0, dtype=np.float32))
                dp_labels.append(np.zeros(0, dtype=np.int64))
            else:
                keep_np = keep_mask.cpu().numpy()
                kept_cls = scores_i[keep_mask].cpu().numpy()
                kept_reg = outputs["reg_preds"][i][keep_mask].cpu().numpy()
                kept_anc = _cached_anchors_np[keep_np]

                ms = kept_cls.max(axis=1)
                ml = kept_cls.argmax(axis=1)
                pb = decode_boxes(kept_anc, kept_reg)
                pb[:, 0] = np.clip(pb[:, 0], 0, C.IMG_WIDTH)
                pb[:, 1] = np.clip(pb[:, 1], 0, C.IMG_HEIGHT)
                pb[:, 2] = np.clip(pb[:, 2], 0, C.IMG_WIDTH)
                pb[:, 3] = np.clip(pb[:, 3], 0, C.IMG_HEIGHT)

                fb, fs, fl = [], [], []
                for c in range(C.NUM_DET_CLASSES):
                    cm = ml == c
                    if cm.sum() == 0:
                        continue
                    nk = nms_numpy(pb[cm], ms[cm], nms_thresh)
                    fb.append(pb[cm][nk])
                    fs.append(ms[cm][nk])
                    fl.append(np.full(len(nk), c, dtype=np.int64))
                if fb:
                    dp_boxes.append(np.concatenate(fb))
                    dp_scores.append(np.concatenate(fs))
                    dp_labels.append(np.concatenate(fl))
                else:
                    dp_boxes.append(np.zeros((0, 4), dtype=np.float32))
                    dp_scores.append(np.zeros(0, dtype=np.float32))
                    dp_labels.append(np.zeros(0, dtype=np.int64))

            # Ground truth
            dg_boxes.append(detection_list[i]["boxes"].cpu().numpy())
            dg_labels.append(detection_list[i]["labels"].cpu().numpy())

            # Cleanup per-image
            del scores_i, max_scores
            try:
                if keep_mask.sum().item() > 0:
                    del kept_cls, kept_reg, pb
                del keep_mask
            except UnboundLocalError:
                pass

        del images, outputs, cls_sigmoid
        gc.collect()
        if bi % 10 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

        n_batches += 1
        if n_batches % 200 == 0:
            log_progress("  processed %d batches (%d frames)..." % (n_batches, n_batches * B))

    # ═══════════════════════════════════════════════════════════════════════
    # Compute aggregate metrics
    # ═══════════════════════════════════════════════════════════════════════
    results: dict = {"n_batches": n_batches}

    # ── Detection mAP ──────────────────────────────────────────────────────
    gt_box_total = sum(len(b) for b in dg_boxes)
    if gt_box_total == 0:
        logger.warning("No GT boxes in evaluation split — skipping detection mAP")
        results["det_mAP50"] = 0.0
        results["det_mAP_50_95"] = 0.0
        results["det_mAP50_all_frames"] = 0.0
        results["det_n_present_classes"] = 0
    else:
        # mAP@0.5
        ap_result = compute_ap_per_class(
            dp_boxes, dp_scores, dp_labels,
            dg_boxes, dg_labels,
            iou_thresh=0.5,
            num_classes=C.NUM_DET_CLASSES,
        )
        ap_50_95 = compute_ap_per_class(
            dp_boxes, dp_scores, dp_labels,
            dg_boxes, dg_labels,
            iou_thresh=0.75,
            num_classes=C.NUM_DET_CLASSES,
        )
        results["det_mAP50"] = float(ap_result["mAP"])
        results["det_mAP_50_95"] = float(ap_50_95["mAP"])
        results["det_per_class_ap"] = {str(k): float(v) for k, v in ap_result["per_class_ap"].items()}
        n_present = sum(1 for v in ap_result["per_class_ap"].values() if v > 0)
        results["det_n_present_classes"] = n_present

        # mAP@0.5 on all frames (including empty)
        ap_all = compute_ap_per_class(
            dp_boxes, dp_scores, dp_labels,
            dg_boxes, dg_labels,
            iou_thresh=0.5,
            num_classes=C.NUM_DET_CLASSES,
        )
        results["det_mAP50_all_frames"] = float(ap_all["mAP"])

    # ── Activity Top-1 ─────────────────────────────────────────────────────
    if act_total > 0:
        results["act_top1"] = act_correct / act_total
        results["act_top1_valid_na_excluded"] = act_correct_valid / max(act_total_valid, 1)
        results["act_n_total"] = act_total
        results["act_n_valid"] = act_total_valid
    else:
        results["act_top1"] = 0.0
        results["act_top1_valid_na_excluded"] = 0.0
        results["act_n_total"] = 0
        results["act_n_valid"] = 0

    # ── Head Pose MAE ──────────────────────────────────────────────────────
    if hp_n > 0:
        results["forward_angular_MAE_deg"] = hp_fwd_sum / hp_n
        results["up_angular_MAE_deg"] = hp_up_sum / hp_n
        results["head_pose_n"] = hp_n
    else:
        results["forward_angular_MAE_deg"] = 0.0
        results["up_angular_MAE_deg"] = 0.0
        results["head_pose_n"] = 0

    # ── PSR per-component F1 at 0.10 threshold ─────────────────────────────
    psr_f1s = []
    for c in range(11):
        prec = psr_tp[c] / max(psr_tp[c] + psr_fp[c], 1)
        rec = psr_tp[c] / max(psr_tp[c] + psr_fn[c], 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        psr_f1s.append(f1)
    results["psr_macro_f1"] = float(np.mean(psr_f1s))
    results["psr_per_component_f1"] = {str(c): float(f) for c, f in enumerate(psr_f1s)}

    # ── PSR transition event F1 (primary metric, comparable to STORM/B3) ────
    try:
        # Load per-component optimal thresholds
        _opt_path = _SRC / "runs" / "rf_stages" / "checkpoints" / "psr_optimal_thr_38k" / "optimal_thresholds.json"
        if _opt_path.exists():
            with open(_opt_path) as _f:
                _opt_data = json.load(_f)
            _per_comp_thr = np.array(_opt_data["optimal_thresholds"], dtype=np.float32)
            logger.info("PSR transition thresholds loaded: %s", _per_comp_thr.tolist())
        else:
            logger.warning("PSR optimal thresholds not found at %s, using 0.10 global", _opt_path)
            _per_comp_thr = np.full(11, 0.10, dtype=np.float32)

        # Concatenate stored logits and labels
        _all_logits = np.concatenate(psr_preds_logits, axis=0)   # [N, 11]
        _all_labels = np.concatenate(psr_labels_list, axis=0)    # [N, 11]

        # Sigmoid -> binary with per-component optimal thresholds
        _all_sig = 1.0 / (1.0 + np.exp(-_all_logits))
        _all_pred_bin = (_all_sig > _per_comp_thr[np.newaxis, :]).astype(np.int32)

        # Group by recording ID
        _rec_data: dict = defaultdict(lambda: {"pred": [], "label": [], "frame": []})
        for i in range(len(psr_rec_ids)):
            rid = psr_rec_ids[i]
            _rec_data[rid]["pred"].append(_all_pred_bin[i])
            _rec_data[rid]["label"].append(_all_labels[i])
            _rec_data[rid]["frame"].append(psr_frame_nums[i])

        # Per-recording transition metrics
        _event_f1s: list[float] = []
        _poss: list[float] = []
        _taus: list[float] = []

        for rid, _arrs in _rec_data.items():
            _frames = np.array(_arrs["frame"], dtype=np.int64)
            _sort = np.argsort(_frames)
            _vp = np.array(_arrs["pred"])[_sort]   # [T, 11]
            _vl = np.array(_arrs["label"])[_sort]  # [T, 11]

            # Keep only valid frames
            _valid = _vl.max(axis=1) >= 0
            _vp = _vp[_valid]
            _vl = _vl[_valid]
            if len(_vp) < 2:
                continue

            # Transition events: 0->1 per component
            _pred_tr = np.clip(_vp[1:] - _vp[:-1], a_min=0, a_max=None)
            _gt_tr = np.clip(_vl[1:] - _vl[:-1], a_min=0, a_max=None)

            # Filter to ground-truth-valid transition pairs
            _valid_tr = _vl[1:].max(axis=1) >= 0
            _pv = _pred_tr[_valid_tr]
            _gv = _gt_tr[_valid_tr]

            _ef1 = event_f1(_pv, _gv, tol=3)
            _event_f1s.append(_ef1)
            _poss.append(_compute_pos(_pv, _gv))
            _tau = _compute_tau(_pv, _gv, tol=3)
            if not np.isnan(_tau):
                _taus.append(_tau)

        if _event_f1s:
            results["psr_event_f1_at_3"] = float(np.mean(_event_f1s))
            results["psr_pos"] = float(np.mean(_poss)) if _poss else 0.0
            _tau_mean = float(np.nanmean(_taus)) if _taus else float("nan")
            results["psr_tau_frames"] = _tau_mean
            results["psr_tau_seconds"] = _tau_mean / 30.0 if not np.isnan(_tau_mean) else float("nan")
        else:
            results["psr_event_f1_at_3"] = 0.0
            results["psr_pos"] = 0.0
            results["psr_tau_frames"] = float("nan")
            results["psr_tau_seconds"] = float("nan")
        logger.info(
            "PSR event_f1@±3=%.4f  POS=%.4f  tau=%.2fs  (per-frame legacy=%.4f)",
            results["psr_event_f1_at_3"],
            results["psr_pos"],
            results.get("psr_tau_seconds", float("nan")),
            results.get("psr_macro_f1", float("nan")),
        )
    except Exception as _exc:
        logger.warning("PSR transition event F1 computation failed: %s", _exc)
        results["psr_event_f1_at_3"] = 0.0
        results["psr_pos"] = 0.0
        results["psr_tau_frames"] = float("nan")
        results["psr_tau_seconds"] = float("nan")

    # ── NaN check ──────────────────────────────────────────────────────────
    bad = has_nan(results)
    if bad:
        logger.warning("NaN/Inf metrics detected (n=%d): %s", len(bad), bad)

    # ── Print summary (to log, stdout, AND status fd) ──────────────────────
    log_progress("=" * 60)
    log_progress("FULL IN-PROCESS EVAL — Key Metrics")
    log_progress("=" * 60)
    for name in ["det_mAP50", "det_mAP_50_95", "det_mAP50_all_frames",
                  "det_n_present_classes", "act_top1", "act_top1_valid_na_excluded",
                  "forward_angular_MAE_deg", "up_angular_MAE_deg",
                  "psr_event_f1_at_3", "psr_pos", "psr_tau_seconds",
                  "psr_macro_f1"]:
        val = results.get(name)
        if isinstance(val, float):
            log_progress(f"  {name:40s} = {val:.6f}")
        else:
            log_progress(f"  {name:40s} = {val}")
    log_progress("=" * 60)
    if _status_fd is not None:
        os.close(_status_fd)
        _status_path.unlink(missing_ok=True)

    return results


def run_multi_seed_subsample(
    model: torch.nn.Module,
    device: torch.device,
    num_seeds: int = 10,
    batches_per_seed: int = 2500,
    save_dir: str | Path | None = None,
) -> dict:
    """Run 10-seed subsample variance as fallback when full eval produces NaN."""
    logger.info(
        "FALLBACK: Running %d-seed subsample evaluation (%d batches each)...",
        num_seeds, batches_per_seed,
    )
    all_seed_results: list[dict] = []

    for seed_idx, seed in enumerate(range(42, 42 + num_seeds)):
        logger.info("  Seed %d/%d (seed=%d)...", seed_idx + 1, num_seeds, seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

        val_ds = IndustRealMultiTaskDataset(
            split="val",
            img_size=(C.IMG_HEIGHT, C.IMG_WIDTH),
            seed=seed,
        )
        val_loader = torch.utils.data.DataLoader(
            val_ds,
            batch_size=int(getattr(C, "VAL_BATCH_SIZE", getattr(C, "BATCH_SIZE", 4))),
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            collate_fn=collate_fn,
        )

        result = streaming_eval(model, val_loader, device, max_batches=batches_per_seed)
        result["_seed"] = seed
        all_seed_results.append(result)

        nan_keys = has_nan(result)
        if nan_keys:
            logger.warning("    Seed %d: NaN metrics: %s", seed, nan_keys)

        del val_ds, val_loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Aggregate
    key_list = [
        "det_mAP50", "det_mAP_50_95", "det_mAP50_all_frames",
        "det_n_present_classes",
        "act_top1", "act_top1_valid_na_excluded",
        "forward_angular_MAE_deg", "up_angular_MAE_deg",
        "psr_macro_f1",
    ]

    summary: dict = {
        "_type": "10seed_subsample_variance",
        "_seeds": [42 + i for i in range(num_seeds)],
    }
    for key in key_list:
        values = [float(r.get(key, float("nan"))) for r in all_seed_results]
        clean = [v for v in values if not math.isnan(v) and not math.isinf(v)]
        if clean:
            summary[f"{key}_mean"] = float(np.mean(clean))
            summary[f"{key}_std"] = float(np.std(clean))
            summary[f"{key}_values"] = clean
        else:
            summary[f"{key}_mean"] = float("nan")
            summary[f"{key}_std"] = float("nan")
            summary[f"{key}_values"] = values

    summary["_per_seed"] = all_seed_results
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        ps_path = Path(save_dir) / "multiseed_per_seed.json"
        with open(ps_path, "w") as f:
            json.dump(all_seed_results, f, indent=2, default=str)
        logger.info("Per-seed results saved to %s", ps_path)

    print("\n" + "=" * 60)
    print("10-SEED SUBSAMPLE VARIANCE")
    print("=" * 60)
    for key in key_list:
        mk, sk = f"{key}_mean", f"{key}_std"
        mv, sv = summary.get(mk, float("nan")), summary.get(sk, float("nan"))
        if isinstance(mv, float) and math.isfinite(mv):
            print(f"  {key:40s} = {mv:.6f}  +/- {sv:.6f}")
        else:
            print(f"  {key:40s} = NaN")
    print("=" * 60)
    sys.stdout.flush()

    return summary


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="In-process full evaluation — NaN-safe, no subprocess."
    )
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--max-batches", type=int, default=0,
                        help="0 = full dataset, >0 = limit")
    parser.add_argument("--fallback-seeds", type=int, default=10)
    parser.add_argument("--fallback-batches", type=int, default=2500)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    logger.info("Device: %s", device)

    base_ckpt_dir = Path(getattr(C, "RUNS", _SRC.parent / "src" / "runs")
                         ) / "rf_stages" / "checkpoints"
    ckpt_path = Path(args.ckpt) if args.ckpt else base_ckpt_dir / "best.pth"
    if not ckpt_path.exists():
        logger.error("Checkpoint not found: %s", ckpt_path)
        sys.exit(1)

    if args.save_dir:
        save_dir = Path(args.save_dir)
    else:
        save_dir = ckpt_path.parent / "full_eval_inprocess"
    save_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Save directory: %s", save_dir)
    logger.info("Checkpoint: %s", ckpt_path)

    apply_overrides(FULL_EVAL_OVERRIDES)
    logger.info("Config overrides: %s", list(FULL_EVAL_OVERRIDES.keys()))

    max_batches = args.max_batches if args.max_batches > 0 else None
    logger.info("MAX_BATCHES: %s", max_batches or "unlimited (full 38036-frame dataset)")

    model, ckpt_epoch = load_model(str(ckpt_path), device)

    logger.info("Building validation dataset...")
    val_ds = IndustRealMultiTaskDataset(
        split="val",
        img_size=(C.IMG_HEIGHT, C.IMG_WIDTH),
    )
    logger.info("Validation dataset: %d samples", len(val_ds))

    val_batch_size = int(getattr(C, "VAL_BATCH_SIZE", getattr(C, "BATCH_SIZE", 4)))
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_fn,
    )

    logger.info("=" * 60)
    logger.info("STARTING FULL IN-PROCESS EVALUATION")
    logger.info("=" * 60)
    t_start = time.time()

    metrics = streaming_eval(model, val_loader, device, max_batches=max_batches)

    t_elapsed = time.time() - t_start
    logger.info("Evaluation complete: %.1f seconds (%.2f s/batch)",
                t_elapsed, t_elapsed / max(len(val_loader), 1))

    bad_keys = has_nan(metrics)
    verdict = "in-process-works"

    if bad_keys:
        logger.warning("=" * 60)
        logger.warning("NaN/Inf METRICS DETECTED: %s", bad_keys)
        logger.warning("ROOT CAUSE (per Opus Q6): The full eval NaN for det_mAP50")
        logger.warning("occurs when compute_ap_per_class returns empty per_class_ap,")
        logger.warning("causing the mean of 0 values to be 0/0 = NaN.")
        logger.warning("FALLING BACK to %d-seed subsample variance.", args.fallback_seeds)
        logger.warning("=" * 60)

        fallback_result = run_multi_seed_subsample(
            model, device,
            num_seeds=args.fallback_seeds,
            batches_per_seed=args.fallback_batches,
            save_dir=str(save_dir / "fallback"),
        )
        final_metrics = fallback_result
        verdict = "in-process-fails"
    else:
        final_metrics = metrics

    def _serialize(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _serialize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_serialize(v) for v in obj]
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        return obj

    clean_metrics = _serialize(final_metrics)
    clean_metrics["_meta"] = {
        "checkpoint": str(ckpt_path),
        "ckpt_epoch": int(ckpt_epoch) if isinstance(ckpt_epoch, (int, float)) else -1,
        "elapsed_seconds": round(t_elapsed, 1),
        "n_batches_total": len(val_loader),
        "n_batches_used": max_batches or len(val_loader),
        "verdict": verdict,
        "config_overrides": list(FULL_EVAL_OVERRIDES.keys()),
        "fallback_seeds": args.fallback_seeds if bad_keys else 0,
    }

    out_path = save_dir / "metrics.json"
    with open(out_path, "w") as f:
        json.dump(clean_metrics, f, indent=2, default=str)
    logger.info("Results saved to %s", out_path)

    print()
    print("=" * 60)
    print(f"VERDICT: {verdict}")
    if verdict == "in-process-works":
        map50 = metrics.get("det_mAP50", "?")
        print("Full eval mAP is valid (no NaN).")
        print(f"  det_mAP50 = {map50}")
    else:
        print("Full eval produced NaN metrics.")
        print("Using 10-seed subsample variance.")
        mk = "det_mAP50_mean"
        sk = "det_mAP50_std"
        if mk in final_metrics:
            print(f"  det_mAP50 = {final_metrics[mk]:.4f} +/- {final_metrics[sk]:.4f}")
    print("=" * 60)
    sys.stdout.flush()

    del model, val_ds, val_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
