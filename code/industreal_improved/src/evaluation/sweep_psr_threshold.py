"""PSR Mode A rescue: sweep threshold on best.pt checkpoint.

[Opus 126 §0.2 #4 Mode A] Inference-only fix: find thresholds for sigmoid>HI
that give meaningful PSR metrics (currently F1=0 from all-ones collapse).

Tracks recording_id + frame_num so we can sort temporally for decoder eval.

Usage: python3 src/evaluation/sweep_psr_threshold.py
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def decode_monotone(probs, sustain_hi=0.5, sustain_lo=0.3, sustain_min=3):
    """Simplified monotone decoder with hysteresis. Matches MonotonicDecoder."""
    T, C = probs.shape
    states = torch.zeros(T, C)
    current = torch.zeros(C)
    counter = torch.zeros(C)
    for t in range(T):
        above_lo = (probs[t] > sustain_lo).float()
        counter = counter * above_lo + above_lo
        sustained = (counter >= sustain_min)
        high_now = (probs[t] > sustain_hi)
        transition = high_now & sustained & (current == 0)
        current = (current + transition.float()).clamp(max=1.0)
        states[t] = current
    return states


def event_f1(pred_states, gt_states, tol=3):
    """F1 on transition events within +/-tol frames."""
    pred_tr = (pred_states[1:] - pred_states[:-1]).clamp(min=0)
    gt_tr = (gt_states[1:] - gt_states[:-1]).clamp(min=0)
    if not pred_tr.any() and not gt_tr.any():
        return 1.0
    if not pred_tr.any() or not gt_tr.any():
        return 0.0
    C = pred_tr.shape[1]
    tp, fp, fn_tot = 0, 0, 0
    for c in range(C):
        p_frames = torch.where(pred_tr[:, c])[0].tolist()
        g_frames = torch.where(gt_tr[:, c])[0].tolist()
        matched = set()
        for pf in p_frames:
            for gi, gf in enumerate(g_frames):
                if gi not in matched and abs(pf - gf) <= tol:
                    matched.add(gi); tp += 1; break
            else:
                fp += 1
        fn_tot += len(g_frames) - len(matched)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn_tot, 1)
    return 2 * prec * rec / max(prec + rec, 1e-9)


def ordered_pair_fraction(pred_states, gt_states):
    pred_pairs = (pred_states[1:] - pred_states[:-1])
    gt_pairs = (gt_states[1:] - gt_states[:-1])
    return float((torch.sign(pred_pairs) == torch.sign(gt_pairs)).float().mean())


def per_frame_f1(binary, labels, valid):
    """Per-frame F1 (not transition-based)."""
    tp = ((binary == 1) & (labels == 1) & valid).sum()
    fp = ((binary == 1) & (labels == 0) & valid).sum()
    fn = ((binary == 0) & (labels == 1) & valid).sum()
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return 2 * prec * rec / max(prec + rec, 1e-9)


CKPT_PATH = "src/runs/rf_stages/checkpoints/best.pth"
CACHE_PATH = Path("src/runs/rf_stages/checkpoints/psr_data_cache.pt")


def collect_data():
    """Run inference on val set, collect per-recording logits/labels."""
    ckpt_path = CKPT_PATH
    print(f"Loading {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    print(f"Epoch: {ckpt.get('epoch', 'unknown')}, combined: {ckpt.get('best_combined', 'unknown')}")

    from src.models.model import POPWMultiTaskModel
    model = POPWMultiTaskModel(
        pretrained=True,
        backbone_type='convnext_tiny',
        use_hand_film=True,
        use_headpose_film=True,
        use_videomae=False,
        train_pose=False,
    )
    state_dict = {k: v for k, v in ckpt["model"].items()
                  if 'total_ops' not in k and 'total_params' not in k}
    model.load_state_dict(state_dict)
    model._seq_len = 1
    model = model.cuda().eval()

    from src.data.industreal_dataset import IndustRealMultiTaskDataset, collate_fn
    from torch.utils.data import DataLoader
    val_ds = IndustRealMultiTaskDataset(split="val", sequence_mode=False)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=0,
                            collate_fn=collate_fn, shuffle=False)

    rec_logits = defaultdict(list)
    rec_labels = defaultdict(list)
    rec_frames = defaultdict(list)

    n_batches = 0
    for batch in val_loader:
        images, targets = batch
        images = images.cuda().float()
        if images.max() > 1.0:
            images = images.div_(255.0)
        mean = torch.tensor(_IMAGENET_MEAN, device=images.device).view(1, 3, 1, 1)
        std = torch.tensor(_IMAGENET_STD, device=images.device).view(1, 3, 1, 1)
        images = (images - mean) / std

        with torch.no_grad():
            outputs = model(images)

        psr_logits = outputs.get("psr_logits", None)
        psr_labels = targets.get("psr_labels", None)
        if psr_logits is None or psr_labels is None:
            continue

        psr_logits = psr_logits.cpu()
        metadata_list = targets.get("metadata", [])

        for b in range(images.shape[0]):
            rec_id = "unknown"
            frame_num = 0
            if b < len(metadata_list):
                rec_id = metadata_list[b].get("recording_id", "unknown")
                frame_num = metadata_list[b].get("frame_num", 0)
            rec_logits[rec_id].append(psr_logits[b].unsqueeze(0))
            rec_labels[rec_id].append(psr_labels[b].unsqueeze(0))
            rec_frames[rec_id].append(frame_num)

        n_batches += 1
        if n_batches % 200 == 0:
            print(f"  processed {n_batches} batches ({n_batches} frames)...")

    # Sort each recording by frame_num and concatenate
    for rec_id in list(rec_logits.keys()):
        frames = torch.tensor(rec_frames[rec_id])
        sort_idx = frames.argsort()
        rec_logits[rec_id] = torch.cat([rec_logits[rec_id][i] for i in sort_idx], dim=0)
        rec_labels[rec_id] = torch.cat([rec_labels[rec_id][i] for i in sort_idx], dim=0)

    total_frames = sum(t.shape[0] for t in rec_logits.values())
    n_recs = len(rec_logits)
    print(f"\nCollected {n_batches} batches, {n_recs} recordings, {total_frames} frames")

    # Cache to disk
    torch.save({"rec_logits": dict(rec_logits), "rec_labels": dict(rec_labels),
                "n_batches": n_batches, "total_frames": total_frames, "n_recs": n_recs}, CACHE_PATH)
    print(f"Cached to {CACHE_PATH}")
    return dict(rec_logits), dict(rec_labels), n_batches, total_frames, n_recs


def main():
    if CACHE_PATH.exists():
        print(f"Loading cached data from {CACHE_PATH}...")
        cache = torch.load(CACHE_PATH, weights_only=False)
        rec_logits = cache["rec_logits"]
        rec_labels = cache["rec_labels"]
        n_batches = cache["n_batches"]
        total_frames = cache["total_frames"]
        n_recs = cache["n_recs"]
        print(f"Loaded: {n_batches} batches, {n_recs} recordings, {total_frames} frames")
    else:
        rec_logits, rec_labels, n_batches, total_frames, n_recs = collect_data()

    # Flat arrays for per-frame analysis
    all_logits = np.concatenate([rec_logits[r].numpy() for r in rec_logits], axis=0)
    all_labels = np.concatenate([rec_labels[r].numpy() for r in rec_labels], axis=0)
    valid_all = all_labels != -1

    print(f"\n--- Global stats ---")
    print(f"Logits: min={all_logits.min():.3f}, max={all_logits.max():.3f}, mean={all_logits.mean():.3f}")
    print(f"Labels: min={all_labels.min()}, max={all_labels.max()}")
    print(f"  pos_frac={all_labels[valid_all].mean():.3f}")
    sig = 1 / (1 + np.exp(-all_logits))
    print(f"Sigmoid: min={sig.min():.3f}, max={sig.max():.3f}, mean={sig.mean():.3f}")

    # Per-component bias
    print(f"\n--- Per-component bias ---")
    bias_per_comp = {}
    for c in range(11):
        valid = all_labels[:, c] != -1
        gt_frac = all_labels[valid, c].mean() if valid.sum() > 0 else 0
        mean_l = float(all_logits[:, c].mean())
        pred_frac = float((1/(1+np.exp(-all_logits[:, c]))).mean())
        bias_per_comp[c] = mean_l
        print(f"  comp={c}: mean_logit={mean_l:.3f}, pred_pos={pred_frac:.3f}, "
              f"gt_pos={gt_frac:.3f}")

    # Global threshold sweep (per-frame F1)
    print(f"\n--- Global threshold sweep (per-frame F1) ---")
    print(f"{'thresh':<8} {'F1':<9} {'POS':<9} {'n_unique':<10} {'pos_frac':<9}")
    global_results = []
    for thresh in np.arange(0.1, 0.96, 0.05):
        binary = (sig > thresh).astype(np.int32)
        f1 = per_frame_f1(binary, all_labels, valid_all)
        # POS on flat data (not temporal, just per-frame ordering)
        n_unique = len(np.unique(binary, axis=0))
        pred_pos_frac = binary[valid_all].mean()
        global_results.append((float(thresh), float(f1), n_unique, float(pred_pos_frac)))
        print(f"{thresh:<8.2f} {f1:<9.4f} {n_unique:<10} {pred_pos_frac:<9.4f}")

    best_global = max(global_results, key=lambda r: r[1])
    print(f"  Best: thresh={best_global[0]:.2f}, F1={best_global[1]:.4f}")

    # Per-component threshold sweep (per-frame F1 per component)
    print(f"\n--- Per-component optimal thresholds (per-frame F1) ---")
    per_comp_best = {}
    for c in range(11):
        c_sig = sig[:, c]
        c_labels = all_labels[:, c]
        valid = c_labels != -1
        best_f1, best_t = 0, 0
        for thresh in np.arange(0.1, 0.96, 0.05):
            binary = (c_sig > thresh).astype(np.int32)
            tp = ((binary == 1) & (c_labels == 1) & valid).sum()
            fp = ((binary == 1) & (c_labels == 0) & valid).sum()
            fn = ((binary == 0) & (c_labels == 1) & valid).sum()
            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-9)
            if f1 > best_f1:
                best_f1 = f1
                best_t = thresh
        per_comp_best[c] = (best_t, best_f1)
        print(f"  comp={c}: best_thresh={best_t:.2f}, F1={best_f1:.4f}")

    # Bias calibration sweep
    print(f"\n--- Bias calibration ---")
    bias_tensor = torch.tensor([bias_per_comp[c] for c in range(11)])
    cal_results = []
    for rec_id in rec_logits:
        logits = rec_logits[rec_id] - bias_tensor
        labels = rec_labels[rec_id]
        if logits.shape[0] < 2:
            continue
        cal_sig = torch.sigmoid(logits).numpy()
        cal_labels = labels.numpy()
        cal_valid = cal_labels != -1
        f1 = per_frame_f1((cal_sig > 0.5).astype(np.int32), cal_labels, cal_valid)
        cal_results.append(f1)
    cal_avg_f1 = float(np.mean(cal_results)) if cal_results else 0
    print(f"  Bias-calibrated (thresh=0.5): per-frame F1={cal_avg_f1:.4f}")

    # Decoder sweep on sequence data (frame_num sorted within each recording)
    print(f"\n--- Decoder threshold sweep (transition-based F1 on {n_recs} recordings) ---")
    print(f"{'hi':<6} {'lo':<6} {'min':<5} {'F1':<9} {'POS':<9}")
    decoder_results = []
    sweep_configs = []
    for hi in [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]:
        for lo_offs in [0.2, 0.1, 0.05]:
            lo = round(max(hi - lo_offs, 0.1), 2)
            for smin in [1, 3, 5]:
                sweep_configs.append((hi, lo, smin))

    for hi, lo, smin in sweep_configs:
        f1s, poss = [], []
        for rec_id in rec_logits:
            logits = rec_logits[rec_id]
            labels = rec_labels[rec_id]
            if logits.shape[0] < 2:
                continue
            probs = torch.sigmoid(logits)
            pred_states = decode_monotone(probs, sustain_hi=hi, sustain_lo=lo, sustain_min=smin)
            f1s.append(event_f1(pred_states, labels))
            poss.append(ordered_pair_fraction(pred_states, labels))
        avg_f1 = float(np.mean(f1s)) if f1s else 0.0
        avg_pos = float(np.mean(poss)) if poss else 0.0
        decoder_results.append((hi, lo, smin, avg_f1, avg_pos))
        print(f"{hi:<6.2f} {lo:<6.2f} {smin:<5} {avg_f1:<9.4f} {avg_pos:<9.4f}")

    best_decoder = max(decoder_results, key=lambda r: r[3])
    print(f"  Best decoder: hi={best_decoder[0]:.2f}, lo={best_decoder[1]:.2f}, "
          f"min={best_decoder[2]} => F1={best_decoder[3]:.4f}")

    # Bias-calibrated decoder
    cal_dec_f1s, cal_dec_poss = [], []
    for rec_id in rec_logits:
        logits = rec_logits[rec_id] - bias_tensor
        labels = rec_labels[rec_id]
        if logits.shape[0] < 2:
            continue
        probs = torch.sigmoid(logits)
        pred_states = decode_monotone(probs, sustain_hi=0.5, sustain_lo=0.3, sustain_min=3)
        cal_dec_f1s.append(event_f1(pred_states, labels))
        cal_dec_poss.append(ordered_pair_fraction(pred_states, labels))
    cal_dec_f1 = float(np.mean(cal_dec_f1s)) if cal_dec_f1s else 0
    cal_dec_pos = float(np.mean(cal_dec_poss)) if cal_dec_poss else 0
    print(f"  Bias-calibrated decoder: F1={cal_dec_f1:.4f}, POS={cal_dec_pos:.4f}")

    # Note: decoder sweep results are INVALID with random-frame data (frames not consecutive).
    # Use eval_psr_sequence.py with sequence_mode=True for proper transition-based evaluation.
    print("\n[NOTE] Decoder results are invalid with random-frame data.")
    print("  Run eval_psr_sequence.py with sequence_mode=True for proper transition eval.")

    # Save
    out_path = Path("src/runs/rf_stages/checkpoints/psr_threshold_sweep.json")
    json.dump({
        "model": CKPT_PATH,
        "total_frames": total_frames,
        "n_recordings": n_recs,
        "n_batches": n_batches,
        "bias_per_component": bias_per_comp,
        "global_threshold_sweep": [
            {"thresh": t, "f1": f1, "n_unique": nu, "pred_pos_frac": ppf}
            for t, f1, nu, ppf in global_results
        ],
        "best_global_threshold": best_global[0],
        "best_global_f1": best_global[1],
        "per_component_thresholds": {str(k): {"threshold": v[0], "f1": v[1]} for k, v in per_comp_best.items()},
        "bias_calibrated_f1": cal_avg_f1,
        "decoder_sweep": [
            {"hi": hi, "lo": lo, "min_sustained": smin, "f1": f1, "pos": pos}
            for hi, lo, smin, f1, pos in decoder_results
        ],
        "best_decoder": {"hi": best_decoder[0], "lo": best_decoder[1],
                         "min_sustained": best_decoder[2], "f1": best_decoder[3],
                         "pos": best_decoder[4]},
        "bias_calibrated_decoder": {"f1": cal_dec_f1, "pos": cal_dec_pos},
    }, open(out_path, "w"), indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
