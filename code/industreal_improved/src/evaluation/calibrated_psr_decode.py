"""PSR Mode A2 rescue: per-component threshold + decoder.

Uses per-component optimal thresholds (from val sweep) combined with the
(now-fixed) MonotonicDecoder that respects procedure order + hysteresis.

Sequence-aware: requires consecutive frames per recording (sequence_mode=True).
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


def decode_with_per_comp_thresholds(probs, thresholds, sustain_min=3):
    """Decode with per-component thresholds AND hysteresis.

    Args:
        probs: [T, 11] sigmoid probabilities
        thresholds: [11] per-component thresholds
        sustain_min: min consecutive frames above threshold before firing
    Returns:
        states: [T, 11] binary monotone predictions
    """
    T, C = probs.shape
    states = torch.zeros(T, C)
    current = torch.zeros(C)
    counter = torch.zeros(C)
    for t in range(T):
        # Per-component threshold check (different threshold per component)
        above_thresh = (probs[t] > thresholds).float()  # [C]
        counter = counter * above_thresh + above_thresh
        sustained = counter >= sustain_min
        # Must not already be active
        transition = sustained & (current == 0)
        current = (current + transition.float()).clamp(max=1.0)
        states[t] = current
    return states


def event_f1(pred_states, gt_states, tol=3):
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
                    matched.add(gi)
                    tp += 1
                    break
            else:
                fp += 1
        fn_tot += len(g_frames) - len(matched)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn_tot, 1)
    return 2 * prec * rec / max(prec + rec, 1e-9)


def ordered_pair_fraction(pred_states, gt_states):
    pred_pairs = pred_states[1:] - pred_states[:-1]
    gt_pairs = gt_states[1:] - gt_states[:-1]
    return float((torch.sign(pred_pairs) == torch.sign(gt_pairs)).float().mean())


def main():
    ckpt_path = "src/runs/rf_stages/checkpoints/best.pth"
    print(f"Loading {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    from src.models.model import POPWMultiTaskModel

    model = POPWMultiTaskModel(
        pretrained=True,
        backbone_type="convnext_tiny",
        use_hand_film=True,
        use_headpose_film=True,
        use_videomae=False,
        train_pose=False,
    )
    state_dict = {
        k: v for k, v in ckpt["model"].items() if "total_ops" not in k and "total_params" not in k
    }
    model.load_state_dict(state_dict)
    model._seq_len = 1
    model = model.cuda().eval()

    # Load val dataset — SEQUENCE mode for consecutive frames
    from src.data.industreal_dataset import IndustRealMultiTaskDataset, collate_fn_sequences
    from torch.utils.data import DataLoader

    val_ds = IndustRealMultiTaskDataset(split="val", sequence_mode=True, sequence_length=8)
    val_loader = DataLoader(
        val_ds, batch_size=1, num_workers=0, collate_fn=collate_fn_sequences, shuffle=False
    )

    # Collect per-recording logits and labels
    rec_logits = defaultdict(list)
    rec_labels = defaultdict(list)

    n_batches = 0
    for batch in val_loader:
        images, targets = batch  # images [B, T, 3, H, W]
        B, T = images.shape[:2]

        images = images.cuda().float()
        if images.max() > 1.0:
            images = images.div_(255.0)
        mean = torch.tensor(_IMAGENET_MEAN, device=images.device).view(1, 1, 3, 1, 1)
        std = torch.tensor(_IMAGENET_STD, device=images.device).view(1, 1, 3, 1, 1)
        images = (images - mean) / std

        with torch.no_grad():
            outputs = model(images)

        psr_logits = outputs.get("psr_logits", None)  # [B*T, 11]
        psr_labels = targets.get("psr_labels", None)  # [B, T, 11]
        if psr_logits is None or psr_labels is None:
            continue

        psr_logits = psr_logits.view(B, T, -1).cpu()
        metadata_list = targets.get("metadata", [])

        for b in range(B):
            rec_id = "unknown"
            if b < len(metadata_list):
                rec_id = metadata_list[b].get("recording_id", "unknown")
            rec_logits[rec_id].append(psr_logits[b])
            rec_labels[rec_id].append(psr_labels[b])

        n_batches += 1
        if n_batches % 50 == 0:
            print(
                f"  processed {n_batches} batches ({sum(v[0].shape[0] for v in rec_logits.values())} frames)..."
            )

    # Concatenate per-recording
    for rec_id in list(rec_logits.keys()):
        rec_logits[rec_id] = torch.cat(rec_logits[rec_id], dim=0)
        rec_labels[rec_id] = torch.cat(rec_labels[rec_id], dim=0)

    total_frames = sum(t.shape[0] for t in rec_logits.values())
    n_recs = len(rec_logits)
    print(f"\nCollected {n_batches} batches, {n_recs} recordings, {total_frames} frames")

    # Per-component best thresholds from val sweep
    thresholds = [0.05, 0.05, 0.05, 0.80, 0.95, 0.80, 0.65, 0.95, 0.95, 0.95, 0.95]
    thresholds = torch.tensor(thresholds)

    # Sweep sustain_min
    print("\n--- Per-component threshold decoder sweep (consecutive frames) ---")
    print(f"{'sustain_min':<13} {'F1':<9} {'POS':<9} {'n_recs':<7}")
    results = []
    for sustain_min in [1, 2, 3, 5, 8]:
        f1s, poss = [], []
        for rec_id in rec_logits:
            logits = rec_logits[rec_id]
            labels = rec_labels[rec_id]
            if logits.shape[0] < 2:
                continue
            probs = torch.sigmoid(logits)
            pred_states = decode_with_per_comp_thresholds(
                probs, thresholds, sustain_min=sustain_min
            )
            f1s.append(event_f1(pred_states, labels))
            poss.append(ordered_pair_fraction(pred_states, labels))
        avg_f1 = float(np.mean(f1s)) if f1s else 0.0
        avg_pos = float(np.mean(poss)) if poss else 0.0
        results.append((sustain_min, avg_f1, avg_pos))
        print(f"{sustain_min:<13} {avg_f1:<9.4f} {avg_pos:<9.4f} {len(f1s):<7}")

    best = max(results, key=lambda r: r[1])
    print(f"\nBest: sustain_min={best[0]} => F1={best[1]:.4f}, POS={best[2]:.4f}")

    # Save
    out_path = Path("src/runs/rf_stages/checkpoints/psr_calibrated_decode.json")
    json.dump(
        {
            "model": ckpt_path,
            "total_frames": total_frames,
            "n_recordings": n_recs,
            "thresholds": thresholds.tolist(),
            "results": [{"sustain_min": sm, "f1": f1, "pos": pos} for sm, f1, pos in results],
            "best": {"sustain_min": best[0], "f1": best[1], "pos": best[2]},
        },
        open(out_path, "w"),
        indent=2,
    )
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
