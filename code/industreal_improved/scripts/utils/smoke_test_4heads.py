#!/usr/bin/env python3
"""
Smoke Test — 4-Head Multi-Task Overfit (175 §2 Preflight Exit Gate)

Overfits a 50-sample subset to near-zero loss on all four heads simultaneously
using the POPWMultiTaskModel (ConvNeXt-Tiny backbone, ~28M params).

If any head can't overfit 50 samples, its plumbing is still broken — do not scale up.

Usage:
    python scripts/smoke_test_4heads.py

Output:
    src/runs/rf_stages/checkpoints/smoke_test_4heads/metrics.json
    PASS/FAIL diagnostic on stdout
"""

import sys, os, json, time
from pathlib import Path

# ── Path setup ──────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_WORK_DIR = os.path.normpath(os.path.join(_SCRIPT_DIR, os.pardir))
_SRC_DIR = os.path.join(_WORK_DIR, "src")
sys.path.insert(0, _WORK_DIR)
sys.path.insert(1, os.path.join(_SRC_DIR, "models"))
sys.path.insert(2, os.path.join(_SRC_DIR, "training"))
sys.path.insert(3, os.path.join(_SRC_DIR, "evaluation"))
sys.path.insert(4, _SRC_DIR)

import torch
import torch.nn.functional as F

# ── Model loading ───────────────────────────────────────────────────────────
from src import config as C
from models import model as model_module


# ── Data loading ────────────────────────────────────────────────────────────
def load_50_samples(split="train", num_samples=50):
    """Load num_samples single frames from the dataset."""
    from src.data.industreal_dataset import IndustRealMultiTaskDataset as Dataset

    # Disable RAM cache for quick module-level loading
    _orig = getattr(C, "RAM_CACHE_MAX_IMAGES", 8000)
    C.RAM_CACHE_MAX_IMAGES = 0

    ds = Dataset(
        split=split,
        augment=False,
        sequence_mode=False,
        max_recordings=None,
    )
    C.RAM_CACHE_MAX_IMAGES = _orig

    total = len(ds)
    if total == 0:
        raise RuntimeError(f"Dataset has 0 samples (split={split})")
    indices = torch.linspace(0, max(0, total - 1), min(num_samples, total)).long().tolist()
    indices = list(set(indices))[:num_samples]
    while len(indices) < num_samples:
        indices.append(indices[len(indices) % len(indices)])

    samples = [ds[i] for i in indices]
    print(f"[DATA] Loaded {len(samples)} samples from {total} total (split={split})")
    return samples


def prepare_frame_batch(samples, device):
    """Build a batch of frames + targets from sample list.

    Returns:
        frames: [B, 3, H, W] float32 tensor normalized to [0,1]
        targets: dict for loss functions
    """
    frame_list = []
    act_list = []
    psr_list = []
    hp_list = []
    det_targets = []

    for sample in samples:
        rgb = sample["images"]["rgb"]  # [3, H, W] uint8

        # Normalize to [0, 1]
        frame = rgb.float().div(255.0)
        frame_list.append(frame.unsqueeze(0))

        # Activity label — clamp -1 to 0 (handles unlabeled sentinel)
        al = sample["action_label"].item()
        act_list.append(max(0, min(al, C.NUM_CLASSES_ACT - 1)))

        # PSR labels [11]
        psr_list.append(sample["psr_labels"].unsqueeze(0))

        # Head pose [9]
        hp_list.append(sample["head_pose"].unsqueeze(0))

        # Detection
        det_targets.append(
            {
                "boxes": sample["gt_boxes"]["rgb"],
                "labels": sample["gt_classes"]["rgb"],
            }
        )

    frames = torch.cat(frame_list, dim=0).to(device)
    targets = {
        "activity": torch.tensor(act_list, dtype=torch.long, device=device),
        "psr_labels": torch.cat(psr_list, dim=0).to(device),
        "head_pose": torch.cat(hp_list, dim=0).to(device),
        "detection": det_targets,
    }
    return frames, targets


# ── Lightweight detection loss — plumbing check ─────────────────────────────
def detection_loss_for_smoke(model_outputs, det_targets_list, device):
    """Simplified detection loss: BCE on cls logits.

    The POPWMultiTaskModel returns cls_preds as [B, total_anchors, num_classes].
    We take the mean over all anchors as a weak class-presence signal.

    This is NOT the real Focal+GIoU — it's a plumbing check that verifies
    gradient flows through the detection branch.
    """
    cls_preds = model_outputs["cls_preds"]  # [B, total_anchors, num_classes]
    cls_gap = cls_preds.mean(dim=1)  # [B, num_classes]
    B, num_classes = cls_gap.shape

    # Build per-sample multi-label targets
    cls_targets = torch.zeros(B, num_classes, device=device)
    for b, dt in enumerate(det_targets_list):
        lbls = dt["labels"].to(device).long()
        for lbl in lbls:
            if 0 <= lbl < num_classes:
                cls_targets[b, int(lbl)] = 1.0

    # Fallback: if no GT boxes in batch, force class 0
    if cls_targets.sum() == 0:
        cls_targets[0, 0] = 1.0

    return F.binary_cross_entropy_with_logits(cls_gap, cls_targets)


# ── Main smoke test ─────────────────────────────────────────────────────────
def run_smoke_test():
    print("=" * 65)
    print("  SMOKE TEST — 4-Head Multi-Task Overfit (175 §2)")
    print("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[ENV] Device: {device}")
    if device.type == "cuda":
        torch.cuda.empty_cache()
        props = torch.cuda.get_device_properties(0)
        print(f"[ENV] GPU: {torch.cuda.get_device_name(0)}")
        print(f"[ENV] VRAM: {props.total_memory / 1e9:.1f} GB")

    # Prevent data loader RAM cache from consuming memory before training
    C.RAM_CACHE_MAX_IMAGES = 0

    # ── Model ────────────────────────────────────────────────────────────────
    print("\n[MODEL] Building POPWMultiTaskModel (ConvNeXt-Tiny)...")
    model = model_module.POPWMultiTaskModel(
        backbone_type=C.BACKBONE,
        pretrained=False,
        use_videomae=False,
    ).to(device)
    model.train()
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"  Params: {total_params:.2f}M total, {trainable:.2f}M trainable")
    print(f"  Backbone: {C.BACKBONE}")

    # ── Data ─────────────────────────────────────────────────────────────────
    print("\n[DATA] Loading 50 frames...")
    samples = load_50_samples(split="train", num_samples=50)
    num_samples = len(samples)
    print(f"  Loaded {num_samples} samples")

    # ── Optimizer ────────────────────────────────────────────────────────────
    # Override config for fast overfitting (random-init backbone needs larger LR)
    C.ACTIVITY_GRAD_BLEND_RATIO = 1.0  # Allow full gradient flow from activity head
    C.ACTIVITY_HEAD_GRAD_CLIP = 50.0  # Loosen activity grad clip for overfit

    backbone_params = []
    head_params = []
    activity_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "backbone" in name:
            backbone_params.append(p)
        elif "activity_head" in name or "feature_bank" in name:
            activity_params.append(p)
        else:
            head_params.append(p)
    print(f"\n[OPTIM] backbone_lr=1e-3, head_lr=1e-3, activity_lr=5e-3, AdamW (overfit)")
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": 1e-3},
            {"params": head_params, "lr": 1e-3},
            {"params": activity_params, "lr": 5e-3},
        ],
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=10, threshold=0.01, min_lr=1e-5
    )

    # ── Training ─────────────────────────────────────────────────────────────
    num_epochs = 200
    batch_size = 1  # safe on 16GB VRAM
    steps_per_epoch = max(1, num_samples // batch_size)

    print(f"\n[TRAIN] {num_epochs} epochs, {steps_per_epoch} steps/epoch, batch={batch_size}")
    print(f"  {'=' * 50}")

    history = {"det": [], "act": [], "psr": [], "pose": [], "total": [], "epoch": []}
    stall_counter = 0
    best_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        epoch_losses = {"det": 0.0, "act": 0.0, "psr": 0.0, "pose": 0.0, "total": 0.0}
        epoch_steps = 0
        t0 = time.time()
        perm = torch.randperm(num_samples).tolist()

        optimizer.zero_grad()
        accum_steps = 1  # no accumulation at batch=1

        for step_idx in range(steps_per_epoch):
            start_i = step_idx * batch_size
            end_i = min(start_i + batch_size, num_samples)
            idxs = perm[start_i:end_i]
            batch_samples = [samples[i] for i in idxs]

            frames, targets = prepare_frame_batch(batch_samples, device)
            outputs = model(frames)

            # ── Detection loss ──
            loss_det = detection_loss_for_smoke(outputs, targets["detection"], device)

            # ── Head pose loss ── (MSE on 6D from [:, :6])
            loss_pose = F.mse_loss(outputs["head_pose"][:, :6], targets["head_pose"][:, :6])

            # ── Activity loss ──
            loss_act = F.cross_entropy(outputs["act_logits"], targets["activity"])

            # ── PSR loss ──
            loss_psr = F.binary_cross_entropy_with_logits(
                outputs["psr_logits"], targets["psr_labels"]
            )

            # Sum + backward (with gradient accumulation)
            total_loss = loss_det + loss_act + loss_psr + loss_pose
            total_loss = total_loss / accum_steps
            total_loss.backward()

            epoch_losses["det"] += loss_det.item()
            epoch_losses["act"] += loss_act.item()
            epoch_losses["psr"] += loss_psr.item()
            epoch_losses["pose"] += loss_pose.item()
            epoch_losses["total"] += total_loss.item() * accum_steps
            epoch_steps += 1

            # Gradient accumulation: step every accum_steps batches
            if (step_idx + 1) % accum_steps == 0 or (step_idx + 1) == steps_per_epoch:
                # Clip + NaN rescue
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                for name, p in model.named_parameters():
                    if p.grad is not None and (
                        torch.isnan(p.grad).any() or torch.isinf(p.grad).any()
                    ):
                        print(f"  [WARN] NaN/Inf grad in {name}, zeroing")
                        p.grad = None
                optimizer.step()
                optimizer.zero_grad()

        # ── End of epoch ──
        avg = {k: v / max(epoch_steps, 1) for k, v in epoch_losses.items()}
        dt = time.time() - t0

        # LR scheduler step
        scheduler.step(avg["total"])

        # Free GPU cache between epochs to prevent fragmentation
        if device.type == "cuda" and epoch % 5 == 0:
            torch.cuda.empty_cache()

        history["det"].append(avg["det"])
        history["act"].append(avg["act"])
        history["psr"].append(avg["psr"])
        history["pose"].append(avg["pose"])
        history["total"].append(avg["total"])
        history["epoch"].append(epoch)

        if epoch <= 20 or epoch % 20 == 0 or epoch == num_epochs:
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"  ep {epoch:4d}/{num_epochs} lr {lr:.2e}"
                f" | det {avg['det']:.4f}  act {avg['act']:.4f}"
                f" | psr {avg['psr']:.4f}  pose {avg['pose']:.6f}"
                f" | total {avg['total']:.4f}  ({dt:.1f}s)"
            )

        # Track stall
        if avg["total"] < best_loss:
            best_loss = avg["total"]
            stall_counter = 0
        else:
            stall_counter += 1

        # Early stop if all thresholds met
        if avg["det"] < 0.8 and avg["act"] < 0.8 and avg["psr"] < 0.8 and avg["pose"] < 0.08:
            print(f"\n  [EARLY STOP] All heads below threshold at epoch {epoch}")
            break

        # Hard stop if no improvement for 50 epochs
        if stall_counter >= 50:
            print(f"\n  [EARLY STOP] No improvement for 50 epochs (loss={avg['total']:.4f})")
            break

    # ── Final assessment ─────────────────────────────────────────────────────
    print(f"\n{'=' * 65}")
    print("  FINAL ASSESSMENT")
    print(f"{'=' * 65}")

    final = {
        "det": history["det"][-1],
        "act": history["act"][-1],
        "psr": history["psr"][-1],
        "pose": history["pose"][-1],
    }

    thresholds = {
        "det": ("loss_det < 1.0", lambda v: v < 1.0),
        "act": ("loss_act < 1.0", lambda v: v < 1.0),
        "psr": ("loss_psr < 1.0", lambda v: v < 1.0),
        "pose": ("loss_pose < 0.1", lambda v: v < 0.1),
    }

    all_passed = True
    failures = []
    for head, (desc, check) in thresholds.items():
        result = check(final[head])
        status = "PASS" if result else "FAIL"
        if not result:
            all_passed = False
            failures.append({"head": head, "desc": desc, "value": final[head]})
        print(f"  [{status}] {desc}  (final={final[head]:.6f})")

    print(f"\n  Trajectory (epoch 1 -> final):")
    print(f"    det:   {history['det'][0]:.4f}  ->  {history['det'][-1]:.4f}")
    print(f"    act:   {history['act'][0]:.4f}  ->  {history['act'][-1]:.4f}")
    print(f"    psr:   {history['psr'][0]:.4f}  ->  {history['psr'][-1]:.4f}")
    print(f"    pose:  {history['pose'][0]:.6f}  ->  {history['pose'][-1]:.6f}")

    if all_passed:
        print(f"\n  {'=' * 45}")
        print("  RESULT: PASS -- All four heads overfit 50 frames to threshold.")
        print(f"  {'=' * 45}")
    else:
        print(f"\n  {'=' * 45}")
        print(f"  RESULT: FAIL -- {len(failures)} head(s) below threshold:")
        print(f"  {'=' * 45}")
        for f in failures:
            print(f"    {f['head']}: final={f['value']:.6f}  need={f['desc']}")

    # ── Save report ──
    output_dir = (
        Path(_WORK_DIR) / "src" / "runs" / "rf_stages" / "checkpoints" / "smoke_test_4heads"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "model": "POPWMultiTaskModel (ConvNeXt-Tiny)",
        "device": str(device),
        "num_params_m": total_params,
        "num_samples": num_samples,
        "num_epochs": epoch,
        "thresholds": {k: v[0] for k, v in thresholds.items()},
        "final_losses": final,
        "trajectory": {
            "det": [round(v, 6) for v in history["det"]],
            "act": [round(v, 6) for v in history["act"]],
            "psr": [round(v, 6) for v in history["psr"]],
            "pose": [round(v, 6) for v in history["pose"]],
        },
        "passed": all_passed,
        "failures": failures,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    report_path = output_dir / "metrics.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Report saved: {report_path}")

    return all_passed, report


if __name__ == "__main__":
    passed, report = run_smoke_test()
    sys.exit(0 if passed else 1)
