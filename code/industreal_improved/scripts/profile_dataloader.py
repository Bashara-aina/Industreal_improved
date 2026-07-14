#!/usr/bin/env python3
"""
Profile the existing dataloader to quantify the 3.5s/step bottleneck.

Measures wall-clock breakdown for 100 batches:
  - Image loading (disk I/O + JPEG decode + resize)
  - Annotation loading (COCO parsing, AR/PSR/Pose/Hands lookup)
  - Tensor creation / normalization
  - Collation
  - GPU transfer

Output: JSON file with per-operation timings + stdout report.
"""

import sys, os, json, time, warnings, gc
from pathlib import Path

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_WORK_DIR = os.path.normpath(os.path.join(_SCRIPT_DIR, os.pardir))
sys.path.insert(0, _WORK_DIR)
sys.path.insert(1, os.path.join(_WORK_DIR, "src"))

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Use RTX 5060 Ti for profiling

import torch
import numpy as np
from src.data.industreal_dataset import (
    IndustRealMultiTaskDataset as Dataset,
    collate_fn,
)
from src import config as C

# Force RAM cache OFF to measure worst-case disk-bound behavior (as smoke test runs)
C.RAM_CACHE_MAX_IMAGES = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[PROFILE] Device: {device}")
print(f"[PROFILE] RAM_CACHE_MAX_IMAGES=0 (worst-case disk-bound)")
print(f"[PROFILE] NUM_WORKERS={C.NUM_WORKERS}")

# ---- Dataset ----
t0 = time.time()
ds = Dataset(
    split="train",
    augment=False,
    sequence_mode=False,
    max_recordings=6,  # ~6 recordings for profile (50 samples per epoch)
)
print(f"[PROFILE] Dataset init: {time.time() - t0:.1f}s, {len(ds)} samples")

# ---- Batch sampler for fixed-index replay ----
num_samples = min(200, len(ds))
indices = list(range(num_samples))
batch_size = 2
num_batches = num_samples // batch_size
print(f"[PROFILE] Profiling {num_batches} batches of size {batch_size}")

# ---- Warm up (1 batch to avoid cold-start bias) ----
_ = collate_fn([ds[i] for i in indices[0:batch_size]])

gc.collect()

# ---- Timed run with per-operation instrumentation ----
timings = {
    "load_image": [],
    "annotations": [],  # PSR + pose + hands lookup
    "coco_boxes": [],
    "collate": [],
    "gpu_transfer": [],
    "total_step": [],
}

for batch_idx in range(num_batches):
    start_i = batch_idx * batch_size
    end_i = min(start_i + batch_size, num_samples)
    batch_ids = indices[start_i:end_i]

    step_t0 = time.time()

    samples = []
    for idx in batch_ids:
        sample_t0 = time.time()

        s = ds.samples[idx]
        recording_id = s["recording_id"]
        frame_num = s["frame_num"]
        img_path = s["img_path"]

        # Time image loading
        t_load = time.time()
        rgb_tensor = ds._load_image(img_path)
        dt_load = time.time() - t_load

        # Time annotation lookups (already cached in _anno_cache)
        cache = ds._anno_cache[recording_id]

        t_ann = time.time()
        action_label = torch.tensor(int(s["action_label"]), dtype=torch.long)
        psr_labels = torch.from_numpy(cache.psr_per_frame[frame_num]).float()
        head_pose = torch.from_numpy(cache.pose[frame_num]).float()
        hand_joints = torch.from_numpy(cache.hands[frame_num]).float()
        dt_ann = time.time() - t_ann

        # Time COCO box extraction
        t_coco = time.time()
        gt_boxes, gt_classes = ds._extract_boxes_from_coco(recording_id, frame_num)
        dt_coco = time.time() - t_coco

        sample_dict = {
            "images": {"rgb": rgb_tensor},
            "gt_boxes": {"rgb": gt_boxes},
            "gt_classes": {"rgb": gt_classes},
            "head_pose": head_pose,
            "psr_labels": psr_labels,
            "hand_joints": hand_joints,
            "action_label": action_label,
            "clip_rgb": torch.zeros(0, 3, 224, 224, dtype=torch.float32),
            "metadata": {"recording_id": recording_id, "frame_num": frame_num},
        }
        samples.append(sample_dict)

        timings["load_image"].append(dt_load)
        timings["annotations"].append(dt_ann)
        timings["coco_boxes"].append(dt_coco)

    # Time collation
    t_coll = time.time()
    images, targets = collate_fn(samples)
    timings["collate"].append(time.time() - t_coll)

    # Time GPU transfer
    t_gpu = time.time()
    images = images.to(device, non_blocking=True)
    for k in ["head_pose", "psr_labels", "hand_joints", "activity"]:
        if k in targets:
            targets[k] = targets[k].to(device, non_blocking=True)
    timings["gpu_transfer"].append(time.time() - t_gpu)

    timings["total_step"].append(time.time() - step_t0)

    if (batch_idx + 1) % 20 == 0:
        print(
            f"  batch {batch_idx + 1}/{num_batches} — "
            f"total={timings['total_step'][-1]:.4f}s, "
            f"load={dt_load:.4f}s, ann={dt_ann:.4f}s, coco={dt_coco:.4f}s"
        )

# ---- Aggregate ----
print("\n" + "=" * 65)
print("  PROFILE RESULTS — Per-Batch Breakdown (averaged over batch_size=2)")
print("=" * 65)

# Per-batch (averaged)
stats = {}
for key in timings:
    arr = np.array(timings[key])
    # For per-sample ops, convert to per-batch by summing per-sample within batch
    if key in ("load_image", "annotations", "coco_boxes"):
        # Already per-sample; group into batches and sum per batch
        per_batch = []
        for b in range(num_batches):
            s = b * batch_size
            e = s + batch_size
            per_batch.append(sum(timings[key][s:e]))
        arr = np.array(per_batch)
    else:
        arr = np.array(timings[key])

    stats[key] = {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "total": float(arr.sum()),
    }

total_mean = stats["total_step"]["mean"]
total_pct = 0.0

ordered = ["load_image", "annotations", "coco_boxes", "collate", "gpu_transfer"]
print(f"\n  {'Operation':<20s} {'Mean (s)':<12s} {'%Total':<10s} {'Min (s)':<12s} {'Max (s)':<12s}")
print(f"  {'-' * 66}")
for key in ordered:
    s = stats[key]
    pct = s["mean"] / total_mean * 100 if total_mean > 0 else 0
    print(f"  {key:<20s} {s['mean']:<12.4f} {pct:<10.1f} {s['min']:<12.4f} {s['max']:<12.4f}")
    total_pct += pct

print(f"  {'-' * 66}")
print(f"  {'total_step':<20s} {total_mean:<12.4f} {total_pct:<10.1f} {'':<12s} {'':<12s}")
print(f"\n  Steps/sec (throughput): {1.0 / total_mean:.2f}")
print(f"  Batches profiled: {num_batches}")

# ---- Step/sec with full GPU (model forward + backward) ----
print(f"\n  {'=' * 65}")
print(f"  FULL STEP (model forward+backward included) — 10 batches")
print(f"  {'=' * 65}")

from models import model as model_module

model = model_module.POPWMultiTaskModel(
    backbone_type=C.BACKBONE,
    pretrained=False,
    use_videomae=False,
).to(device)
model.train()

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

full_timings = []
for step in range(10):
    batch_ids = indices[(step % num_batches) * batch_size : (step % num_batches + 1) * batch_size]
    batch_samples = [ds[i] for i in batch_ids]

    t0 = time.time()
    frames, targets = collate_fn(batch_samples)
    frames = frames.to(device)
    targets_d = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in targets.items()}

    # Filter detection list (non-tensor, stays on CPU)
    outputs = model(frames)

    loss_det = outputs["cls_preds"].mean()  # placeholder
    loss_act = torch.tensor(0.0, device=device)
    if "act_logits" in outputs and "activity" in targets_d:
        loss_act = torch.nn.functional.cross_entropy(outputs["act_logits"], targets_d["activity"])
    loss_psr = torch.nn.functional.binary_cross_entropy_with_logits(
        outputs["psr_logits"], targets_d["psr_labels"]
    )
    loss_pose = torch.nn.functional.mse_loss(
        outputs["head_pose"][:, :6], targets_d["head_pose"][:, :6]
    )

    total_loss = loss_det + loss_act + loss_psr + loss_pose
    total_loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    dt = time.time() - t0
    full_timings.append(dt)
    if step < 3 or step == 9:
        print(f"    full step {step + 1:2d}: {dt:.4f}s")

print(
    f"\n  Full step mean: {np.mean(full_timings):.4f}s ({1.0 / np.mean(full_timings):.2f} steps/sec)"
)
print(f"  vs data-only step: {total_mean:.4f}s ({1.0 / total_mean:.2f} steps/sec)")
print(f"  Data as % of full step: {total_mean / np.mean(full_timings) * 100:.1f}%")

# ---- Save ----
results = {
    "device": str(device),
    "num_batches": num_batches,
    "batch_size": batch_size,
    "num_workers": C.NUM_WORKERS,
    "ram_cache": C.RAM_CACHE_MAX_IMAGES,
    "per_batch_stats": stats,
    "full_step_mean_s": float(np.mean(full_timings)),
    "data_only_mean_s": total_mean,
    "steps_per_sec_full": 1.0 / float(np.mean(full_timings)),
    "steps_per_sec_data": 1.0 / max(total_mean, 1e-8),
}

out_path = Path(_WORK_DIR) / "src" / "runs" / "rf_stages" / "checkpoints" / "profile_results.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\n[PROFILE] Saved to {out_path}")
