#!/usr/bin/env python3
"""Debug: trace canonical POS computation on a tiny sample."""
import csv
import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.evaluate import _compute_psr_pos_vectorized, _compute_psr_pos_canonical

# ---- Test 1: Toy 5-frame, 3-component case ----
print("=" * 60)
print("Test 1: Toy 5-frame, 3-component canonical")
gt = np.array([
    [0, 0, 0],  # K=0 -> canon_pred=[0,0,0]
    [1, 0, 0],  # K=1 -> canon_pred=[1,0,0]
    [1, 1, 0],  # K=2 -> canon_pred=[1,1,0]
    [1, 1, 1],  # K=3 -> canon_pred=[1,1,1]
    [1, 1, 1],  # K=3 -> canon_pred=[1,1,1]
], dtype=np.int64)
vm = np.ones_like(gt, dtype=bool)

canon_pred = np.zeros_like(gt, dtype=np.int64)
C = 3
for t in range(5):
    k_done = 0
    for c in range(C):
        if vm[t, c] and gt[t, c] == 1:
            k_done += 1
    for i in range(min(k_done, C)):
        canon_pred[t, i] = 1

print("GT:\n", gt)
print("Canonical pred:\n", canon_pred)
pos = _compute_psr_pos_vectorized(canon_pred, gt, vm)
print(f"POS (expected ~1.0): {pos:.6f}")

# ---- Test 2: Non-canonical GT ----
print()
print("Test 2: Non-canonical GT (comp1 before comp0)")
gt2 = np.array([
    [0, 0, 0],
    [0, 1, 0],  # comp1 done before comp0
    [1, 1, 0],
    [1, 1, 1],
    [1, 1, 1],
], dtype=np.int64)

canon_pred2 = np.zeros_like(gt2, dtype=np.int64)
for t in range(5):
    k_done = 0
    for c in range(3):
        if vm[t, c] and gt2[t, c] == 1:
            k_done += 1
    for i in range(min(k_done, 3)):
        canon_pred2[t, i] = 1

print("GT2:\n", gt2)
print("Canonical pred2:\n", canon_pred2)
pos2 = _compute_psr_pos_vectorized(canon_pred2, gt2, vm)
print(f"POS: {pos2:.6f}")

# ---- Test 3: Real data first recording first 20 frames ----
print()
print("=" * 60)
print("Test 3: Real data first 20 frames from 05_assy_0_1")

rec_dir = Path('/media/newadmin/master/POPW/datasets/industreal/recordings/val/05_assy_0_1')
psr_file = rec_dir / 'PSR_labels_raw.csv'
num_comp = 11

sparse = []
with open(psr_file, encoding='utf-8') as f:
    reader = csv.reader(f)
    for row in reader:
        if len(row) < 12:
            continue
        try:
            frame_num = int(Path(row[0]).stem)
            values = np.array([float(v) for v in row[1:12]], dtype=np.float64)
            sparse.append((frame_num, values))
        except (ValueError, IndexError):
            continue
sparse.sort(key=lambda x: x[0])

rgb_dir = rec_dir / 'rgb'
num_frames = len(sorted(rgb_dir.glob('*.jpg')))

labels = np.zeros((num_frames, 11), dtype=np.float64)
_last_valid = np.zeros(11, dtype=np.float64)
sparse_idx = 0
for frame in range(num_frames):
    if sparse_idx < len(sparse) and frame == sparse[sparse_idx][0]:
        _new = sparse[sparse_idx][1].copy()
        sparse_idx += 1
        _valid_mask = _new >= 0
        _last_valid[_valid_mask] = _new[_valid_mask]
    labels[frame] = _last_valid.copy()

vm_real = labels >= 0
gt_real = labels.copy()
gt_real[~vm_real] = 0.0

# First 20 frames
print("First 20 frames GT:")
print(gt_real[:20].astype(int))

# First 20 frames canonical pred
canon_real20 = np.zeros_like(gt_real[:20], dtype=np.int64)
for t in range(20):
    k_done = 0
    for c in range(11):
        if vm_real[t, c] and gt_real[t, c] == 1:
            k_done += 1
    for i in range(min(k_done, 11)):
        canon_real20[t, i] = 1

print("\nFirst 20 frames canonical pred:")
print(canon_real20)

pos_real20 = _compute_psr_pos_vectorized(canon_real20, gt_real[:20].astype(np.int64), vm_real[:20])
print(f"\nPOS for first 20 frames: {pos_real20:.6f}")

# ---- Test 4: Whole recording ----
print()
print("=" * 60)
print("Test 4: Full recording 05_assy_0_1")
canon_all = np.zeros_like(gt_real, dtype=np.int64)
for t in range(num_frames):
    k_done = 0
    for c in range(11):
        if vm_real[t, c] and gt_real[t, c] == 1:
            k_done += 1
    for i in range(min(k_done, 11)):
        canon_all[t, i] = 1

pos_all = _compute_psr_pos_vectorized(canon_all, gt_real.astype(np.int64), vm_real)
print(f"POS for full recording: {pos_all:.6f}")

# Debug: per-component
N, C = gt_real.shape
for c in range(11):
    vm_c = vm_real[:, c]
    gt_c = gt_real[vm_c, c].astype(np.int8)
    pred_c = canon_all[vm_c, c].astype(np.int8)
    gt_diff = np.diff(gt_c, prepend=gt_c[0:1])
    run_starts = np.where(gt_diff != 0)[0]
    run_vals = gt_c[run_starts]

    if len(run_vals) < 2:
        print(f"  comp{c}: fewer than 2 runs -> skipped")
        continue

    total_pairs = len(run_vals) - 1
    correct_pairs = 0
    for k in range(total_pairs):
        val_a = run_vals[k]
        val_b = run_vals[k + 1]
        pos_a = np.where(pred_c == val_a)[0]
        pos_b = np.where(pred_c == val_b)[0]
        if len(pos_a) > 0 and len(pos_b) > 0 and pos_a.max() < pos_b.min():
            correct_pairs += 1
    print(f"  comp{c}: {correct_pairs}/{total_pairs} pairs correct = {correct_pairs/total_pairs:.4f} (runs: {run_vals})")

# ---- Test 5: Use the canonical function directly ----
print()
print("=" * 60)
print("Test 5: Using _compute_psr_pos_canonical function directly")
pos_canon = _compute_psr_pos_canonical(gt_real.astype(np.int64), vm_real)
print(f"POS via canonical function: {pos_canon:.6f}")
