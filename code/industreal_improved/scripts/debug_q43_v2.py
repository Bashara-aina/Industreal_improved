#!/usr/bin/env python3
"""Debug Q43: per-recording POS breakdown."""
import csv
import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.evaluate import _compute_psr_pos_vectorized, _compute_psr_pos_canonical

VAL_ROOT = Path('/media/newadmin/master/POPW/datasets/industreal/recordings/val')
NUM_COMPONENTS = 11

def load_psr_labels(rec_dir: Path):
    psr_file = rec_dir / 'PSR_labels_raw.csv'
    rgb_dir = rec_dir / 'rgb'
    num_frames = len(sorted(rgb_dir.glob('*.jpg')))

    sparse = []
    with open(psr_file, encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < NUM_COMPONENTS + 1:
                continue
            try:
                frame_num = int(Path(row[0]).stem)
                values = np.array([float(v) for v in row[1:NUM_COMPONENTS + 1]], dtype=np.float64)
                sparse.append((frame_num, values))
            except (ValueError, IndexError):
                continue

    sparse.sort(key=lambda x: x[0])

    labels = np.zeros((num_frames, NUM_COMPONENTS), dtype=np.float64)
    _last_valid = np.zeros(NUM_COMPONENTS, dtype=np.float64)
    sparse_idx = 0
    for frame in range(num_frames):
        if sparse_idx < len(sparse) and frame == sparse[sparse_idx][0]:
            _new = sparse[sparse_idx][1].copy()
            sparse_idx += 1
            _valid_mask = _new >= 0
            _last_valid[_valid_mask] = _new[_valid_mask]
        labels[frame] = _last_valid.copy()

    valid_mask = labels >= 0
    gt_safe = labels.copy()
    gt_safe[~valid_mask] = 0.0
    return gt_safe, valid_mask

def per_component_pos_debug(gt_safe, valid_mask, canon_pred):
    """Debug POS per component."""
    N, C = gt_safe.shape
    results = {}
    for c in range(C):
        vm = valid_mask[:, c]
        gt_c = gt_safe[vm, c].astype(np.int8)
        pred_c = canon_pred[vm, c].astype(np.int8)

        gt_diff = np.diff(gt_c, prepend=gt_c[0:1])
        run_starts = np.where(gt_diff != 0)[0]
        run_vals = gt_c[run_starts]

        if len(run_vals) < 2:
            results[c] = {'status': 'skipped', 'runs': len(run_vals), 'run_vals': run_vals.tolist()}
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

        results[c] = {
            'status': 'evaluated',
            'correct': int(correct_pairs),
            'total': int(total_pairs),
            'pos': correct_pairs / total_pairs if total_pairs > 0 else 0.0,
            'run_vals': run_vals.tolist(),
        }
    return results

# ========== MAIN ==========
print("=" * 72)
print("Q43 Debug: Per-recording POS analysis")
print("=" * 72)

# Option A: Concatenated (like eval code)
print("\n--- Option A: Concatenated across all recordings ---")
all_gt = []
all_vm = []
rec_dirs = sorted(VAL_ROOT.iterdir())
total_rec = 0

for rec_dir in rec_dirs:
    if not rec_dir.is_dir():
        continue
    gt, vm = load_psr_labels(rec_dir)
    if gt is None:
        continue
    all_gt.append(gt)
    all_vm.append(vm)
    total_rec += 1

gt_safe_concat = np.concatenate(all_gt, axis=0)
valid_mask_concat = np.concatenate(all_vm, axis=0)

# Build canonical pred for concatenated
canon_concat = np.zeros_like(gt_safe_concat, dtype=np.int64)
N, C = gt_safe_concat.shape
for t in range(N):
    k_done = int(gt_safe_concat[t, :].sum())
    for i in range(min(k_done, C)):
        canon_concat[t, i] = 1

print(f"Total frames: {N}, total recordings: {total_rec}")
pos_concat = _compute_psr_pos_vectorized(canon_concat, gt_safe_concat.astype(np.int64), valid_mask_concat)
print(f"Concatenated POS: {pos_concat:.6f}")

# Debug components in concatenated
print("\nPer-component (concatenated):")
dbg_concat = per_component_pos_debug(gt_safe_concat.astype(np.int64), valid_mask_concat, canon_concat)
for c in range(C):
    r = dbg_concat[c]
    if r['status'] == 'evaluated':
        print(f"  comp{c}: POS={r['pos']:.4f} ({r['correct']}/{r['total']}) runs={r['run_vals']}")
    else:
        print(f"  comp{c}: SKIPPED ({r['runs']} run vals)")

# Option B: Per-recording then average
print("\n--- Option B: Per-recording POS, then macro-average ---")
rec_pos_values = []
comp_pos_matrix = {c: [] for c in range(C)}

for rec_dir in rec_dirs:
    if not rec_dir.is_dir():
        continue
    gt, vm = load_psr_labels(rec_dir)
    if gt is None:
        continue

    # Build canonical pred
    canon = np.zeros_like(gt, dtype=np.int64)
    n_rec = gt.shape[0]
    for t in range(n_rec):
        k_done = int(gt[t, :].sum())
        for i in range(min(k_done, C)):
            canon[t, i] = 1

    pos_rec = _compute_psr_pos_vectorized(canon, gt.astype(np.int64), vm)
    rec_pos_values.append(pos_rec)

    dbg = per_component_pos_debug(gt.astype(np.int64), vm, canon)
    for c in range(C):
        if dbg[c]['status'] == 'evaluated':
            comp_pos_matrix[c].append(dbg[c]['pos'])

# Compute macro-average
rec_macro = np.mean(rec_pos_values) if rec_pos_values else 0.0
print(f"Per-recording POS values: {[f'{v:.4f}' for v in rec_pos_values]}")
print(f"Per-recording macro-average: {rec_macro:.6f}")

comp_macro = {c: np.mean(vals) for c, vals in comp_pos_matrix.items() if vals}
print(f"\nPer-component macro-average (across recordings with 3+ runs):")
for c in sorted(comp_macro.keys()):
    print(f"  comp{c}: {comp_macro[c]:.4f} (evaluated in {len(comp_pos_matrix[c])}/{total_rec} recordings)")

# Use the actual function
pos_blind_concat = _compute_psr_pos_canonical(
    gt_safe_concat.astype(np.int64), valid_mask_concat
)
print(f"\n_compute_psr_pos_canonical (concat): {pos_blind_concat:.6f}")

# Compute per-recording using the function
per_rec_func = []
for rec_dir in rec_dirs:
    if not rec_dir.is_dir():
        continue
    gt, vm = load_psr_labels(rec_dir)
    if gt is None:
        continue
    pos = _compute_psr_pos_canonical(gt.astype(np.int64), vm)
    per_rec_func.append(pos)

print(f"Per-recording _compute_psr_pos_canonical: {[f'{v:.4f}' for v in per_rec_func]}")
print(f"Per-recording mean: {np.mean(per_rec_func):.6f}")
