#!/usr/bin/env python3
"""Analyze GT distribution and compute present-class mAP."""
import json, os, sys, numpy as np
sys.path.insert(0, '.')
os.chdir('.')
from src.evaluation.evaluate import compute_ap_per_class

print('Loading predictions...')
with open('src/runs/rf_stages/checkpoints/d3_full_eval/per_frame_predictions.json') as f:
    data = json.load(f)

num_det_classes = data['num_det_classes']
gt_labels_list = data['det_gt_labels']

# Count GT boxes per class
gt_per_class = {}
for frame_labels in gt_labels_list:
    for lbl in frame_labels:
        gt_per_class[lbl] = gt_per_class.get(lbl, 0) + 1

present_classes = sorted([c for c in range(num_det_classes) if gt_per_class.get(c, 0) > 0])
zero_gt_classes = sorted([c for c in range(num_det_classes) if gt_per_class.get(c, 0) == 0])

print(f'Present classes ({len(present_classes)}): {present_classes}')
print(f'Zero-GT classes ({len(zero_gt_classes)}): {zero_gt_classes}')
for c in present_classes:
    print(f'  Class {c}: {gt_per_class[c]} GT boxes')

# Convert to numpy arrays
pb = [np.array(b, dtype=np.float32) for b in data['det_pred_boxes']]
ps = [np.array(s, dtype=np.float32) for s in data['det_pred_scores']]
pl = [np.array(l, dtype=np.int64) for l in data['det_pred_labels']]
gb = [np.array(b, dtype=np.float32) for b in data['det_gt_boxes']]
gl = [np.array(l, dtype=np.int64) for l in data['det_gt_labels']]
del data

print('\nComputing full-set mAP50 (all 24 classes)...')
result = compute_ap_per_class(pb, ps, pl, gb, gl, iou_thresh=0.5, num_classes=num_det_classes)
mAP50_all = result['mAP']
print(f'Full-set mAP50 (24 classes): {mAP50_all:.6f}')

print('\nComputing present-class mAP50...')
result_pc = compute_ap_per_class(pb, ps, pl, gb, gl, iou_thresh=0.5, num_classes=num_det_classes)
# For present-class mAP, we only average over present classes
pc_aps = [result_pc['per_class_ap'][c] for c in present_classes]
mAP50_pc = float(np.mean(pc_aps)) if pc_aps else 0.0
print(f'Present-class mAP50 ({len(present_classes)} classes): {mAP50_pc:.6f}')

print('\nAll per-class AP50:')
for c in range(num_det_classes):
    ap = result_pc['per_class_ap'].get(c, 0.0)
    marker = ' (present)' if c in present_classes else ' (zero-GT)'
    print(f'  Class {c:2d}: {ap:.6f}{marker}')

# Also compute mAP@0.5:0.95
print('\nComputing mAP@0.5:0.95...')
iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
map_50_95_all = 0.0
map_50_95_pc = 0.0
per_class_50_95_all = {}
for t in iou_thresholds:
    r = compute_ap_per_class(pb, ps, pl, gb, gl, iou_thresh=t, num_classes=num_det_classes)
    map_50_95_all += r['mAP']
    pc_vals = [r['per_class_ap'][c] for c in present_classes]
    map_50_95_pc += float(np.mean(pc_vals)) if pc_vals else 0.0
    for c, v in r['per_class_ap'].items():
        per_class_50_95_all[c] = per_class_50_95_all.get(c, 0.0) + v
map_50_95_all /= len(iou_thresholds)
map_50_95_pc /= len(iou_thresholds)
for c in per_class_50_95_all:
    per_class_50_95_all[c] /= len(iou_thresholds)

del pb, ps, pl, gb, gl

print(f'\nFull-set mAP@0.5:0.95: {map_50_95_all:.6f}')
print(f'Present-class mAP@0.5:0.95: {map_50_95_pc:.6f}')

# Save comprehensive results
out_dir = 'src/runs/rf_stages/checkpoints/d3_full_38k'
out = {
    'full_set': {
        'mAP50': mAP50_all,
        'mAP_50_95': map_50_95_all,
        'num_classes': num_det_classes,
    },
    'present_class': {
        'mAP50': mAP50_pc,
        'mAP_50_95': map_50_95_pc,
        'num_classes': len(present_classes),
        'class_ids': present_classes,
    },
    'per_class_AP50': {str(c): result_pc['per_class_ap'].get(c, 0.0) for c in range(num_det_classes)},
    'per_class_AP_50_95': {str(c): per_class_50_95_all[c] for c in range(num_det_classes)},
    'zero_gt_classes': zero_gt_classes,
    'present_classes': present_classes,
    'num_images': 38036,
    'gt_boxes_per_class': {str(c): gt_per_class.get(c, 0) for c in range(num_det_classes)},
    'total_gt_boxes': sum(gt_per_class.values()),
    'iou_thresholds': iou_thresholds,
    'source': 'per_frame_predictions.json from d3_full_eval (v1, 38k frames)',
}
with open(os.path.join(out_dir, 'detection_mAP.json'), 'w') as f:
    json.dump(out, f, indent=2)

# Write markdown
total_gt = sum(gt_per_class.values())
md = f'''# D3 Full-38k Detection mAP

Computed from per-frame predictions on all 38,036 validation frames.

| Metric | Value |
|--------|-------|
| Full-set mAP@0.5 (24 classes) | {mAP50_all:.6f} |
| Full-set mAP@0.5:0.95 | {map_50_95_all:.6f} |
| Present-class mAP@0.5 ({len(present_classes)} classes) | {mAP50_pc:.6f} |
| Present-class mAP@0.5:0.95 | {map_50_95_pc:.6f} |
| Total frames | 38036 |
| Total GT boxes | {total_gt} |
| Present classes | {len(present_classes)} |
| Zero-GT classes | {len(zero_gt_classes)} |

## Per-Class AP@0.5

| Class ID | AP@0.5 | AP@0.5:0.95 | GT boxes | Present? |
|----------|--------|------------|----------|----------|
'''
for c in range(num_det_classes):
    ap50 = result_pc['per_class_ap'].get(c, 0.0)
    ap5095 = per_class_50_95_all[c]
    gt_cnt = gt_per_class.get(c, 0)
    marker = 'YES' if c in present_classes else '—'
    md += f'| {c} | {ap50:.6f} | {ap5095:.6f} | {gt_cnt} | {marker} |\n'

md += f'''
COCO-style all-point interpolation.
mAP@0.5:0.95 averaged over 10 thresholds (0.50:0.05:0.95).
Present-class mAP excludes {len(zero_gt_classes)} classes with zero GT boxes.

**Note**: The full-set mAP is very low ({mAP50_all:.4f}) because only {total_gt} GT boxes exist
across 38,036 frames, while the model produces ~3.98M predictions. Most frames have
zero GT boxes, so most predictions are false positives. The model never saw real
detection training — it was trained as a multi-task model with detection as an auxiliary head.
'''

with open(os.path.join(out_dir, 'detection_mAP.md'), 'w') as f:
    f.write(md)

print(f'\nResults saved to {out_dir}/')
