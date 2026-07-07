#!/usr/bin/env python3
"""CPU-only script to compute detection mAP from per_frame_predictions.json.

Loads pre-computed predictions and GT from the v1 full eval output,
computes per-class AP and mAP50 using the same compute_ap_per_class function
used in the evaluation pipeline. No GPU needed.
"""

import json
import sys
import os
import numpy as np

# Add project root to path
project_root = '/media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved'
sys.path.insert(0, project_root)
os.chdir(project_root)

from src.evaluation.evaluate import compute_ap_per_class


def main():
    predictions_path = 'src/runs/rf_stages/checkpoints/d3_full_eval/per_frame_predictions.json'
    out_dir = 'src/runs/rf_stages/checkpoints/d3_full_38k'
    os.makedirs(out_dir, exist_ok=True)

    print(f'Loading {predictions_path} ...')
    with open(predictions_path) as f:
        data = json.load(f)

    num_images = data['num_images']
    num_det_classes = data['num_det_classes']
    pred_boxes = data['det_pred_boxes']
    pred_scores = data['det_pred_scores']
    pred_labels = data['det_pred_labels']
    gt_boxes = data['det_gt_boxes']
    gt_labels = data['det_gt_labels']

    print(f'Loaded {num_images} images, {num_det_classes} classes')
    total_pred = sum(len(b) for b in pred_boxes)
    total_gt = sum(len(b) for b in gt_boxes)
    print(f'Total predictions: {total_pred}, Total GT boxes: {total_gt}')

    # Convert lists to numpy arrays per image
    pred_boxes_np = [np.array(b, dtype=np.float32) for b in pred_boxes]
    pred_scores_np = [np.array(s, dtype=np.float32) for s in pred_scores]
    pred_labels_np = [np.array(l, dtype=np.int64) for l in pred_labels]
    gt_boxes_np = [np.array(b, dtype=np.float32) for b in gt_boxes]
    gt_labels_np = [np.array(l, dtype=np.int64) for l in gt_labels]

    # Compute mAP50
    print('Computing detection mAP@0.5 ...')
    result = compute_ap_per_class(
        pred_boxes_np, pred_scores_np, pred_labels_np,
        gt_boxes_np, gt_labels_np,
        iou_thresh=0.5,
        num_classes=num_det_classes,
        interpolation_mode='coco',
    )

    mAP50 = result['mAP']
    per_class_ap = result['per_class_ap']

    # Also compute mAP@0.5:0.95 by averaging multiple IoU thresholds
    iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    map_50_95 = 0.0
    per_class_ap_50_95 = {}
    for t in iou_thresholds:
        r = compute_ap_per_class(
            pred_boxes_np, pred_scores_np, pred_labels_np,
            gt_boxes_np, gt_labels_np,
            iou_thresh=t,
            num_classes=num_det_classes,
            interpolation_mode='coco',
        )
        map_50_95 += r['mAP']
        for c, v in r['per_class_ap'].items():
            per_class_ap_50_95[c] = per_class_ap_50_95.get(c, 0.0) + v
    map_50_95 /= len(iou_thresholds)
    for c in per_class_ap_50_95:
        per_class_ap_50_95[c] /= len(iou_thresholds)

    print(f'\n=== D3 Full-38k Detection Results ===')
    print(f'mAP@0.5:      {mAP50:.4f}')
    print(f'mAP@0.5:0.95: {map_50_95:.4f}')
    print(f'Per-class AP@0.5:')
    for c in sorted(per_class_ap.keys()):
        print(f'  Class {c:2d}: {per_class_ap[c]:.4f}')
    print(f'Total GT boxes: {total_gt} across {num_images} frames')

    # Save JSON
    out_json = {
        'mAP50': mAP50,
        'mAP_50_95': map_50_95,
        'per_class_AP50': {str(k): v for k, v in per_class_ap.items()},
        'per_class_AP_50_95': {str(k): v for k, v in per_class_ap_50_95.items()},
        'num_images': num_images,
        'num_det_classes': num_det_classes,
        'total_gt_boxes': total_gt,
        'iou_thresholds_used': iou_thresholds,
        'interpolation_mode': 'coco',
        'source': 'per_frame_predictions.json from d3_full_eval (v1, 38k frames)',
    }
    json_path = os.path.join(out_dir, 'detection_mAP.json')
    with open(json_path, 'w') as f:
        json.dump(out_json, f, indent=2)
    print(f'\nSaved {json_path}')

    # Save Markdown report
    md_lines = [
        '# D3 Full-38k Detection mAP',
        '',
        f'Computed from `per_frame_predictions.json` (v1 full eval on all 38,036 frames).',
        '',
        f'| Metric | Value |',
        f'|--------|-------|',
        f'| mAP@0.5 | {mAP50:.4f} |',
        f'| mAP@0.5:0.95 | {map_50_95:.4f} |',
        f'| Total frames | {num_images} |',
        f'| Total GT boxes | {total_gt} |',
        f'| Detection classes | {num_det_classes} |',
        '',
        '## Per-Class AP@0.5',
        '',
        '| Class ID | AP@0.5 |',
        '|----------|--------|',
    ]
    for c in sorted(per_class_ap.keys()):
        md_lines.append(f'| {c} | {per_class_ap[c]:.4f} |')

    md_lines.extend([
        '',
        '## Per-Class AP@0.5:0.95',
        '',
        '| Class ID | AP@0.5:0.95 |',
        '|----------|------------|',
    ])
    for c in sorted(per_class_ap_50_95.keys()):
        md_lines.append(f'| {c} | {per_class_ap_50_95[c]:.4f} |')

    md_lines.append('')
    md_lines.append(f'Computed with COCO-style all-point interpolation.')
    md_lines.append(f'Average of 10 IoU thresholds (0.50:0.05:0.95) for mAP@0.5:0.95.')
    md_lines.append(f'GPU not required — all computation on CPU from saved predictions.')

    md_path = os.path.join(out_dir, 'detection_mAP.md')
    with open(md_path, 'w') as f:
        f.write('\n'.join(md_lines) + '\n')
    print(f'Saved {md_path}')

    print('\nDone.')


if __name__ == '__main__':
    main()
