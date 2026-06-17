#!/usr/bin/env python3
"""
Deep investigation: verify all data and annotations are correctly used.
Checks:
1. All sample entries have valid file paths
2. All annotation types per sample are non-empty
3. Correct placement of annotations in every data item
4. Per-recording annotation completeness
"""

import csv
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

POPW_ROOT = Path('/media/newadmin/master/POPW/datasets/industreal')
RECORDINGS_ROOT = POPW_ROOT / 'recordings'
TRAIN_CSV = POPW_ROOT / 'splits' / 'train.csv'
VAL_CSV = POPW_ROOT / 'splits' / 'val.csv'

# COCO 24-class names
DET_CLASS_NAMES = [
    'background', 'Cabinet', 'CabinetDoor', 'Door', 'Drawer', 'Handle',
    'Hinge', 'Wall', 'Shelf', 'LEDStrip', 'PowerController', 'PowerOutlet',
    'PowerSwitch', 'Controller', 'Sensor', 'Touch_sensor', 'Button',
    'Display', 'Marker', 'Camera', 'Cable', 'CableTie', 'Screwdriver', 'Workbench'
]

issues = []
warnings = []

def load_csv_samples(csv_path):
    """Load all sample entries from split CSV."""
    samples = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rec_id = row['recording_id']
            action_label = int(row['action_class_id'])
            samples.append({
                'recording_id': rec_id,
                'action_label': action_label,
            })
    return samples

def get_recording_frames(rec_dir):
    """Get all frame numbers for a recording."""
    rgb_dir = rec_dir / 'rgb'
    if not rgb_dir.exists():
        return []
    frames = []
    for f in sorted(rgb_dir.glob('*.jpg')):
        try:
            frames.append(int(f.stem))
        except:
            pass
    return sorted(frames)

def check_pose_annotation(rec_dir, num_frames):
    """Check pose.csv — all 9 DoF present and non-zero."""
    pose_file = rec_dir / 'pose.csv'
    if not pose_file.exists():
        return {'status': 'missing', 'num_frames': num_frames}

    fwd_norms = []
    pos_vals = []
    up_norms = []
    parsed = 0
    all_zeros = 0

    with open(pose_file, encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 10:
                continue
            try:
                frame_num = int(Path(row[0]).stem)
                values = [float(v) for v in row[1:10]]
                fwd = values[0:3]
                pos = values[3:6]
                up = values[6:9]
                if not np.allclose(fwd, 0):
                    fwd_norms.append(np.linalg.norm(fwd))
                if not np.allclose(pos, 0):
                    pos_vals.append(np.max(np.abs(pos)))
                if not np.allclose(up, 0):
                    up_norms.append(np.linalg.norm(up))
                parsed += 1
            except:
                continue

    return {
        'status': 'ok',
        'parsed_rows': parsed,
        'fwd_norm_mean': np.mean(fwd_norms) if fwd_norms else 0,
        'fwd_norm_std': np.std(fwd_norms) if fwd_norms else 0,
        'pos_max_mean': np.mean(pos_vals) if pos_vals else 0,
        'up_norm_mean': np.mean(up_norms) if up_norms else 0,
        'fwd_nonzero_count': len(fwd_norms),
        'up_nonzero_count': len(up_norms),
    }

def check_hands_annotation(rec_dir, num_frames):
    """Check hands.csv — all 52 joints present and non-zero."""
    hands_file = rec_dir / 'hands.csv'
    if not hands_file.exists():
        return {'status': 'missing'}

    parsed = 0
    nonzero_joints = 0

    with open(hands_file, encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 53:
                continue
            try:
                frame_num = int(Path(row[0]).stem)
                values = [float(v) for v in row[1:53]]
                if not np.allclose(values, 0):
                    nonzero_joints += 1
                parsed += 1
            except:
                continue

    return {
        'status': 'ok',
        'parsed_rows': parsed,
        'nonzero_frames': nonzero_joints,
    }

def check_psr_annotation(rec_dir, num_frames):
    """Check PSR_labels_raw.csv — fill-forward produces dense per-frame."""
    psr_file = rec_dir / 'PSR_labels_raw.csv'
    if not psr_file.exists():
        return {'status': 'missing'}

    parsed = 0
    changes = 0
    current = np.zeros(11, dtype=np.float32)

    with open(psr_file, encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 12:
                continue
            try:
                frame_num = int(Path(row[0]).stem)
                vals = [float(v) for v in row[1:12]]
                for j in range(11):
                    if vals[j] != -2:
                        if vals[j] != current[j]:
                            changes += 1
                        current[j] = vals[j]
                parsed += 1
            except:
                continue

    return {
        'status': 'ok',
        'parsed_rows': parsed,
        'changes': changes,
    }

def check_ar_annotation(rec_dir, num_frames):
    """Check AR_labels.csv — sparse spans interpolated to per-frame."""
    ar_file = rec_dir / 'AR_labels.csv'
    if not ar_file.exists():
        return {'status': 'missing'}

    spans = []
    with open(ar_file, encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 5:
                continue
            try:
                action_id = int(row[1])
                start_frame = int(Path(row[3]).stem)
                end_frame = int(Path(row[4]).stem)
                spans.append((start_frame, end_frame, action_id))
            except:
                continue

    labels = np.zeros(num_frames, dtype=np.int64)
    for start, end, action_id in spans:
        end = min(end, num_frames - 1)
        if start < num_frames:
            labels[start:end+1] = action_id

    zero_labels = np.sum(labels == 0)

    return {
        'status': 'ok',
        'num_spans': len(spans),
        'labeled_frames': num_frames - zero_labels,
        'zero_frames': zero_labels,
    }

def check_od_annotation(rec_dir, num_frames):
    """Check OD_labels.json (COCO format) — object detections per frame."""
    od_file = rec_dir / 'OD_labels.json'
    if not od_file.exists():
        return {'status': 'missing'}

    with open(od_file, 'r') as f:
        data = json.load(f)

    images = {img['id']: img for img in data.get('images', [])}
    annotations = data.get('annotations', [])

    frame_to_annots = defaultdict(list)
    for ann in annotations:
        img_id = ann.get('image_id', 0)
        frame_num_str = images.get(img_id, {}).get('file_name', '')
        if frame_num_str:
            try:
                fn = int(Path(frame_num_str).stem)
                frame_to_annots[fn].append(ann)
            except:
                continue

    frames_with_annots = len(frame_to_annots)
    total_objects = sum(len(v) for v in frame_to_annots.values())
    empty_frames = sum(1 for fn in range(num_frames) if fn not in frame_to_annots)

    class_counts = defaultdict(int)
    for anns in frame_to_annots.values():
        for ann in anns:
            cat = ann.get('category_id', 0)
            class_counts[cat] += 1

    return {
        'status': 'ok',
        'frames_with_annots': frames_with_annots,
        'total_objects': total_objects,
        'empty_frames': empty_frames,
        'class_counts': dict(class_counts),
    }

def load_recording_data(rec_id, rec_dir):
    """Load all annotation data for a recording."""
    frames = get_recording_frames(rec_dir)
    num_frames = len(frames)

    # pose
    pose_data = np.zeros((num_frames, 9), dtype=np.float32)
    pose_file = rec_dir / 'pose.csv'
    if pose_file.exists():
        with open(pose_file, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 10:
                    continue
                try:
                    fn = int(Path(row[0]).stem)
                    pose_data[fn] = [float(v) for v in row[1:10]]
                except:
                    continue

    # hands
    hands_data = np.zeros((num_frames, 52), dtype=np.float32)
    hands_file = rec_dir / 'hands.csv'
    if hands_file.exists():
        with open(hands_file, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 53:
                    continue
                try:
                    fn = int(Path(row[0]).stem)
                    hands_data[fn] = [float(v) for v in row[1:53]]
                except:
                    continue

    # PSR
    psr_data = np.zeros((num_frames, 11), dtype=np.float32)
    current = np.zeros(11, dtype=np.float32)
    psr_file = rec_dir / 'PSR_labels_raw.csv'
    if psr_file.exists():
        with open(psr_file, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 12:
                    continue
                try:
                    fn = int(Path(row[0]).stem)
                    vals = [float(v) for v in row[1:12]]
                    for j in range(11):
                        if vals[j] != -2:
                            current[j] = vals[j]
                    psr_data[fn] = current
                except:
                    continue

    # AR
    ar_data = np.zeros(num_frames, dtype=np.int64)
    ar_file = rec_dir / 'AR_labels.csv'
    if ar_file.exists():
        spans = []
        with open(ar_file, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 5:
                    continue
                try:
                    action_id = int(row[1])
                    start = int(Path(row[3]).stem)
                    end = int(Path(row[4]).stem)
                    spans.append((start, end, action_id))
                except:
                    continue
        for start, end, action_id in spans:
            end = min(end, num_frames - 1)
            if start < num_frames:
                ar_data[start:end+1] = action_id

    # OD
    od_data = defaultdict(list)
    od_file = rec_dir / 'OD_labels.json'
    if od_file.exists():
        with open(od_file, 'r') as f:
            data = json.load(f)
        images = {img['id']: img for img in data.get('images', [])}
        for ann in data.get('annotations', []):
            img_id = ann.get('image_id', 0)
            fname = images.get(img_id, {}).get('file_name', '')
            if fname:
                try:
                    fn = int(Path(fname).stem)
                    od_data[fn].append(ann)
                except:
                    continue

    return {
        'num_frames': num_frames,
        'pose': pose_data,
        'hands': hands_data,
        'psr': psr_data,
        'ar': ar_data,
        'od': od_data,
    }

def check_rgb_exists(samples):
    """Check all RGB frames exist."""
    print("\n=== CHECK 1: RGB Frame Existence ===")
    missing = 0
    for s in samples:
        rec_id = s['recording_id']
        rec_dir = RECORDINGS_ROOT / 'train' / rec_id
        if not rec_dir.exists():
            rec_dir = RECORDINGS_ROOT / 'val' / rec_id
        rgb_dir = rec_dir / 'rgb'
        if not rgb_dir.exists():
            missing += 1
            if missing <= 5:
                issues.append(f"Missing RGB dir: {rgb_dir}")
            continue
        frames = get_recording_frames(rec_dir)
        if not frames:
            missing += 1
            if missing <= 5:
                issues.append(f"No frames in: {rgb_dir}")
    print(f"  Recordings missing RGB: {missing}/{len(samples)}")
    return missing

def analyze_recording(rec_id, rec_dir):
    """Analyze all annotations for one recording."""
    frames = get_recording_frames(rec_dir)
    num_frames = len(frames)
    if num_frames == 0:
        return {'error': 'no frames'}

    pose_info = check_pose_annotation(rec_dir, num_frames)
    hands_info = check_hands_annotation(rec_dir, num_frames)
    psr_info = check_psr_annotation(rec_dir, num_frames)
    ar_info = check_ar_annotation(rec_dir, num_frames)
    od_info = check_od_annotation(rec_dir, num_frames)

    return {
        'num_frames': num_frames,
        'pose': pose_info,
        'hands': hands_info,
        'psr': psr_info,
        'ar': ar_info,
        'od': od_info,
    }

def check_annotation_placement(samples):
    """Verify every frame in every recording has all annotation types."""
    print("\n=== CHECK 3: Per-Frame Annotation Placement ===")

    # Group samples by recording
    by_rec = defaultdict(list)
    for s in samples:
        by_rec[s['recording_id']].append(s)

    total_empty = {
        'pose': 0, 'hands': 0, 'psr': 0, 'ar': 0, 'od': 0
    }
    total_frames = 0

    for rec_id in sorted(by_rec.keys()):
        rec_dir = RECORDINGS_ROOT / 'train' / rec_id
        if not rec_dir.exists():
            rec_dir = RECORDINGS_ROOT / 'val' / rec_id

        data = load_recording_data(rec_id, rec_dir)
        num_frames = data['num_frames']
        total_frames += num_frames

        for fn in range(num_frames):
            # Pose: all 9 DoF must be non-zero
            if np.allclose(data['pose'][fn], 0):
                total_empty['pose'] += 1

            # Hands: all 52 joints must be non-zero
            if np.allclose(data['hands'][fn], 0):
                total_empty['hands'] += 1

            # PSR: all 11 components must be non-zero
            if np.allclose(data['psr'][fn], 0):
                total_empty['psr'] += 1

            # AR: label must be non-zero
            if data['ar'][fn] == 0:
                total_empty['ar'] += 1

            # OD: at least one detection
            if fn not in data['od'] or len(data['od'][fn]) == 0:
                total_empty['od'] += 1

    print(f"  Analyzing {total_frames} total frames across {len(by_rec)} recordings")
    for k, v in total_empty.items():
        pct = 100 * v / total_frames if total_frames > 0 else 0
        print(f"  Empty {k}: {v}/{total_frames} ({pct:.1f}%)")

    return total_empty, total_frames

def main():
    print("=" * 80)
    print("INDUSTREAL DATASET — DEEP ANNOTATION INVESTIGATION")
    print("=" * 80)

    train_samples = load_csv_samples(TRAIN_CSV)
    val_samples = load_csv_samples(VAL_CSV)
    all_samples = train_samples + val_samples

    print(f"\nLoaded {len(train_samples)} train + {len(val_samples)} val = {len(all_samples)} total recordings")
    unique_recs = set(s['recording_id'] for s in all_samples)
    print(f"Unique recordings: {len(unique_recs)}")

    # Check 1: RGB existence
    missing_rgb = check_rgb_exists(all_samples)

    # Check 2: Per-recording annotation quality
    print("\n=== CHECK 2: Per-Recording Annotation Quality ===")
    for rec_id in sorted(unique_recs):
        rec_dir = RECORDINGS_ROOT / 'train' / rec_id
        if not rec_dir.exists():
            rec_dir = RECORDINGS_ROOT / 'val' / rec_id

        info = analyze_recording(rec_id, rec_dir)

        if 'error' in info:
            print(f"\n  [{rec_id}] ERROR: {info['error']}")
            continue

        print(f"\n  [{rec_id}] frames={info['num_frames']}")

        # Pose
        p = info['pose']
        if p.get('status') == 'missing':
            print(f"    pose: MISSING")
        else:
            print(f"    pose: rows={p['parsed_rows']}, fwd_nonzero={p['fwd_nonzero_count']}, "
                  f"up_nonzero={p['up_nonzero_count']}, pos_max_mean={p['pos_max_mean']:.1f}")

        # Hands
        h = info['hands']
        if h.get('status') == 'missing':
            print(f"    hands: MISSING")
        else:
            print(f"    hands: rows={h['parsed_rows']}, nonzero_frames={h['nonzero_frames']}")

        # PSR
        psr = info['psr']
        if psr.get('status') == 'missing':
            print(f"    psr: MISSING")
        else:
            print(f"    psr: rows={psr['parsed_rows']}, changes={psr['changes']}")

        # AR
        ar = info['ar']
        if ar.get('status') == 'missing':
            print(f"    ar: MISSING")
        else:
            print(f"    ar: spans={ar['num_spans']}, labeled_frames={ar['labeled_frames']}, "
                  f"zero_frames={ar['zero_frames']}")

        # OD
        od = info['od']
        if od.get('status') == 'missing':
            print(f"    od: MISSING")
        else:
            print(f"    od: frames={od['frames_with_annots']}/{info['num_frames']}, "
                  f"total_objs={od['total_objects']}, empty={od['empty_frames']}")
            cc = od.get('class_counts', {})
            if cc:
                top3 = sorted(cc.items(), key=lambda x: -x[1])[:3]
                print(f"       top_classes: {top3}")

    # Check 3: Per-frame annotation placement
    placement, total_frames = check_annotation_placement(all_samples)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total recordings: {len(all_samples)}")
    print(f"Missing RGB dirs: {missing_rgb}")
    print(f"\nPer-frame annotation empties ({total_frames} total frames):")
    for k, v in placement.items():
        pct = 100 * v / total_frames if total_frames > 0 else 0
        print(f"  {k}: {v}/{total_frames} ({pct:.1f}%)")

    if issues:
        print(f"\nCRITICAL ISSUES ({len(issues)}):")
        for i in issues[:20]:
            print(f"  - {i}")
    else:
        print("\nNo critical issues found.")

if __name__ == '__main__':
    main()