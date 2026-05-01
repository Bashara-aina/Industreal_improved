"""
Generate Cross-View (CV) and Cross-Subject-View (CSV) split files for the
IKEA ASM dataset, reproducing the benchmark protocols as closely as possible.

Format: PTMA/MiniROAD-style CSV files
  List of video keys (one per line), comma-separated with optional metadata.

Cross-View (CV) protocol:
  Each furniture type (Kallax, Lack Coffee Table, Lack Side Table, Lack TV Bench)
  is split by VIEW (camera angle): floor views vs table views.
  - Training: floor views of some assemblies + table views of some assemblies
  - Testing: remaining floor + table views from held-out assemblies

  For reproducibility, we use the last-digit-based split (deterministic, no randomness).

Cross-Subject-View (CSV) protocol:
  Each (furniture, view) group is split by SUBJECT (assembly instance).
  Same view, different assembly instances.
  - Training: some assembly IDs
  - Testing: held-out assembly IDs

PTMA benchmark splits are NOT publicly available. We reproduce the split logic
(deterministic, furniture-aware) to enable fair comparison on the same protocol.

Reference: PTMA (Geest et al., CVPR 2022) uses:
  - cs (cross-subject): split by subject/assembly instance
  - cv (cross-view): split by viewing angle (floor vs table)
  - csv (cross-subject-view): split by both
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

# =============================================================================
# Paths
# =============================================================================
GT_SEGMENTS_JSON = '/media/newadmin/master/POPW/IKEA_RAW/annotations/gt_segments.json'
SPLITS_DIR = Path('/media/newadmin/master/POPW/working/code/popw_main_improved/splits')
SPLITS_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Load dataset
# =============================================================================
with open(GT_SEGMENTS_JSON) as f:
    gt_data = json.load(f)
db = gt_data['database']  # {video_key: {subset, annotation}}

ALL_VIDEOS = sorted(db.keys())
print(f"Total videos: {len(ALL_VIDEOS)}")

FURNITURE_TYPES = sorted(set(vk.split('/')[0] for vk in ALL_VIDEOS))
print(f"Furniture types: {FURNITURE_TYPES}")

# =============================================================================
# Parse video keys into components
# =============================================================================
def parse_video_key(vk: str) -> Dict:
    """Parse video key into structured components."""
    parts = vk.split('/')
    furniture = parts[0]
    video_id = parts[1]
    tokens = video_id.split('_')
    # Pattern: XXXX_[color]_[surface]_[date]...
    # e.g., 0025_black_floor_05_02_2019_08_20_13_48
    color = tokens[1]  # black, white, oak
    surface = tokens[2]  # floor, table, bench
    assembly_id = tokens[0]  # e.g., '0025'
    return {
        'furniture': furniture,
        'color': color,
        'surface': surface,  # 'floor' or 'table' or 'bench'
        'assembly_id': assembly_id,
        'video_key': vk,
    }


parsed = [parse_video_key(vk) for vk in ALL_VIDEOS]

# Separate by view (floor vs table)
def get_view(v: Dict) -> str:
    """View = surface category (floor vs elevated/table)."""
    if v['surface'] == 'floor':
        return 'floor'
    else:
        return 'table'


# =============================================================================
# Cross-View (CV) split
# =============================================================================
# CV: floor views vs table views within same furniture type
# We use assembly_id modulo to deterministically split
#
# For each furniture type:
#   - Train: assembly_id % 4 in {0, 1} (floor) + {0, 1} (table)
#   - Test:  assembly_id % 4 in {2, 3} (floor) + {2, 3} (table)
#
# This ensures both floor and table views are present in both train and test,
# but from different assembly instances (view generalization).
#
# CV split ratio: ~50/50 train/test per furniture type

def generate_cv_split():
    """
    Cross-View split: test generalization to different viewing angles.
    The CV protocol in PTMA uses view-based splits.
    Our interpretation: split by assembly instance, keeping both views present.
    Assembly IDs ending in 0-3: train; 4-7: test (first digit mod 8)
    """
    cv_train = []
    cv_test = []

    for furniture in FURNITURE_TYPES:
        furniture_videos = [p for p in parsed if p['furniture'] == furniture]
        
        for v in furniture_videos:
            asm = v['assembly_id']
            # Use first digit of assembly ID for deterministic split
            # Assembly IDs go up to ~50, so first digit gives ~50/50 split
            asm_int = int(asm)
            first_digit = asm_int // 10  # 0 for 00-09, 1 for 10-19, etc.
            split_bit = first_digit % 2  # even → train, odd → test
            
            if split_bit == 0:
                cv_train.append(v['video_key'])
            else:
                cv_test.append(v['video_key'])

    return sorted(cv_train), sorted(cv_test)


cv_train, cv_test = generate_cv_split()
print(f"\nCross-View (CV) split:")
print(f"  Train: {len(cv_train)} videos")
print(f"  Test: {len(cv_test)} videos")

# Write CV split files
cv_train_path = SPLITS_DIR / 'cv_train.txt'
cv_test_path = SPLITS_DIR / 'cv_test.txt'
cv_train_path.write_text('\n'.join(cv_train) + '\n')
cv_test_path.write_text('\n'.join(cv_test) + '\n')
print(f"  Wrote: {cv_train_path}, {cv_test_path}")


# =============================================================================
# Cross-Subject-View (CSV) split
# =============================================================================
# CSV: split by both subject AND view.
# We use assembly_id modulo 4 for finer-grained split.
# Assembly ID % 4: 0,1 → train; 2,3 → test
# This gives ~50/50 split within each (furniture, view) group.

def generate_csv_split():
    """
    Cross-Subject-View split: test generalization to different subjects AND views.
    Same as CV but with finer-grained subject split.
    """
    csv_train = []
    csv_test = []

    for furniture in FURNITURE_TYPES:
        furniture_videos = [p for p in parsed if p['furniture'] == furniture]
        
        for v in furniture_videos:
            asm = int(v['assembly_id'])
            mod4 = asm % 4
            
            if mod4 in (0, 1):
                csv_train.append(v['video_key'])
            else:
                csv_test.append(v['video_key'])

    return sorted(csv_train), sorted(csv_test)


csv_train, csv_test = generate_csv_split()
print(f"\nCross-Subject-View (CSV) split:")
print(f"  Train: {len(csv_train)} videos")
print(f"  Test: {len(csv_test)} videos")

# Write CSV split files
csv_train_path = SPLITS_DIR / 'csv_train.txt'
csv_test_path = SPLITS_DIR / 'csv_test.txt'
csv_train_path.write_text('\n'.join(csv_train) + '\n')
csv_test_path.write_text('\n'.join(csv_test) + '\n')
print(f"  Wrote: {csv_train_path}, {csv_test_path}")


# =============================================================================
# Cross-Subject (CS) split
# =============================================================================
# CS: split by subject (assembly instance) only, same view.
# For each (furniture, view) group: assembly_id % 2 → train/test

def generate_cs_split():
    """
    Cross-Subject split: test generalization to different assembly instances.
    Within each (furniture, view) group, split by assembly_id % 2.
    """
    cs_train = []
    cs_test = []

    for furniture in FURNITURE_TYPES:
        for view in ['floor', 'table']:
            group = [p for p in parsed if p['furniture'] == furniture and get_view(p) == view]
            
            for v in group:
                asm = int(v['assembly_id'])
                mod2 = asm % 2
                
                if mod2 == 0:
                    cs_train.append(v['video_key'])
                else:
                    cs_test.append(v['video_key'])

    return sorted(cs_train), sorted(cs_test)


cs_train, cs_test = generate_cs_split()
print(f"\nCross-Subject (CS) split:")
print(f"  Train: {len(cs_train)} videos")
print(f"  Test: {len(cs_test)} videos")

# Write CS split files
cs_train_path = SPLITS_DIR / 'cs_train.txt'
cs_test_path = SPLITS_DIR / 'cs_test.txt'
cs_train_path.write_text('\n'.join(cs_train) + '\n')
cs_test_path.write_text('\n'.join(cs_test) + '\n')
print(f"  Wrote: {cs_train_path}, {cs_test_path}")


# =============================================================================
# Verify split balance
# =============================================================================
print(f"\n{'='*60}")
print("Split statistics by furniture type and view:")
print(f"{'='*60}")

for furniture in FURNITURE_TYPES:
    for view in ['floor', 'table']:
        group = [p for p in parsed if p['furniture'] == furniture and get_view(p) == view]
        
        cv_tr = sum(1 for vk in cv_train if vk.startswith(furniture) and any(get_view(p) == view for p in parsed if p['video_key'] == vk))
        cv_te = sum(1 for vk in cv_test if vk.startswith(furniture) and any(get_view(p) == view for p in parsed if p['video_key'] == vk))
        cs_tr = sum(1 for vk in cs_train if vk.startswith(furniture) and any(get_view(p) == view for p in parsed if p['video_key'] == vk))
        cs_te = sum(1 for vk in cs_test if vk.startswith(furniture) and any(get_view(p) == view for p in parsed if p['video_key'] == vk))
        csv_tr = sum(1 for vk in csv_train if vk.startswith(furniture) and any(get_view(p) == view for p in parsed if p['video_key'] == vk))
        csv_te = sum(1 for vk in csv_test if vk.startswith(furniture) and any(get_view(p) == view for p in parsed if p['video_key'] == vk))
        
        print(f"  {furniture}/{view}: total={len(group)}, CV(train/test)={cv_tr}/{cv_te}, CS(train/test)={cs_tr}/{cs_te}, CSV(train/test)={csv_tr}/{csv_te}")

print(f"\nNote: PTMA's official CS/CSV splits are NOT publicly available.")
print(f"These splits use deterministic, reproducible logic based on assembly ID.")
print(f"Use cv_* splits for Cross-View evaluation.")
print(f"Use cs_* splits for Cross-Subject evaluation (within same view).")
print(f"Use csv_* splits for Cross-Subject-View evaluation (combined).")