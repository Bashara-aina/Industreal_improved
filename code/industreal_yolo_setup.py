#!/usr/bin/env python3
"""Convert IndustReal COCO labels to YOLOv8m format and start training.

[Opus 126 Decision 5] D1-R: retrain YOLOv8m on IndustReal.
Output: yolo_dataset/{images,labels}/<rec>_<id>.{jpg,txt} + data.yaml
"""
import json
import os
import shutil
from pathlib import Path
from collections import defaultdict

DATA_ROOT = Path("/media/newadmin/master/POPW/datasets/industreal/recordings")
OUT_ROOT = Path("/media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/yolo_dataset")
IMG_DIR = OUT_ROOT / "images"
LBL_DIR = OUT_ROOT / "labels"
IMG_DIR.mkdir(parents=True, exist_ok=True)
LBL_DIR.mkdir(parents=True, exist_ok=True)

def coco_to_yolo(bbox, img_w, img_h):
    """COCO [x_tl, y_tl, w, h] -> YOLO [x_center, y_center, w, h] all normalized."""
    x, y, w, h = bbox
    x_c = (x + w / 2) / img_w
    y_c = (y + h / 2) / img_h
    return (x_c, y_c, w / img_w, h / img_h)

def process_recording(rec_dir: Path, split: str):
    od_path = rec_dir / "OD_labels.json"
    if not od_path.exists():
        return 0, 0
    data = json.load(open(od_path))
    images = {img["id"]: img for img in data["images"]}
    n_imgs = 0
    n_boxes = 0
    for ann in data["annotations"]:
        img = images.get(ann["image_id"])
        if not img:
            continue
        x_c, y_c, w, h = coco_to_yolo(ann["bbox"], img["width"], img["height"])
        if not (0 <= x_c <= 1 and 0 <= y_c <= 1 and 0 < w <= 1 and 0 < h <= 1):
            continue  # skip out-of-bounds
        cls = ann["category_id"] - 1  # COCO is 1-indexed; YOLO is 0-indexed
        out_name = f"{rec_dir.name}_{img['id']:06d}"
        # Symlink image
        src_img = rec_dir / "rgb" / img["file_name"]
        dst_img = IMG_DIR / f"{out_name}.jpg"
        if src_img.exists() and not dst_img.exists():
            os.symlink(src_img, dst_img)
        # Write label
        lbl_file = LBL_DIR / f"{out_name}.txt"
        with open(lbl_file, "a") as f:
            f.write(f"{cls} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")
        n_boxes += 1
    n_imgs = len(set(ann["image_id"] for ann in data["annotations"]))
    return n_imgs, n_boxes

# Discover category names
cat_names = {}
for split in ["train", "test"]:
    for rec in (DATA_ROOT / split).iterdir():
        od = rec / "OD_labels.json"
        if od.exists():
            data = json.load(open(od))
            for c in data["categories"]:
                cat_names[c["id"]] = c["name"]
            break
    if cat_names:
        break
print(f"Found {len(cat_names)} categories: {list(cat_names.items())[:3]}...")

# Process all recordings
total_imgs, total_boxes = 0, 0
for split in ["train", "test"]:
    print(f"Processing {split}/...")
    for rec in sorted((DATA_ROOT / split).iterdir()):
        n_imgs, n_boxes = process_recording(rec, split)
        total_imgs += n_imgs
        total_boxes += n_boxes
print(f"Done: {total_imgs} images, {total_boxes} boxes")

# Write YOLO data.yaml
yaml_path = OUT_ROOT / "data.yaml"
with open(yaml_path, "w") as f:
    f.write(f"""# YOLOv8m data.yaml for IndustReal (D1-R)
path: {OUT_ROOT}
train: images  # all images in single dir
val: images
nc: {len(cat_names)}
names:
""")
    for cid in sorted(cat_names.keys()):
        f.write(f"  {cid-1}: {cat_names[cid]}\n")
print(f"Wrote {yaml_path}")
