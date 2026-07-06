# IKEA ASM T3 Cross-Dataset Training Plan

**Status**: Planning phase (D1-R YOLOv8m training in progress on GPU 0)
**Prepared**: 2026-07-05
**Author**: Monitoring agent

---

## 1. D1-R YOLOv8m Training Status

**Current state (2026-07-05 20:25 JST):**

| Metric | Value |
|--------|-------|
| PID | 563272 (GPU 0: RTX 3060) |
| Epochs completed | 2 / 25 (8% complete) |
| Current mAP50 | 0.856 (epoch 2) |
| mAP50-95 | 0.630 (epoch 2) |
| Precision/Recall | 0.874 / 0.867 (epoch 2) |
| Epoch time | ~9 min (epoch 1: 515s, epoch 2: 542s) |
| ETA completion | ~01:20 JST Jul 6 (~3.5h remaining) |
| Batch/device | batch=8, imgsz=640, GPU 0 |
| Split | Proper 80/20: 18867 train, 4956 val |
| Classes | 24 IndustReal assembly states |

**Expected final mAP50**: ~0.99 (typical for YOLOv8m on this dataset with proper split, based on prior runs). The dataset has 24 fine-grained assembly states (binary-coded assembly progress) with clear visual differences, making detection relatively straightforward for a medium-sized model like YOLOv8m.

**Post-completion steps:**
1. Read final results.csv for mAP50 and mAP50-95
2. Run `model.val()` on the proper val split to confirm metrics
3. Compare against the previous D1-R run (which was on the broken 50/50 split where both train and val pointed to the same images)

---

## 2. IKEA ASM Dataset Overview

**Source**: `/media/newadmin/master/ikea_asm_dataset_public/`
**Paper**: Ben-Shabat et al., WACV 2021 (arXiv:2007.00394)
**Website**: https://ikeaasm.github.io/

### Scale

| Dimension | Count |
|-----------|-------|
| Total frames (all views) | 4,055,984 JPG |
| dev3 (top view, used for segmentation) | 1,013,700 |
| dev1 (front view) | 1,015,132 |
| dev2 (side view) | 1,015,062 |
| Total recording sessions | 371 |
| Train sessions | 254 (68%) |
| Test sessions | 117 (32%) |

### Furniture Categories (4)

| Category | Sessions | dev3 Frames | Notes |
|----------|----------|-------------|-------|
| Lack_TV_Bench | 91 | 277,250 | TV bench/table |
| Lack_Coffee_Table | 95 | 321,045 | Coffee table |
| Lack_Side_Table | 95 | 250,836 | Side table |
| Kallax_Shelf_Drawer | 90 | 164,569 | Shelf with drawer |

### Instance Segmentation Annotations (COCO Format)

| Split | Images | Annotations | Sources |
|-------|--------|-------------|---------|
| Train | 7,710 | 41,216 | 1% manual + 99% pseudo-GT (Mask RCNN) |
| Test | 3,635 | 19,220 | 1% manual + 99% pseudo-GT |

**7 part categories**: table_top, leg, shelf, side_panel, front_panel, bottom_panel, rear_panel

| Part | Train anns | Test anns | Description |
|------|------------|-----------|-------------|
| leg | 23,013 | 10,379 | Most common (4 per table) |
| table_top | 6,082 | 2,788 | 1 per assembly |
| shelf | 4,291 | 1,928 | Kallax only |
| side_panel | 3,157 | 1,663 | |
| front_panel | 1,593 | 844 | |
| bottom_panel | 1,562 | 796 | |
| rear_panel | 1,518 | 822 | |

**Annotation format**: Standard COCO (bbox: [x,y,w,h], segmentation: polygon), 1920x1080 images. Bbox statistics: mean size ~124x126 pixels, median area 3,936 px^2. Average 5.3 annotations per image.

### Action Annotations

31 atomic action types (32 including "NA" for no-action), including:
- pick up leg/table_top/shelf/side_panel/front_panel/back_panel/bottom_panel/pin
- lay down leg/table_top/shelf/side_panel/front_panel/back_panel/bottom_panel
- spin leg, tighten leg, rotate table, flip table/top/shelf
- align leg screw with table thread, attach shelf/drawer components
- slide bottom of drawer, insert drawer pin, position drawer
- other (catch-all)

### Available Pretrained Models

Located at `/media/newadmin/master/ikea_asm_dataset_public/trained_models_extracted/`:

**Instance Segmentation** (Detectron2/Mask RCNN):
- `Res50.pth` — ResNet-50 backbone, 7 classes
- `Res101.pth` — ResNet-101 backbone, 7 classes
- `ResNeXt.pth` — ResNeXt-101-32x8d backbone, 7 classes

**Action Recognition** (not directly usable for YOLO, but as reference):
- ResNet18/ResNet50 (frame-based, top1: 27-30%)
- C3D/P3D (clip-based, top1: 46-60%)
- I3D (clip-based, top1: 58-63%)
- HCN/ST-GCN (pose-based, top1: 39-43%)

---

## 3. T3: YOLOv8m Training on IKEA ASM

### 3.1 Dataset Preparation

**Goal**: Convert IKEA ASM COCO-format instance segmentation annotations to YOLO-format detection labels for the 7 part categories.

**Steps:**

1. **Extract bboxes from COCO polygons**: Each COCO annotation already has a `bbox` field in [x, y, w, h] format. Convert to YOLO normalized format: `<class_id> <x_center> <y_center> <width> <height>` (all normalized by image width/height: 1920x1080).

2. **Create train/val split**: Use the existing COCO train/test splits (7710/3635) directly.

3. **Image organization**: Copy or symlink images from:
   - `ANU_ikea_dataset_frames/{furniture}/{session}/dev3/images/` to YOLO-style `images/train/` and `images/val/`

4. **Filter multi-label images**: Since IKEA has multiple parts per image (avg 5.3), all parts should be retained as separate objects in YOLO format.

5. **data.yaml structure**:
   ```yaml
   path: /media/newadmin/master/ikea_asm_dataset_public/yolo_dataset/
   train: images/train
   val: images/val
   nc: 7
   names:
     0: table_top
     1: leg
     2: shelf
     3: side_panel
     4: front_panel
     5: bottom_panel
     6: rear_panel
   ```

**Estimated dataset size**: 7710 train / 3635 val images, ~535 MB total (JPG at 1920x1080, ~50 KB each). This is manageable for a single GPU.

### 3.2 Conversion Script

A Python script (`scripts/convert_ikea_to_yolo.py`) will:
1. Load COCO JSON train/test annotations
2. For each image, create YOLO-format `.txt` label file
3. Symlink or copy image files to YOLO directory structure
4. Verify all labels are valid (bbox within [0,1] range, non-negative dimensions)

The script is estimated at ~80 lines and runs in <5 minutes.

### 3.3 Training Configuration

**Model**: YOLOv8m (pretrained on COCO), using the same weights file at `weights/yolov8m.pt`

**Hyperparameters**:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| epochs | 50 | Moderate for 7-class detection with ~7.7k images |
| batch | 8 | Matches D1-R (GPU 0: 12GB VRAM, GPU 1: 16GB) |
| imgsz | 640 | Standard YOLOv8m, matches D1-R |
| optimizer | AdamW (lr0=0.001) | Better convergence than SGD for small-medium datasets |
| weight_decay | 0.0005 | Standard |
| warmup_epochs | 3 | Standard |
| mosaic | 1.0 | Effective for small-object detection (parts are ~124x126 px on 1920x1080) |
| mixup | 0.1 | Mild mixup for regularization |
| patience | 15 | Early stopping |
| augment | randaugment | Beneficial for small-scale dataset |
| label_smoothing | 0.0 | Not needed for 7 well-defined classes |

**Device**: GPU 1 (RTX 5060 Ti, 16GB, CC 12.0) — currently idle after D1-R completes. If GPU 0 (RTX 3060) is free, also viable (12GB, batch=8 worked in D1-R).

### 3.4 Expected Training Time

| Component | Time Estimate |
|-----------|---------------|
| Dataset conversion | ~5 minutes |
| YOLOv8m training (50 epochs, batch=8, 7710 images) | ~7-8 hours on RTX 5060 Ti |
| YOLOv8m training (50 epochs, batch=8, 7710 images) | ~10-12 hours on RTX 3060 |
| Total wall time | ~0.5 GPU-days on RTX 5060 Ti |

**Calculation**: 7710 images / 8 batch = 964 steps/epoch. At ~0.25s/step (RTX 5060 Ti): 241s/epoch. 50 epochs = 12,050s = 3.4h. Add val (3635 images / 8 = 454 steps at ~0.15s = 68s) per epoch. 50 * (241 + 68) ≈ 4.3h. With I/O overhead: ~6h.

### 3.5 Expected Performance

**Expected validation mAP50** (7 part categories):
- table_top: 0.95+ (large, distinct shape)
- leg: 0.90+ (many instances, consistent shape)
- shelf: 0.85+ (moderate variation)
- side_panel, front_panel, bottom_panel, rear_panel: 0.80-0.90 (thin panels, harder)
- Overall mAP50: ~0.88-0.92

The instance segmentation baseline (Mask RCNN ResNeXt, COCO-pretrained) achieved strong results on the 1% manually annotated set. YOLOv8m with the full pseudo-GT set (99% of 7710 images) should match or exceed this.

---

## 4. T3: Cross-Dataset Evaluation Protocol

### 4.1 Evaluation Tasks

Three evaluations to produce the T3 comparison table:

**Task A: POPW-on-IKEA (zero-shot)**
- Take the D1-R YOLOv8m best.pt (trained on 24 IndustReal assembly states)
- Run inference on IKEA ASM dev3 test set (3635 images)
- Map 24 IndustReal classes to 7 IKEA part categories (requires class remapping)
- Report: mAP50, mAP50-95, per-class AP

**Challenge**: The 24 IndustReal states describe assembly progress (e.g., "10010110000" = binary assembly flags), not object parts. Direct class mapping is not meaningful. Instead, we can group the states by which parts are present: if both left and right legs are assembled (bit pattern includes 10010010000 or similar), remap to "leg" class. This requires a semantic mapping table.

**Alternative**: Train YOLOv8m from scratch on IKEA ASM (Task B) and use POPW-on-IKEA as an ablation showing the domain gap between egocentric IndustReal and third-person IKEA ASM.

**Task B: IKEA-only YOLOv8m (our baseline)**
- Train YOLOv8m on converted IKEA ASM dataset (Section 3)
- Evaluate on IKEA ASM test set (3635 images)
- This serves as our method's IKEA performance
- Report: mAP50, mAP50-95, per-class AP, FPS

**Task C: WACV 2021 baselines (literature comparison)**
- Mask RCNN with ResNet-50/101/ResNeXt backbones
- Results from the IKEA ASM paper (using Detectron2, trained on same data)
- The Detectron2 models at `trained_models_extracted/instance_segmentation/` can be evaluated on the test set for direct comparison
- Report: mask AP (their metric) vs bbox AP (our metric) — note the metric difference

### 4.2 Evaluation Metrics

| Metric | Applicable To | Notes |
|--------|---------------|-------|
| mAP50 | Tasks A, B, C | Primary metric for YOLO |
| mAP50-95 | Tasks A, B, C | Strict metric |
| Per-class AP | Tasks A, B, C | Identify weak categories |
| FPS (GPU) | Task B | Practical deployment metric |

### 4.3 Comparison Table Template

| Method | Dataset | Backbone | Eval Data | mAP50 | mAP50-95 | Notes |
|--------|---------|----------|-----------|-------|----------|-------|
| Mask RCNN | IKEA ASM | ResNet-50 | IKEA test | TBD | TBD | WACV baseline |
| Mask RCNN | IKEA ASM | ResNet-101 | IKEA test | TBD | TBD | WACV baseline |
| Mask RCNN | IKEA ASM | ResNeXt-101 | IKEA test | TBD | TBD | WACV baseline |
| YOLOv8m (scratch) | IKEA ASM | YOLOv8m | IKEA test | TBD | TBD | Our T3 |
| YOLOv8m (POPW) | IndustReal | YOLOv8m | IKEA test | TBD | TBD | Zero-shot |

---

## 5. Execution Plan

### Phase 1: D1-R Completion (current, ~3.5h remaining)
- [ ] Wait for D1-R YOLOv8m to reach 25 epochs
- [ ] Record final mAP50 and mAP50-95
- [ ] Run `model.val()` on proper val split for confirmation
- [ ] Save best.pt path: `runs/detect/src/runs/yolov8m_industreal/d1r_proper/weights/best.pt`

### Phase 2: IKEA ASM Dataset Conversion (~1h)
- [ ] Write `scripts/convert_ikea_to_yolo.py`
- [ ] Convert COCO annotations to YOLO format (7 classes)
- [ ] Organize images into train/val directories
- [ ] Create data.yaml
- [ ] Verify labels (check bbox bounds, count per class)
- [ ] Run validation: load YOLO format, verify parseability

### Phase 3: IKEA-only YOLOv8m Training (~6-8h on GPU 1)
- [ ] Launch YOLOv8m training on IKEA ASM (50 epochs, batch=8)
- [ ] Monitor training (mAP50 trajectory, loss curves)
- [ ] Save best.pt
- [ ] Run model.val() on IKEA test set
- [ ] Record final metrics: mAP50, mAP50-95, per-class AP

### Phase 4: POPW-on-IKEA Zero-shot Evaluation (~30min)
- [ ] Load D1-R best.pt
- [ ] Create class remapping: 24 IndustReal states -> 7 IKEA part categories
- [ ] Run model.val() with remapped classes on IKEA test set
- [ ] Record metrics

### Phase 5: WACV Baseline Comparison (~2h)
- [ ] Run Detectron2 evaluation with `trained_models_extracted/instance_segmentation/` checkpoints on IKEA test set
- [ ] Convert mask AP to bbox AP for fair comparison
- [ ] Populate comparison table

### Phase 6: Documentation (~1h)
- [ ] Write T3 results report in `analyses/consult_2026_06_10/`
- [ ] Include comparison table, per-class breakdown, training curves
- [ ] Document findings for the POPW paper

### Total Timeline

| Phase | Duration | Dependencies |
|-------|----------|--------------|
| Phase 1 | ~3.5h (waiting) | None |
| Phase 2 | ~1h | Phase 1 (GPU 0 frees up) |
| Phase 3 | ~6-8h | Phase 2 |
| Phase 4 | ~30min | Phase 1 (D1-R model) |
| Phase 5 | ~2h | Phase 2 |
| Phase 6 | ~1h | Phases 3-5 |
| **Total** | **~14-16h wall time** | Sequential with overlap possible |

### GPU Allocation Strategy

| GPU | Phase 1 | Phase 2 | Phase 3 | Phase 4 | Phase 5 |
|-----|---------|---------|---------|---------|---------|
| GPU 0 (RTX 3060) | D1-R training | Idle | Benchmarks | POPW evaluation | Idle |
| GPU 1 (RTX 5060 Ti) | Idle | Dataset prep | IKEA training | IKEA evaluation | WACV eval |

Phase 3 (IKEA training) on GPU 1 can overlap with Phase 4/5 benchmark evaluations on GPU 0, saving ~3h of wall time.

---

## 6. Key Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| 24-to-7 class remapping ambiguous | High | Medium | Define explicit binary-state-to-part mapping; consider evaluating at furniture level (4 classes) instead |
| Pseudo-GT labels have systematic errors | Medium | Medium | Spot-check 100 random train images; compare manual vs pseudo-GT bbox quality |
| YOLOv8m struggles with small parts | Medium | Medium | Use imgsz=640 vs 1920; consider imgsz=1280 for small parts; evaluate per-class separately |
| GPU 0 training fails mid-way | Low | High | Check nvidia-smi periodically; auto-resume set up? (YOLO saves last.pt every epoch) |
| GPU memory insufficient for batch=8 on IKEA images | Low | Low | IKEA images are 1920x1080; YOLO resizes to 640x640 for training; batch=8 uses ~8-10GB. Reduce to batch=4 if needed |
| Detectron2 environment not set up | Medium | Low | WACV comparison is optional; can skip if environment setup takes >2h |

---

## 7. Appendix: Proposed Class Remapping (24 IndustReal -> 7 IKEA Parts)

The 24 IndustReal classes represent binary-coded assembly state vectors. Each bit position corresponds to a subtask completion. By reading the bit pattern, we can infer which parts are present. Example:

| Bit pattern | Meaning | Remaps To |
|-------------|---------|-----------|
| 10000000000 | Bit 0: left_leg_attached | leg |
| 10010010000 | Bits 0,3,6: three legs attached | leg |
| 10010111111 | All 4 legs + table_top + shelf fully attached | table_top, leg, shelf |
| 11000000000 | Bit 0 + Bit 9: leg + side panel | leg, side_panel |

A full 24-row remapping table will be created during Phase 4 based on the actual bit-to-component mapping defined in the POPW project's `config.py` or dataset definition files.
