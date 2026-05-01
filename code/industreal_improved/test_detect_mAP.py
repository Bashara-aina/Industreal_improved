import torch, numpy as np, sys, time
sys.path.insert(0, '.')
from model import POPWMultiTaskModel, EMA
from industreal_dataset import IndustRealMultiTaskDataset, collate_fn
import config as cfg
from torch.utils.data import DataLoader
from evaluate import _prepare_images, decode_boxes, nms_numpy, compute_ap_per_class

device = torch.device('cuda')
model = POPWMultiTaskModel(pretrained=False, backbone_type='convnext_tiny', use_headpose_film=False, use_videomae=False).to(device)
ckpt = torch.load('runs/pretrain_synthetic/checkpoints/latest.pth', map_location='cpu', weights_only=False)
from train import _load_model_compat
_load_model_compat(model, ckpt['model'])
ema = EMA(model, decay=0.999); ema.get_ema()
model.eval()

val_ds = IndustRealMultiTaskDataset(split='val', img_size=cfg.IMG_SIZE, augment=False, seed=42)
val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0, collate_fn=collate_fn)

dg_boxes, dg_labels = [], []
dp_boxes, dp_scores, dp_labels = [], [], []
_anchors_np = None

print("Starting detection collection on 100 batches...", flush=True)
t0 = time.time()

for bi, (images, targets) in enumerate(val_loader):
    if bi >= 100:
        break
    if bi % 20 == 0:
        print(f"  Batch {bi}/100...", flush=True)

    images = _prepare_images(images, device)
    detection_list = targets['detection']
    for i in range(len(detection_list)):
        detection_list[i]['boxes'] = detection_list[i]['boxes'].to(device)
        detection_list[i]['labels'] = detection_list[i]['labels'].to(device)

    with torch.no_grad():
        outputs = model(images, clip_rgb=None)

    if _anchors_np is None:
        _anchors_np = outputs['anchors'].cpu().numpy()
    cls_sigmoid = torch.sigmoid(outputs['cls_preds'])

    for i in range(images.shape[0]):
        scores_i = cls_sigmoid[i]
        max_scores = scores_i.max(dim=1).values

        keep_mask = max_scores > 0.005

        if keep_mask.sum().item() == 0:
            dp_boxes.append(np.zeros((0,4)))
            dp_scores.append(np.zeros(0))
            dp_labels.append(np.zeros(0,dtype=np.int64))
        else:
            keep_np = keep_mask.cpu().numpy()
            kept_cls = scores_i[keep_mask].cpu().numpy()
            kept_reg = outputs['reg_preds'][i][keep_mask].cpu().numpy()
            kept_anc = _anchors_np[keep_np]
            ms = kept_cls.max(axis=1)
            ml = kept_cls.argmax(axis=1)
            pb = decode_boxes(kept_anc, kept_reg)
            pb[:, 0] = np.clip(pb[:, 0], 0, cfg.IMG_WIDTH)
            pb[:, 1] = np.clip(pb[:, 1], 0, cfg.IMG_HEIGHT)
            pb[:, 2] = np.clip(pb[:, 2], 0, cfg.IMG_WIDTH)
            pb[:, 3] = np.clip(pb[:, 3], 0, cfg.IMG_HEIGHT)

            fb, fs, fl = [], [], []
            for c in range(cfg.NUM_DET_CLASSES):
                cm = ml == c
                if cm.sum() == 0:
                    continue
                nk = nms_numpy(pb[cm], ms[cm], cfg.DET_EVAL_NMS_IOU_THRESH)
                fb.append(pb[cm][nk])
                fs.append(ms[cm][nk])
                fl.append(np.full(len(nk), c, dtype=np.int64))

            if fb:
                dp_boxes.append(np.concatenate(fb))
                dp_scores.append(np.concatenate(fs))
                dp_labels.append(np.concatenate(fl))
            else:
                dp_boxes.append(np.zeros((0,4)))
                dp_scores.append(np.zeros(0))
                dp_labels.append(np.zeros(0,dtype=np.int64))

        dg_boxes.append(detection_list[i]['boxes'].cpu().numpy())
        dg_labels.append(detection_list[i]['labels'].cpu().numpy())

    del images, outputs, cls_sigmoid

t1 = time.time()
print(f"Done in {t1-t0:.1f}s", flush=True)
print(f"GT boxes: {sum(len(x) for x in dg_boxes)}", flush=True)
print(f"Pred boxes (thresh=0.005): {sum(len(x) for x in dp_boxes)}", flush=True)

# Compute AP at thresh=0.005
gt_boxes_by_cls = {}
for boxes, labels in zip(dg_boxes, dg_labels):
    for box, label in zip(boxes, labels):
        l = int(label)
        if l not in gt_boxes_by_cls:
            gt_boxes_by_cls[l] = []
        gt_boxes_by_cls[l].append(box)

aps = []
for c in sorted(gt_boxes_by_cls.keys()):
    gt_b = np.array(gt_boxes_by_cls[c])
    pm = np.array([l == c for l in dp_labels])
    pb = np.concatenate([dp_boxes[j] for j in range(len(pm)) if pm[j]]) if any(pm) else np.zeros((0,4))
    ps = np.concatenate([dp_scores[j] for j in range(len(pm)) if pm[j]]) if any(pm) else np.zeros(0)
    ap = compute_ap_per_class(pb, ps, gt_b, iou_thresh=0.5) if len(pb) > 0 else 0.0
    aps.append(ap)
    print(f"  Class {c}: {len(gt_b)} GT, {len(pb)} pred, AP={ap:.4f}", flush=True)

if aps:
    print(f"mAP@0.5 (at thresh=0.005): {np.mean(aps):.4f}", flush=True)
else:
    print("No APs computed", flush=True)

# What about at thresh=0.5?
print("\n--- At default thresh=0.5 ---", flush=True)
pred_boxes_05, pred_scores_05, pred_labels_05 = [], [], []
for boxes, scores, labels in zip(dp_boxes, dp_scores, dp_labels):
    mask = scores >= 0.5
    pred_boxes_05.append(boxes[mask] if mask.any() else np.zeros((0,4)))
    pred_scores_05.append(scores[mask] if mask.any() else np.zeros(0))
    pred_labels_05.append(labels[mask] if mask.any() else np.zeros(0,dtype=np.int64))

pred_b_05 = np.concatenate(pred_boxes_05)
pred_s_05 = np.concatenate(pred_scores_05)
pred_l_05 = np.concatenate(pred_labels_05)
print(f"Pred boxes at thresh=0.5: {len(pred_b_05)}", flush=True)

aps_05 = []
for c in sorted(gt_boxes_by_cls.keys()):
    gt_b = np.array(gt_boxes_by_cls[c])
    pm = pred_l_05 == c
    pb = pred_b_05[pm]
    ps = pred_s_05[pm]
    ap = compute_ap_per_class(pb, ps, gt_b, iou_thresh=0.5) if len(pb) > 0 else 0.0
    aps_05.append(ap)
    print(f"  Class {c}: {len(gt_b)} GT, {len(pb)} pred, AP={ap:.4f}", flush=True)

if aps_05:
    print(f"mAP@0.5 (at thresh=0.5): {np.mean(aps_05):.4f}", flush=True)
else:
    print("No APs at thresh=0.5 (expected)", flush=True)