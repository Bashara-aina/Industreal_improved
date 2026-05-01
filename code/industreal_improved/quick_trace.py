import torch, numpy as np, sys
sys.path.insert(0, '.')
from model import POPWMultiTaskModel, EMA
from losses import MultiTaskLoss
from industreal_dataset import IndustRealMultiTaskDataset, collate_fn
import config as cfg
from torch.utils.data import DataLoader
from evaluate import _prepare_images, decode_boxes, nms_numpy

device = torch.device('cuda')
model = POPWMultiTaskModel(pretrained=False, backbone_type='convnext_tiny', use_headpose_film=False, use_videomae=False).to(device)
ckpt = torch.load('runs/pretrain_synthetic/checkpoints/latest.pth', map_location='cpu', weights_only=False)
from train import _load_model_compat
_load_model_compat(model, ckpt['model'])
ema = EMA(model, decay=0.999)
ema.get_ema()
model.eval()
model._evaluate_all_active = True

criterion = MultiTaskLoss(num_classes_act=cfg.NUM_CLASSES_ACT, num_psr_components=cfg.NUM_PSR_COMPONENTS,
    train_det=True, train_pose=False, train_act=True, train_psr=True, use_kendall=True).to(device)

val_ds = IndustRealMultiTaskDataset(split='val', img_size=cfg.IMG_SIZE, augment=False, seed=42)
val_loader = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=0, collate_fn=collate_fn)

dg_boxes, dg_labels = [], []
dp_boxes, dp_scores, dp_labels = [], [], []
_anchors_np = None

losses, lc_count = [], 0
act_preds, act_labels = [], []
head_pose_preds, head_pose_gts = [], []
psr_logits_list, psr_labels_list = [], []
act_logits_all = []

for bi, (images, targets) in enumerate(val_loader):
    if bi >= 5:
        break

    images = _prepare_images(images, device)
    det = targets['detection']
    for i in range(len(det)):
        det[i]['boxes'] = det[i]['boxes'].to(device)
        det[i]['labels'] = det[i]['labels'].to(device)
    targets['head_pose'] = targets['head_pose'].to(device)
    targets['psr_labels'] = targets['psr_labels'].to(device)
    targets['activity'] = targets['activity'].to(device)
    clip_rgb = targets.get('clip_rgb')
    if clip_rgb is not None:
        clip_rgb = clip_rgb.to(device)

    B = images.shape[0]

    def run_model(inp, clip=None):
        inp = inp.to(device)
        if inp.dtype == torch.uint8:
            inp = inp.float().div_(255.0)
            mean = torch.tensor(cfg.IMAGENET_MEAN, device=device, dtype=torch.float32).view(1, 3, 1, 1)
            std = torch.tensor(cfg.IMAGENET_STD, device=device, dtype=torch.float32).view(1, 3, 1, 1)
            inp = (inp - mean) / std
        elif inp.dtype != torch.float32:
            inp = inp.float()
        out = model(inp, clip_rgb=clip)
        for _k in out:
            if isinstance(out[_k], torch.Tensor):
                out[_k] = out[_k].float()
        return out

    outputs = run_model(images, clip_rgb)

    loss, _ = criterion(outputs, targets)
    losses.append(float('nan') if not torch.isfinite(loss) else loss.item())
    if torch.isfinite(loss):
        lc_count += 1

    act_logits_batch = outputs['act_logits'].detach().detach().cpu().numpy()
    if act_logits_all is not None:
        act_logits_all.append(act_logits_batch)
    act_preds.append(act_logits_batch.argmax(axis=1))
    act_labels.append(targets['activity'].detach().cpu().numpy())

    head_pose_preds.append(outputs['head_pose'].detach().cpu().numpy())
    head_pose_gts.append(targets['head_pose'].detach().cpu().numpy())

    psr_logits_list.append(outputs['psr_logits'].detach().cpu().numpy())
    psr_labels_list.append(targets['psr_labels'].detach().cpu().numpy())

    if _anchors_np is None:
        _anchors_np = outputs['anchors'].detach().cpu().numpy()

    cls_sigmoid = torch.sigmoid(outputs['cls_preds'])

    for i in range(B):
        scores_i = cls_sigmoid[i]
        max_scores = scores_i.max(dim=1).values
        score_thresh = float(getattr(cfg, 'DET_EVAL_SCORE_THRESH', 0.5))
        keep_mask = max_scores > score_thresh

        max_keep = int(getattr(cfg, 'DET_EVAL_MAX_PER_IMAGE', 300))
        if max_keep > 0 and keep_mask.sum().item() > max_keep:
            topk_idx = torch.topk(max_scores, k=max_keep, largest=True, sorted=False).indices
            topk_mask = torch.zeros_like(keep_mask)
            topk_mask[topk_idx] = True
            keep_mask = keep_mask & topk_mask

        if keep_mask.sum().item() == 0:
            dp_boxes.append(np.zeros((0, 4)))
            dp_scores.append(np.zeros(0))
            dp_labels.append(np.zeros(0, dtype=np.int64))
        else:
            keep_np = keep_mask.detach().cpu().numpy()
            kept_cls = scores_i[keep_mask].detach().cpu().numpy()
            kept_reg = outputs['reg_preds'][i][keep_mask].detach().cpu().numpy()
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
                dp_boxes.append(np.zeros((0, 4)))
                dp_scores.append(np.zeros(0))
                dp_labels.append(np.zeros(0, dtype=np.int64))

        dg_boxes.append(det[i]['boxes'].detach().cpu().numpy())
        dg_labels.append(det[i]['labels'].detach().cpu().numpy())

    del images, outputs, cls_sigmoid
    print(f'Batch {bi}: loss={losses[-1]:.4f} lc={lc_count}/{(bi+1)*B} GT={sum(len(x) for x in dg_boxes)} DP={sum(len(x) for x in dp_boxes)}', flush=True)

print(f'\nFinal after {bi+1} batches:')
print(f'  Loss finite: {sum(1 for l in losses if not np.isnan(l))}/{len(losses)}')
print(f'  GT boxes: {sum(len(x) for x in dg_boxes)}')
print(f'  DP boxes: {sum(len(x) for x in dp_boxes)}')
print(f'  dg_labels sample: {[x.tolist() for x in dg_labels[:5]]}')
print(f'  dp_labels sample: {[x.tolist() for x in dp_labels[:5]]}')
print(f'  act_preds: {len(act_preds)}, head_pose: {len(head_pose_preds)}, psr: {len(psr_logits_list)}')