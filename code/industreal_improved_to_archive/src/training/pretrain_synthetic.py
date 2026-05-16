"""
Pretrain Detection Head on IndustReal Synthetic Data
==================================================
Doc 01 B.1: Synthetic pretraining for the detection backbone.

This script pretrains ONLY the backbone + FPN + detection head
for 20 epochs on synthetic/real detection data before the full
multi-task training begins.

Usage:
    python pretrain_synthetic.py
    python pretrain_synthetic.py --epochs 20 --lr 5e-4
    python pretrain_synthetic.py --resume runs/pretrain_synthetic/checkpoints/latest.pth

Architecture trained:
    - ResNet-50 or ConvNeXt-Tiny backbone (pretrained on ImageNet)
    - FPN neck (P3-P7, 256ch)
    - RetinaNet detection head (24 ASD classes)

NOT trained (frozen / not initialized):
    - Pose head
    - PoseFiLM / HeadPoseFiLM
    - Head pose head
    - Activity head
    - PSR head
    - Feature bank

After pretraining, load the checkpoint with:
    checkpoint = torch.load('runs/pretrain_synthetic/checkpoints/best.pth')
    model.load_state_dict(checkpoint['model'], strict=False)

Author: Bashara
Date: April 2026
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.cuda.amp as amp
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

import config as C
import losses as _losses_module
import model as _model_module

POPWMultiTaskModel = getattr(_model_module, 'POPWMultiTaskModel')
build_backbone = getattr(_model_module, 'build_backbone')
count_parameters = getattr(_model_module, 'count_parameters')
FocalLoss = getattr(_losses_module, 'FocalLoss')

logger = logging.getLogger(__name__)

IMG_HEIGHT = 360   # Reduced from 720 — 3× faster CPU processing, no impact on backbone learning
IMG_WIDTH = 640    # Reduced from 1280
NUM_DET_CLASSES = 24
PRETRAIN_BATCH_SIZE = 12  # Sweet spot: ~5GB VRAM, fits alongside other GPU processes
PRETRAIN_NUM_WORKERS = 12  # Parallel prefetch to saturate GPU at 640x360


class DetectionPretrainDataset(torch.utils.data.Dataset):
    """
    Detection-only dataset for pretraining.
    Loads only RGB frames and OD_labels.json (COCO format).
    Ignores AR, PSR, pose labels.
    """
    def __init__(self, split: str = 'train', img_size: tuple = (IMG_WIDTH, IMG_HEIGHT),
                 augment: bool = True, max_recordings: Optional[int] = None,
                 frame_stride: Optional[int] = None):
        from pathlib import Path
        self.recordings_root = Path(C.POPW_ROOT) / 'recordings'
        self.split = split
        self.img_size = img_size
        self.augment = augment

        if frame_stride is None:
            if split == 'train' and not C.DEBUG_MODE:
                frame_stride = int(getattr(C, 'PRETRAIN_DET_FRAME_STRIDE', C.TRAIN_FRAME_STRIDE))
            else:
                frame_stride = 1
        self.frame_stride = frame_stride

        self.samples: List[Dict] = []

        split_csv = getattr(C, f'{split.upper()}_CSV', None)
        if split_csv and Path(split_csv).exists():
            with open(split_csv) as f:
                recording_ids = [line.strip().split(',')[0] for line in f if line.strip()]
        else:
            recording_ids = sorted([
                d.name for d in (self.recordings_root / split).iterdir()
                if d.is_dir()
            ])

        if max_recordings:
            recording_ids = recording_ids[:max_recordings]

        for rec_id in recording_ids:
            rec_path = self.recordings_root / split / rec_id
            rgb_dir = rec_path / 'rgb'
            od_file = rec_path / 'OD_labels.json'

            if not rgb_dir.exists() or not od_file.exists():
                continue

            with open(od_file) as f:
                od_data = json.load(f)

            images = sorted(rgb_dir.glob('*.jpg'))
            if not images:
                continue

            for img_path in images:
                frame_idx = int(img_path.stem)
                if frame_idx % self.frame_stride != 0:
                    continue
                anns_for_frame = [
                    ann for ann in od_data.get('annotations', [])
                    if ann.get('image_id') == frame_idx
                ]

                if not anns_for_frame and not self.augment:
                    continue

                self.samples.append({
                    'rec_id': rec_id,
                    'frame_idx': frame_idx,
                    'rgb_path': str(img_path),
                    'annotations': anns_for_frame,
                })

        logger.info(f'[{split}] Detection dataset: {len(self.samples)} frames from {len(recording_ids)} recordings')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        import cv2
        sample = self.samples[idx]

        img = cv2.imread(sample['rgb_path'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.img_size)
        anns = [ann.copy() for ann in sample['annotations']]

        if self.augment:
            mosaic_prob = float(getattr(C, 'PRETRAIN_MOSAIC_PROB', 0.3))
            mixup_prob = float(getattr(C, 'PRETRAIN_MIXUP_PROB', 0.2))
            hflip_prob = float(getattr(C, 'PRETRAIN_HFLIP_PROB', 0.5))

            if random.random() < mosaic_prob:
                mosaic_size = (self.img_size[0] * 2, self.img_size[1] * 2)
                mosaic_img = np.zeros((mosaic_size[1], mosaic_size[0], 3), dtype=np.uint8)
                mosaic_anns = []

                quad_samples = [sample] + [
                    self.samples[random.randint(0, len(self.samples) - 1)]
                    for _ in range(3)
                ]
                half_w = self.img_size[0]
                half_h = self.img_size[1]
                offsets = [(0, 0), (half_w, 0), (0, half_h), (half_w, half_h)]

                for qs, (ox, oy) in zip(quad_samples, offsets):
                    q_img = cv2.imread(qs['rgb_path'])
                    q_img = cv2.cvtColor(q_img, cv2.COLOR_BGR2RGB)
                    q_img = cv2.resize(q_img, self.img_size)
                    mosaic_img[oy:oy + half_h, ox:ox + half_w] = q_img
                    for ann in qs['annotations']:
                        x1, y1, w, h = ann['bbox']
                        x2, y2 = x1 + w, y1 + h
                        remapped = [x1 + ox, y1 + oy, x2 + ox, y2 + oy]
                        mosaic_anns.append({'bbox': remapped, 'category_id': ann['category_id']})

                img = cv2.resize(mosaic_img, self.img_size)
                anns = mosaic_anns
                scale_x = self.img_size[0] / mosaic_size[0]
                scale_y = self.img_size[1] / mosaic_size[1]
                for ann in anns:
                    ann['bbox'][0] *= scale_x
                    ann['bbox'][1] *= scale_y
                    ann['bbox'][2] *= scale_x
                    ann['bbox'][3] *= scale_y

            if random.random() < mixup_prob:
                mix_idx = random.randint(0, len(self.samples) - 1)
                mix_sample = self.samples[mix_idx]
                mix_img = cv2.imread(mix_sample['rgb_path'])
                mix_img = cv2.cvtColor(mix_img, cv2.COLOR_BGR2RGB)
                mix_img = cv2.resize(mix_img, self.img_size)
                mix_anns = [ann.copy() for ann in mix_sample['annotations']]
                alpha = np.random.beta(0.5, 0.5)
                img = (img * alpha + mix_img * (1 - alpha)).astype(np.uint8)
                anns = anns + mix_anns

            if random.random() < hflip_prob:
                img = np.fliplr(img)
                for ann in anns:
                    x1, y1, x2, y2 = ann['bbox']
                    ann['bbox'] = [self.img_size[0] - x2, y1, self.img_size[0] - x1, y2]

        img_tensor = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor(C.IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(C.IMAGENET_STD).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std

        boxes, labels = [], []
        for ann in anns:
            x1, y1, x2, y2 = ann['bbox']
            x1 = max(0.0, min(x1, self.img_size[0] - 1))
            y1 = max(0.0, min(y1, self.img_size[1] - 1))
            x2 = max(x1 + 1.0, min(x2, self.img_size[0]))
            y2 = max(y1 + 1.0, min(y2, self.img_size[1]))
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])
                labels.append(ann['category_id'])

        if boxes:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.long)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.long)

        return img_tensor, {'boxes': boxes, 'labels': labels, 'image_id': sample['frame_idx']}


def detection_collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    targets = [item[1] for item in batch]
    return images, targets


def build_detection_model(backbone_type: str = 'resnet50'):
    """Build model with only detection components trainable."""
    model = POPWMultiTaskModel(
        pretrained=True,
        backbone_type=backbone_type,
        use_headpose_film=False,
        use_videomae=False,
        train_pose=False,
    )

    for name, param in model.named_parameters():
        if any(h in name for h in ['backbone', 'fpn', 'detection_head', 'anchor_gen']):
            param.requires_grad = False

    from model import set_backbone_stage_requires_grad
    set_backbone_stage_requires_grad(model, backbone_type, stage=1, requires_grad=True)

    return model


def train_one_epoch_det(
    model,
    loader,
    optimizer,
    scaler,
    device,
    epoch: int,
    accum_steps: int = 8,
):
    """Train detection-only for one epoch."""
    model.train()

    criterion = FocalLoss(alpha=0.25, gamma=2.0)

    running_loss = 0.0
    num_batches = 0
    nan_skips = 0

    pbar = tqdm(loader, desc=f'Pretrain Det Epoch {epoch}', leave=True)

    for step, (images, targets) in enumerate(pbar):
        images = images.to(device)

        for i in range(len(targets)):
            targets[i]['boxes'] = targets[i]['boxes'].to(device)
            targets[i]['labels'] = targets[i]['labels'].to(device)

        with amp.autocast(enabled=C.MIXED_PRECISION):
            outputs = model(images)

        cls_preds = outputs['cls_preds']
        reg_preds = outputs['reg_preds']
        anchors = outputs['anchors']

        loss_cls, loss_reg = criterion(cls_preds, reg_preds, anchors, targets)
        loss = loss_cls + loss_reg

        if not torch.isfinite(loss):
            nan_skips += 1
            optimizer.zero_grad(set_to_none=True)
            continue

        loss = loss / accum_steps
        scaler.scale(loss).backward()

        if (step + 1) % accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), C.GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        running_loss += loss.item() * accum_steps
        num_batches += 1

        pbar.set_postfix({'loss': f'{loss.item() * accum_steps:.3f}'})

    if nan_skips > 0:
        logger.warning(f'Epoch {epoch}: skipped {nan_skips} NaN batches')

    return {'loss': running_loss / max(num_batches, 1), 'nan_skips': nan_skips}


@torch.no_grad()
def evaluate_detection(model, loader, device, max_batches: int = 400):
    """Evaluate detection mAP."""
    from torchmetrics.detection import MeanAveragePrecision

    model.eval()
    metric = MeanAveragePrecision(iou_type='bbox')

    for batch_idx, (images, targets) in enumerate(tqdm(loader, desc='Evaluating')):
        if batch_idx >= max_batches:
            break

        images = images.to(device)
        outputs = model(images)

        preds = []
        for i in range(images.shape[0]):
            cls_pred = outputs['cls_preds'][i].cpu()
            reg_pred = outputs['reg_preds'][i].cpu()
            anchor = outputs['anchors'].cpu()

            scores, labels = cls_pred.softmax(dim=-1).max(dim=-1)
            boxes = decode_boxes(anchor, reg_pred)

            keep = scores > 0.1
            preds.append({
                'boxes': boxes[keep],
                'scores': scores[keep],
                'labels': labels[keep],
            })

        targets_formatted = []
        for i in range(len(targets)):
            targets_formatted.append({
                'boxes': targets[i]['boxes'],
                'labels': targets[i]['labels'],
            })

        metric.update(preds, targets_formatted)

    return metric.compute()


def decode_boxes(anchors, deltas):
    """Decode anchor deltas to xyxy boxes."""
    a_cx = (anchors[:, 0] + anchors[:, 2]) / 2
    a_cy = (anchors[:, 1] + anchors[:, 3]) / 2
    a_w = anchors[:, 2] - anchors[:, 0]
    a_h = anchors[:, 3] - anchors[:, 1]

    dx = deltas[:, 0]
    dy = deltas[:, 1]
    dw = deltas[:, 2].clamp(-4, 4)
    dh = deltas[:, 3].clamp(-4, 4)

    pred_w = torch.exp(dw) * a_w
    pred_h = torch.exp(dh) * a_h
    pred_cx = dx * a_w + a_cx
    pred_cy = dy * a_h + a_cy

    return torch.stack([
        pred_cx - pred_w / 2,
        pred_cy - pred_h / 2,
        pred_cx + pred_w / 2,
        pred_cy + pred_h / 2,
    ], dim=1)


def main(args):
    output_dir = Path(C.OUTPUT_ROOT) / 'pretrain_synthetic'
    ckpt_dir = output_dir / 'checkpoints'
    log_dir = output_dir / 'logs'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'pretrain.log'),
            logging.StreamHandler(),
        ],
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Device: {device}')

    seed_everything(C.SEED)

    logger.info('Building detection-only dataset...')
    _max_rec = getattr(args, 'max_recordings', None)
    train_ds = DetectionPretrainDataset(
        split='train',
        img_size=(IMG_WIDTH, IMG_HEIGHT),
        augment=True,
        max_recordings=_max_rec,
    )
    val_ds = DetectionPretrainDataset(
        split='val',
        img_size=(IMG_WIDTH, IMG_HEIGHT),
        augment=False,
        max_recordings=_max_rec,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=PRETRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=PRETRAIN_NUM_WORKERS,
        collate_fn=detection_collate_fn,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
        prefetch_factor=4,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=PRETRAIN_BATCH_SIZE,
        shuffle=False,
        num_workers=PRETRAIN_NUM_WORKERS,
        collate_fn=detection_collate_fn,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    backbone_type = str(getattr(C, 'BACKBONE', 'resnet50'))
    logger.info(f'Building detection model with {backbone_type} backbone...')
    model = build_detection_model(backbone_type=backbone_type).to(device)

    params = count_parameters(model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'Total params: {params["total_all"]:,}')
    logger.info(f'Trainable params (detection only): {trainable:,}')

    lr = float(getattr(C, 'PRETRAIN_DET_LR', 5e-4))
    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=C.WEIGHT_DECAY,
    )

    epochs = int(getattr(args, 'epochs', None) or int(getattr(C, 'PRETRAIN_DET_EPOCHS', 20)))
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    scaler = amp.GradScaler(enabled=C.MIXED_PRECISION)

    best_map = 0.0
    start_epoch = 0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'], strict=False)
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1
        best_map = float(ckpt.get('best_map', 0.0))
        logger.info(f'Resumed from epoch {start_epoch}, best_mAP={best_map:.4f}')

    logger.info(f'Starting detection pretraining for {epochs} epochs...')

    for epoch in range(start_epoch, epochs):
        logger.info(f'\n--- Epoch {epoch}/{epochs - 1} ---')

        train_metrics = train_one_epoch_det(
            model, train_loader, optimizer, scaler, device, epoch,
            accum_steps=C.GRAD_ACCUM_STEPS,
        )
        scheduler.step()

        logger.info(
            f'Train loss: {train_metrics["loss"]:.4f}  '
            f'lr: {optimizer.param_groups[0]["lr"]:.2e}'
        )

        save_ckpt_path = ckpt_dir / 'latest.pth'
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_map': best_map,
        }, save_ckpt_path)
        logger.info(f'Saved checkpoint: {save_ckpt_path}')

        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            logger.info('Running detection evaluation...')
            map_metrics = evaluate_detection(model, val_loader, device)

            map50 = map_metrics['map_50'].item()
            map5095 = map_metrics['map'].item()

            logger.info(f'mAP@0.5: {map50:.4f}  mAP@[0.5:0.95]: {map5095:.4f}')

            if map50 > best_map:
                best_map = map50
                torch.save({
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'best_map': best_map,
                }, ckpt_dir / 'best.pth')
                logger.info(f'New best mAP@0.5: {best_map:.4f}')

    logger.info(f'\nPretraining complete. Best mAP@0.5: {best_map:.4f}')
    logger.info(f'Checkpoint: {ckpt_dir / "best.pth"}')


def seed_everything(seed: int = C.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    import sys
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--max-recordings', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    args = parser.parse_args()
    sys.argv = [sys.argv[0]]  # reset argv for nested argparse in main
    main(args)