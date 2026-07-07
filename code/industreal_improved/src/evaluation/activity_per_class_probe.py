"""
Per-class linear probe accuracy + per-recording majority baseline.
Simple direct-iteration approach to avoid DataLoader/SHM issues.
"""
import json
import logging
import os
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src import config as C
from src.data.industreal_dataset import IndustRealMultiTaskDataset
from src.models.model import ConvNeXtBackbone

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger('per_class_probe')


def main():
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f'Device: {device}')

        project_root = Path(__file__).resolve().parent.parent.parent
        checkpoint_dir = project_root / 'src' / 'runs' / 'rf_stages' / 'checkpoints'
        checkpoint_path = checkpoint_dir / 'best.pth'
        output_json = checkpoint_dir / 'per_class_probe.json'

        # Load backbone
        logger.info(f'Loading checkpoint from {checkpoint_path}')
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        model_state = checkpoint['model']

        backbone_state = {k[len('backbone.'):]: v for k, v in model_state.items() if k.startswith('backbone.')}
        backbone = ConvNeXtBackbone(pretrained=False)
        backbone.load_state_dict(backbone_state, strict=False)
        backbone = backbone.to(device).eval()
        for p in backbone.parameters():
            p.requires_grad = False
        logger.info(f'Backbone loaded ({sum(p.numel() for p in backbone.parameters()):,} params, frozen)')

        num_classes = 69
        backbone_dim = 768

        # Load datasets without multiprocessing
        C.RAM_CACHE_MAX_IMAGES = 200
        train_ds = IndustRealMultiTaskDataset(split='train', augment=False, subset_ratio=1.0)
        val_ds = IndustRealMultiTaskDataset(split='val', augment=False, subset_ratio=1.0)
        logger.info(f'Train: {len(train_ds)}, Val: {len(val_ds)}')

        # Normalize function
        def normalize(img):
            if img.dtype == torch.uint8:
                img = img.float().div_(255.0)
                mean = torch.tensor(C.IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
                std = torch.tensor(C.IMAGENET_STD, device=device).view(1, 3, 1, 1)
                img = (img - mean) / std
            return img

        def extract_all(ds, desc):
            """Simple direct iteration without DataLoader."""
            all_feats, all_labels, all_rec_ids = [], [], []
            skipped = 0
            N = len(ds)
            for i in range(N):
                sample = ds[i]
                label = sample['activity']
                if label.item() < 0:
                    skipped += 1
                    continue
                img = sample['images']['rgb'].unsqueeze(0).to(device)  # [1, 3, H, W]
                img = normalize(img)
                with torch.no_grad():
                    _, _, _, c5 = backbone(img)
                    feat = F.adaptive_avg_pool2d(c5, 1).flatten(1).cpu()  # [1, 768]
                all_feats.append(feat)
                all_labels.append(label.cpu())
                all_rec_ids.append(sample['metadata']['recording_id'])
                if (i + 1) % 1000 == 0:
                    logger.info(f'  {desc} {i+1}/{N} ({skipped} skipped)')
                if (i + 1) % 5000 == 0:
                    torch.cuda.empty_cache()
            if not all_feats:
                return torch.empty(0, 768), torch.empty(0, dtype=torch.long), [], skipped
            return torch.cat(all_feats, dim=0), torch.tensor(all_labels), all_rec_ids, skipped

        # Extract features
        logger.info('Extracting train features (direct iteration)...')
        t0 = time.time()
        train_feats, train_labels, train_rec_ids, train_skip = extract_all(train_ds, 'Train')
        logger.info(f'Train features: {train_feats.shape} ({train_skip} skipped) in {time.time()-t0:.0f}s')

        logger.info('Extracting val features (direct iteration)...')
        t0 = time.time()
        val_feats, val_labels, val_rec_ids, val_skip = extract_all(val_ds, 'Val')
        logger.info(f'Val features: {val_feats.shape} ({val_skip} skipped) in {time.time()-t0:.0f}s')

        # Per-recording majority baseline on VAL
        rec_groups = defaultdict(list)
        for rec_id, lbl in zip(val_rec_ids, val_labels.tolist()):
            rec_groups[rec_id].append(lbl)
        rec_correct = 0
        rec_total = 0
        rec_majorities = {}
        for rec_id, labels in rec_groups.items():
            counts = np.bincount(labels)
            majority = int(counts.argmax())
            rec_majorities[rec_id] = majority
            rec_correct += int((np.array(labels) == majority).sum())
            rec_total += len(labels)
        per_rec_baseline = rec_correct / rec_total
        logger.info(f'Per-recording majority baseline: {per_rec_baseline:.4f} ({rec_correct}/{rec_total})')

        # Global majority baseline
        valid_labels = val_labels.numpy()
        global_counts = np.bincount(valid_labels)
        global_majority_class = int(global_counts.argmax())
        global_majority_baseline = global_counts[global_majority_class] / len(valid_labels)
        logger.info(f'Global majority baseline: {global_majority_baseline:.4f} (class {global_majority_class})')

        # Train linear probe
        classifier = nn.Linear(backbone_dim, num_classes).to(device)
        optim = torch.optim.AdamW(classifier.parameters(), lr=1e-3, weight_decay=1e-4)
        bs = 256
        best_val_acc = 0.0
        best_state = None

        for epoch in range(5):
            classifier.train()
            perm = torch.randperm(len(train_feats))
            for i in range(0, len(train_feats), bs):
                idx = perm[i:i+bs]
                x = train_feats[idx].to(device)
                y = train_labels[idx].to(device)
                logits = classifier(x)
                loss = F.cross_entropy(logits, y)
                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)
                optim.step()

            # Val
            classifier.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for i in range(0, len(val_feats), bs):
                    x = val_feats[i:i+bs].to(device)
                    y = val_labels[i:i+bs].to(device)
                    logits = classifier(x)
                    preds = logits.argmax(-1)
                    correct += (preds == y).sum().item()
                    total += y.size(0)
            val_acc = correct / total
            logger.info(f'  Epoch {epoch}: val_acc={val_acc:.4f}')
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = classifier.state_dict()

        # Per-class accuracy with best model
        classifier.load_state_dict(best_state)
        classifier.eval()
        all_preds = []
        with torch.no_grad():
            for i in range(0, len(val_feats), bs):
                x = val_feats[i:i+bs].to(device)
                logits = classifier(x)
                preds = logits.argmax(-1).cpu().numpy()
                all_preds.extend(preds.tolist())
        all_preds = np.array(all_preds)
        all_labels_np = val_labels.numpy()

        per_class_correct = defaultdict(int)
        per_class_total = defaultdict(int)
        for lbl, pred in zip(all_labels_np, all_preds):
            per_class_total[int(lbl)] += 1
            if lbl == pred:
                per_class_correct[int(lbl)] += 1

        classes_with_signal = 0
        classes_with_zero = 0
        per_class_acc = {}
        for c in range(num_classes):
            total_c = per_class_total.get(c, 0)
            if total_c > 0:
                acc = per_class_correct.get(c, 0) / total_c
                per_class_acc[str(c)] = {
                    'accuracy': round(acc, 4),
                    'correct': per_class_correct.get(c, 0),
                    'total': total_c,
                    'has_signal': acc > 1.0 / num_classes
                }
                if acc == 0.0:
                    classes_with_zero += 1
                if acc > 1.0 / num_classes:
                    classes_with_signal += 1
            else:
                per_class_acc[str(c)] = {'accuracy': 0.0, 'correct': 0, 'total': 0, 'has_signal': False}

        # Per-recording accuracy
        rec_pred_groups = defaultdict(lambda: {'correct': 0, 'total': 0})
        for i, (rec_id, lbl, pred) in enumerate(zip(val_rec_ids, all_labels_np, all_preds)):
            rec_pred_groups[rec_id]['total'] += 1
            if lbl == pred:
                rec_pred_groups[rec_id]['correct'] += 1
        per_rec_accuracy = {rec: round(d['correct'] / d['total'], 4) for rec, d in rec_pred_groups.items()}

        logger.info(f'\n{"="*60}')
        logger.info(f'PER-CLASS PROBE RESULTS')
        logger.info(f'{"="*60}')
        logger.info(f'Global majority baseline: {global_majority_baseline:.4f}')
        logger.info(f'Per-recording majority baseline: {per_rec_baseline:.4f}')
        logger.info(f'Best val per-frame top-1: {best_val_acc:.4f}')
        logger.info(f'Classes with signal (acc > 1/{num_classes}): {classes_with_signal}/{num_classes}')
        logger.info(f'Classes with zero accuracy: {classes_with_zero}/{num_classes}')
        logger.info(f'Mean per-class acc: {np.mean([v["accuracy"] for v in per_class_acc.values()]):.4f}')

        results = {
            'global_majority_baseline': global_majority_baseline,
            'global_majority_class': global_majority_class,
            'per_recording_majority_baseline': per_rec_baseline,
            'per_recording_majority_correct': rec_correct,
            'per_recording_majority_total': rec_total,
            'best_val_per_frame_top1': best_val_acc,
            'num_classes': num_classes,
            'classes_with_signal': classes_with_signal,
            'classes_with_zero_accuracy': classes_with_zero,
            'mean_per_class_accuracy': round(float(np.mean([v['accuracy'] for v in per_class_acc.values()])), 4),
            'per_class': per_class_acc,
            'per_recording_accuracy': per_rec_accuracy,
            'per_recording_majority_class': {k: int(v) for k, v in rec_majorities.items()},
        }

        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f'Saved {output_json}')

    except Exception:
        logger.error(f"Fatal: {traceback.format_exc()}")
        sys.exit(1)


if __name__ == '__main__':
    main()
