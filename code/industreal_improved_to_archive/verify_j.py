import sys, os, torch, warnings, math, gc
warnings.filterwarnings('ignore')
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
gc.collect()
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

sys.path.insert(0,'.')
from src import config as C
C.DEBUG_MODE = True
C.MIXED_PRECISION = True
C.USE_KENDALL = True
C.USE_VIDEOMAE = False

from src.models.model import POPWMultiTaskModel
from src.training.losses import MultiTaskLoss
from src.data.industreal_dataset import IndustRealMultiTaskDataset, collate_fn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

print('=== J-1: No NaN loss in first 10 batches ===')
try:
    model = POPWMultiTaskModel(pretrained=False, backbone_type='convnext_tiny', use_headpose_film=True, use_videomae=False, train_pose=True).cuda()
    model.train()
    loss_fn = MultiTaskLoss(num_classes_act=C.NUM_ACT_CLASSES, num_psr_components=C.NUM_PSR_COMPONENTS,
        train_det=True, train_pose=False, train_act=True, train_psr=True, use_kendall=True).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = GradScaler('cuda')

    ds = IndustRealMultiTaskDataset(split='train', max_recordings=2, subset_ratio=0.05, sequence_mode=False)
    dl = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0, collate_fn=collate_fn)

    losses = []
    nan_count = 0
    mean = torch.tensor(C.IMAGENET_MEAN).view(1, 3, 1, 1).cuda()
    std = torch.tensor(C.IMAGENET_STD).view(1, 3, 1, 1).cuda()

    for batch_idx, (images, targets) in enumerate(dl):
        if batch_idx >= 10:
            break
        images_norm = (images.cuda().float() / 255.0 - mean) / std
        for i in range(len(targets['detection'])):
            targets['detection'][i]['boxes'] = targets['detection'][i]['boxes'].cuda()
            targets['detection'][i]['labels'] = targets['detection'][i]['labels'].cuda()
        targets['head_pose'] = targets['head_pose'].cuda()
        targets['psr_labels'] = targets['psr_labels'].cuda()
        targets['activity'] = targets['activity'].cuda()
        if 'hand_joints' in targets:
            targets['hand_joints'] = targets['hand_joints'].cuda()

        optimizer.zero_grad(set_to_none=True)
        with autocast('cuda'):
            outputs = model(images_norm)
            loss, _ = loss_fn(outputs, targets)
        if not math.isfinite(loss.item()):
            nan_count += 1
            print(f'  Batch {batch_idx}: NaN/Inf detected! loss={loss.item()}')
            optimizer.zero_grad(set_to_none=True)
            continue
        losses.append(loss.item())
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    print(f'  J-1: {nan_count} NaN batches out of {len(losses)} [{len(losses)} total]')
    if losses:
        print(f'  Losses: {[f"{l:.4f}" for l in losses[:5]]}...')
    status = "PASS" if nan_count == 0 and len(losses) > 0 else "FAIL"
    print(f'  J-1: [{status}]')
except Exception as e:
    import traceback; traceback.print_exc()
    print(f'  J-1: FAIL - {e}')

print()
print('=== J-2: No CUDA OOM on first training step ===')
try:
    torch.cuda.empty_cache()
    mem_before = torch.cuda.memory_allocated() / 1024**3
    mem_before_allocated = torch.cuda.memory_reserved() / 1024**3
    print(f'  GPU mem before: allocated={mem_before:.2f}GB, reserved={mem_before_allocated:.2f}GB')

    # Single step with B=4
    model2 = POPWMultiTaskModel(pretrained=False, backbone_type='convnext_tiny', use_headpose_film=True, use_videomae=False, train_pose=True).cuda()
    model2.train()
    loss_fn2 = MultiTaskLoss(num_classes_act=C.NUM_ACT_CLASSES, num_psr_components=C.NUM_PSR_COMPONENTS,
        train_det=True, train_pose=False, train_act=True, train_psr=True, use_kendall=True).cuda()
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-4)
    scaler2 = GradScaler('cuda')

    # B=4 batch
    images_b4 = torch.randn(4, 3, 720, 1280).cuda()

    with autocast('cuda'):
        outputs = model2(images_b4)
    target_b4 = {
        'heatmaps': torch.randn(4, 17, 96, 160).cuda(), 'keypoints': torch.randn(4, 17, 2).cuda(),
        'detection': [{'boxes': torch.randn(5, 4).abs().cuda(), 'labels': torch.randint(0, 24, (5,)).cuda()} for _ in range(4)],
        'head_pose': torch.randn(4, 9).cuda(),
        'psr_labels': torch.randint(0, 2, (4, 11)).float().cuda(),
        'activity': torch.randint(0, C.NUM_ACT_CLASSES, (4,)).cuda(),
        'hand_joints': torch.randn(4, 52).cuda(),
    }
    loss2, _ = loss_fn2(outputs, target_b4)
    scaler2.scale(loss2).backward()
    scaler2.step(optimizer2)
    scaler2.update()

    mem_after = torch.cuda.memory_allocated() / 1024**3
    peak = torch.cuda.max_memory_allocated() / 1024**3
    print(f'  J-2: After B=4 step: allocated={mem_after:.2f}GB, peak={peak:.2f}GB')
    status2 = "PASS" if peak < 12.0 else "WARN"
    print(f'  J-2: [{status2}] (under 12GB for RTX 3060)')
    print(f'  (Note: limit is RTX 3060 12GB - warning at 10GB)')
except RuntimeError as e:
    if 'out of memory' in str(e).lower():
        print(f'  J-2: FAIL - CUDA OOM: {e}')
    else:
        print(f'  J-2: FAIL - {e}')
except Exception as e:
    print(f'  J-2: FAIL - {e}')