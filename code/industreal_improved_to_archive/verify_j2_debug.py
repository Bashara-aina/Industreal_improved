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
from torch.amp import autocast, GradScaler

print('=== J-2 Debug: Trace dtype mismatch ===')
try:
    torch.cuda.empty_cache()

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
        print(f'  Model output dtypes:')
        for k, v in outputs.items():
            if isinstance(v, torch.Tensor):
                print(f'    {k}: {v.dtype}')
            elif isinstance(v, (list, tuple)):
                print(f'    {k}: list/tuple len={len(v)}, elems={[type(x) for x in v]}')
                if isinstance(v[0], torch.Tensor):
                    print(f'      [0] dtype={v[0].dtype}')

    target_b4 = {
        'heatmaps': torch.randn(4, 17, 96, 160).cuda(), 'keypoints': torch.randn(4, 17, 2).cuda(),
        'detection': [{'boxes': torch.randn(5, 4).abs().cuda(), 'labels': torch.randint(0, 24, (5,)).cuda()} for _ in range(4)],
        'head_pose': torch.randn(4, 9).cuda(),
        'psr_labels': torch.randint(0, 2, (4, 11)).float().cuda(),
        'activity': torch.randint(0, C.NUM_ACT_CLASSES, (4,)).cuda(),
        'hand_joints': torch.randn(4, 52).cuda(),
    }
    print(f'  Target dtypes:')
    for k, v in target_b4.items():
        if isinstance(v, torch.Tensor):
            print(f'    {k}: {v.dtype}')
        elif isinstance(v, (list, tuple)):
            print(f'    {k}: list len={len(v)}')
            if isinstance(v[0], torch.Tensor):
                print(f'      [0] dtype={v[0].dtype}')

    print(f'  Calling loss_fn...')
    loss2, _ = loss_fn2(outputs, target_b4)
    print(f'  Loss computed: {loss2.item()}')

    print(f'  Running backward...')
    scaler2.scale(loss2).backward()
    print(f'  Backward succeeded')
    scaler2.step(optimizer2)
    scaler2.update()
    print(f'  Optimizer step succeeded')

    peak = torch.cuda.max_memory_allocated() / 1024**3
    print(f'  J-2: Peak={peak:.2f}GB')
    status2 = "PASS" if peak < 12.0 else "WARN"
    print(f'  J-2: [{status2}] (under 12GB for RTX 3060)')

except RuntimeError as e:
    if 'out of memory' in str(e).lower():
        print(f'  J-2: FAIL - CUDA OOM')
    else:
        print(f'  J-2: FAIL - {e}')
        import traceback; traceback.print_exc()
except Exception as e:
    print(f'  J-2: FAIL - {e}')
    import traceback; traceback.print_exc()