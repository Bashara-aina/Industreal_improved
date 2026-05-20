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

print('=== J-2 Debug v2: Full backward trace ===')
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

    target_b4 = {
        'heatmaps': torch.randn(4, 17, 96, 160).cuda(), 'keypoints': torch.randn(4, 17, 2).cuda(),
        'detection': [{'boxes': torch.randn(5, 4).abs().cuda(), 'labels': torch.randint(0, 24, (5,)).cuda()} for _ in range(4)],
        'head_pose': torch.randn(4, 9).cuda(),
        'psr_labels': torch.randint(0, 2, (4, 11)).float().cuda(),
        'activity': torch.randint(0, C.NUM_ACT_CLASSES, (4,)).cuda(),
        'hand_joints': torch.randn(4, 52).cuda(),
    }

    print(f'  Output dtypes (sample):')
    for k in ['cls_preds', 'act_logits', 'psr_logits', 'head_pose', 'heatmaps']:
        if k in outputs and isinstance(outputs[k], torch.Tensor):
            print(f'    {k}: {outputs[k].dtype}')

    print(f'  Target dtypes (sample):')
    for k in ['head_pose', 'psr_labels', 'activity']:
        if k in target_b4 and isinstance(target_b4[k], torch.Tensor):
            print(f'    {k}: {target_b4[k].dtype}')

    print(f'  Moving Kendall params to fp16...')
    # Check Kendall param dtypes
    print(f'    log_var_det: {loss_fn2.log_var_det.dtype}')
    print(f'    log_var_pose: {loss_fn2.log_var_pose.dtype}')
    print(f'    log_var_act: {loss_fn2.log_var_act.dtype}')
    print(f'    log_var_psr: {loss_fn2.log_var_psr.dtype}')

    # Force move Kendall params to cuda
    loss_fn2.log_var_det.data = loss_fn2.log_var_det.data.cuda()
    loss_fn2.log_var_pose.data = loss_fn2.log_var_pose.data.cuda()
    loss_fn2.log_var_act.data = loss_fn2.log_var_act.data.cuda()
    loss_fn2.log_var_psr.data = loss_fn2.log_var_psr.data.cuda()

    print(f'  After moving to cuda:')
    print(f'    log_var_det: {loss_fn2.log_var_det.dtype} on {loss_fn2.log_var_det.device}')

    print(f'  Computing loss...')
    with autocast('cuda'):
        loss2, loss_dict = loss_fn2(outputs, target_b4)
    print(f'  Loss: {loss2.item():.4f}, dtype: {loss2.dtype}')

    print(f'  Checking loss components...')
    for k, v in loss_dict.items():
        if isinstance(v, float):
            print(f'    {k}: {v:.4f}')
        elif isinstance(v, torch.Tensor):
            print(f'    {k}: {v.item():.4f} dtype={v.dtype}')
        else:
            print(f'    {k}: {v}')

    print(f'  Running backward with GradScaler...')
    scaler2.scale(loss2).backward()
    print(f'  Backward succeeded!')

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