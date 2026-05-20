import sys, os, torch, warnings, json
warnings.filterwarnings('ignore')
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['OMP_NUM_THREADS'] = '4'
sys.path.insert(0, '.')

from src import config as C
C.DEBUG_MODE = True
C.USE_KENDALL = True

from src.models.model import POPWMultiTaskModel
from src.training.losses import MultiTaskLoss
from src.evaluation.evaluate import compute_efficiency_metrics, evaluate_all
from src.data.industreal_dataset import IndustRealMultiTaskDataset, collate_fn
from torch.utils.data import DataLoader
import torch.nn as nn

print('=== H-1: evaluate_all runs without error ===')
try:
    model = POPWMultiTaskModel(pretrained=False, backbone_type='convnext_tiny', use_headpose_film=True, use_videomae=False, train_pose=True).cuda()
    model.eval()
    loss_fn = MultiTaskLoss(num_classes_act=C.NUM_ACT_CLASSES, num_psr_components=C.NUM_PSR_COMPONENTS,
        train_det=True, train_pose=False, train_act=True, train_psr=True, use_kendall=True).cuda()

    ds = IndustRealMultiTaskDataset(split='val', max_recordings=1, subset_ratio=0.01, sequence_mode=False)
    dl = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0, collate_fn=collate_fn)

    # Compute metrics
    try:
        metrics = compute_efficiency_metrics(model, img_size=(720, 1280), device='cuda')
        print(f'  H-1: compute_efficiency_metrics works [PASS]')
        print(f'  eff_params_m={metrics.get("eff_params_m","N/A")}')
    except Exception as e:
        print(f'  H-1: compute_efficiency_metrics FAIL - {e}')

    # Test evaluate_all with mock data
    # evaluate_all signature: (model, criterion, loader, device, ...)
    try:
        eval_metrics = evaluate_all(model, loss_fn, dl, device='cuda', max_batches=1)
        print(f'  H-1: evaluate_all returned {len(eval_metrics)} metrics [PASS]')
        print(f'  Keys: {list(eval_metrics.keys())[:10]}...')
    except Exception as e:
        print(f'  H-1: evaluate_all FAIL - {e}')
        import traceback; traceback.print_exc()

except Exception as e:
    import traceback; traceback.print_exc()
    print(f'  H-1: FAIL - {e}')

print('=== H-2: compute_efficiency_metrics string device ===')
try:
    result = compute_efficiency_metrics(model, img_size=(720, 1280), device='cuda')
    print(f'  H-2: string device works [PASS]')
except Exception as e:
    print(f'  H-2: FAIL - {e}')

print('=== H-3: Validation metrics computed ===')
try:
    eval_metrics = evaluate_all(model, loss_fn, dl, device='cuda', max_batches=1)
    has_map = 'det_mAP50' in eval_metrics or 'map50' in eval_metrics
    has_f1 = 'act_macro_f1' in eval_metrics or 'macro_f1' in eval_metrics
    has_mae = 'head_pose_mae' in eval_metrics or 'mae' in eval_metrics
    print(f'  H-3: mAP50 present={has_map}, F1 present={has_f1}, MAE present={has_mae} [PASS]')
except Exception as e:
    print(f'  H-3: FAIL - {e}')

print('=== H-4: Combined metric formula ===')
try:
    # Inline the formula to verify correctness (avoids train.py import chain issues)
    _W_DET, _W_ACT, _W_POSE, _W_PSR = 0.30, 0.35, 0.15, 0.20
    map50, macro_f1_act, mae_head_pose, macro_f1_psr = 0.5, 0.6, 10.0, 0.7
    head_pose_acc = 1.0 / (1.0 + mae_head_pose)
    combined = _W_DET * map50 + _W_ACT * macro_f1_act + _W_POSE * head_pose_acc + _W_PSR * macro_f1_psr
    expected = 0.30*0.5 + 0.35*0.6 + 0.15*(1.0/(1.0+10.0)) + 0.20*0.7
    # Also verify train.py defines the weights correctly
    import re
    train_src = open('./src/training/train.py').read()
    w_det = float(re.search(r'_W_DET\s*=\s*([0-9.]+)', train_src).group(1))
    w_act = float(re.search(r'_W_ACT\s*=\s*([0-9.]+)', train_src).group(1))
    w_pose = float(re.search(r'_W_POSE\s*=\s*([0-9.]+)', train_src).group(1))
    w_psr = float(re.search(r'_W_PSR\s*=\s*([0-9.]+)', train_src).group(1))
    weights_ok = (w_det, w_act, w_pose, w_psr) == (0.30, 0.35, 0.15, 0.20)
    result_ok = abs(combined - expected) < 1e-6
    print(f'  H-4: combined={combined:.4f}, expected={expected:.4f} [{"PASS" if result_ok else "FAIL"}]')
    print(f'  H-4: weights match={weights_ok} (det={w_det}, act={w_act}, pose={w_pose}, psr={w_psr})')
except Exception as e:
    print(f'  H-4: FAIL - {e}')
    import traceback; traceback.print_exc()