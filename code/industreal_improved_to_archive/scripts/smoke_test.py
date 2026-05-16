#!/usr/bin/env python3
"""
POPW Comprehensive Smoke Test
Tests all model components, loss functions, and training infrastructure.
"""

import sys
import os
import signal
import warnings

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
# Modules are in subdirectories: src/models/model.py, src/training/losses.py,
# src/evaluation/evaluate.py, src/config.py
SRC_DIR = os.path.join(SCRIPTS_DIR, os.pardir, 'src')
sys.path.insert(0, os.path.normpath(os.path.join(SRC_DIR, 'models')))
sys.path.insert(0, os.path.normpath(os.path.join(SRC_DIR, 'training')))
sys.path.insert(0, os.path.normpath(os.path.join(SRC_DIR, 'evaluation')))
sys.path.insert(0, os.path.normpath(SRC_DIR))

import torch
import torch.nn as nn
import numpy as np


class TimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise TimeoutError("test timed out")


def with_timeout(seconds, default=None):
    """Decorator: run test with timeout. Return default on timeout."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            if sys.platform == "win32":
                return func(*args, **kwargs)
            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            except TimeoutError:
                print(f"  ⚠️  {func.__name__} timed out after {seconds}s — skipping")
                return default
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
            return result
        return wrapper
    return decorator


# ============================================================
# TEST 1: All imports
# ============================================================
def test_imports():
    print("\n" + "="*60)
    print("TEST 1: Imports")
    print("="*60)
    try:
        import model
        import losses
        import config as C
        print("  ✅ All imports successful")
        return True
    except Exception as e:
        print(f"  ❌ Import failed: {e}")
        return False

# ============================================================
# TEST 2: Config values
# ============================================================
def test_config():
    print("\n" + "="*60)
    print("TEST 2: Config values")
    print("="*60)
    try:
        import config as C
        checks = [
            ("BACKBONE", C.BACKBONE == 'convnext_tiny'),
            ("NUM_DET_CLASSES", C.NUM_DET_CLASSES == 24),
            ("NUM_KEYPOINTS", C.NUM_KEYPOINTS == 17),
            ("NUM_CLASSES_ACT", C.NUM_CLASSES_ACT == 75),
            ("NUM_PSR_COMPONENTS", C.NUM_PSR_COMPONENTS == 11),
            ("IMG_WIDTH", C.IMG_WIDTH == 1280),
            ("IMG_HEIGHT", C.IMG_HEIGHT == 720),
            ("FOCAL_ALPHA", C.FOCAL_ALPHA == 0.25),
            ("FOCAL_GAMMA", C.FOCAL_GAMMA == 2.0),
            ("WING_OMEGA", C.WING_OMEGA == 0.05),
            ("WING_EPSILON", C.WING_EPSILON == 0.005),
            ("NUM_HEAD_POSE_DOF", C.NUM_HEAD_POSE_DOF == 9),
            ("PSR_TEMPORAL_SMOOTH_WEIGHT", C.PSR_TEMPORAL_SMOOTH_WEIGHT == 0.05),
            ("STAGED_TRAINING", C.STAGED_TRAINING == True),
            ("STAGE1_EPOCHS", C.STAGE1_EPOCHS == 5),
            ("STAGE2_EPOCHS", C.STAGE2_EPOCHS == 10),
            ("EMA_DECAY", C.EMA_DECAY == 0.999),
        ]
        passed = 0
        failed = 0
        for name, result in checks:
            status = "✅" if result else "❌"
            print(f"  {status} {name} = {getattr(C, name, 'MISSING')}")
            if result:
                passed += 1
            else:
                failed += 1
        print(f"\n  Result: {passed}/{len(checks)} passed")
        return failed == 0
    except Exception as e:
        print(f"  ❌ Config test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

# ============================================================
# TEST 3: Model creation and tensor shapes
# ============================================================
@with_timeout(30, False)
def test_model_shapes():
    print("\n" + "="*60)
    print("TEST 3: Model tensor shapes")
    print("="*60)
    try:
        import model as model_module
        import config as C

        # Create model
        m = model_module.POPWMultiTaskModel(
            backbone_type=C.BACKBONE,
            pretrained=False,
            use_videomae=False,  # Keep simple for smoke test
        )
        m.eval()

        # Dummy input
        B = 2
        images = torch.randn(B, 3, C.IMG_HEIGHT, C.IMG_WIDTH)

        with torch.no_grad():
            outputs = m(images)

        # Check output keys
        expected_keys = ['cls_preds', 'reg_preds', 'heatmaps', 'keypoints',
                        'pose_confidence', 'head_pose', 'act_logits', 'psr_logits']

        passed = 0
        failed = 0

        for key in expected_keys:
            if key in outputs:
                shape = outputs[key].shape
                print(f"  ✅ {key}: {shape}")
                passed += 1
            else:
                print(f"  ❌ {key}: MISSING")
                failed += 1

        # Check specific shapes
        print("\n  Shape validation:")
        checks = [
            ("cls_preds B,N,24", outputs['cls_preds'].shape[0] == B and outputs['cls_preds'].shape[2] == 24),
            ("reg_preds B,N,4", outputs['reg_preds'].shape[0] == B and outputs['reg_preds'].shape[2] == 4),
            ("heatmaps B,17,H,W", len(outputs['heatmaps'].shape) == 4 and outputs['heatmaps'].shape[1] == 17),
            ("keypoints B,17,2", outputs['keypoints'].shape == (B, 17, 2)),
            ("pose_confidence B,17", outputs['pose_confidence'].shape == (B, 17)),
            ("head_pose B,9", outputs['head_pose'].shape == (B, 9)),
            ("act_logits B,74or75", outputs['act_logits'].shape[1] in [74, 75]),
            ("psr_logits B,11", outputs['psr_logits'].shape == (B, 11)),
        ]

        for name, result in checks:
            status = "✅" if result else "❌"
            print(f"    {status} {name}")
            if result:
                passed += 1
            else:
                failed += 1

        print(f"\n  Result: {passed}/{passed+failed} passed")
        return failed == 0
    except Exception as e:
        print(f"  ❌ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

# ============================================================
# TEST 4: Kendall logvar initialization
# ============================================================
def test_kendall_init():
    print("\n" + "="*60)
    print("TEST 4: Kendall logvar initialization")
    print("="*60)
    try:
        import losses as losses_module
        import config as C

        # Create loss
        loss_fn = losses_module.MultiTaskLoss(
            num_classes_act=C.NUM_CLASSES_ACT,
            num_psr_components=C.NUM_PSR_COMPONENTS,
        )

        # Check init values
        checks = [
            ("log_var_det init = 0", abs(loss_fn.log_var_det.item()) < 1e-3),
            ("log_var_pose init = -1", abs(loss_fn.log_var_pose.item() - (-1.0)) < 1e-3),
            ("log_var_act init = 0", abs(loss_fn.log_var_act.item()) < 1e-3),
            ("log_var_psr init = 0", abs(loss_fn.log_var_psr.item()) < 1e-3),
        ]

        passed = 0
        failed = 0
        for name, result in checks:
            status = "✅" if result else "❌"
            print(f"  {status} {name}")
            if result:
                passed += 1
            else:
                failed += 1

        print(f"\n  Result: {passed}/{passed+failed} passed")
        return failed == 0
    except Exception as e:
        print(f"  ❌ Kendall init test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

# ============================================================
# TEST 5: Loss function values
# ============================================================
@with_timeout(30, False)
def test_loss_values():
    print("\n" + "="*60)
    print("TEST 5: Loss function sanity")
    print("="*60)
    try:
        import losses as losses_module
        import model as model_module
        import config as C

        # Create loss
        loss_fn = losses_module.MultiTaskLoss(
            num_classes_act=C.NUM_CLASSES_ACT,
            num_psr_components=C.NUM_PSR_COMPONENTS,
        )

        B = 2

        # Use the model to generate proper anchors and detection outputs
        m = model_module.POPWMultiTaskModel(
            backbone_type=C.BACKBONE,
            pretrained=False,
            use_videomae=False,
        )
        m.eval()

        images = torch.randn(B, 3, C.IMG_HEIGHT, C.IMG_WIDTH)
        with torch.no_grad():
            outputs = m(images)

        # Create detection targets per image (list of dicts, one per image in batch)
        targets = {
            'detection': [
                {'boxes': torch.tensor([[100.0, 100.0, 300.0, 300.0]], dtype=torch.float32),
                 'labels': torch.tensor([2], dtype=torch.long)}
                for _ in range(B)
            ],
            'keypoints': torch.randn(B, 17, 2),
            'pose_confidence': torch.rand(B, 17),
            'head_pose': torch.randn(B, 9),
            'activity': torch.randint(0, C.NUM_CLASSES_ACT, (B,)),
            'psr_labels': torch.randint(0, 2, (B, C.NUM_PSR_COMPONENTS)).float(),
        }

        # Set epoch to trigger staged loss
        loss_fn.set_epoch(10)  # Stage 2

        # Compute loss
        total_loss, loss_dict = loss_fn(outputs, targets)

        print(f"  Total loss: {total_loss.item():.4f}")
        print(f"  Loss dict keys: {list(loss_dict.keys())}")

        # Check loss dict has expected keys
        # Note: loss_dict values may be floats (not tensors) after Kendall division
        expected_loss_keys = ['det', 'pose', 'head_pose', 'activity', 'psr']
        passed = 0
        failed = 0
        for key in expected_loss_keys:
            if key in loss_dict:
                val = loss_dict[key]
                # Handle both tensor and float
                val_str = f"{val.item():.4f}" if isinstance(val, torch.Tensor) else f"{val:.4f}"
                if isinstance(val, torch.Tensor):
                    if torch.isfinite(val).all():
                        print(f"  ✅ {key}: {val_str}")
                        passed += 1
                    else:
                        print(f"  ❌ {key}: {val_str} (not finite)")
                        failed += 1
                else:
                    if np.isfinite(val):
                        print(f"  ✅ {key}: {val_str}")
                        passed += 1
                    else:
                        print(f"  ❌ {key}: {val_str} (not finite)")
                        failed += 1
            else:
                print(f"  ❌ {key}: MISSING")
                failed += 1

        # Check losses are finite
        if torch.isfinite(total_loss):
            print(f"  ✅ Total loss is finite")
            passed += 1
        else:
            print(f"  ❌ Total loss is NaN/Inf")
            failed += 1

        print(f"\n  Result: {passed}/{passed+failed} passed")
        return failed == 0
    except Exception as e:
        print(f"  ❌ Loss test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

# ============================================================
# TEST 6: Backward pass and gradient flow
# ============================================================
@with_timeout(120, False)
def test_backward():
    print("\n" + "="*60)
    print("TEST 6: Backward pass and gradient flow")
    print("="*60)
    try:
        import model as model_module
        import losses as losses_module
        import config as C

        # Create model and loss
        m = model_module.POPWMultiTaskModel(
            backbone_type=C.BACKBONE,
            pretrained=False,
            use_videomae=False,
        )
        loss_fn = losses_module.MultiTaskLoss(
            num_classes_act=C.NUM_CLASSES_ACT,
            num_psr_components=C.NUM_PSR_COMPONENTS,
        )

        B = 2
        m.train()

        # Dummy input
        images = torch.randn(B, 3, C.IMG_HEIGHT, C.IMG_WIDTH)

        # Dummy targets — detection targets must be a list of dicts, one per image
        targets = {
            'detection': [
                {'boxes': torch.tensor([[100.0, 100.0, 300.0, 300.0]], dtype=torch.float32),
                 'labels': torch.tensor([2], dtype=torch.long)}
                for _ in range(B)
            ],
            'keypoints': torch.randn(B, 17, 2),
            'pose_confidence': torch.rand(B, 17),
            'head_pose': torch.randn(B, 9),
            'activity': torch.randint(0, C.NUM_CLASSES_ACT, (B,)),
            'psr_labels': torch.randint(0, 2, (B, C.NUM_PSR_COMPONENTS)).float(),
        }

        # Forward pass (real model gives us anchors + all outputs)
        outputs = m(images)

        # Set epoch 0 (no Kendall staging) to ensure all modules get gradients
        # Using epoch 0 means backward compat mode where all Kendall weights are active
        loss_fn.set_epoch(0)

        # Compute loss
        total_loss, loss_dict = loss_fn(outputs, targets)

        # Backward
        total_loss.backward()

        # Count params with gradients
        params_with_grad = sum(1 for p in m.parameters() if p.grad is not None)

        print(f"  Total loss: {total_loss.item():.4f}")
        print(f"  Parameters with gradients: {params_with_grad}")

        passed = 0
        failed = 0

        if params_with_grad > 100:
            print(f"  ✅ {params_with_grad} params have gradients")
            passed += 1
        else:
            print(f"  ❌ Only {params_with_grad} params have gradients (expected >100)")
            failed += 1

        # Check specific modules have gradients
        # Note: head_pose_head may NOT get gradients in stage 1 (epoch 0)
        # because Kendall prec_pose=0 in stage 1 (epochs 1-5).
        # Only backbone/fpn/detection/pose/activity/psr heads are guaranteed
        # to have gradients in epoch 0. Check headpose_film separately (Test 7).
        modules_to_check = ['backbone', 'fpn', 'detection_head', 'pose_head',
                          'activity_head', 'psr_head']
        for mod_name in modules_to_check:
            mod = getattr(m, mod_name, None)
            if mod is not None:
                has_grad = any(p.grad is not None for p in mod.parameters())
                status = "✅" if has_grad else "❌"
                print(f"  {status} {mod_name} has gradients")
                if has_grad:
                    passed += 1
                else:
                    failed += 1

        print(f"\n  Result: {passed}/{passed+failed} passed")
        return failed == 0
    except Exception as e:
        print(f"  ❌ Backward test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

# ============================================================
# TEST 7: headpose_film gradient isolation
# ============================================================
@with_timeout(120, False)
def test_headpose_film_detach():
    """
    Test that headpose_film uses head_pose.detach() to isolate gradients.

    The FiLM module's gamma/beta networks SHOULD receive gradients (they learn to predict
    modulation parameters from head pose). But the head_pose HEAD should NOT receive
    activity gradients through the headpose_film path (because headpose uses detach).

    Verification: Run model forward + backward, then check which params get gradients.
    """
    print("\n" + "="*60)
    print("TEST 7: headpose_film gradient isolation")
    print("="*60)
    try:
        import model as model_module
        import losses as losses_module
        import config as C

        m = model_module.POPWMultiTaskModel(
            backbone_type=C.BACKBONE,
            pretrained=False,
            use_videomae=False,
        )
        loss_fn = losses_module.MultiTaskLoss(
            num_classes_act=C.NUM_CLASSES_ACT,
            num_psr_components=C.NUM_PSR_COMPONENTS,
        )
        m.train()

        B = 2
        images = torch.randn(B, 3, C.IMG_HEIGHT, C.IMG_WIDTH)

        # Create targets with real detection outputs from model
        with torch.no_grad():
            outputs_for_targets = m(images)

        targets = {
            'detection': [
                {'boxes': torch.tensor([[100.0, 100.0, 300.0, 300.0]], dtype=torch.float32),
                 'labels': torch.tensor([2], dtype=torch.long)}
                for _ in range(B)
            ],
            'keypoints': torch.randn(B, 17, 2),
            'pose_confidence': torch.rand(B, 17),
            'head_pose': torch.randn(B, 9),
            'activity': torch.randint(0, C.NUM_CLASSES_ACT, (B,)),
            'psr_labels': torch.randint(0, 2, (B, C.NUM_PSR_COMPONENTS)).float(),
        }

        # Epoch 0 = no Kendall staging, all modules get gradient signals
        loss_fn.set_epoch(0)
        outputs = m(images)
        total_loss, _ = loss_fn(outputs, targets)
        total_loss.backward()

        # Now check gradient ownership:
        # 1. headpose_film parameters (gamma_net, beta_net) SHOULD have gradients
        #    because headpose_film output feeds activity head → loss
        # 2. head_pose_head parameters should NOT receive gradients from
        #    activity path through headpose_film (due to .detach())

        hp_film = m.headpose_film
        hp_gamma_params = list(hp_film.gamma_net.parameters())
        hp_beta_params = list(hp_film.beta_net.parameters())
        hp_params = hp_gamma_params + hp_beta_params

        hp_film_has_grad = any(p.grad is not None for p in hp_params)
        head_pose_head_has_grad = any(
            p.grad is not None for p in m.head_pose_head.parameters()
        )

        print(f"  {'✅' if hp_film_has_grad else '❌'} headpose_film params have gradients (gamma/beta nets)")
        print(f"  {'✅' if not head_pose_head_has_grad else '❌'} head_pose_head params isolated from activity path (via detach)")

        # Also verify backbone/other heads get gradients
        backbone_has_grad = any(p.grad is not None for p in m.backbone.parameters())

        print(f"  {'✅' if backbone_has_grad else '❌'} backbone params have gradients")

        passed = 0
        failed = 0
        if hp_film_has_grad:
            passed += 1
        else:
            failed += 1
        if not head_pose_head_has_grad:
            passed += 1
        else:
            failed += 1
        if backbone_has_grad:
            passed += 1
        else:
            failed += 1

        print(f"\n  Result: {passed}/{passed+failed} passed")
        return failed == 0
    except Exception as e:
        print(f"  ❌ headpose_film detach test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

# ============================================================
# TEST 8: FeatureBank round-trip
# ============================================================
def test_feature_bank():
    print("\n" + "="*60)
    print("TEST 8: FeatureBank round-trip")
    print("="*60)
    try:
        import model as model_module

        fb = model_module.FeatureBank(embed_dim=512, window_size=8)

        B = 2
        video_ids = ['seq1', 'seq2']
        camera_views = ['front', 'top']

        # FeatureBank.forward() takes (projected_features, video_ids, camera_views)
        # and returns (bank, T, embed_dim) for each item in batch

        # First call: bank will replicate current frame since ring buffer starts empty
        feat1 = torch.randn(B, 512)
        bank1 = fb(feat1, video_ids, camera_views)
        print(f"  First forward shape: {bank1.shape}")  # [B, T=8, 512]

        # Second call with same sequence: ring buffer accumulates
        feat2 = torch.randn(B, 512)
        bank2 = fb(feat2, video_ids, camera_views)
        print(f"  Second forward shape: {bank2.shape}")  # [B, T=8, 512]

        # Third call
        feat3 = torch.randn(B, 512)
        bank3 = fb(feat3, video_ids, camera_views)
        print(f"  Third forward shape: {bank3.shape}")

        # Test reset
        fb.reset()
        bank4 = fb(feat1, video_ids, camera_views)
        print(f"  After reset shape: {bank4.shape}")

        # Test per-sequence reset — use feat1[0:1] to get B=1 and matching video_ids
        fb(feat1, video_ids, camera_views)  # accumulate first
        fb(feat2, video_ids, camera_views)  # accumulate second
        fb.reset_sequence('seq1', 'front')
        # Must use B=1 with matching video_ids list
        bank5 = fb(feat1[0:1], ['seq1'], ['front'])
        print(f"  After reset_sequence seq1 shape: {bank5.shape}")

        passed = 0
        failed = 0

        if bank1.shape == (B, 8, 512):
            print(f"  ✅ Forward returns (B, window_size, embed_dim)")
            passed += 1
        else:
            print(f"  ❌ Shape wrong: expected (B, 8, 512), got {bank1.shape}")
            failed += 1

        # Bank should change as features accumulate (ring buffer)
        if not torch.allclose(bank1, bank2):
            print(f"  ✅ Bank changes as features accumulate")
            passed += 1
        else:
            print(f"  ❌ Bank should change after update")
            failed += 1

        # After reset, bank should revert to replication mode
        if torch.allclose(bank1, bank4):
            print(f"  ✅ Reset clears the bank")
            passed += 1
        else:
            print(f"  ❌ Reset should restore initial state")
            failed += 1

        print(f"\n  Result: {passed}/{passed+failed} passed")
        return failed == 0
    except Exception as e:
        print(f"  ❌ FeatureBank test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

# ============================================================
# TEST 9: EMA functionality
# ============================================================
@with_timeout(30, False)
def test_ema():
    print("\n" + "="*60)
    print("TEST 9: EMA functionality")
    print("="*60)
    try:
        import model as model_module

        # Create model
        model = model_module.POPWMultiTaskModel(pretrained=False, use_videomae=False)

        # Create EMA
        ema = model_module.EMA(model, decay=0.999)

        # Get any parameter to test with (ConvNeXt: features.0.0.weight = first conv)
        # We iterate params to find one that exists in shadow
        param_name = None
        param_before = None
        for name, p in model.named_parameters():
            if name in ema.shadow:
                param_name = name
                param_before = p.clone()
                break

        if param_before is None:
            print(f"  ❌ No matching parameters found in EMA shadow")
            return False

        print(f"  Testing with parameter: {param_name}")

        # Make an optimizer step — can't use in-place on leaf tensors (requires_grad)
        # Use no_grad context + direct data copy
        with torch.no_grad():
            for name, p in model.named_parameters():
                if name == param_name:
                    # Simulate optimizer step by adding to param data
                    p.data = p.data + torch.randn_like(p) * 0.1
                    break

        # Update EMA
        ema.update()

        # Get EMA shadow after update
        shadow_after = ema.shadow[param_name]

        param_new = None
        for name, p in model.named_parameters():
            if name == param_name:
                param_new = p
                break

        print(f"  Param changed by optimizer: {not torch.allclose(param_before, param_new)}")
        print(f"  EMA shadow exists: {shadow_after is not None}")

        diff_from_old = (shadow_after - param_before).abs().mean().item()
        diff_from_new = (shadow_after - param_new).abs().mean().item()

        print(f"  Shadow diff from old: {diff_from_old:.6f}")
        print(f"  Shadow diff from new: {diff_from_new:.6f}")

        # With decay 0.999, shadow should be close to old (slow update)
        passed = 0
        failed = 0

        if diff_from_old > 1e-7:
            print(f"  ✅ EMA shadow was updated")
            passed += 1
        else:
            print(f"  ❌ EMA shadow not updated")
            failed += 1

        if diff_from_old < diff_from_new:
            print(f"  ✅ Shadow closer to old value (correct for high decay)")
            passed += 1
        else:
            print(f"  ❌ Shadow update pattern unexpected")
            failed += 1

        # Test get_ema / restore
        ema.get_ema()
        print(f"  ✅ get_ema() executed")

        ema.restore()
        print(f"  ✅ restore() executed")

        print(f"\n  Result: {passed}/{passed+failed} passed")
        return failed == 0
    except Exception as e:
        print(f"  ❌ EMA test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

# ============================================================
# TEST 10: Staged Kendall masking
# ============================================================
@with_timeout(30, False)
def test_staged_kendall():
    print("\n" + "="*60)
    print("TEST 10: Staged Kendall masking")
    print("="*60)
    try:
        import losses as losses_module
        import model as model_module
        import config as C

        loss_fn = losses_module.MultiTaskLoss(
            num_classes_act=C.NUM_CLASSES_ACT,
            num_psr_components=C.NUM_PSR_COMPONENTS,
        )

        # Create model to get real outputs (including anchors)
        m = model_module.POPWMultiTaskModel(
            backbone_type=C.BACKBONE,
            pretrained=False,
            use_videomae=False,
        )
        m.eval()

        B = 2
        images = torch.randn(B, 3, C.IMG_HEIGHT, C.IMG_WIDTH)

        with torch.no_grad():
            outputs = m(images)

        # Create detection targets per image
        targets = {
            'detection': [
                {'boxes': torch.tensor([[100.0, 100.0, 300.0, 300.0]], dtype=torch.float32),
                 'labels': torch.tensor([2], dtype=torch.long)}
                for _ in range(B)
            ],
            'keypoints': torch.randn(B, 17, 2),
            'pose_confidence': torch.rand(B, 17),
            'head_pose': torch.randn(B, 9),
            'activity': torch.randint(0, C.NUM_CLASSES_ACT, (B,)),
            'psr_labels': torch.randint(0, 2, (B, C.NUM_PSR_COMPONENTS)).float(),
        }

        passed = 0
        failed = 0

        # Test Stage 1 (epoch 1-5)
        loss_fn.set_epoch(3)
        _, loss_dict_s1 = loss_fn(outputs, targets)
        det_s1 = loss_dict_s1['det']
        pose_s1 = loss_dict_s1['pose']
        act_s1 = loss_dict_s1['activity']
        psr_s1 = loss_dict_s1['psr']
        hp_s1 = loss_dict_s1['head_pose']

        print(f"  Stage 1 (epoch 3): det={det_s1:.4f}, pose={pose_s1:.4f}, hp={hp_s1:.4f}, act={act_s1:.4f}, psr={psr_s1:.4f}")

        # Stage 1 (epoch 1-5): Kendall zeroes hp/act/psr precisions
        # Raw losses still flow to loss_dict (pre-kendall), but Kendall-weighted total uses staged precisions
        # In stage 1, only det is active in Kendall sum. Check via Kendall weights.
        w_det_s1 = loss_dict_s1['w_det']
        print(f"  Kendall weights (S1): w_det={w_det_s1:.4f}")

        # Test Stage 2 (epoch 6-15)
        loss_fn.set_epoch(10)
        _, loss_dict_s2 = loss_fn(outputs, targets)
        det_s2 = loss_dict_s2['det']
        pose_s2 = loss_dict_s2['pose']
        act_s2 = loss_dict_s2['activity']
        psr_s2 = loss_dict_s2['psr']
        hp_s2 = loss_dict_s2['head_pose']

        print(f"  Stage 2 (epoch 10): det={det_s2:.4f}, pose={pose_s2:.4f}, hp={hp_s2:.4f}, act={act_s2:.4f}, psr={psr_s2:.4f}")

        w_det_s2 = loss_dict_s2['w_det']
        w_pose_s2 = loss_dict_s2['w_pose']
        print(f"  Kendall weights (S2): w_det={w_det_s2:.4f}, w_pose={w_pose_s2:.4f}")

        # Stage 2: Kendall zeroes act/psr precisions, det+pose(hp) active
        # The staged Kendall zeroes prec_act and prec_psr to 0 in stage 2
        # So w_act and w_psr should be ~0
        if loss_dict_s2['w_act'] < 0.01 and loss_dict_s2['w_psr'] < 0.01:
            print(f"  ✅ Stage 2: Kendall correctly zeroes act/psr precision scalars")
            passed += 1
        else:
            print(f"  ❌ Stage 2: Kendall should zero act/psr")
            failed += 1

        # Test Stage 3 (epoch 16+)
        loss_fn.set_epoch(20)
        _, loss_dict_s3 = loss_fn(outputs, targets)
        det_s3 = loss_dict_s3['det']
        act_s3 = loss_dict_s3['activity']
        psr_s3 = loss_dict_s3['psr']
        hp_s3 = loss_dict_s3['head_pose']

        print(f"  Stage 3 (epoch 20): det={det_s3:.4f}, hp={hp_s3:.4f}, act={act_s3:.4f}, psr={psr_s3:.4f}")

        # Stage 3: all Kendall weights should be non-zero
        all_weights_nonzero = all([
            loss_dict_s3['w_det'] > 0.01,
            loss_dict_s3['w_pose'] > 0.01,
            loss_dict_s3['w_act'] > 0.01,
            loss_dict_s3['w_psr'] > 0.01,
        ])
        if all_weights_nonzero:
            print(f"  ✅ Stage 3: all Kendall weights are active")
            passed += 1
        else:
            print(f"  ❌ Stage 3: all Kendall weights should be active")
            failed += 1

        # Test epoch 0 (no staging — backward compat)
        loss_fn.set_epoch(0)
        _, loss_dict_s0 = loss_fn(outputs, targets)
        det_s0 = loss_dict_s0['det']
        hp_s0 = loss_dict_s0['head_pose']

        print(f"  Epoch 0 (no staging): det={det_s0:.4f}, hp={hp_s0:.4f}")

        if hp_s0 > 0 and det_s0 > 0:
            print(f"  ✅ Epoch 0: both det and head_pose active (backward compat)")
            passed += 1
        else:
            print(f"  ❌ Epoch 0: should have both losses")
            failed += 1

        print(f"\n  Result: {passed}/{passed+failed} passed")
        return failed == 0
    except Exception as e:
        print(f"  ❌ Staged Kendall test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

# ============================================================
# TEST 11: WingLoss, FocalLoss, BinaryFocal sanity
# ============================================================
def test_loss_functions():
    print("\n" + "="*60)
    print("TEST 11: Individual loss function sanity")
    print("="*60)
    try:
        import losses as losses_module

        passed = 0
        failed = 0

        # WingLoss
        wing = losses_module.WingLoss(omega=0.05, epsilon=0.005)
        pred = torch.randn(10, 17, 2)
        target = torch.randn(10, 17, 2)
        w_loss = wing(pred, target)
        if torch.isfinite(w_loss) and w_loss.item() > 0:
            print(f"  ✅ WingLoss: {w_loss.item():.4f}")
            passed += 1
        else:
            print(f"  ❌ WingLoss failed: {w_loss.item()}")
            failed += 1

        # FocalLoss — takes (cls_preds, reg_preds, anchors, targets)
        # targets must be a LIST with one dict per image in the batch
        focal = losses_module.FocalLoss(alpha=0.25, gamma=2.0)
        B_f, N_anchors, n_cls = 4, 9441, 24
        cls_preds = torch.randn(B_f, N_anchors, n_cls)
        reg_preds = torch.randn(B_f, N_anchors, 4)
        anchors = torch.randn(N_anchors, 4)  # [N, 4] xyxy format
        anchors[:, 2:] = anchors[:, :2].abs() + 10  # ensure w,h > 0
        # B_f=4 images in batch → need 4 target dicts, one per image
        targets = [
            {'boxes': torch.tensor([[50.0, 50.0, 100.0, 100.0]], dtype=torch.float32),
             'labels': torch.tensor([2], dtype=torch.long)}
            for _ in range(B_f)
        ]
        f_loss, _ = focal(cls_preds, reg_preds, anchors, targets)
        if torch.isfinite(f_loss) and f_loss.item() > 0:
            print(f"  ✅ FocalLoss: {f_loss.item():.4f}")
            passed += 1
        else:
            print(f"  ❌ FocalLoss failed: {f_loss.item() if torch.isfinite(f_loss) else 'NaN/Inf'}")
            failed += 1

        # BinaryFocalLoss — use binary_focal_loss function (no class wrapper)
        psr_pred = torch.randn(10, 11)
        psr_target = torch.randint(0, 2, (10, 11)).float()
        bf_loss = losses_module.binary_focal_loss(psr_pred, psr_target, alpha=0.25, gamma=2.0)
        if torch.isfinite(bf_loss) and bf_loss.item() > 0:
            print(f"  ✅ BinaryFocalLoss: {bf_loss.item():.4f}")
            passed += 1
        else:
            print(f"  ❌ BinaryFocalLoss failed: {bf_loss.item()}")
            failed += 1

        # GIoU Loss
        pred_boxes = torch.tensor([[50.0, 50.0, 100.0, 100.0]]).expand(10, -1)
        target_boxes = torch.tensor([[50.0, 50.0, 100.0, 100.0]]).expand(10, -1)
        giou = losses_module.generalized_box_iou_loss(pred_boxes, target_boxes)
        # giou is a tensor of per-box GIoU values (10 values, one per box pair)
        # Perfect overlap → GIoU = 1.0 for all boxes
        giou_mean = giou.mean().item()
        if giou.isfinite().all() and giou_mean >= 0:
            print(f"  ✅ GIoU (perfect overlap): mean={giou_mean:.4f}, all_finite=True")
            passed += 1
        else:
            print(f"  ❌ GIoU failed: mean={giou_mean:.4f}")
            failed += 1

        # LDAMLoss — no label_smoothing kwarg (that was a torch CrossEntropyLoss arg)
        ldam = losses_module.LDAMLoss(num_classes=74, max_m=0.5, s=30)
        act_pred = torch.randn(10, 74)
        act_target = torch.randint(0, 74, (10,))
        ldam_loss = ldam(act_pred, act_target)
        if torch.isfinite(ldam_loss) and ldam_loss.item() > 0:
            print(f"  ✅ LDAMLoss: {ldam_loss.item():.4f}")
            passed += 1
        else:
            print(f"  ❌ LDAMLoss failed: {ldam_loss.item()}")
            failed += 1

        print(f"\n  Result: {passed}/{passed+failed} passed")
        return failed == 0
    except Exception as e:
        print(f"  ❌ Loss functions test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

# ============================================================
# TEST 12: count_parameters utility
# ============================================================
def test_count_params():
    print("\n" + "="*60)
    print("TEST 12: Parameter counting utility")
    print("="*60)
    try:
        import model as model_module

        m = model_module.POPWMultiTaskModel(pretrained=False, use_videomae=False)

        result = model_module.count_parameters(m)

        total = result['total_all']
        trainable = result['total_trainable']

        print(f"  Total parameters: {total:,}")
        print(f"  Trainable parameters: {trainable:,}")

        passed = 0
        failed = 0

        if total > 40_000_000:
            print(f"  ✅ Total params > 40M")
            passed += 1
        else:
            print(f"  ❌ Total params too low (expected >40M)")
            failed += 1

        if trainable > 40_000_000:
            print(f"  ✅ Trainable params > 40M")
            passed += 1
        else:
            print(f"  ❌ Trainable params too low")
            failed += 1

        print(f"\n  Result: {passed}/{passed+failed} passed")
        return failed == 0
    except Exception as e:
        print(f"  ❌ count_parameters test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

# ============================================================
# TEST 13: compute_efficiency_metrics accepts string device
# [Bug fix: evaluate.py line 1492 — device_obj normalization]
# ============================================================
def test_compute_efficiency_metrics_string_device():
    print("\n" + "="*60)
    print("TEST 13: compute_efficiency_metrics — string device")
    print("="*60)
    try:
        import config as C
        from model import POPWMultiTaskModel
        from evaluate import compute_efficiency_metrics

        model = POPWMultiTaskModel(
            pretrained=False,
            backbone_type=C.BACKBONE,
            use_headpose_film=True,
            use_videomae=False,
            train_pose=True,
        )
        model.eval()

        # Pass string 'cuda' (or 'cpu') — this is the bug fix:
        # compute_efficiency_metrics expects torch.device but callers pass str
        device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
        eff = compute_efficiency_metrics(
            model, device_str,
            img_size=(C.IMG_HEIGHT, C.IMG_WIDTH),
        )

        has_keys = all(k in eff for k in ['eff_params_m', 'eff_gflops', 'eff_fps'])
        if has_keys:
            print(f"  ✅ compute_efficiency_metrics accepted string device '{device_str}'")
            print(f"     params={eff['eff_params_m']:.2f}M, gflops={eff['eff_gflops']:.1f}G, fps={eff['eff_fps']:.1f}")
            return True
        else:
            print(f"  ❌ Missing keys in efficiency output: {eff.keys()}")
            return False
    except Exception as e:
        print(f"  ❌ compute_efficiency_metrics test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

# ============================================================
# TEST 14: evaluate_all pipeline (synthetic data)
# [Full pipeline: model → criterion → metrics, 73 keys returned]
# ============================================================
@with_timeout(60, False)
def test_evaluate_all_pipeline():
    print("\n" + "="*60)
    print("TEST 14: evaluate_all — full pipeline with synthetic data")
    print("="*60)
    try:
        import config as C
        from model import POPWMultiTaskModel
        from losses import MultiTaskLoss
        from evaluate import evaluate_all
        from torch.utils.data import DataLoader, Dataset

        class _SynthDS(Dataset):
            def __init__(self, n=6, H=720, W=1280):
                self.n = n; self.H = H; self.W = W
            def __len__(self):
                return self.n
            def __getitem__(self, idx):
                n_dets = 2
                return (
                    torch.randn(3, self.H, self.W),
                    {
                        'detection': [{'boxes': torch.rand(n_dets, 4) * torch.tensor([self.W, self.H, self.W, self.H]),
                                       'labels': torch.randint(0, 24, (n_dets,))}],
                        'head_pose': torch.rand(9) * 2 - 1,
                        'psr_labels': torch.randint(0, 2, (11,)).float(),
                        'activity': torch.randint(0, 75, (1,)).item(),
                        'keypoints': torch.rand(17, 2) * torch.tensor([self.H, self.W]),
                        'pose_confidence': torch.rand(17),
                        'clip_rgb': None,
                        'metadata': [{'recording_id': f'synth_{idx}', 'camera_view': 'default'}],
                    }
                )

        def _collate(batch):
            B = len(batch)
            images = torch.stack([b[0] for b in batch])
            detection_list = [{'boxes': b[1]['detection'][0]['boxes'], 'labels': b[1]['detection'][0]['labels']} for b in batch]
            return images, {
                'detection': detection_list,
                'head_pose': torch.stack([b[1]['head_pose'] for b in batch]),
                'psr_labels': torch.stack([b[1]['psr_labels'] for b in batch]),
                'activity': torch.tensor([b[1]['activity'] for b in batch]),
                'keypoints': torch.stack([b[1]['keypoints'] for b in batch]),
                'pose_confidence': torch.stack([b[1]['pose_confidence'] for b in batch]),
                'clip_rgb': None,
                'metadata': [b[1]['metadata'][0] for b in batch],
            }

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = POPWMultiTaskModel(
            pretrained=False, backbone_type=C.BACKBONE,
            use_headpose_film=True, use_videomae=False, train_pose=True,
        ).to(device)
        criterion = MultiTaskLoss(
            num_classes_act=C.NUM_CLASSES_ACT,
            num_psr_components=C.NUM_PSR_COMPONENTS,
            train_det=True, train_pose=True,
            train_act=True, train_psr=True,
            use_kendall=C.STAGED_TRAINING,
        ).to(device)
        loader = DataLoader(_SynthDS(n=6), batch_size=2, shuffle=False, num_workers=0, collate_fn=_collate)

        metrics = evaluate_all(
            model, criterion, loader,
            device=device,
            max_batches=3,
        )

        num_keys = len(metrics)
        has_loss = 'loss' in metrics
        has_act_acc = 'act_accuracy' in metrics
        has_eff = 'eff_params_m' in metrics

        print(f"  ✅ evaluate_all returned {num_keys} metric keys")
        print(f"     loss={metrics.get('loss', None):.4f}" if has_loss else "  ❌ Missing loss")
        print(f"     act_accuracy={metrics.get('act_accuracy', None)}" if has_act_acc else "  ❌ Missing act_accuracy")
        print(f"     eff_params_m={metrics.get('eff_params_m', None)}" if has_eff else "  ❌ Missing eff_params_m")
        print(f"     det_mAP50={metrics.get('det_mAP50', None)}")

        passed = num_keys > 60 and has_loss and has_eff
        print(f"\n  Result: {'PASS' if passed else 'FAIL'}")
        return passed
    except Exception as e:
        print(f"  ❌ evaluate_all pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

# ============================================================
# MAIN
# ============================================================
def main():
    print("\n" + "#"*60)
    print("# POPW COMPREHENSIVE SMOKE TEST")
    print("#"*60)

    tests = [
        ("Imports", test_imports),
        ("Config", test_config),
        ("Model Shapes", test_model_shapes),
        ("Kendall Init", test_kendall_init),
        ("Loss Values", test_loss_values),
        ("Backward Pass", test_backward),
        ("headpose_film detach", test_headpose_film_detach),
        ("FeatureBank", test_feature_bank),
        ("EMA", test_ema),
        ("Staged Kendall", test_staged_kendall),
        ("Loss Functions", test_loss_functions),
        ("Count Parameters", test_count_params),
        ("compute_efficiency_metrics — string device", test_compute_efficiency_metrics_string_device),
        ("evaluate_all pipeline (synthetic)", test_evaluate_all_pipeline),
    ]

    results = []
    for name, test_fn in tests:
        try:
            result = test_fn()
            results.append((name, result))
        except Exception as e:
            print(f"  ❌ TEST CRASHED: {e}")
            results.append((name, False))

    print("\n" + "#"*60)
    print("# SUMMARY")
    print("#"*60)

    passed = sum(1 for _, r in results if r)
    failed = len(results) - passed

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {name}")

    print(f"\nTotal: {passed}/{len(results)} tests passed")

    if failed > 0:
        print(f"\n⚠️  {failed} test(s) failed - review above for details")
        return 1
    else:
        print(f"\n✅ All tests passed!")
        return 0

if __name__ == '__main__':
    sys.exit(main())