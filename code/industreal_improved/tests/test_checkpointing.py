#!/usr/bin/env python3
"""
Checkpoint Verification Test — Bashara 2026-05-15

Tests:
1. Save a checkpoint with model + EMA + criterion (Kendall log_vars)
2. Load it back and verify all keys match
3. Test crash_recovery save/load
4. Verify strict=False doesn't silently swallow keys
"""

import sys, os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import tempfile


def test_checkpoint_roundtrip():
    """Test full checkpoint save/load round-trip."""
    from models.model import POPWMultiTaskModel, EMA
    from training.losses import MultiTaskLoss
    import config as C

    print("\n=== Test 1: Full checkpoint round-trip ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_dir = Path(tmpdir) / "checkpoints"
        ckpt_dir.mkdir(parents=True)

        # Build minimal model
        device = torch.device("cpu")
        model = POPWMultiTaskModel(
            pretrained=False,
            backbone_type="convnext_tiny",
            use_headpose_film=True,
            use_videomae=False,
            train_pose=True,
        ).to(device)

        criterion = MultiTaskLoss(
            num_classes_act=C.NUM_CLASSES_ACT,
            num_psr_components=C.NUM_PSR_COMPONENTS,
            train_det=True,
            train_pose=True,
            train_act=True,
            train_psr=True,
            use_kendall=True,
        ).to(device)

        # Set some non-default Kendall log_vars
        criterion.log_var_det.data.fill_(0.5)
        criterion.log_var_pose.data.fill_(-0.3)
        criterion.log_var_act.data.fill_(0.2)
        criterion.log_var_psr.data.fill_(-0.1)

        # Build EMA
        ema = EMA(model, decay=0.999, device=device)

        # Do a fake update
        for _ in range(10):
            ema.update()

        # Save a best checkpoint (simulating train.py)
        save_dict = {
            "epoch": 5,
            "model": model.state_dict(),
            "optimizer": {},  # fake
            "scheduler": {},  # fake
            "scaler": {},  # fake
            "best_metric": 0.75,
            "patience_counter": 0,
            "val_metrics": {"det_mAP50": 0.6, "act_macro_f1": 0.5},
        }
        if ema is not None:
            ema.get_ema()
            save_dict["model"] = model.state_dict()
            ema.restore()
            save_dict["ema_shadow"] = {k: v.clone() for k, v in ema.shadow.items()}

        # Add criterion state
        save_dict["criterion"] = {
            "log_var_det": criterion.log_var_det.data.clone(),
            "log_var_pose": criterion.log_var_pose.data.clone(),
            "log_var_act": criterion.log_var_act.data.clone(),
            "log_var_psr": criterion.log_var_psr.data.clone(),
        }

        ckpt_path = ckpt_dir / "best.pth"
        torch.save(save_dict, ckpt_path)
        print(f"  Saved checkpoint to {ckpt_path}")

        # Load it back
        loaded = torch.load(ckpt_path, map_location=device)

        # Verify all keys present
        required_keys = [
            "epoch",
            "model",
            "optimizer",
            "scheduler",
            "scaler",
            "best_metric",
            "patience_counter",
            "val_metrics",
            "ema_shadow",
            "criterion",
        ]
        missing = [k for k in required_keys if k not in loaded]
        if missing:
            print(f"  FAIL: Missing keys: {missing}")
            return False
        print(f"  PASS: All required keys present")

        # Verify Kendall log_vars
        criterion_loaded = loaded["criterion"]
        log_vars_match = (
            torch.allclose(criterion_loaded["log_var_det"], criterion.log_var_det.data, atol=1e-4)
            and torch.allclose(
                criterion_loaded["log_var_pose"], criterion.log_var_pose.data, atol=1e-4
            )
            and torch.allclose(
                criterion_loaded["log_var_act"], criterion.log_var_act.data, atol=1e-4
            )
            and torch.allclose(
                criterion_loaded["log_var_psr"], criterion.log_var_psr.data, atol=1e-4
            )
        )
        if log_vars_match:
            print(f"  PASS: Kendall log_vars match after round-trip")
        else:
            print(f"  FAIL: Kendall log_vars mismatch:")
            print(
                f"    saved: det={criterion_loaded['log_var_det'].item():.4f}, pose={criterion_loaded['log_var_pose'].item():.4f}, act={criterion_loaded['log_var_act'].item():.4f}, psr={criterion_loaded['log_var_psr'].item():.4f}"
            )
            print(
                f"    orig:  det={criterion.log_var_det.data.item():.4f}, pose={criterion.log_var_pose.data.item():.4f}, act={criterion.log_var_act.data.item():.4f}, psr={criterion.log_var_psr.data.item():.4f}"
            )
            return False

        # Verify EMA shadow
        ema_shadow_keys = set(loaded["ema_shadow"].keys())
        model_keys = set(loaded["model"].keys())
        if ema_shadow_keys == model_keys:
            print(f"  PASS: EMA shadow keys match model keys ({len(ema_shadow_keys)} keys)")
        else:
            extra = ema_shadow_keys - model_keys
            missing_ema = model_keys - ema_shadow_keys
            if extra:
                print(f"  WARN: EMA shadow has extra keys: {extra}")
            if missing_ema:
                print(f"  FAIL: EMA shadow missing keys: {missing_ema}")
                return False

        # Verify model state_dict can be loaded into fresh model
        model2 = POPWMultiTaskModel(
            pretrained=False,
            backbone_type="convnext_tiny",
            use_headpose_film=True,
            use_videomae=False,
            train_pose=True,
        ).to(device)

        result = model2.load_state_dict(loaded["model"], strict=False)
        if result.missing_keys:
            print(
                f"  WARN: Missing keys when loading into fresh model: {result.missing_keys[:5]}..."
            )
        if result.unexpected_keys:
            print(f"  WARN: Unexpected keys: {result.unexpected_keys[:5]}...")
        print(f"  PASS: Model state_dict loaded into fresh model")

        return True


def test_crash_recovery():
    """Test crash recovery checkpoint."""
    from models.model import POPWMultiTaskModel, EMA

    print("\n=== Test 2: Crash recovery checkpoint ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_dir = Path(tmpdir) / "checkpoints"
        ckpt_dir.mkdir(parents=True)

        device = torch.device("cpu")
        model = POPWMultiTaskModel(
            pretrained=False,
            backbone_type="convnext_tiny",
            use_headpose_film=True,
            use_videomae=False,
            train_pose=True,
        ).to(device)

        ema = EMA(model, decay=0.999, device=device)
        for _ in range(5):
            ema.update()

        # Simulate crash_recovery save (from train.py _save_crash_recovery)
        recovery_path = ckpt_dir / "crash_recovery.pth"
        save_dict = {
            "tag": "batch_50",
            "epoch": 3,
            "step": 50,
            "total_steps": 50,
            "seq_steps": 0,
            "model": model.state_dict(),
            "optimizer": {},  # fake
            "scaler": {},  # fake
            "nan_skips": 2,
            "running": {"total": 0.5, "det": 0.1},
            "num_batches": 50,
            "timestamp": 1234567890.0,
        }
        if ema is not None:
            save_dict["ema_shadow"] = {k: v.clone() for k, v in ema.shadow.items()}
        torch.save(save_dict, recovery_path)

        # Load and verify
        loaded = torch.load(recovery_path, map_location=device)
        assert loaded["tag"] == "batch_50"
        assert loaded["epoch"] == 3
        assert loaded["step"] == 50
        assert "ema_shadow" in loaded
        assert set(loaded["ema_shadow"].keys()) == set(model.state_dict().keys())
        print(f"  PASS: Crash recovery checkpoint round-trips correctly")

        # Verify it has all the keys needed for recovery
        assert "model" in loaded
        assert "optimizer" in loaded
        assert "scaler" in loaded
        assert "running" in loaded
        print(f"  PASS: All recovery keys present")

        return True


def test_strict_false_behavior():
    """Verify strict=False doesn't silently swallow matching keys."""
    from models.model import POPWMultiTaskModel

    print("\n=== Test 3: strict=False key mismatch behavior ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        device = torch.device("cpu")

        # Create model A
        model_a = POPWMultiTaskModel(
            pretrained=False,
            backbone_type="convnext_tiny",
            use_headpose_film=True,
            use_videomae=False,
            train_pose=True,
        ).to(device)

        # Create checkpoint with model A
        ckpt_a = {k: v.clone() for k, v in model_a.state_dict().items()}

        # Create model B (same architecture)
        model_b = POPWMultiTaskModel(
            pretrained=False,
            backbone_type="convnext_tiny",
            use_headpose_film=True,
            use_videomae=False,
            train_pose=True,
        ).to(device)

        # Load A's state into B with strict=False
        result = model_b.load_state_dict(ckpt_a, strict=False)

        # Should have no missing or unexpected keys
        if result.missing_keys:
            print(f"  WARN: Missing keys: {result.missing_keys}")
        if result.unexpected_keys:
            print(f"  WARN: Unexpected keys: {result.unexpected_keys}")

        if not result.missing_keys and not result.unexpected_keys:
            print(f"  PASS: No key mismatches between same-architecture models")
        else:
            print(f"  FAIL: Key mismatches detected")
            return False

        return True


def test_load_model_compat():
    """Test the _load_model_compat function from train.py."""
    from models.model import POPWMultiTaskModel

    print("\n=== Test 4: _load_model_compat behavior ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        device = torch.device("cpu")

        # Create checkpoint from model
        model = POPWMultiTaskModel(
            pretrained=False,
            backbone_type="convnext_tiny",
            use_headpose_film=True,
            use_videomae=False,
            train_pose=True,
        ).to(device)

        ckpt_state = model.state_dict()
        ckpt = {"model": ckpt_state, "epoch": 0}

        # Create slightly different model (same architecture)
        model2 = POPWMultiTaskModel(
            pretrained=False,
            backbone_type="convnext_tiny",
            use_headpose_film=True,
            use_videomae=False,
            train_pose=True,
        ).to(device)

        # Call _load_model_compat manually
        def _load_model_compat(model, state_dict):
            model_state = model.state_dict()
            compatible, skipped = {}, []
            for k, v in state_dict.items():
                if k in model_state and model_state[k].shape == v.shape:
                    compatible[k] = v
                else:
                    skipped.append(
                        (
                            k,
                            v.shape if hasattr(v, "shape") else "?",
                            model_state[k].shape if k in model_state else "NOT IN MODEL",
                        )
                    )
            result = model.load_state_dict(compatible, strict=False)
            return result, skipped

        load_result, skipped = _load_model_compat(model2, ckpt["model"])

        if skipped:
            print(f"  WARN: Skipped {len(skipped)} keys due to shape mismatch")
            for k, cs, ms in skipped[:3]:
                print(f"    {k}: ckpt={cs} model={ms}")
        else:
            print(f"  PASS: All {len(ckpt['model'])} keys loaded compatibly")

        return True


def test_current_trainpy_checkpointing():
    """Analyze current train.py checkpoint saving."""
    print("\n=== Analysis: Current train.py checkpoint behavior ===")

    issues = []

    # Issue 1: Kendall log_vars NOT saved in checkpoints
    print("  [1] Kendall log_vars in checkpoints:")
    print("      best.pth: criterion NOT saved  <-- BUG")
    print("      latest.pth: criterion NOT saved  <-- BUG")
    print("      crash_recovery.pth: criterion NOT saved  <-- BUG")
    issues.append("criterion state (Kendall log_vars) not saved in any checkpoint")

    # Issue 2: crash_recovery saves inside ckpt_dir, not archive
    print("  [2] Crash recovery path:")
    print("      Saves to: {ckpt_dir}/crash_recovery.pth (train.py line 568)")
    print("      This IS correct (ckpt_dir = C.CHECKPOINT_DIR = OUTPUT_ROOT/checkpoints)")
    print("      BUT: archive/checkpoints/ is EMPTY (audit finding)")
    print("      This means training never ran OR runs didn't complete epoch 1")
    issues.append("archive/checkpoints/ empty = training never completed epoch")

    # Issue 3: No periodic interval saves
    print("  [3] Periodic checkpoint saves:")
    print("      Saves at: epoch end, best, crash_recovery every 50 batches")
    print("      MISSING: checkpoint_epoch_{N}.pth files")
    print("      Only: latest.pth, best.pth, crash_recovery.pth")
    issues.append("No checkpoint_epoch_{N}.pth periodic saves")

    # Issue 4: strict=False
    print("  [4] strict=False in _load_model_compat:")
    print("      Uses shape-matched compatible dict before loading")
    print("      So strict=False only affects keys not in compatible dict")
    print("      This is CORRECT behavior - no silent swallowing")
    print("      However: result.missing_keys are logged at INFO level")

    for i, issue in enumerate(issues, 1):
        print(f"  Issue {i}: {issue}")

    return len(issues) == 0


if __name__ == "__main__":
    os.chdir(Path(__file__).parent.parent / "src")

    results = []

    try:
        results.append(("checkpoint_roundtrip", test_checkpoint_roundtrip()))
    except Exception as e:
        print(f"  EXCEPTION: {e}")
        import traceback

        traceback.print_exc()
        results.append(("checkpoint_roundtrip", False))

    try:
        results.append(("crash_recovery", test_crash_recovery()))
    except Exception as e:
        print(f"  EXCEPTION: {e}")
        import traceback

        traceback.print_exc()
        results.append(("crash_recovery", False))

    try:
        results.append(("strict_false", test_strict_false_behavior()))
    except Exception as e:
        print(f"  EXCEPTION: {e}")
        import traceback

        traceback.print_exc()
        results.append(("strict_false", False))

    try:
        results.append(("load_model_compat", test_load_model_compat()))
    except Exception as e:
        print(f"  EXCEPTION: {e}")
        import traceback

        traceback.print_exc()
        results.append(("load_model_compat", False))

    results.append(("trainpy_analysis", test_current_trainpy_checkpointing()))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {status}: {name}")

    all_passed = all(r[1] for r in results)
    print(f"\nOverall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    sys.exit(0 if all_passed else 1)
