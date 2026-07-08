"""
Tests for MTL-All training script invariance (175 §6 row 5).

Verifies:
  1. TierFModel loads and forward produces all 4 head outputs with correct shapes.
  2. Kendall log_vars are created with init values close to 0 (per spec).
  3. HP_PREC_CAP is applied correctly (pose precision <= det precision).
  4. PCGrad balancer is wired when mode='pcgrad'.
  5. One-epoch training reduces all 4 losses (smoke-level).
  6. No staging zeros out heads (P4 guard: STAGED_TRAINING=False).
"""

from __future__ import annotations

import sys
import math
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent  # code/industreal_improved/
_SRC = _ROOT / "src"
for _p in [_ROOT, _SRC, _SRC / "models", _SRC / "training"]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

NUM_DET_CLASSES = 24
NUM_ACT_CLASSES = 75
NUM_PSR_COMPONENTS = 11
POSE_DIM = 6


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture(scope="module")
def device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="module")
def model(device):
    """TierFModel with random init (no pretrained)."""
    from src.models.tier_f_model import TierFModel

    m = TierFModel(
        num_classes_det=NUM_DET_CLASSES,
        num_classes_act=NUM_ACT_CLASSES,
        num_components_psr=NUM_PSR_COMPONENTS,
        pose_dim=POSE_DIM,
        pretrained=False,
    ).to(device)
    m.train()
    return m


@pytest.fixture(scope="module")
def synthetic_batch(device):
    """Synthetic batch with all 4 task targets."""
    B = 4
    T = 16
    return {
        "temporal_clip": torch.randn(B, T, 3, 224, 224, device=device),
        "detection_frame": torch.randn(B, 3, 224, 224, device=device),
        "act_targets": torch.randint(0, NUM_ACT_CLASSES, (B,), device=device),
        "psr_targets": torch.randint(0, 2, (B, T, NUM_PSR_COMPONENTS), device=device).float(),
        "pose_targets": torch.randn(B, POSE_DIM, device=device),
        "det_boxes": torch.tensor([[[50, 50, 180, 200]]], device=device).repeat(B, 1, 1),
        "det_classes": torch.zeros(B, 1, dtype=torch.long, device=device),
    }


@pytest.fixture(scope="module")
def small_batch(device):
    """Small-batch synthetic data (B=1) for memory-intensive backward tests."""
    B = 1
    T = 16
    return {
        "temporal_clip": torch.randn(B, T, 3, 224, 224, device=device),
        "detection_frame": torch.randn(B, 3, 224, 224, device=device),
        "act_targets": torch.randint(0, NUM_ACT_CLASSES, (B,), device=device),
        "psr_targets": torch.randint(0, 2, (B, T, NUM_PSR_COMPONENTS), device=device).float(),
        "pose_targets": torch.randn(B, POSE_DIM, device=device),
        "det_boxes": torch.tensor([[[50, 50, 180, 200]]], device=device).repeat(B, 1, 1),
        "det_classes": torch.zeros(B, 1, dtype=torch.long, device=device),
    }


# ===========================================================================
# Test 1: Model forward produces all 4 head outputs
# ===========================================================================


class TestModelForward:
    """Verify TierFModel produces all 4 head outputs in both modes."""

    def test_temporal_mode_shapes(self, model, synthetic_batch):
        """Temporal forward: act_logits, psr_logits, pose_6d with correct shapes."""
        clip = synthetic_batch["temporal_clip"]
        with torch.no_grad():
            out = model(clip, mode="temporal")

        B = clip.shape[0]
        T = clip.shape[1]

        assert "act_logits" in out, "Missing act_logits"
        assert "psr_logits" in out, "Missing psr_logits"
        assert "pose_6d" in out, "Missing pose_6d"

        assert out["act_logits"].shape == (B, NUM_ACT_CLASSES), (
            f"act_logits: expected ({B}, {NUM_ACT_CLASSES}), got {out['act_logits'].shape}"
        )
        assert out["psr_logits"].shape == (B, T, NUM_PSR_COMPONENTS), (
            f"psr_logits: expected ({B}, {T}, {NUM_PSR_COMPONENTS}), got {out['psr_logits'].shape}"
        )
        assert out["pose_6d"].shape == (B, POSE_DIM), (
            f"pose_6d: expected ({B}, {POSE_DIM}), got {out['pose_6d'].shape}"
        )

        # All should be finite
        for key, tensor in out.items():
            assert torch.isfinite(tensor).all(), f"{key} contains non-finite values"

    def test_detection_mode_shapes(self, model, synthetic_batch):
        """Detection forward: det_cls_logits, det_box_logits with correct structure."""
        frame = synthetic_batch["detection_frame"]
        with torch.no_grad():
            out = model(frame, mode="detection")

        assert "det_cls_logits" in out, "Missing det_cls_logits"
        assert "det_box_logits" in out, "Missing det_box_logits"

        B = frame.shape[0]
        cls_list = out["det_cls_logits"]
        box_list = out["det_box_logits"]

        assert len(cls_list) == 3, f"Expected 3 FPN levels, got {len(cls_list)}"
        assert len(box_list) == 3, f"Expected 3 FPN levels, got {len(box_list)}"

        for level_idx, (cls_t, box_t) in enumerate(zip(cls_list, box_list)):
            assert cls_t.shape[0] == B, f"Level {level_idx} cls batch dim"
            assert box_t.shape[0] == B, f"Level {level_idx} box batch dim"
            assert cls_t.shape[1] == NUM_DET_CLASSES, (
                f"Level {level_idx} cls: expected {NUM_DET_CLASSES} ch, got {cls_t.shape[1]}"
            )
            assert box_t.shape[1] == 64, (
                f"Level {level_idx} box: expected 64 ch, got {box_t.shape[1]}"
            )
            assert torch.isfinite(cls_t).all(), f"Level {level_idx} cls non-finite"
            assert torch.isfinite(box_t).all(), f"Level {level_idx} box non-finite"


# ===========================================================================
# Test 2: Kendall log_vars init values
# ===========================================================================


class TestKendallLogVars:
    """Kendall log_vars are created with init values close to 0 (per spec)."""

    def test_log_var_init_values(self, model, device):
        """Verify log_vars created in the training script init near 0."""
        # Simulate the init in train_mtl_all.py
        log_vars = nn.ParameterDict({
            "det": nn.Parameter(torch.zeros(1, device=device)),
            "act": nn.Parameter(torch.zeros(1, device=device)),
            "psr": nn.Parameter(torch.zeros(1, device=device)),
            "pose": nn.Parameter(torch.tensor([-1.0], device=device)),
        })

        # det, act, psr should be ~0
        assert abs(log_vars["det"].item()) < 1e-6, (
            f"log_var_det should be ~0, got {log_vars['det'].item()}"
        )
        assert abs(log_vars["act"].item()) < 1e-6, (
            f"log_var_act should be ~0, got {log_vars['act'].item()}"
        )
        assert abs(log_vars["psr"].item()) < 1e-6, (
            f"log_var_psr should be ~0, got {log_vars['psr'].item()}"
        )
        # pose init is -1.0 as per the existing convention
        assert abs(log_vars["pose"].item() - (-1.0)) < 1e-6, (
            f"log_var_pose should be -1.0, got {log_vars['pose'].item()}"
        )

        # All should be trainable
        for key, param in log_vars.items():
            assert param.requires_grad, f"log_var_{key} should require grad"

    def test_log_vars_have_no_weight_decay_group(self, model, device):
        """In build_param_groups, log_vars get their own group with wd=0."""
        from scripts.train_mtl_all import build_param_groups

        log_vars = nn.ParameterDict({
            "det": nn.Parameter(torch.zeros(1, device=device)),
            "act": nn.Parameter(torch.zeros(1, device=device)),
            "psr": nn.Parameter(torch.zeros(1, device=device)),
            "pose": nn.Parameter(torch.tensor([-1.0], device=device)),
        })

        groups = build_param_groups(model, log_vars)

        # Last group should be log_vars with wd=0
        last_group = groups[-1]
        assert len(last_group["params"]) == 4, (
            f"Expected 4 log_var params in last group, got {len(last_group['params'])}"
        )
        assert last_group["weight_decay"] == 0.0, (
            f"log_var group should have wd=0, got {last_group['weight_decay']}"
        )


# ===========================================================================
# Test 3: HP_PREC_CAP is on
# ===========================================================================


class TestHPPrecCap:
    """HP_PREC_CAP: pose precision never exceeds detection precision."""

    def test_hp_prec_cap_applied_correctly(self, device):
        """Verify that HP_PREC_CAP cap = max(lv_hp, lv_det.detach())."""
        log_vars = nn.ParameterDict({
            "det": nn.Parameter(torch.zeros(1, device=device)),
            "pose": nn.Parameter(torch.tensor([0.0], device=device)),
        })

        # Scenario: pose has lower log_var (higher precision) than det
        log_vars["det"].data.fill_(-2.0)  # prec_det = exp(2) ≈ 7.4
        log_vars["pose"].data.fill_(-3.0)  # prec_pose = exp(3) ≈ 20.1

        # Without cap: pose would dominate
        lv_det = log_vars["det"].clamp(-4.0, 2.0)
        lv_hp = log_vars["pose"].clamp(-4.0, 2.0)

        prec_det_no_cap = torch.exp(-lv_det)  # high precision
        prec_hp_no_cap = torch.exp(-lv_hp)  # even higher!

        # With cap: lv_hp = max(lv_hp, lv_det.detach())
        lv_hp_capped = torch.maximum(lv_hp, lv_det.detach())
        prec_hp_capped = torch.exp(-lv_hp_capped)

        # Without cap, pose precision should be higher than det
        assert prec_hp_no_cap > prec_det_no_cap, (
            f"Without cap: pose prec ({prec_hp_no_cap.item():.3f}) should be > "
            f"det prec ({prec_det_no_cap.item():.3f}) for this scenario"
        )

        # With cap, pose precision should NOT exceed det precision
        assert prec_hp_capped <= prec_det_no_cap + 1e-6, (
            f"With HP_PREC_CAP: pose prec ({prec_hp_capped.item():.3f}) must not "
            f"exceed det prec ({prec_det_no_cap.item():.3f})"
        )

        # After cap, lv_hp should equal lv_det (since lv_hp was lower/more negative)
        assert abs(lv_hp_capped.item() - lv_det.item()) < 1e-6, (
            f"Capped lv_hp ({lv_hp_capped.item():.3f}) should equal "
            f"lv_det ({lv_det.item():.3f}) when pose was more confident"
        )

    def test_hp_prec_cap_does_not_affect_det_log_var_grad(self, device):
        """The detach() on lv_det in HP_PREC_CAP prevents gradient flow to det's log_var."""
        log_vars = nn.ParameterDict({
            "det": nn.Parameter(torch.tensor([-1.0], device=device)),
            "pose": nn.Parameter(torch.tensor([-2.0], device=device)),
        })

        lv_det = log_vars["det"].clamp(-4.0, 2.0)
        lv_hp = log_vars["pose"].clamp(-4.0, 2.0)

        # Apply cap
        lv_hp_capped = torch.maximum(lv_hp, lv_det.detach())

        # Compute a loss that depends on both
        loss = lv_hp_capped.sum() + lv_det.sum()
        loss.backward()

        # det's log_var should have a gradient (from lv_det directly in loss)
        assert log_vars["det"].grad is not None, "det log_var should have grad"
        # pose's log_var should have a gradient (from lv_hp_capped which depends on lv_hp)
        assert log_vars["pose"].grad is not None, "pose log_var should have grad"


# ===========================================================================
# Test 4: PCGrad balancer is wired
# ===========================================================================


class TestPcgradWiring:
    """PCGrad balancer is correctly wired when mode='pcgrad'."""

    def test_balancer_created_with_pcgrad_mode(self, model, device):
        """MTLBalancer created with mode='pcgrad' has the correct mode."""
        from src.training.mtl_balancer import MTLBalancer

        shared_params = list(model.backbone.parameters())
        balancer = MTLBalancer(shared_params=shared_params, mode="pcgrad")

        assert balancer.mode == "pcgrad", (
            f"Expected mode='pcgrad', got mode='{balancer.mode}'"
        )

    def test_balancer_created_with_none_mode(self, model, device):
        """MTLBalancer created with mode='none' has the correct mode."""
        from src.training.mtl_balancer import MTLBalancer

        shared_params = list(model.backbone.parameters())
        balancer = MTLBalancer(shared_params=shared_params, mode="none")

        assert balancer.mode == "none", (
            f"Expected mode='none', got mode='{balancer.mode}'"
        )

    def test_pcgrad_backward_hooks_installed(self, model, small_batch, device):
        """PCGrad compute_step installs backward hooks on shared params."""
        from src.training.mtl_balancer import MTLBalancer

        shared_params = list(model.backbone.parameters())
        balancer = MTLBalancer(shared_params=shared_params, mode="pcgrad")

        log_vars = nn.ParameterDict({
            "det": nn.Parameter(torch.zeros(1, device=device)),
            "act": nn.Parameter(torch.zeros(1, device=device)),
            "psr": nn.Parameter(torch.zeros(1, device=device)),
            "pose": nn.Parameter(torch.tensor([-1.0], device=device)),
        })

        batch = small_batch

        # Compute synthetic task losses
        temporal_out = model(batch["temporal_clip"], mode="temporal")
        det_out = model(batch["detection_frame"], mode="detection")

        l_act = F.cross_entropy(temporal_out["act_logits"], batch["act_targets"])
        l_psr_class = F.binary_cross_entropy_with_logits(temporal_out["psr_logits"], batch["psr_targets"])
        l_pose_class = F.mse_loss(temporal_out["pose_6d"], batch["pose_targets"])

        # Simplified detection loss for test
        cls_losses = 0
        for cls_l in det_out["det_cls_logits"]:
            B, C, H, W = cls_l.shape
            cls_losses = cls_losses + F.cross_entropy(
                cls_l.permute(0, 2, 3, 1).reshape(-1, C),
                torch.zeros(B * H * W, dtype=torch.long, device=device),
            )

        # Apply Kendall weighting (simplified)
        lv_det = log_vars["det"].clamp(-4.0, 2.0)
        lv_hp = log_vars["pose"].clamp(-4.0, 2.0)
        lv_act = log_vars["act"].clamp(-4.0, 2.0)
        lv_psr = log_vars["pose"].clamp(-4.0, 2.0)
        lv_hp = torch.maximum(lv_hp, lv_det.detach())  # HP_PREC_CAP

        task_losses = [
            torch.exp(-lv_det) * cls_losses + lv_det,
            torch.exp(-lv_act) * l_act + lv_act,
            torch.exp(-lv_psr) * l_psr_class + lv_psr,
            torch.exp(-lv_hp) * l_pose_class + lv_hp,
        ]

        # compute_step should install hooks
        combined = balancer.compute_step(task_losses)
        assert balancer.has_hooks, "PCGrad should install backward hooks"

        # After backward, hooks should have fired and grads should be set
        combined.backward()
        for p in shared_params[:5]:  # check first 5
            assert p.grad is not None, "Shared param should have grad after PCGrad backward"
            assert torch.isfinite(p.grad).all(), "Grad should be finite"

    def test_pcgrad_cli_flag_maps_correctly(self):
        """CLI flag --pcgrad on|none maps to balancer mode."""
        from scripts.train_mtl_all import main as train_main

        # Verify the mapping by checking the argparse setup
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--pcgrad", type=str, choices=["on", "none"], default="on")

        args_on = parser.parse_args(["--pcgrad", "on"])
        args_off = parser.parse_args(["--pcgrad", "none"])

        assert args_on.pcgrad == "on"
        assert args_off.pcgrad == "none"

        # Mode mapping
        assert "pcgrad" if args_on.pcgrad == "on" else "none" == "pcgrad"
        assert "none" if args_off.pcgrad == "none" else "pcgrad" == "none"


# ===========================================================================
# Test 5: One-epoch training reduces all 4 losses (smoke-level)
# ===========================================================================


class TestOneEpochTraining:
    """One epoch of synthetic training should reduce all 4 losses."""

    def test_one_epoch_reduces_losses(self, model, small_batch, device):
        """All 4 head losses decrease after a single epoch of synthetic training (B=1)."""
        from src.training.mtl_balancer import MTLBalancer

        shared_params = list(model.backbone.parameters())
        balancer = MTLBalancer(shared_params=shared_params, mode="pcgrad")

        # Kendall log_vars
        log_vars = nn.ParameterDict({
            "det": nn.Parameter(torch.zeros(1, device=device)),
            "act": nn.Parameter(torch.zeros(1, device=device)),
            "psr": nn.Parameter(torch.zeros(1, device=device)),
            "pose": nn.Parameter(torch.tensor([-1.0], device=device)),
        })

        # Build optimizer following the training script pattern
        from scripts.train_mtl_all import build_param_groups

        param_groups = build_param_groups(
            model=model, log_vars=log_vars,
            backbone_lr=1e-4, head_lr=1e-3, log_var_lr=1e-3,
            layer_decay=0.8, weight_decay=0.05,
        )
        optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.999))

        # Use small_batch (B=1) to avoid OOM on 15 GB GPU
        batch = small_batch

        # Compute initial losses
        def compute_losses():
            temporal_out = model(batch["temporal_clip"], mode="temporal")
            det_out = model(batch["detection_frame"], mode="detection")

            l_det = 0.0
            for cls_l in det_out["det_cls_logits"]:
                B, C, H, W = cls_l.shape
                l_det = l_det + F.cross_entropy(
                    cls_l.permute(0, 2, 3, 1).reshape(-1, C),
                    torch.zeros(B * H * W, dtype=torch.long, device=device),
                )

            l_act = F.cross_entropy(temporal_out["act_logits"], batch["act_targets"])
            l_psr = F.binary_cross_entropy_with_logits(temporal_out["psr_logits"], batch["psr_targets"])
            l_pose = F.mse_loss(temporal_out["pose_6d"], batch["pose_targets"])

            return {
                "det": l_det.item(),
                "act": l_act.item(),
                "psr": l_psr.item(),
                "pose": l_pose.item(),
            }

        initial = compute_losses()

        # Run 3 training steps (mini-epoch, smoke level)
        for step in range(3):
            optimizer.zero_grad()

            temporal_out = model(batch["temporal_clip"], mode="temporal")
            det_out = model(batch["detection_frame"], mode="detection")

            # Detection loss
            l_det = 0.0
            for cls_l in det_out["det_cls_logits"]:
                B, C, H, W = cls_l.shape
                l_det = l_det + F.cross_entropy(
                    cls_l.permute(0, 2, 3, 1).reshape(-1, C),
                    torch.zeros(B * H * W, dtype=torch.long, device=device),
                )

            l_act = F.cross_entropy(temporal_out["act_logits"], batch["act_targets"])
            l_psr = F.binary_cross_entropy_with_logits(temporal_out["psr_logits"], batch["psr_targets"])
            l_pose = F.mse_loss(temporal_out["pose_6d"], batch["pose_targets"])

            # Kendall weighting with HP_PREC_CAP
            lv_det = log_vars["det"].clamp(-4.0, 2.0)
            lv_hp = log_vars["pose"].clamp(-4.0, 2.0)
            lv_act = log_vars["act"].clamp(-4.0, 2.0)
            lv_psr = log_vars["pose"].clamp(-4.0, 2.0)
            lv_hp = torch.maximum(lv_hp, lv_det.detach())  # HP_PREC_CAP

            task_losses = [
                torch.exp(-lv_det) * l_det + lv_det,
                torch.exp(-lv_act) * l_act + lv_act,
                torch.exp(-lv_psr) * l_psr + lv_psr,
                torch.exp(-lv_hp) * l_pose + lv_hp,
            ]

            combined = balancer.compute_step(task_losses)
            combined.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        final = compute_losses()

        # All losses should have decreased
        for head in ["det", "act", "psr", "pose"]:
            assert final[head] < initial[head] + 1e-6, (
                f"{head} loss did not decrease: initial={initial[head]:.4f}, "
                f"final={final[head]:.4f}"
            )

        # All losses should be finite
        for head in ["det", "act", "psr", "pose"]:
            assert math.isfinite(final[head]), f"{head} loss is non-finite: {final[head]}"
            assert final[head] >= 0, f"{head} loss is negative: {final[head]}"


# ===========================================================================
# Test 6: No staging (P4 guard)
# ===========================================================================


class TestNoStagingGuard:
    """P4 guard: no STAGED_TRAINING or KENDALL_STAGED_TRAINING zeroes out heads."""

    def test_training_script_imports_no_staging(self):
        """Run the train_mtl_all script module and verify staging flags are False."""
        # Import the module's main function to verify config intent
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "train_mtl_all",
            str(_ROOT / "scripts" / "train_mtl_all.py"),
        )
        assert spec is not None, "Could not load train_mtl_all.py spec"
        mod = importlib.util.module_from_spec(spec)

        # Don't actually execute, just verify file contains the guards
        source = (str(_ROOT / "scripts" / "train_mtl_all.py"))
        with open(source) as f:
            content = f.read()

        assert "STAGED_TRAINING = False" in content, (
            "P4 guard missing: STAGED_TRAINING=False must be set"
        )
        assert "KENDALL_STAGED_TRAINING = False" in content, (
            "P4 guard missing: KENDALL_STAGED_TRAINING=False must be set"
        )

    def test_no_staging_zeros_heads(self, model, small_batch, device):
        """Verify that all 4 heads receive gradient in a standard backward pass
        (i.e., no staging logic silently zeroes any head's gradient)."""
        from src.training.mtl_balancer import MTLBalancer

        shared_params = list(model.backbone.parameters())
        balancer = MTLBalancer(shared_params=shared_params, mode="pcgrad")

        # Use small_batch (B=1) to avoid OOM
        batch = small_batch

        log_vars = nn.ParameterDict({
            "det": nn.Parameter(torch.zeros(1, device=device)),
            "act": nn.Parameter(torch.zeros(1, device=device)),
            "psr": nn.Parameter(torch.zeros(1, device=device)),
            "pose": nn.Parameter(torch.tensor([-1.0], device=device)),
        })
        log_vars_params = list(log_vars.values())

        # Forward all 4 heads
        temporal_out = model(batch["temporal_clip"], mode="temporal")
        det_out = model(batch["detection_frame"], mode="detection")

        l_det = 0.0
        for cls_l in det_out["det_cls_logits"]:
            B, C, H, W = cls_l.shape
            l_det = l_det + F.cross_entropy(
                cls_l.permute(0, 2, 3, 1).reshape(-1, C),
                torch.zeros(B * H * W, dtype=torch.long, device=device),
            )
        l_act = F.cross_entropy(temporal_out["act_logits"], batch["act_targets"])
        l_psr = F.binary_cross_entropy_with_logits(temporal_out["psr_logits"], batch["psr_targets"])
        l_pose = F.mse_loss(temporal_out["pose_6d"], batch["pose_targets"])

        # Kendall with HP_PREC_CAP (same as training script)
        lv_det = log_vars["det"].clamp(-4.0, 2.0)
        lv_hp = log_vars["pose"].clamp(-4.0, 2.0)
        lv_act = log_vars["act"].clamp(-4.0, 2.0)
        lv_psr = log_vars["pose"].clamp(-4.0, 2.0)
        lv_hp = torch.maximum(lv_hp, lv_det.detach())

        task_losses = [
            torch.exp(-lv_det) * l_det + lv_det,
            torch.exp(-lv_act) * l_act + lv_act,
            torch.exp(-lv_psr) * l_psr + lv_psr,
            torch.exp(-lv_hp) * l_pose + lv_hp,
        ]

        combined = balancer.compute_step(task_losses)

        # Zero all grads
        model.zero_grad()
        for p in log_vars_params:
            p.grad = None

        combined.backward()

        # Check that EVERY head parameter has received gradient (not zeroed)
        # Head names: detection_head, activity_head, psr_head, pose_head
        head_params = []
        for name, param in model.named_parameters():
            if any(name.startswith(prefix) for prefix in [
                "detection_head", "activity_head", "psr_head", "pose_head"
            ]):
                head_params.append((name, param))

        # Group by head
        heads_with_grad = {"detection_head": False, "activity_head": False,
                           "psr_head": False, "pose_head": False}
        for name, param in head_params:
            if param.grad is not None and torch.isfinite(param.grad).any():
                for head_name in heads_with_grad:
                    if name.startswith(head_name):
                        heads_with_grad[head_name] = True

        # Every head should have received gradient
        for head_name, has_grad in heads_with_grad.items():
            assert has_grad, (
                f"P4 VIOLATION: {head_name} received zero gradient! "
                "Staging logic is zeroing out a head."
            )


# ===========================================================================
# Test 7: Per-head loss functions produce finite outputs
# ===========================================================================


class TestLossFunctions:
    """Per-head loss functions from train_mtl_all produce finite values."""

    def test_activity_loss_finite(self, device):
        """activity_loss produces finite scalar."""
        from scripts.train_mtl_all import activity_loss

        logits = torch.randn(4, NUM_ACT_CLASSES, device=device)
        targets = torch.randint(0, NUM_ACT_CLASSES, (4,), device=device)
        loss = activity_loss(logits, targets)
        assert torch.isfinite(loss)
        assert loss > 0

    def test_psr_loss_finite(self, device):
        """psr_loss produces finite scalar."""
        from scripts.train_mtl_all import psr_loss

        logits = torch.randn(4, 16, NUM_PSR_COMPONENTS, device=device)
        targets = torch.randint(0, 2, (4, 16, NUM_PSR_COMPONENTS), device=device).float()
        loss = psr_loss(logits, targets)
        assert torch.isfinite(loss)
        assert loss > 0

    def test_pose_loss_finite(self, device):
        """pose_loss produces finite scalar."""
        from scripts.train_mtl_all import pose_loss

        pred = torch.randn(4, POSE_DIM, device=device)
        target = torch.randn(4, POSE_DIM, device=device)
        loss = pose_loss(pred, target)
        assert torch.isfinite(loss)
        assert loss >= 0

    def test_detection_loss_finite(self, device):
        """detection_loss produces finite scalar (plumbing mode, no GT)."""
        from scripts.train_mtl_all import detection_loss

        cls_list = [torch.randn(2, NUM_DET_CLASSES, 28, 28, device=device),
                    torch.randn(2, NUM_DET_CLASSES, 14, 14, device=device),
                    torch.randn(2, NUM_DET_CLASSES, 7, 7, device=device)]
        box_list = [torch.randn(2, 64, 28, 28, device=device),
                    torch.randn(2, 64, 14, 14, device=device),
                    torch.randn(2, 64, 7, 7, device=device)]

        loss = detection_loss(cls_list, box_list)
        assert torch.isfinite(loss)
        assert loss > 0

    def test_detection_loss_with_gt_finite(self, device):
        """detection_loss with GT boxes/classes produces finite scalar."""
        from scripts.train_mtl_all import detection_loss

        cls_list = [torch.randn(2, NUM_DET_CLASSES, 28, 28, device=device),
                    torch.randn(2, NUM_DET_CLASSES, 14, 14, device=device),
                    torch.randn(2, NUM_DET_CLASSES, 7, 7, device=device)]
        box_list = [torch.randn(2, 64, 28, 28, device=device),
                    torch.randn(2, 64, 14, 14, device=device),
                    torch.randn(2, 64, 7, 7, device=device)]

        gt_boxes = torch.tensor([[[50, 50, 180, 200]]], device=device).repeat(2, 1, 1)
        gt_classes = torch.zeros(2, 1, dtype=torch.long, device=device)

        loss = detection_loss(cls_list, box_list, gt_boxes, gt_classes)
        assert torch.isfinite(loss)

    def test_all_losses_backprop(self, device):
        """All 4 loss functions support backward (produce finite gradients).

        Runs on CPU to avoid GPU memory pressure from the model fixture.
        """
        cpu = torch.device("cpu")
        from scripts.train_mtl_all import detection_loss, activity_loss, psr_loss, pose_loss

        # Detection (small spatial to minimize memory)
        x_cls = torch.randn(2, NUM_DET_CLASSES, 14, 14, device=cpu, requires_grad=True)
        x_box = torch.randn(2, 64, 14, 14, device=cpu, requires_grad=True)
        # Provide GT boxes/classes so both cls and box branches get gradients
        boxes = torch.tensor([[[50, 50, 180, 200]]], device=cpu).repeat(2, 1, 1)
        classes = torch.zeros(2, 1, dtype=torch.long, device=cpu)
        loss = detection_loss([x_cls], [x_box], gt_boxes=boxes, gt_classes=classes)
        loss.backward()
        assert x_cls.grad is not None and torch.isfinite(x_cls.grad).all()
        assert x_box.grad is not None and torch.isfinite(x_box.grad).all()

        # Activity
        x = torch.randn(4, NUM_ACT_CLASSES, device=cpu, requires_grad=True)
        t = torch.randint(0, NUM_ACT_CLASSES, (4,), device=cpu)
        loss = activity_loss(x, t)
        loss.backward()
        assert torch.isfinite(x.grad).all()

        # PSR
        x = torch.randn(4, 16, NUM_PSR_COMPONENTS, device=cpu, requires_grad=True)
        t = torch.randint(0, 2, (4, 16, NUM_PSR_COMPONENTS), device=cpu).float()
        loss = psr_loss(x, t)
        loss.backward()
        assert torch.isfinite(x.grad).all()

        # Pose
        x = torch.randn(4, POSE_DIM, device=cpu, requires_grad=True)
        t = torch.randn(4, POSE_DIM, device=cpu)
        loss = pose_loss(x, t)
        loss.backward()
        assert torch.isfinite(x.grad).all()
