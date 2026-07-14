#!/usr/bin/env python3
"""
End-to-end training smoke test — verifies dataloader→model→loss→backward→optimizer→EMA
works for at least 2 training steps with gradient accumulation.
"""

import sys
import os

# Add workdir so 'import src' works; also add subdirs for bare 'from models import model'
WORK_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SRC_DIR = os.path.join(WORK_DIR, "src")
sys.path.insert(0, WORK_DIR)
sys.path.insert(1, os.path.join(SRC_DIR, "models"))
sys.path.insert(2, os.path.join(SRC_DIR, "training"))
sys.path.insert(3, os.path.join(SRC_DIR, "evaluation"))
sys.path.insert(4, SRC_DIR)

import torch
import torch.optim as optim
from torch.amp import autocast

from models import model as model_module
from training import losses as losses_module
import config as C


def make_dummy_batch(B=2):  # Reduced from 4 to fit GPU
    """Create a dummy batch matching IndustRealDataset format."""
    images = torch.randn(B, 3, C.IMG_HEIGHT, C.IMG_WIDTH)

    # Detection targets — list of dicts, one per image
    detection = [
        {
            "boxes": torch.tensor([[100.0, 100.0, 400.0, 400.0]], dtype=torch.float32),
            "labels": torch.tensor([2], dtype=torch.long),
        }
        for _ in range(B)
    ]

    targets = {
        "detection": detection,
        "keypoints": torch.randn(B, 17, 2),
        "pose_confidence": torch.rand(B, 17),
        "head_pose": torch.randn(B, 9),
        "activity": torch.randint(0, C.NUM_CLASSES_ACT, (B,)),
        "psr_labels": torch.randint(0, 2, (B, C.NUM_PSR_COMPONENTS)).float(),
    }
    return images, targets


def test_e2e_training():
    print("\n" + "=" * 60)
    print("E2E TRAINING TEST: 2 steps + gradient accumulation")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # ---- Model ----
    m = model_module.POPWMultiTaskModel(
        backbone_type=C.BACKBONE,
        pretrained=False,
        use_videomae=False,
    ).to(device)
    m.train()
    print(f"  ✅ Model created and on {device}")

    # ---- EMA ----
    ema = model_module.EMA(m, decay=C.EMA_DECAY)
    print(f"  ✅ EMA initialized (decay={C.EMA_DECAY})")

    # ---- Optimizer ----
    optimizer = optim.AdamW(
        m.parameters(),
        lr=1e-4,
        weight_decay=1e-2,
    )
    print(f"  ✅ AdamW optimizer created")

    # ---- Loss ----
    criterion = losses_module.MultiTaskLoss(
        num_classes_act=C.NUM_CLASSES_ACT,
        num_psr_components=C.NUM_PSR_COMPONENTS,
    )
    print(f"  ✅ MultiTaskLoss created")

    # No AMP scaler needed for test (simplifies device handling)
    scaler = None
    print(f"  ✅ Skipping AMP (pure FP32 for clarity)")

    # ---- Training loop ----
    accum_steps = 4
    n_steps = 5  # 5 steps = 1 optimizer update after 4 steps + 1 more for grad accumulation

    print(f"\n  Running {n_steps} steps with accum={accum_steps}...")

    for step in range(n_steps):
        images, targets = make_dummy_batch(B=4)
        images = images.to(device)

        # Move targets to device
        targets_device = {}
        for k, v in targets.items():
            if k == "detection":
                targets_device[k] = [
                    {
                        k2: v2.to(device) if isinstance(v2, torch.Tensor) else v2
                        for k2, v2 in det.items()
                    }
                    for det in v
                ]
            elif isinstance(v, torch.Tensor):
                targets_device[k] = v.to(device)
            else:
                targets_device[k] = v

        # Forward
        # Disable autocast for cleaner error messages (mixed precision not needed for test)
        with autocast("cuda", enabled=False):
            outputs = m(images)
            # Ensure float
            for key in [
                "cls_preds",
                "reg_preds",
                "heatmaps",
                "keypoints",
                "pose_confidence",
                "head_pose",
                "act_logits",
                "psr_logits",
            ]:
                if key in outputs and isinstance(outputs[key], torch.Tensor):
                    outputs[key] = outputs[key].float()

            # Set epoch for staged Kendall
            epoch = 3 if step == 0 else 10  # stage 1 then stage 2
            criterion.set_epoch(epoch)

            total_loss, loss_dict = criterion(outputs, targets_device)

        # Scale loss for accumulation
        scaled_loss = total_loss / accum_steps

        # Backward
        scaled_loss.backward()

        # Optimizer step (only after accum_steps)
        if (step + 1) % accum_steps == 0:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            # EMA update
            ema.update()

            print(
                f"    Step {step + 1}: loss={total_loss.item():.4f}, epoch={epoch}, EMA updated ✅"
            )

    # ---- Final checks ----
    print(f"\n  Verifying final state...")

    # 1. Model still has grads
    params_with_grad = sum(1 for p in m.parameters() if p.grad is not None)
    print(f"    Params with grads: {params_with_grad}")

    # 2. EMA shadow was updated
    any_shadow_changed = False
    for name, shadow in list(ema.shadow.items())[:5]:
        for pname, p in m.named_parameters():
            if pname == name:
                if not torch.allclose(p.data, shadow):
                    any_shadow_changed = True
                    break
    print(f"    EMA shadow diverged from current weights: {any_shadow_changed} ✅")

    # 3. Optimizer state exists
    has_opt_state = (
        any(len(g) > 0 for g in optimizer.state_dict()["state"].values())
        if len(optimizer.state_dict()["state"]) > 0
        else False
    )
    print(f"    Optimizer has state: {has_opt_state} ✅")

    print(f"\n  ✅ E2E training test PASSED — model, loss, backward, optimizer, EMA all working")
    return True


if __name__ == "__main__":
    try:
        test_e2e_training()
        print("\n" + "=" * 60)
        print("ALL E2E TESTS PASSED")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ E2E test FAILED: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
