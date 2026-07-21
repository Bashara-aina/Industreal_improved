#!/usr/bin/env python3
"""st_act_standalone.py — Single-task activity training without DataLoader.

Bypasses the collate_fn_sequences DataLoader hang by iterating the dataset
directly. Same logic as train_st.py --task act but without the hang.
"""

# DEPRECATED: This script uses the legacy MTLMViTModel. Use POPWMultiTaskModel from src/models/model.py instead.
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch, torch.nn.functional as F

_CODE_ROOT = Path(__file__).resolve().parent.parent
for _p in [str(_CODE_ROOT), str(_CODE_ROOT / "src")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import src.config as C

C.NUM_ACT_OUTPUTS = 75
C.ACT_CLASS_GROUPING = "none"
C.RAM_CACHE_MAX_IMAGES = 0

from src.data.industreal_dataset import IndustRealMultiTaskDataset
from src.models.mvit_mtl_model import MTLMViTModel

_MEAN = torch.tensor([0.45, 0.45, 0.45])
_STD = torch.tensor([0.225, 0.225, 0.225])


def ensure_cuda(t):
    return t.cuda() if isinstance(t, torch.Tensor) else t


def eval_act(model, ds, max_batches=2000):
    """Quick activity top-1 accuracy on val set."""
    model.eval()
    correct, total = 0, 0
    n = min(max_batches, len(ds))
    with torch.no_grad():
        for i in range(n):
            s = ds[i]
            act = s.get("action_label")
            if act is None or act.item() < 0:
                continue
            img = s["images"]["rgb"].unsqueeze(0).cuda().float() / 255.0
            img = (
                (img - _MEAN.cuda().view(1, 1, 3, 1, 1)) / _STD.cuda().view(1, 1, 3, 1, 1)
            ).permute(0, 2, 1, 3, 4)
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(img)
                pred = out["activity"].argmax(dim=-1)
            correct += pred.cpu() == act.item()
            total += 1
    model.train()
    return correct / max(total, 1)


def main():
    p = argparse.ArgumentParser(description="ST-act standalone trainer")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--batches-per-epoch", type=int, default=8000)
    p.add_argument(
        "--subset-windows",
        type=int,
        default=10000,
        help="Number of dataset windows to cycle through (default 10k; 0 = all 78k)",
    )
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    # ── Data ──────────────────────────────────────────────────────────
    print("Loading train dataset...")
    train_ds = IndustRealMultiTaskDataset(
        split="train", img_size=(224, 224), augment=False, sequence_mode=True, sequence_length=16
    )
    print("Loading val dataset...")
    val_ds = IndustRealMultiTaskDataset(
        split="val", img_size=(224, 224), augment=False, sequence_mode=True, sequence_length=16
    )
    n_train = min(args.subset_windows if args.subset_windows > 0 else len(train_ds), len(train_ds))
    print(f"Train: {len(train_ds)} total windows, cycling through {n_train}")
    print(f"Val: {len(val_ds)} windows")

    # ── Model ─────────────────────────────────────────────────────────
    print("Building model...")
    model = MTLMViTModel(num_act_classes=75).to(device)
    total_p = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  {total_p:.1f}M params")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda")

    _mean = _MEAN.to(device).view(1, 1, 3, 1, 1)
    _std = _STD.to(device).view(1, 1, 3, 1, 1)
    best_acc, best_epoch = 0.0, 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        epoch_loss, n_steps = 0.0, 0

        for bi in range(args.batches_per_epoch):
            idx = np.random.randint(0, n_train)
            s = train_ds[idx]
            act = s.get("action_label")
            if act is None or act.item() < 0:
                continue

            img = s["images"]["rgb"].unsqueeze(0).to(device).float() / 255.0
            img = ((img - _mean) / _std).permute(0, 2, 1, 3, 4)
            act_t = act.to(device)
            if act_t.dim() == 0:
                act_t = act_t.unsqueeze(0)

            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = F.cross_entropy(model(img)["activity"], act_t)

            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            epoch_loss += loss.item()
            n_steps += 1

            if bi % 1000 == 0:
                print(
                    f"  epoch {epoch}/{args.epochs} batch {bi}/{args.batches_per_epoch}: loss={loss.item():.4f}"
                )

        avg_loss = epoch_loss / max(n_steps, 1)
        sched.step()
        dt = time.time() - t0
        print(
            f"Epoch {epoch:3d}/{args.epochs}: loss={avg_loss:.4f}  lr={opt.param_groups[0]['lr']:.2e}  {dt:.0f}s"
        )

        # Eval
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            acc = eval_act(model, val_ds)
            print(f"  Eval: act_top1={acc:.4f}")
            if acc > best_acc:
                best_acc, best_epoch = acc, epoch
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "task": "act",
                        "metric": acc,
                        "metric_key": "act_top1",
                    },
                    out_dir / "best.pt",
                )
                print(f"  New best: {acc:.4f}")

        # Save latest
        torch.save({"epoch": epoch, "model_state_dict": model.state_dict()}, out_dir / "latest.pt")

    # ── Final ─────────────────────────────────────────────────────────
    print(f"\nTraining complete. Best acc: {best_acc:.4f} at epoch {best_epoch}")
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(
            {"best_act_top1": best_acc, "best_epoch": best_epoch, "epochs": args.epochs},
            f,
            indent=2,
        )

    # Symlink for MTL warm-start
    st_dir = Path("src/runs/st_checkpoints")
    st_dir.mkdir(parents=True, exist_ok=True)
    dst = st_dir / "st_act_best.pt"
    if dst.exists():
        dst.unlink()
    dst.symlink_to((out_dir / "best.pt").resolve())
    print(f"Symlinked: {dst}")


if __name__ == "__main__":
    main()
