#!/usr/bin/env python3
"""
Tier 2.4 — Two-Stage Embedding Cache Pipeline
==============================================
Stage A: Train backbone + spatial heads (det, head pose) on the frames that matter.
Stage B: Run the FROZEN backbone over the full dataset, cache per-frame embeddings
         (512-D × ~2M frames ≈ 4GB on disk), then train the activity and PSR
         temporal heads over long real sequences (T=64-256) from cache at
         hundreds of epochs per hour.

This turns the dead FeatureBank into the centerpiece, fixes RC-18 by construction,
and makes the streaming-FPS efficiency story real.

At inference: one backbone pass per frame, temporal heads run on cached embeddings at O(1).

Design:
  - Cache stores: proj_feat (512-D activity projection) per frame + metadata
  - Cache format: HDF5 with per-recording groups, per-frame datasets
  - CacheLoader: Dataset that reads from HDF5, supports per-task sampling
  - CacheTraining: Trains activity + PSR heads from cache at 100× speed

Usage:
  # Stage A: train full model normally (existing train.py)
  # Stage B: cache embeddings
  python src/training/embedding_cache.py --cache --ckpt runs/.../best.pth
  # Stage B: train temporal heads from cache
  python src/training/embedding_cache.py --train --cache-dir runs/cache/
"""

import argparse
import h5py
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

_SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_SRC / "models"))
sys.path.insert(0, str(_SRC / "data"))

from src import config as C

logger = logging.getLogger(__name__)


# ============================================================================
# HDF5 Embedding Cache
# ============================================================================
class EmbeddingCache:
    """Disk cache for per-frame backbone embeddings.

    Stores: proj_feat [512] + det_conf [24] + image feature [768] per frame,
    plus metadata (recording_id, camera_view, frame_idx, activity_label, psr_labels).

    Layout:
        /recordings/{rec_id}/
            proj_feat [N, 512]  (float16 — 4GB for 2M frames)
            det_conf   [N, 24]
            c5_gap     [N, 768]
            p4_gap     [N, 256]
            activity   [N]   (int64 label)
            psr        [N, 11] (float32 binary)
            frame_idx  [N]   (int64)
            camera_view [N]  (S10 string)
    """

    FEATURE_DIM = 512  # proj_feat dimension
    DET_DIM = 24  # det_conf dimension

    def __init__(self, cache_dir: str, mode: str = "write"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.mode = mode
        self._file: Optional[h5py.File] = None
        self._groups: Dict[str, h5py.Group] = {}
        self._buffers: Dict[str, Dict[str, list]] = defaultdict(lambda: defaultdict(list))
        self._counts: Dict[str, int] = defaultdict(int)

    def open(self):
        if self.mode == "write":
            self._file = h5py.File(self.cache_dir / "embeddings.h5", "w")
        elif self.mode == "read":
            self._file = h5py.File(self.cache_dir / "embeddings.h5", "r")
        return self

    def close(self):
        if self._file:
            self._file.close()
            self._file = None

    def add_frame(
        self,
        rec_id: str,
        proj_feat: torch.Tensor,
        det_conf: torch.Tensor,
        c5_gap: torch.Tensor,
        p4_gap: torch.Tensor,
        activity: int,
        psr: torch.Tensor,
        frame_idx: int,
        camera: str,
    ):
        """Buffer a frame; flushed in batches for HDF5 write efficiency."""
        buf = self._buffers[rec_id]
        buf["proj_feat"].append(proj_feat.cpu().float16().numpy())
        buf["det_conf"].append(det_conf.cpu().float16().numpy())
        buf["c5_gap"].append(c5_gap.cpu().float16().numpy())
        buf["p4_gap"].append(p4_gap.cpu().float16().numpy())
        buf["activity"].append(activity)
        buf["psr"].append(psr.cpu().numpy())
        buf["frame_idx"].append(frame_idx)
        buf["camera"] = camera  # constant per recording
        self._counts[rec_id] += 1

    def flush_recording(self, rec_id: str):
        """Write buffered frames for one recording to HDF5."""
        if rec_id not in self._buffers or self._counts[rec_id] == 0:
            return

        buf = self._buffers[rec_id]
        N = self._counts[rec_id]
        grp = self._file.create_group(f"recordings/{rec_id}")

        for key, dtype in [
            ("proj_feat", np.float16),
            ("det_conf", np.float16),
            ("c5_gap", np.float16),
            ("p4_gap", np.float16),
        ]:
            data = np.stack(buf[key], axis=0)
            grp.create_dataset(key, data=data, dtype=dtype)

        grp.create_dataset("activity", data=np.array(buf["activity"], dtype=np.int64))
        grp.create_dataset("psr", data=np.stack(buf["psr"], axis=0).astype(np.float32))
        grp.create_dataset("frame_idx", data=np.array(buf["frame_idx"], dtype=np.int64))
        grp.attrs["camera_view"] = buf["camera"]
        grp.attrs["num_frames"] = N

        logger.info(f"[CACHE] Saved {rec_id}: {N} frames")

        # Clear buffer
        self._buffers.pop(rec_id)
        self._counts.pop(rec_id)

    def flush_all(self):
        for rec_id in list(self._buffers.keys()):
            self.flush_recording(rec_id)

    @property
    def recording_ids(self) -> List[str]:
        if self.mode == "read":
            return list(self._file["recordings"].keys())
        return list(self._buffers.keys())

    def get_recording(self, rec_id: str) -> Dict[str, np.ndarray]:
        """Get all frames for a recording as numpy arrays."""
        grp = self._file[f"recordings/{rec_id}"]
        return {
            "proj_feat": grp["proj_feat"][:],
            "det_conf": grp["det_conf"][:],
            "c5_gap": grp["c5_gap"][:],
            "p4_gap": grp["p4_gap"][:],
            "activity": grp["activity"][:],
            "psr": grp["psr"][:],
            "frame_idx": grp["frame_idx"][:],
            "camera_view": grp.attrs["camera_view"],
            "num_frames": grp.attrs["num_frames"],
        }

    def total_frames(self) -> int:
        if self.mode == "read":
            return sum(g.attrs["num_frames"] for g in self._file["recordings"].values())
        return sum(self._counts.values())


# ============================================================================
# Cache Dataset — optimized for fast loading from HDF5
# ============================================================================
class CacheDataset(Dataset):
    """Loads temporal sequences from the embedding cache.

    Each item is a (T, feature_dim) sequence from a specific recording.
    Supports per-task sampling: activity clips, PSR transition windows, etc.
    """

    def __init__(
        self,
        cache_dir: str,
        seq_len: int = 64,
        stride: int = 8,
        task_mode: str = "all",
        split: str = "train",
    ):
        self.cache = EmbeddingCache(cache_dir, mode="read")
        self.cache.open()
        self.seq_len = seq_len
        self.stride = stride
        self.task_mode = task_mode

        # Build index: list of (rec_id, start_frame) pairs
        self._index = []
        rec_ids = sorted(self.cache.recording_ids)
        # Use official train/val/test recording lists from config CSVs
        split_csv = {
            "train": C.TRAIN_CSV,
            "val": C.VAL_CSV,
            "test": C.TEST_CSV,
        }.get(split)
        if split_csv and split_csv.exists():
            official_ids = set()
            with open(split_csv, encoding="utf-8") as f:
                for line in f:
                    rid = line.strip().split(",")[0]
                    if rid:
                        official_ids.add(rid)
            rec_ids = sorted(official_ids & set(rec_ids))
            if not rec_ids:
                logger.warning(
                    f"[CacheDataset] No cached recordings match split {split}; "
                    f"falling back to all recordings"
                )
                rec_ids = sorted(self.cache.recording_ids)
            else:
                logger.info(
                    f"[CacheDataset] Filtered to {len(rec_ids)} recordings from {split} split"
                )
        else:
            logger.warning(f"[CacheDataset] Split CSV not found: {split_csv}; using all recordings")

        for rec_id in rec_ids:
            data = self.cache.get_recording(rec_id)
            N = data["num_frames"]
            for start in range(0, N - seq_len, stride):
                self._index.append((rec_id, start))

        logger.info(
            f"[CacheDataset] {len(self._index)} sequences, "
            f"seq_len={seq_len}, stride={stride}, task={task_mode}"
        )

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        rec_id, start = self._index[idx]
        data = self.cache.get_recording(rec_id)
        end = min(start + self.seq_len, data["num_frames"])

        seq = {}
        for key in ["proj_feat", "det_conf", "c5_gap", "p4_gap"]:
            seq[key] = torch.from_numpy(data[key][start:end].astype(np.float32))

        # Activity label: majority vote or last frame
        act_labels = data["activity"][start:end]
        seq["activity_seq"] = torch.from_numpy(act_labels).long()
        seq["activity"] = act_labels[-1]  # last frame label

        # PSR labels
        seq["psr_labels"] = torch.from_numpy(data["psr"][start:end].astype(np.float32))

        seq["recording_id"] = rec_id
        seq["start_frame"] = start
        seq["camera_view"] = data["camera_view"]

        return seq


# ============================================================================
# Temporal Head Trainer (Stage B)
# ============================================================================
class CacheTrainer:
    """Train activity + PSR heads from cached embeddings at high speed."""

    def __init__(
        self,
        ckpt_path: str,
        cache_dir: str,
        output_dir: str,
        seq_len: int = 64,
        batch_size: int = 128,
        lr: float = 1e-3,
    ):
        self.ckpt_path = ckpt_path
        self.cache_dir = cache_dir
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.lr = lr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model to extract head-only weights
        self._load_heads()

    def _load_heads(self):
        """Extract activity and PSR heads from the checkpoint."""
        ckpt = torch.load(self.ckpt_path, map_location="cpu")
        state = ckpt.get("model", ckpt)

        # Extract head parameters
        self.act_head_state = {}
        self.psr_head_state = {}
        for k, v in state.items():
            if k.startswith("activity_head."):
                self.act_head_state[k[len("activity_head.") :]] = v
            elif k.startswith("psr_head."):
                self.psr_head_state[k[len("psr_head.") :]] = v

        logger.info(
            f"[CacheTrainer] Loaded {len(self.act_head_state)} activity params, "
            f"{len(self.psr_head_state)} PSR params"
        )

    def _build_act_model(self) -> nn.Module:
        """Build an activity-only model from the extracted head."""
        from src.models.model import ActivityHead

        model = ActivityHead(
            embed_dim=512,
            num_classes=C.NUM_CLASSES_ACT,
            use_videomae=False,
        )
        # Load pre-trained head weights
        model.load_state_dict(self.act_head_state, strict=False)
        return model.to(self.device)

    def _build_psr_model(self) -> nn.Module:
        """Build a PSR-only model from the extracted head."""
        from src.models.model import PSRHead

        model = PSRHead(
            num_components=C.NUM_PSR_COMPONENTS,
            per_scale_ch=768,
            gru_hidden=256,
        )
        # Load pre-trained head weights
        model.load_state_dict(self.psr_head_state, strict=False)
        return model.to(self.device)

    def train_heads(self, num_epochs: int = 50):
        """Train both temporal heads from the cache."""
        train_ds = CacheDataset(
            self.cache_dir, seq_len=self.seq_len, task_mode="all", split="train"
        )
        val_ds = CacheDataset(self.cache_dir, seq_len=self.seq_len, task_mode="all", split="val")

        train_loader = DataLoader(
            train_ds, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True
        )
        val_loader = DataLoader(
            val_ds, batch_size=self.batch_size, shuffle=False, num_workers=2, pin_memory=True
        )

        # Build models
        act_model = self._build_act_model()
        psr_model = self._build_psr_model()

        # Optimizer
        params = list(act_model.parameters()) + list(psr_model.parameters())
        optim = torch.optim.AdamW(params, lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=num_epochs)

        # Losses
        ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
        bce_loss = nn.BCEWithLogitsLoss()

        logger.info(
            f"[CacheTrainer] Starting {num_epochs} epochs, "
            f"{len(train_ds)} train seqs, bs={self.batch_size}"
        )

        best_act_acc = 0.0
        best_psr_f1 = 0.0

        for epoch in range(num_epochs):
            # Training
            act_model.train()
            psr_model.train()
            train_act_loss = 0.0
            train_psr_loss = 0.0

            for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
                proj = batch["proj_feat"].to(self.device)
                act_labels = batch["activity_seq"].to(self.device)
                psr_labels = batch["psr_labels"].to(self.device)

                # Activity: classify each frame
                B, T, D = proj.shape
                proj_flat = proj.reshape(B * T, D)
                act_logits = act_model(proj_flat)
                act_loss = ce_loss(act_logits, act_labels.reshape(-1))

                # PSR: predict from last feature in each sequence
                psr_feat = proj[:, -1, :]  # [B, D]
                psr_logits = psr_model._get_frame_feat(psr_feat, None, None, None)
                psr_loss = bce_loss(psr_logits[..., : C.NUM_PSR_COMPONENTS], psr_labels[:, -1, :])

                loss = act_loss + psr_loss
                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optim.step()

                train_act_loss += act_loss.item()
                train_psr_loss += psr_loss.item()

            scheduler.step()

            # Validation
            act_model.eval()
            psr_model.eval()
            val_act_correct = 0
            val_act_total = 0
            val_psr_tp = 0
            val_psr_fp = 0
            val_psr_fn = 0

            with torch.no_grad():
                for batch in val_loader:
                    proj = batch["proj_feat"].to(self.device)
                    act_labels = batch["activity_seq"].to(self.device)
                    psr_labels = batch["psr_labels"].to(self.device)

                    B, T, D = proj.shape
                    proj_flat = proj.reshape(B * T, D)
                    act_logits = act_model(proj_flat)
                    act_pred = act_logits.argmax(-1)
                    val_act_correct += (act_pred == act_labels.reshape(-1)).sum().item()
                    val_act_total += act_labels.numel()

                    psr_logits = psr_model._get_frame_feat(proj[:, -1, :], None, None, None)
                    psr_pred = (
                        torch.sigmoid(psr_logits[..., : C.NUM_PSR_COMPONENTS]) > 0.3
                    ).float()
                    val_psr_tp += (psr_pred * psr_labels[:, -1, :]).sum().item()
                    val_psr_fp += (psr_pred * (1 - psr_labels[:, -1, :])).sum().item()
                    val_psr_fn += ((1 - psr_pred) * psr_labels[:, -1, :]).sum().item()

            act_acc = val_act_correct / max(val_act_total, 1)
            psr_prec = val_psr_tp / max(val_psr_tp + val_psr_fp, 1)
            psr_rec = val_psr_tp / max(val_psr_tp + val_psr_fn, 1)
            psr_f1 = 2 * psr_prec * psr_rec / max(psr_prec + psr_rec, 1e-8)

            logger.info(
                f"Epoch {epoch + 1:3d}: "
                f"train_act_loss={train_act_loss / len(train_loader):.4f}, "
                f"train_psr_loss={train_psr_loss / len(train_loader):.4f}, "
                f"val_act_acc={act_acc:.4f}, val_psr_f1={psr_f1:.4f}"
            )

            # Save best
            if act_acc > best_act_acc:
                best_act_acc = act_acc
                torch.save(act_model.state_dict(), self.output_dir / "best_activity_head.pth")

            if psr_f1 > best_psr_f1:
                best_psr_f1 = psr_f1
                torch.save(psr_model.state_dict(), self.output_dir / "best_psr_head.pth")

        # Save final
        torch.save(act_model.state_dict(), self.output_dir / "final_activity_head.pth")
        torch.save(psr_model.state_dict(), self.output_dir / "final_psr_head.pth")

        logger.info(
            f"[CacheTrainer] Done. Best act_acc={best_act_acc:.4f}, best_psr_f1={best_psr_f1:.4f}"
        )


# ============================================================================
# Main: Cache embeddings from a trained checkpoint
# ============================================================================
def cache_embeddings(
    ckpt_path: str,
    output_dir: str,
    split: str = "val",
    max_batches: Optional[int] = None,
    subset_ratio: float = 1.0,
):
    """Run the frozen backbone over the dataset and cache embeddings."""
    import data as _ds_module
    from src.models.model import POPWMultiTaskModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = POPWMultiTaskModel(pretrained=False).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model", ckpt)
    state = {k.replace("ema.", ""): v for k, v in state.items() if not k.startswith("ema.")}
    model.load_state_dict(state, strict=False)
    model.eval()

    # Freeze backbone + FPN (only extract features)
    for n, p in model.named_parameters():
        if "backbone" in n or "fpn" in n:
            p.requires_grad = False

    # Dataset
    ds = _ds_module.IndustRealMultiTaskDataset(
        split=split,
        img_size=C.IMG_SIZE,
        augment=False,
        seed=C.SEED,
    )
    collate_fn = _ds_module.collate_fn
    loader = DataLoader(
        ds, batch_size=8, shuffle=False, num_workers=4, collate_fn=collate_fn, pin_memory=True
    )

    # Cache
    cache = EmbeddingCache(output_dir, mode="write")
    cache.open()

    logger.info(f"[Cache] Processing {len(ds)} frames from {split} split...")

    with torch.no_grad():
        for bi, batch in enumerate(tqdm(loader, total=max_batches or len(loader))):
            if max_batches and bi >= max_batches:
                break

            images = batch[0].to(device)
            targets = batch[1]
            clip_rgb = targets.get("clip_rgb")
            if clip_rgb is not None and isinstance(clip_rgb, torch.Tensor):
                clip_rgb = clip_rgb.to(device)

            outputs = model(images, clip_rgb=clip_rgb)

            # Extract features to cache
            proj_feat = outputs.get("proj_feat")
            det_conf = outputs.get("det_conf")
            c5_gap = F.adaptive_avg_pool2d(outputs.get("c5_mod"), 1).flatten(1)
            p4_gap = F.adaptive_avg_pool2d(outputs.get("p4"), 1).flatten(1)

            for i in range(images.shape[0]):
                meta = targets["metadata"][i]
                rec_id = meta.get("recording_id", f"unknown_{i}")
                frame_idx = meta.get("frame_idx", i)
                camera = meta.get("camera_view", "c1")
                activity = targets.get(
                    "activity", torch.zeros(images.shape[0], dtype=torch.long).to(device)
                )[i].item()
                psr = targets.get(
                    "psr_labels", torch.zeros(images.shape[0], C.NUM_PSR_COMPONENTS).to(device)
                )[i]

                cache.add_frame(
                    rec_id,
                    proj_feat[i],
                    det_conf[i],
                    c5_gap[i],
                    p4_gap[i],
                    activity,
                    psr,
                    frame_idx,
                    camera,
                )

    cache.flush_all()
    cache.close()
    logger.info(f"[Cache] Done. {cache.total_frames()} frames cached to {output_dir}")


# ============================================================================
# CLI
# ============================================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Tier 2.4: Embedding Cache Pipeline")
    ap.add_argument("--cache", action="store_true", help="Cache embeddings from checkpoint")
    ap.add_argument("--train", action="store_true", help="Train temporal heads from cache")
    ap.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint")
    ap.add_argument("--cache-dir", type=str, default="runs/cache", help="Cache directory")
    ap.add_argument(
        "--output-dir", type=str, default="runs/cache_training", help="Output for trained heads"
    )
    ap.add_argument("--seq-len", type=int, default=64, help="Temporal sequence length")
    ap.add_argument("--epochs", type=int, default=50, help="Training epochs")
    ap.add_argument("--batch-size", type=int, default=128, help="Batch size")
    ap.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    ap.add_argument("--split", type=str, default="val", help="Dataset split to cache")
    args = ap.parse_args()

    if args.cache:
        cache_embeddings(args.ckpt, args.cache_dir, split=args.split)
    elif args.train:
        trainer = CacheTrainer(
            args.ckpt,
            args.cache_dir,
            args.output_dir,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            lr=args.lr,
        )
        trainer.train_heads(num_epochs=args.epochs)
    else:
        ap.print_help()
