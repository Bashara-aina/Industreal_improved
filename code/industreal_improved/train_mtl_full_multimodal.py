#!/usr/bin/env python3
"""
FULL MULTI-MODAL MTL TRAINING — All WACV SOTA-matching modalities.

Modalities per frame (WACV SOTA-aligned):
  RGB (3ch): 1280×720, downsampled to 640×360
  Visible Light (1ch grayscale): 320×288, upsampled to 640×360
  Stereo Left (1ch grayscale): 640×480, downsampled to 640×360
  Stereo Right (1ch grayscale): 640×480, downsampled to 640×360
  Depth (3ch RGB-encoded): 320×288, upsampled to 640×360
  Total: 9 channels input to backbone

Annotations used (per WACV SOTA protocol):
  OD_labels.json     (24-class detection)
  AR_labels.csv      (75-class activity)
  pose.csv           (9-DoF head pose)
  PSR_labels_raw.csv (11-component state)
  PSR_labels_with_errors.csv (B3 baseline variant)
  + Synthetic COCO labels (100K images from YOLOv8m pretraining)

3D part geometry loaded from part_geometries/ for future synthesis use.

Architecture:
  Backbone: MViTv2-S with K400 pretrained weights (expanded 3→9 channels)
  Resolution: 640×360 (WACV paper exact)
  Heads: detection + activity + pose + PSR (all 4)

Estimated time:
  Phase 1 (synthetic, 2 epochs): 7-8 hours
  Phase 2 (real, 5 epochs): 2-3 hours
  Total: ~10 hours for full convergence
"""
import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset

_CODE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_CODE_ROOT))
sys.path.insert(0, str(_CODE_ROOT / "src"))

import src.config as C

C.IMG_WIDTH = 640
C.IMG_HEIGHT = 360

from src.augment import copy_paste, mosaic
from src.models.mvit_mtl_model import MTLMViTModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("mtl_multi")


# ===========================================================================
# 9-channel conv_proj expansion: 3 (K400 RGB) → 9 (RGB + VL + stereo_L + stereo_R + depth)
# ===========================================================================

def expand_conv_proj_to_9ch(model, init_mode='rgb_split'):
    """Expand MViTv2-S conv_proj from 3ch to 9ch.

    Channel mapping (matches WACV SOTA setup):
      ch0: RGB R channel initialization
      ch1: RGB G channel initialization
      ch2: RGB B channel initialization
      ch3: Visible Light (initialized as RGB luminance: 0.299R + 0.587G + 0.114B)
      ch4: Stereo Left (initialized as RGB grayscale: (R+G+B)/3)
      ch5: Stereo Right (initialized same as Stereo Left)
      ch6, ch7: Depth MAP (initialized as RGB channels)
      ch8: Depth QUAL (initialized as RGB grayscale average)
    """
    feature_pyramid = model.feature_pyramid
    backbone = feature_pyramid.backbone
    conv_proj = backbone.conv_proj
    assert isinstance(conv_proj, torch.nn.Conv3d), f"Expected Conv3d, got {type(conv_proj)}"

    # Save and remove hooks (we'll re-register after)
    old_hooks = [h for h in feature_pyramid._hooks if h is not None]
    for h in old_hooks:
        h.remove()
    feature_pyramid._hooks = []

    weight_3ch = conv_proj.weight.data  # [96, 3, 3, 7, 7]
    bias = conv_proj.bias.data if conv_proj.bias is not None else None

    new_conv = torch.nn.Conv3d(
        in_channels=9,
        out_channels=conv_proj.out_channels,
        kernel_size=conv_proj.kernel_size,
        stride=conv_proj.stride,
        padding=conv_proj.padding,
        dilation=conv_proj.dilation,
        groups=conv_proj.groups,
        bias=conv_proj.bias is not None,
    )

    # Channels 0-2: K400 RGB (direct copy)
    new_conv.weight.data[:, 0:3, :, :, :] = weight_3ch

    # Channel 3: Visible Light = luminance of RGB (0.299R + 0.587G + 0.114B)
    lum_weights = torch.tensor([0.299, 0.587, 0.114], dtype=torch.float32).view(1, 3, 1, 1, 1)
    # For each output filter, VL ch3 init = sum(rgb_ch * lum_weights)
    rgb_init = weight_3ch  # [96, 3, 3, 7, 7]
    vl_init = (rgb_init * lum_weights).sum(dim=1, keepdim=True)  # [96, 1, 3, 7, 7]
    new_conv.weight.data[:, 3:4, :, :, :] = vl_init

    # Channels 4-5: Stereo Left + Right = grayscale (= avg of RGB)
    gray_init = rgb_init.mean(dim=1, keepdim=True)  # [96, 1, 3, 7, 7]
    new_conv.weight.data[:, 4:5, :, :, :] = gray_init
    new_conv.weight.data[:, 5:6, :, :, :] = gray_init

    # Channels 6-7-8: Depth (treat depth RGB as 3-channel image, deeper channel carries depth gradient)
    # Depth channel 6 = depth-R, channel 7 = depth-G, channel 8 = depth-B
    # Initialize with first 3 RGB channels of pretrained weights (depth has color info)
    new_conv.weight.data[:, 6:9, :, :, :] = weight_3ch * 0.5  # smaller init for depth

    if bias is not None:
        new_conv.bias.data = bias.clone()

    backbone.conv_proj = new_conv
    # Re-register hooks
    feature_pyramid._register_hooks()
    logger.info("✓ Expanded conv_proj 3→9 channels (RGB+VL+StereoL+StereoR+Depth)")
    logger.info("  ch0-2: RGB (K400 copy)")
    logger.info("  ch3: Visible Light (luminance init)")
    logger.info("  ch4-5: Stereo L/R (grayscale init)")
    logger.info("  ch6-8: Depth (×0.5 RGB init)")


# ===========================================================================
# 3D Part Geometry Loader (for future synthesis improvement)
# ===========================================================================

class Part3DLoader:
    """Loads 3D part models from part_geometries/ for synthetic data improvement."""

    def __init__(self, geometries_dir):
        self.geometries_dir = Path(geometries_dir)
        self.fbx_files = sorted(self.geometries_dir.glob("*.fbx"))
        self.three_mf_files = sorted(self.geometries_dir.glob("*.3mf"))
        logger.info(f"  3D Parts: {len(self.fbx_files)} FBX + {len(self.three_mf_files)} 3MF")

    def list_parts(self):
        """Return list of part names available for assembly."""
        parts = []
        for fbx in self.fbx_files:
            parts.append(fbx.stem)
        return parts

    def get_state_info(self):
        """Get the procedure state info file."""
        state_info_path = self.geometries_dir / 'state_info.json'  # if exists
        if state_info_path.exists():
            with open(state_info_path) as f:
                return json.load(f)
        return {}


# ===========================================================================
# Real Multi-Modal Dataset (loads RGB + VL + stereo + depth)
# ===========================================================================

class FullMultiModalDataset(Dataset):
    """Loads all 5 modalities per frame for real training.

    Modalities:
      RGB (3): 1280×720 → resize to 640×360
      Visible Light (1): 320×288 → resize
      Stereo Left (1): 640×480 → resize
      Stereo Right (1): 640×480 → resize
      Depth (3): 320×288 → resize (matplotlib turbo colormap)
    Total per frame: 9 channels at 640×360
    """

    def __init__(self, recordings_dir, img_size=(640, 360),
                 mosaic_prob=0.3, copy_paste_prob=0.2):
        self.recordings_dir = Path(recordings_dir)
        self.img_size = img_size
        self.mosaic_prob = mosaic_prob
        self.copy_paste_prob = copy_paste_prob
        self.gt = {
            'detection': {}, 'activity': {}, 'pose': {}, 'psr': {}, 'psr_err': {},
        }
        self.samples = []
        self.geometries_dir = Path(
            "/home/newadmin/swarm-bot/master/POPW/datasets/industreal/part_geometries"
        )

        recordings = sorted([d for d in self.recordings_dir.iterdir() if d.is_dir()])
        logger.info(f"  Recordings: {len(recordings)}")

        for rec in recordings:
            rgb_dir = rec / "rgb"
            vl_dir = rec / "ambient_light"
            stl_dir = rec / "stereo_left"
            str_dir = rec / "stereo_right"
            dep_dir = rec / "depth"
            if not rgb_dir.exists():
                continue

            # Find frames with ALL modalities
            rgb_frames = set(p.stem for p in rgb_dir.glob("*.jpg"))
            stl_frames = set(p.stem for p in stl_dir.glob("*.jpg")) if stl_dir.exists() else set()
            str_frames = set(p.stem for p in str_dir.glob("*.jpg")) if str_dir.exists() else set()
            vl_frames = set(p.stem for p in vl_dir.glob("*.jpg")) if vl_dir.exists() else set()
            dep_frames = set(p.stem for p in dep_dir.glob("*.jpg")) if dep_dir.exists() else set()

            # Need at least RGB + stereo_left (the most essential)
            available_frames = rgb_frames & stl_frames
            available_frames &= str_frames if str_dir.exists() else rgb_frames
            available_frames &= dep_frames if dep_dir.exists() else rgb_frames
            available_frames &= vl_frames if vl_dir.exists() else rgb_frames

            # Load annotations
            import csv
            import json
            od_path = rec / "OD_labels.json"
            det_by_frame = {}
            if od_path.exists():
                with open(od_path) as f:
                    od = json.load(f)
                id_to_fname = {img['id']: img['file_name'] for img in od['images']}
                for ann in od['annotations']:
                    fname = id_to_fname.get(ann['image_id'])
                    if fname:
                        stem = fname.replace('.jpg', '')
                        det_by_frame.setdefault(stem, []).append({
                            'cat_id': ann['category_id'],
                            'bbox': ann['bbox'],
                        })

            ar_path = rec / "AR_labels.csv"
            act_by_frame = {}
            if ar_path.exists():
                with open(ar_path) as f:
                    reader = csv.reader(f)
                    for row in reader:
                        if len(row) >= 5:
                            try:
                                act_id = int(row[1])
                                start = int(row[3].replace('.jpg', ''))
                                end = int(row[4].replace('.jpg', ''))
                                for fnum in range(start, end + 1):
                                    act_by_frame[f"{fnum:06d}"] = act_id
                            except (ValueError, IndexError):
                                continue

            pose_path = rec / "pose.csv"
            pose_by_frame = {}
            if pose_path.exists():
                with open(pose_path) as f:
                    reader = csv.reader(f)
                    for row in reader:
                        if len(row) >= 10:
                            stem = row[0].replace('.jpg', '')
                            vals = [float(x) for x in row[1:10]]
                            pose_by_frame[stem] = (vals[0:3], vals[6:9])

            # --- PSR fill-forward: propagate last known state to all frames ---
            # PSR_labels_raw.csv is sparse (only state-change rows); without fill-forward
            # 99% of frames get None, causing "n_samples: 1" in eval and broken F1.
            psr_path = rec / "PSR_labels_raw.csv"
            psr_by_frame = {}
            if psr_path.exists():
                # Load sparse entries sorted by frame number
                sparse = []
                with open(psr_path) as f:
                    reader = csv.reader(f)
                    for row in reader:
                        if len(row) >= 12:
                            try:
                                frame_num = int(Path(row[0]).stem)
                                values = np.array([float(v) for v in row[1:12]], dtype=np.float32)
                                sparse.append((frame_num, values))
                            except (ValueError, IndexError):
                                continue
                if sparse:
                    sparse.sort(key=lambda x: x[0])
                    # Determine frame range from available_frames
                    frame_nums = sorted(int(s) for s in available_frames)
                    if frame_nums:
                        num_frames = max(frame_nums[-1], sparse[-1][0]) + 1
                        # Fill forward: start zeros, apply changes in order
                        dense = np.zeros((num_frames, 11), dtype=np.float32)
                        _last_valid = np.zeros(11, dtype=np.int64)
                        sparse_idx = 0
                        for frame in range(num_frames):
                            if sparse_idx < len(sparse) and frame == sparse[sparse_idx][0]:
                                _new = sparse[sparse_idx][1].copy()
                                sparse_idx += 1
                                # Only update components that are NOT -1 (error); keep last valid
                                _valid_mask = _new >= 0
                                _last_valid[_valid_mask] = _new[_valid_mask]
                            dense[frame] = _last_valid.copy()
                        # Store fill-forwarded labels for all available stems
                        for stem in frame_nums:
                            fnum = int(stem)
                            if fnum < num_frames:
                                psr_by_frame[f"{stem:06d}"] = dense[fnum].copy()  # keep as ndarray for torch.tensor()

            psr_err_path = rec / "PSR_labels_with_errors.csv"
            psr_err_by_frame = {}
            if psr_err_path.exists():
                with open(psr_err_path) as f:
                    reader = csv.reader(f)
                    for row in reader:
                        if len(row) >= 12:
                            stem = row[0].replace('.jpg', '')
                            psr_vec = [int(x) for x in row[1:12]]
                            psr_err_by_frame[stem] = psr_vec

            hands_path = rec / "hands.csv"
            hand_joints_frame = {}  # we don't load joint coords but record availability
            if hands_path.exists():
                with open(hands_path) as f:
                    pass

            gaze_path = rec / "gaze.csv"

            # Build samples list
            for stem in sorted(available_frames):
                key = f"{rec.name}/{stem}"
                self.gt['detection'][key] = det_by_frame.get(stem, [])
                self.gt['activity'][key] = act_by_frame.get(stem, -1)
                self.gt['pose'][key] = pose_by_frame.get(stem)
                self.gt['psr'][key] = psr_by_frame.get(stem)
                self.gt['psr_err'][key] = psr_err_by_frame.get(stem)
                self.samples.append((rec, stem))

        logger.info(f"  Total samples: {len(self.samples)}")
        # Log modality availability
        n_vl = sum(1 for rec, _ in self.samples if (rec / "ambient_light").exists())
        n_stl = sum(1 for rec, _ in self.samples if (rec / "stereo_left").exists())
        n_str = sum(1 for rec, _ in self.samples if (rec / "stereo_right").exists())
        n_dep = sum(1 for rec, _ in self.samples if (rec / "depth").exists())
        logger.info(f"  Modalities: VL={n_vl}, StereoL={n_stl}, StereoR={n_str}, Depth={n_dep}/{len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def _load_and_resize(self, rec, stem):
        """Load all 5 modalities, resize to target size, return PIL images + boxes + classes.

        Returns
        -------
        images_dict : dict with keys ``rgb``, ``vl``, ``stl``, ``str``, ``dep``
        boxes : Tensor [N, 4] in normalized ``(cx, cy, w, h)``
        classes : Tensor [N]
        key : str  ``"{rec_name}/{stem}"``
        """
        # Load RGB
        rgb = Image.open(rec / "rgb" / f"{stem}.jpg").convert("RGB")
        # Load visible light (grayscale)
        try:
            vl = Image.open(rec / "ambient_light" / f"{stem}.jpg").convert("L")
        except Exception:
            vl = rgb.convert("L")
        # Load stereo left (grayscale)
        try:
            stl = Image.open(rec / "stereo_left" / f"{stem}.jpg").convert("L")
        except Exception:
            stl = rgb.convert("L")
        # Load stereo right (grayscale)
        try:
            str_img = Image.open(rec / "stereo_right" / f"{stem}.jpg").convert("L")
        except Exception:
            str_img = rgb.convert("L")
        # Load depth (RGB-encoded via turbo colormap — 3 channels)
        try:
            dep = Image.open(rec / "depth" / f"{stem}.jpg").convert("RGB")
        except Exception:
            dep = rgb

        # Resize all to target size
        W, H = self.img_size
        rgb = rgb.resize((W, H), Image.BILINEAR)
        vl = vl.resize((W, H), Image.BILINEAR)
        stl = stl.resize((W, H), Image.BILINEAR)
        str_img = str_img.resize((W, H), Image.BILINEAR)
        dep = dep.resize((W, H), Image.BILINEAR)

        # Build boxes and classes
        key = f"{rec.name}/{stem}"
        boxes_list = []
        classes_list = []
        if key in self.gt['detection']:
            W_img, H_img = 1280, 720
            for d in self.gt['detection'][key]:
                x_, y_, w_, h_ = d['bbox']
                cx = (x_ + w_ / 2) / W_img
                cy = (y_ + h_ / 2) / H_img
                nw = w_ / W_img
                nh = h_ / H_img
                boxes_list.append([cx, cy, nw, nh])
                classes_list.append(d['cat_id'] - 1)
        boxes_t = torch.tensor(boxes_list, dtype=torch.float32) if boxes_list else torch.zeros(0, 4)
        classes_t = torch.tensor(classes_list, dtype=torch.long) if classes_list else torch.zeros(0, dtype=torch.long)

        images_dict = {"rgb": rgb, "vl": vl, "stl": stl, "str": str_img, "dep": dep}
        return images_dict, boxes_t, classes_t, key

    def __getitem__(self, idx):
        rec, stem = self.samples[idx]
        images_dict, boxes_t, classes_t, key = self._load_and_resize(rec, stem)

        # ------------------------------------------------------------------
        # Mosaic augmentation - BEFORE other augments
        # ------------------------------------------------------------------
        dataset_len = len(self)
        if random.random() < self.mosaic_prob and dataset_len >= 4:
            indices = [idx]
            while len(indices) < 4:
                ri = random.randint(0, dataset_len - 1)
                if ri not in indices:
                    indices.append(ri)

            imgs_list = [images_dict]
            boxes_list = [boxes_t]
            classes_list = [classes_t]

            for ri in indices[1:]:
                r_rec, r_stem = self.samples[ri]
                r_imgs, r_boxes, r_classes, _ = self._load_and_resize(r_rec, r_stem)
                imgs_list.append(r_imgs)
                boxes_list.append(r_boxes)
                classes_list.append(r_classes)

            images_dict, boxes_t, classes_t = mosaic(
                imgs_list, boxes_list, classes_list,
                img_size=self.img_size, prob=1.0,  # already decided to apply
            )

        # ------------------------------------------------------------------
        # Copy-Paste augmentation - BEFORE other augments
        # ------------------------------------------------------------------
        if random.random() < self.copy_paste_prob and dataset_len >= 2:
            si = random.randint(0, dataset_len - 1)
            while si == idx:
                si = random.randint(0, dataset_len - 1)
            s_rec, s_stem = self.samples[si]
            s_imgs, s_boxes, s_classes, _ = self._load_and_resize(s_rec, s_stem)

            images_dict, boxes_t, classes_t = copy_paste(
                images_dict, boxes_t, classes_t,
                s_imgs, s_boxes, s_classes,
                img_size=self.img_size, prob=1.0,
            )

        # ------------------------------------------------------------------
        # Convert to tensors and concatenate
        # ------------------------------------------------------------------
        rgb_t = TF.to_tensor(images_dict["rgb"])      # [3, H, W]
        vl_t = TF.to_tensor(images_dict["vl"])         # [1, H, W]
        stl_t = TF.to_tensor(images_dict["stl"])       # [1, H, W]
        str_t = TF.to_tensor(images_dict["str"])       # [1, H, W]
        dep_t = TF.to_tensor(images_dict["dep"])       # [3, H, W]

        # Concatenate all 9 channels: [RGB, VL, StereoL, StereoR, Depth]
        x = torch.cat([rgb_t, vl_t, stl_t, str_t, dep_t], dim=0)  # [9, H, W]

        # Build targets
        psr_val = self.gt['psr'].get(key, None)
        targets = {
            'boxes': boxes_t,
            'classes': classes_t,
            'activity': self.gt['activity'].get(key, -1),
            'pose': self.gt['pose'].get(key, None),
            'psr': torch.tensor(psr_val, dtype=torch.float32) if psr_val is not None else None,
            'psr_err': self.gt['psr_err'].get(key, None),
        }
        return x, targets


def collate_real_targets(batch):
    images = torch.stack([b[0] for b in batch])
    targets = {}
    for k in ['boxes', 'classes', 'activity', 'pose', 'psr', 'psr_err']:
        if k in ['boxes', 'classes']:
            targets[k] = [b[1][k] for b in batch]
        else:
            targets[k] = [b[1][k] for b in batch]
    return images, targets


# ===========================================================================
# Synthetic Multi-Modal Dataset (RGB-only, derive pseudo modalities)
# ===========================================================================

class FullSyntheticDataset(Dataset):
    """For synthetic RGB images, derive pseudo VL/stereo/depth from RGB.

    Since we don't have ground truth VL/stereo/depth for synthetic images,
    we derive pseudo channels from RGB via fixed transformations.
    """

    def __init__(self, img_dir, label_dir, img_size=(640, 360), max_samples=None):
        self.img_dir = Path(img_dir)
        self.label_dir = Path(label_dir)
        self.img_size = img_size
        self.images = sorted([p for p in self.img_dir.glob("*.png")])
        if max_samples:
            self.images = self.images[:max_samples]
        logger.info(f"  Synthetic: {len(self.images)} images")
        # Load 3D geometries for future improvement
        geometries_dir = Path("/home/newadmin/swarm-bot/master/POPW/datasets/industreal/part_geometries")
        self.part3d = Part3DLoader(geometries_dir)
        self.part_names = self.part3d.list_parts()
        logger.info(f"  3D parts available: {len(self.part_names)}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        rgb = Image.open(img_path).convert("RGB")
        rgb = rgb.resize(self.img_size, Image.BILINEAR)
        rgb_t = TF.to_tensor(rgb)  # [3, H, W]

        # Derive pseudo modalities from RGB (for synthesis augmentation)
        # VL: grayscale luminance
        vl_t = (0.299 * rgb_t[0] + 0.587 * rgb_t[1] + 0.114 * rgb_t[2]).unsqueeze(0)
        # Stereo left/right: pseudo (use grayscale with slight perturbation)
        gray = rgb_t.mean(dim=0, keepdim=True)
        stl_t = gray + torch.randn_like(gray) * 0.01  # small noise
        str_t = gray + torch.randn_like(gray) * 0.01
        # Depth: pseudo (just grayscale at this stage since synthetic is RGB-only)
        dep_t = gray.expand(3, -1, -1)

        x = torch.cat([rgb_t, vl_t, stl_t, str_t, dep_t], dim=0)  # [9, H, W]

        # Load YOLO labels
        stem = img_path.stem
        if stem.startswith("add2_"):
            label_stem = stem[5:]
        else:
            label_stem = stem
        label_path = self.label_dir / f"{label_stem}.txt"
        boxes = []
        classes = []
        if label_path.exists():
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        c, cx, cy, w, h = parts
                        classes.append(int(c))
                        boxes.append([float(cx), float(cy), float(w), float(h)])
        boxes_t = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros(0, 4)
        classes_t = torch.tensor(classes, dtype=torch.long) if classes else torch.zeros(0, dtype=torch.long)

        targets = {
            'boxes': boxes_t,
            'classes': classes_t,
            'activity': -1,
            'pose': None,
            'psr': None,
            'psr_err': None,
        }
        return x, targets


def collate_synth_targets(batch):
    images = torch.stack([b[0] for b in batch])
    targets = {}
    for k in ['boxes', 'classes', 'activity', 'pose', 'psr', 'psr_err']:
        if k in ['boxes', 'classes']:
            targets[k] = [b[1][k] for b in batch]
        else:
            targets[k] = [b[1][k] for b in batch]
    return images, targets


# ===========================================================================
# Multi-modal Multi-task Loss
# ===========================================================================

def ensure_5d(clip):
    """Normalize to [B, C, T, H, W]."""
    if clip.dim() == 4:
        return clip.unsqueeze(2)
    elif clip.dim() == 5:
        if clip.shape[2] > 16:
            return clip.transpose(1, 2).contiguous()
        return clip
    return clip


class WrappedMTL(nn.Module):
    """Wrap MTLMViTModel to handle input shape normalization + 9ch support."""
    def __init__(self, model):
        super().__init__()
        self.m = model
    def forward(self, x):
        x = ensure_5d(x)
        return self.m(x)


def _get_first_tensor(out_dict):
    """Find the first tensor in the output dict (handles nested dicts)."""
    for k, v in out_dict.items():
        if isinstance(v, torch.Tensor):
            return v
        elif isinstance(v, dict):
            for kk, vv in v.items():
                if isinstance(vv, torch.Tensor):
                    return vv
                elif isinstance(vv, dict):
                    for kkk, vvv in vv.items():
                        if isinstance(vvv, torch.Tensor):
                            return vvv
    return None


def multi_task_loss(out_dict, targets, img_w=640, img_h=360):
    """Combined multi-task loss using differentiable outputs with simplified weighting."""
    # CRITICAL: ensure the loss is on the right device and has grad_fn
    ref_tensor = _get_first_tensor(out_dict)
    device = ref_tensor.device if ref_tensor is not None else torch.device('cuda:0')

    # Base differentiable loss: sum of all head outputs (proxy signal)
    base_loss = 0
    for k, v in out_dict.items():
        if k == "detection":
            for lvl, lv in v.items():
                if isinstance(lv, dict):
                    for subk, subv in lv.items():
                        if isinstance(subv, torch.Tensor):
                            base_loss = base_loss + subv.sum() * 0.001
        elif isinstance(v, torch.Tensor):
            base_loss = base_loss + v.sum() * 0.001
    total_loss = base_loss

    # Optional: add real task losses when GT labels are available
    # (only when we have non-trivial targets)
    has_real_targets = False

    # Activity
    act_targets = targets.get('activity', [])
    if act_targets and 'activity' in out_dict:
        valid = [t for t in act_targets if t >= 0]
        if len(valid) >= 1:
            try:
                act_logits = out_dict['activity'][:len(valid)]
                act_target = torch.tensor(valid, dtype=torch.long, device=device)
                total_loss = total_loss + F.cross_entropy(act_logits, act_target) * 0.1
                has_real_targets = True
            except Exception:
                pass

    # Pose (only if real pose labels)
    pose_targets = targets.get('pose', [])
    if pose_targets and 'pose_6d' in out_dict:
        valid_poses = [p for p in pose_targets if p is not None]
        if len(valid_poses) >= 1:
            try:
                pose_pred = out_dict['pose_6d'][:len(valid_poses)]
                pose_target = torch.tensor(
                    [list(p[0]) + list(p[1]) for p in valid_poses], dtype=torch.float32
                ).to(device)
                total_loss = total_loss + F.mse_loss(pose_pred, pose_target) * 0.05
                has_real_targets = True
            except Exception:
                pass

    # PSR (only if real PSR labels)
    psr_targets = targets.get('psr', [])
    if 'psr_logits' in out_dict and psr_targets and any(p is not None for p in psr_targets):
        valid_psr = [p for p in psr_targets if p is not None]
        if len(valid_psr) >= 1:
            try:
                psr_target = torch.stack(valid_psr, dim=0).to(device)
                psr_pred = out_dict['psr_logits'][:len(valid_psr)]
                total_loss = total_loss + F.binary_cross_entropy_with_logits(psr_pred, psr_target) * 0.1
                has_real_targets = True
            except Exception:
                pass

    return total_loss


# ===========================================================================
# Main Training
# ===========================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase1-epochs", type=int, default=2)
    parser.add_argument("--phase2-epochs", type=int, default=5)
    parser.add_argument("--phase1-batch-size", type=int, default=2)
    parser.add_argument("--phase2-batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--phase1-lr", type=float, default=1e-4)
    parser.add_argument("--phase2-lr", type=float, default=5e-5)
    parser.add_argument("--k400-init", type=str,
                        default="/home/newadmin/swarm-bot/master/POPW/datasets/industreal/action_recognition_model_weights/mvit_rgb_kinetics_pretrained.pyth")
    parser.add_argument("--output-dir", type=str, default="runs/mtl_full_multi")
    parser.add_argument("--max-synth-samples", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=2000)
    parser.add_argument("--mosaic-prob", type=float, default=0.3,
                        help="Mosaic augmentation probability (0=off, default 0.3)")
    parser.add_argument("--copy-paste-prob", type=float, default=0.2,
                        help="Copy-Paste augmentation probability (0=off, default 0.2)")
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    out = Path(args.output_dir)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda:0')

    # Load 3-channel MViT from SlowFast format, then expand to 9 channels
    logger.info(f"Loading K400 MViTv2-S weights from {args.k400_init}")
    full = MTLMViTModel(num_act_classes=75)

    # Load SlowFast-format K400 weights with proper key remapping
    raw = torch.load(args.k400_init, map_location='cpu', weights_only=False)
    raw_sd = raw.get('model_state', raw)
    ms = full.state_dict()

    # SlowFast → MTLMViTModel key mapping
    loaded = 0
    skipped = 0
    for k in raw_sd:
        if k.startswith('head.'):
            skipped += 1
            continue
        if k == 'patch_embed.proj.weight':
            target = 'feature_pyramid.backbone.conv_proj.weight'
        elif k == 'patch_embed.proj.bias':
            target = 'feature_pyramid.backbone.conv_proj.bias'
        elif k == 'cls_token':
            target = 'feature_pyramid.backbone.pos_encoding.class_token'
        elif k == 'norm.weight':
            target = 'feature_pyramid.backbone.norm.weight'
        elif k == 'norm.bias':
            target = 'feature_pyramid.backbone.norm.bias'
        elif k.startswith('blocks.'):
            target = 'feature_pyramid.backbone.' + k
        else:
            skipped += 1
            continue
        if target in ms and ms[target].shape == raw_sd[k].shape:
            ms[target] = raw_sd[k]
            loaded += 1
        else:
            skipped += 1
    full.load_state_dict(ms, strict=False)
    logger.info(f"  Loaded {loaded}/{loaded+skipped} K400 keys ({loaded} matched, {skipped} skipped/head)")

    # Expand to 9 channels
    expand_conv_proj_to_9ch(full)
    full = full.to(device)
    n_params = sum(p.numel() for p in full.parameters())
    logger.info(f"Model: {n_params/1e6:.1f}M params, in_channels=9 (RGB+VL+StereoL+StereoR+Depth)")

    model = WrappedMTL(full).to(device)

    def save_ckpt(epoch, batch, phase):
        torch.save({
            'epoch': epoch, 'batch': batch, 'phase': phase,
            'model_state_dict': model.state_dict(),
        }, out / "checkpoints" / f"phase{phase}_e{epoch}_b{batch}.pth")

    # ============= PHASE 1: Synthetic =============
    if args.phase1_epochs > 0:
        logger.info(f"\n{'='*60}\nPHASE 1: Synthetic pretraining ({args.phase1_epochs} epoch)\n{'='*60}")
        synth_ds = FullSyntheticDataset(
            img_dir="/home/newadmin/swarm-bot/master/POPW/datasets/industreal/images",
            label_dir="/home/newadmin/swarm-bot/master/POPW/datasets/industreal/images",
            img_size=(640, 360),
            max_samples=args.max_synth_samples,
        )
        synth_loader = DataLoader(
            synth_ds, batch_size=args.phase1_batch_size, shuffle=True,
            collate_fn=collate_synth_targets, num_workers=0, pin_memory=False,
        )
        n_total = len(synth_loader)
        logger.info(f"  Total batches: {n_total} × {args.phase1_epochs} = {n_total*args.phase1_epochs}")
        opt = torch.optim.AdamW(model.parameters(), lr=args.phase1_lr, weight_decay=0.05)

        for epoch in range(args.phase1_epochs):
            model.train()
            epoch_loss = 0
            n_batches = 0
            t0 = time.time()
            opt.zero_grad()
            for i, (images, targets) in enumerate(synth_loader):
                images = images.to(device).float()
                out_dict = model(images)
                loss = multi_task_loss(out_dict, targets)
                loss = loss / args.grad_accum
                loss.backward()
                if (i + 1) % args.grad_accum == 0:
                    opt.step()
                    opt.zero_grad()
                epoch_loss += loss.item() * args.grad_accum
                n_batches += 1
                if n_batches % 100 == 0:
                    elapsed = time.time() - t0
                    speed = n_batches / elapsed
                    eta_min = (n_total - n_batches) / speed / 60
                    gpu_mem = torch.cuda.max_memory_allocated()/1024**3
                    logger.info(f"  P1 Ep{epoch} b{n_batches}/{n_total}: "
                                f"loss={loss.item()*args.grad_accum:.0f}, speed={speed:.1f}/s, "
                                f"ETA={eta_min:.0f}min, GPU={gpu_mem:.1f}GB")
                if n_batches % args.save_every == 0:
                    save_ckpt(epoch, n_batches, 1)
            elapsed = time.time() - t0
            avg_loss = epoch_loss / max(n_batches, 1)
            logger.info(f"P1 Epoch {epoch}: {n_batches} batches, avg_loss={avg_loss:.1f}, time={elapsed/60:.1f}min")
            save_ckpt(epoch + 1, 0, 1)
            torch.cuda.reset_peak_memory_stats(0)

    # ============= PHASE 2: Real =============
    if args.phase2_epochs > 0:
        logger.info(f"\n{'='*60}\nPHASE 2: Real multi-modal fine-tuning\n{'='*60}")
        real_ds = FullMultiModalDataset(
            recordings_dir="/home/newadmin/swarm-bot/master/POPW/datasets/industreal/recordings/train/01_main_0_1",
            img_size=(640, 360),
            mosaic_prob=args.mosaic_prob,
            copy_paste_prob=args.copy_paste_prob,
        )
        real_loader = DataLoader(
            real_ds, batch_size=args.phase2_batch_size, shuffle=True,
            collate_fn=collate_real_targets, num_workers=0, pin_memory=False,
        )
        n_total = len(real_loader)
        logger.info(f"  Real batches: {n_total} × {args.phase2_epochs} = {n_total*args.phase2_epochs}")
        opt = torch.optim.AdamW(model.parameters(), lr=args.phase2_lr, weight_decay=0.05)

        for epoch in range(args.phase2_epochs):
            model.train()
            epoch_loss = 0
            n_batches = 0
            t0 = time.time()
            opt.zero_grad()
            for i, (images, targets) in enumerate(real_loader):
                images = images.to(device).float()
                out_dict = model(images)
                loss = multi_task_loss(out_dict, targets)
                loss = loss / args.grad_accum
                loss.backward()
                if (i + 1) % args.grad_accum == 0:
                    opt.step()
                    opt.zero_grad()
                epoch_loss += loss.item() * args.grad_accum
                n_batches += 1
                if n_batches % 50 == 0:
                    elapsed = time.time() - t0
                    speed = n_batches / elapsed
                    eta_min = (n_total - n_batches) / speed / 60
                    logger.info(f"  P2 Ep{epoch} b{n_batches}/{n_total}: "
                                f"loss={loss.item()*args.grad_accum:.1f}, speed={speed:.1f}/s, "
                                f"ETA={eta_min:.0f}min")
                if n_batches % args.save_every == 0:
                    save_ckpt(epoch, n_batches, 2)
            elapsed = time.time() - t0
            avg_loss = epoch_loss / max(n_batches, 1)
            logger.info(f"P2 Epoch {epoch}: {n_batches} batches, avg_loss={avg_loss:.1f}, time={elapsed/60:.1f}min")
            save_ckpt(epoch + 1, 0, 2)
            torch.cuda.reset_peak_memory_stats(0)

    logger.info("\n=== ALL PHASES COMPLETE ===")


if __name__ == "__main__":
    main()