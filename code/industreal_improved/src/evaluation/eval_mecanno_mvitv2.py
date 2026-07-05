"""T3: FULL MViTv2-S inference — SlowFast MViT matching WACV 2024 Meccano.

Implements the EXACT MViTv2 architecture from facebookresearch/SlowFast
(slowfast/models/attention.py) with combined QKV (separate_qkv=False).
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


def attention_pool(tensor, pool, thw_shape, has_cls_embed=True, norm=None):
    """3D pooling from SlowFast — matches checkpoint."""
    if pool is None:
        return tensor, thw_shape
    tensor_dim = tensor.ndim
    if tensor_dim == 4:
        pass
    elif tensor_dim == 3:
        tensor = tensor.unsqueeze(1)
    if has_cls_embed:
        cls_tok, tensor = tensor[:, :, :1, :], tensor[:, :, 1:, :]
    B, N, L, C = tensor.shape
    T, H, W = thw_shape
    tensor = tensor.reshape(B * N, T, H, W, C).permute(0, 4, 1, 2, 3).contiguous()
    tensor = pool(tensor)
    thw_shape = (tensor.shape[2], tensor.shape[3], tensor.shape[4])
    Lp = tensor.shape[2] * tensor.shape[3] * tensor.shape[4]
    tensor = tensor.reshape(B, N, C, Lp).transpose(2, 3)
    if has_cls_embed:
        tensor = torch.cat((cls_tok, tensor), dim=2)
    if norm is not None:
        tensor = norm(tensor)
    if tensor_dim == 3:
        tensor = tensor.squeeze(1)
    return tensor, thw_shape


def get_rel_pos(rel_pos, d):
    if isinstance(d, int):
        if rel_pos.shape[0] == d:
            return rel_pos
        return F.interpolate(
            rel_pos.reshape(1, -1, rel_pos.shape[1]).permute(0, 2, 1),
            size=d, mode="linear",
        ).reshape(rel_pos.shape[1], d).permute(1, 0)
    return rel_pos


def cal_rel_pos_spatial(attn, q, has_cls_embed, q_shape, k_shape, rel_pos_h, rel_pos_w):
    sp_idx = 1 if has_cls_embed else 0
    q_t, q_h, q_w = q_shape
    k_t, k_h, k_w = k_shape
    dh = int(2 * max(q_h, k_h) - 1)
    dw = int(2 * max(q_w, k_w) - 1)

    q_h_ratio = max(k_h / q_h, 1.0)
    k_h_ratio = max(q_h / k_h, 1.0)
    dist_h = (torch.arange(q_h, device=attn.device)[:, None] * q_h_ratio -
              torch.arange(k_h, device=attn.device)[None, :] * k_h_ratio) + (k_h - 1) * k_h_ratio
    q_w_ratio = max(k_w / q_w, 1.0)
    k_w_ratio = max(q_w / k_w, 1.0)
    dist_w = (torch.arange(q_w, device=attn.device)[:, None] * q_w_ratio -
              torch.arange(k_w, device=attn.device)[None, :] * k_w_ratio) + (k_w - 1) * k_w_ratio

    rel_pos_h = get_rel_pos(rel_pos_h, dh)
    rel_pos_w = get_rel_pos(rel_pos_w, dw)
    Rh = rel_pos_h[dist_h.long()]
    Rw = rel_pos_w[dist_w.long()]

    B, nh, qN, dim = q.shape
    r_q = q[:, :, sp_idx:].reshape(B, nh, q_t, q_h, q_w, dim)
    rel_h = torch.einsum("bythwc,hkc->bythwk", r_q, Rh)
    rel_w = torch.einsum("bythwc,wkc->bythwk", r_q, Rw)

    attn[:, :, sp_idx:, sp_idx:] = (
        attn[:, :, sp_idx:, sp_idx:].view(B, -1, q_t, q_h, q_w, k_t, k_h, k_w)
        + rel_h[:, :, :, :, :, None, :, None]
        + rel_w[:, :, :, :, :, None, None, :]
    ).view(B, -1, q_t * q_h * q_w, k_t * k_h * k_w)
    return attn


def cal_rel_pos_temporal(attn, q, has_cls_embed, q_shape, k_shape, rel_pos_t):
    sp_idx = 1 if has_cls_embed else 0
    q_t, q_h, q_w = q_shape
    k_t, k_h, k_w = k_shape
    dt = int(2 * max(q_t, k_t) - 1)

    rel_pos_t = get_rel_pos(rel_pos_t, dt)
    q_t_ratio = max(k_t / q_t, 1.0)
    k_t_ratio = max(q_t / k_t, 1.0)
    dist_t = (torch.arange(q_t, device=attn.device)[:, None] * q_t_ratio -
              torch.arange(k_t, device=attn.device)[None, :] * k_t_ratio) + (k_t - 1) * k_t_ratio
    Rt = rel_pos_t[dist_t.long()]

    B, nh, qN, dim = q.shape
    r_q = q[:, :, sp_idx:].reshape(B, nh, q_t, q_h, q_w, dim)
    r_q = r_q.permute(2, 0, 1, 3, 4, 5).reshape(q_t, B * nh * q_h * q_w, dim)
    rel = torch.matmul(r_q, Rt.transpose(1, 2)).transpose(0, 1)
    rel = rel.view(B, nh, q_h, q_w, q_t, k_t).permute(0, 1, 4, 2, 3, 5)

    attn[:, :, sp_idx:, sp_idx:] = (
        attn[:, :, sp_idx:, sp_idx:].view(B, -1, q_t, q_h, q_w, k_t, k_h, k_w)
        + rel[:, :, :, :, :, :, None, None]
    ).view(B, -1, q_t * q_h * q_w, k_t * k_h * k_w)
    return attn


class MultiScaleAttention(nn.Module):
    """SlowFast MultiScaleAttention with combined QKV (separate_qkv=False)."""

    def __init__(self, dim, dim_out, input_size, num_heads=1, qkv_bias=True,
                 kernel_q=(1,1,1), kernel_kv=(1,1,1),
                 stride_q=(1,1,1), stride_kv=(1,1,1),
                 has_cls_embed=True, mode="conv",
                 rel_pos_spatial=True, rel_pos_temporal=True,
                 residual_pooling=True):
        super().__init__()
        self.num_heads = num_heads
        self.dim_out = dim_out
        head_dim = dim_out // num_heads
        self.scale = head_dim ** -0.5
        self.has_cls_embed = has_cls_embed
        self.mode = mode
        self.residual_pooling = residual_pooling
        padding_q = [int(q // 2) for q in kernel_q]
        padding_kv = [int(kv // 2) for kv in kernel_kv]

        self.qkv = nn.Linear(dim, dim_out * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim_out, dim_out)

        dim_conv = head_dim
        if all(s == 1 for s in kernel_q) and all(s == 1 for s in stride_q):
            self.pool_q = None
            self.norm_q = None
        else:
            self.pool_q = nn.Conv3d(dim_conv, dim_conv, kernel_q, stride_q,
                                     padding_q, groups=dim_conv, bias=False)
            self.norm_q = nn.LayerNorm(dim_conv)

        if all(s == 1 for s in kernel_kv) and all(s == 1 for s in stride_kv):
            self.pool_k = self.pool_v = None
            self.norm_k = self.norm_v = None
        else:
            self.pool_k = nn.Conv3d(dim_conv, dim_conv, kernel_kv, stride_kv,
                                     padding_kv, groups=dim_conv, bias=False)
            self.norm_k = nn.LayerNorm(dim_conv)
            self.pool_v = nn.Conv3d(dim_conv, dim_conv, kernel_kv, stride_kv,
                                     padding_kv, groups=dim_conv, bias=False)
            self.norm_v = nn.LayerNorm(dim_conv)

        # Relative position — SlowFast style
        self.rel_pos_spatial = rel_pos_spatial
        self.rel_pos_temporal = rel_pos_temporal
        if rel_pos_spatial:
            assert input_size[1] == input_size[2]
            size = input_size[1]
            qs = size // stride_q[1] if len(stride_q) > 0 and stride_q[1] > 1 else size
            ks = size // stride_kv[1] if len(stride_kv) > 0 and stride_kv[1] > 1 else size
            rel_d = 2 * max(qs, ks) - 1
            self.rel_pos_h = nn.Parameter(torch.zeros(rel_d, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(rel_d, head_dim))
            nn.init.trunc_normal_(self.rel_pos_h, std=0.02)
            nn.init.trunc_normal_(self.rel_pos_w, std=0.02)
        if rel_pos_temporal:
            self.rel_pos_t = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            nn.init.trunc_normal_(self.rel_pos_t, std=0.02)

    def forward(self, x, thw):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q, q_thw = attention_pool(q, self.pool_q, thw, self.has_cls_embed, self.norm_q)
        k, k_thw = attention_pool(k, self.pool_k, thw, self.has_cls_embed, self.norm_k)
        v, _ = attention_pool(v, self.pool_v, thw, self.has_cls_embed, self.norm_v)

        attn = (q * self.scale) @ k.transpose(-2, -1)
        if hasattr(self, 'rel_pos_h'):
            attn = cal_rel_pos_spatial(attn, q, self.has_cls_embed, q_thw, k_thw,
                                       self.rel_pos_h, self.rel_pos_w)
        if hasattr(self, 'rel_pos_t'):
            attn = cal_rel_pos_temporal(attn, q, self.has_cls_embed, q_thw, k_thw,
                                        self.rel_pos_t)
        attn = attn.softmax(dim=-1)
        x_out = attn @ v

        if self.residual_pooling:
            if self.has_cls_embed:
                x_out[:, :, 1:, :] += q[:, :, 1:, :]
            else:
                x_out = x_out + q

        x_out = x_out.transpose(1, 2).reshape(B, -1, self.dim_out)
        x_out = self.proj(x_out)
        return x_out, q_thw


class MultiScaleBlock(nn.Module):
    def __init__(self, dim, dim_out, num_heads, input_size,
                 mlp_ratio=4.0, qkv_bias=True,
                 kernel_q=(1,1,1), kernel_kv=(1,1,1),
                 stride_q=(1,1,1), stride_kv=(1,1,1),
                 has_cls_embed=True, mode="conv",
                 rel_pos_spatial=True, rel_pos_temporal=True,
                 residual_pooling=True, dim_mul_in_att=True):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out
        self.norm1 = nn.LayerNorm(dim)
        self.dim_mul_in_att = dim_mul_in_att
        attn_dim = dim_out if dim_mul_in_att else dim

        self.attn = MultiScaleAttention(
            dim, attn_dim, input_size, num_heads=num_heads,
            qkv_bias=qkv_bias,
            kernel_q=kernel_q, kernel_kv=kernel_kv,
            stride_q=stride_q, stride_kv=stride_kv,
            has_cls_embed=has_cls_embed, mode=mode,
            rel_pos_spatial=rel_pos_spatial,
            rel_pos_temporal=rel_pos_temporal,
            residual_pooling=residual_pooling,
        )
        self.norm2 = nn.LayerNorm(attn_dim)
        self.mlp = Mlp(attn_dim, int(attn_dim * mlp_ratio), dim_out)
        self.has_cls_embed = has_cls_embed

        if dim != dim_out:
            self.proj = nn.Linear(dim, dim_out)

        # Skip path: MaxPool3d (no channel constraints)
        if any(s > 1 for s in stride_q):
            k_skip = [s + 1 if s > 1 else s for s in stride_q]
            p_skip = [k // 2 for k in k_skip]
            self.pool_skip = nn.MaxPool3d(k_skip, stride_q, p_skip, ceil_mode=False)
        else:
            self.pool_skip = None

    def forward(self, x, thw):
        x_norm = self.norm1(x)
        x_block, thw_new = self.attn(x_norm, thw)
        if self.dim_mul_in_att and self.dim != self.dim_out:
            x = self.proj(x_norm)
        x_res, _ = attention_pool(x, self.pool_skip, thw, self.has_cls_embed)
        x = x_res + x_block
        x = x + self.mlp(self.norm2(x))
        return x, thw_new


class MViTv2S(nn.Module):
    """Full MViTv2-S matching SlowFast MViT (WACV 2024 Meccano)."""

    def __init__(self, num_classes=75):
        super().__init__()
        self.patch_embed_proj = nn.Conv3d(3, 96, kernel_size=(3, 7, 7),
                                           stride=(2, 4, 4), padding=(1, 3, 3))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 96))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Block config: (dim_in, dim_out, nH, q_stride, kv_stride)
        # Input sizes computed: after patch_embed -> (T=8, H=56, W=56)
        # Then each block's Q pooling determines next input size
        configs = [
            (96,   96,   1, (1,1,1), (1,8,8)),   # 0: in=(8,56,56) Q=(8,56,56)
            (96,   192,  2, (1,2,2), (1,4,4)),   # 1: in=(8,56,56) Q=(8,28,28)
            (192,  192,  2, (1,1,1), (1,4,4)),   # 2: in=(8,28,28) Q=(8,28,28)
            (192,  384,  4, (1,2,2), (1,2,2)),   # 3: in=(8,28,28) Q=(8,14,14)
            (384,  384,  4, (1,1,1), (1,2,2)),   # 4: in=(8,14,14) Q=(8,14,14)
            (384,  384,  4, (1,1,1), (1,2,2)),   # 5
            (384,  384,  4, (1,1,1), (1,2,2)),   # 6
            (384,  384,  4, (1,1,1), (1,2,2)),   # 7
            (384,  384,  4, (1,1,1), (1,2,2)),   # 8
            (384,  384,  4, (1,1,1), (1,2,2)),   # 9
            (384,  384,  4, (1,1,1), (1,2,2)),   # 10
            (384,  384,  4, (1,1,1), (1,2,2)),   # 11
            (384,  384,  4, (1,1,1), (1,2,2)),   # 12
            (384,  384,  4, (1,1,1), (1,2,2)),   # 13
            (384,  768,  8, (1,2,2), (1,1,1)),   # 14: in=(8,14,14) Q=(8,7,7)
            (768,  768,  8, (1,1,1), (1,1,1)),   # 15: in=(8,7,7) Q=(8,7,7)
        ]

        # Track input sizes for rel_pos computation
        t, h, w = 8, 56, 56
        self.blocks = nn.ModuleList()
        for dim, dim_out, nH, qs, kvs in configs:
            input_size = (t, h, w)
            self.blocks.append(MultiScaleBlock(
                dim=dim, dim_out=dim_out, num_heads=nH,
                input_size=input_size,
                kernel_q=(3,3,3), kernel_kv=(3,3,3),
                stride_q=qs, stride_kv=kvs,
                rel_pos_spatial=True, rel_pos_temporal=True,
                residual_pooling=True, dim_mul_in_att=True,
            ))
            # Update for next block
            t = t // qs[0] if qs[0] > 1 else t
            h = h // qs[1] if qs[1] > 1 else h
            w = w // qs[2] if qs[2] > 1 else w

        self.norm = nn.LayerNorm(768)
        self.head_projection = nn.Linear(768, num_classes)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed_proj(x)
        T, H, W = x.shape[2], x.shape[3], x.shape[4]
        x = x.flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        thw = (T, H, W)
        for blk in self.blocks:
            x, thw = blk(x, thw)
        x = self.norm(x)
        x = x[:, 0]
        x = self.head_projection(x)
        return x


def load_mecanno_mvitv2_full(weights_path, device='cpu'):
    ckpt = torch.load(weights_path, map_location='cpu', weights_only=False)
    state = ckpt.get('model_state', ckpt)

    model = MViTv2S(num_classes=75)
    model.eval()

    # Key remap: only patch_embed.proj -> patch_embed_proj, head.projection -> head_projection
    own_state = model.state_dict()
    loaded = set()
    for ckpt_key, param in state.items():
        model_key = ckpt_key.replace('patch_embed.proj.', 'patch_embed_proj.')
        model_key = model_key.replace('head.projection.', 'head_projection.')
        if model_key in own_state and param.shape == own_state[model_key].shape:
            own_state[model_key].copy_(param)
            loaded.add(model_key)

    n_loaded = len(loaded)
    n_model = len(own_state)
    print(f"  Loaded {n_loaded}/{n_model} keys")
    if n_loaded < n_model:
        missing = sorted(set(own_state.keys()) - loaded)
        print(f"  Missing ({len(missing)}): {missing[:20]}")

    model = model.to(device)
    return model


# =========================================================================
# Dataset + Main
# =========================================================================

def build_clip_dataset(ar_csv, image_dir, clip_frames=16):
    pairs = []
    with open(ar_csv) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 5:
                continue
            action_id = int(parts[1])
            start_f = int(Path(parts[3]).stem)
            end_f = int(Path(parts[4]).stem)
            if action_id < 0 or end_f < start_f:
                continue
            span = end_f - start_f + 1
            if span < clip_frames:
                continue
            indices = (list(range(start_f, end_f + 1)) if span == clip_frames else
                       [start_f + int(round(i * (span - 1) / (clip_frames - 1)))
                        for i in range(clip_frames)])
            rgb_d = image_dir / parts[0] / "rgb"
            if rgb_d.exists() and all((rgb_d / f"{f:06d}.jpg").exists() for f in indices):
                pairs.append((parts[0], action_id, indices))
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--max_segments", type=int, default=0)
    parser.add_argument("--ckpt", default="/media/newadmin/master/POPW/datasets/industreal/action_recognition_model_weights/mvit_rgb_meccano_pretrained.pyth")
    parser.add_argument("--out", default="/tmp/t3_full_eval.json")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    print(f"Device: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'}")

    BASE = Path(__file__).resolve().parent.parent
    remap = json.load(open(BASE / "runs/rf_stages/checkpoints/act_remap_75_to_69.json"))
    id_to_group = remap["id_to_group"]
    num_groups = remap["num_groups"]
    print(f"Remap: 75 -> {num_groups} groups\n")

    print(f"Loading {args.ckpt}...")
    model = load_mecanno_mvitv2_full(args.ckpt, device=device)
    model.eval()
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}\n")

    from PIL import Image
    import torchvision.transforms as T
    transform = T.Compose([
        T.Resize(256), T.CenterCrop(224), T.ToTensor(),
        T.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225]),
    ])

    val_root = Path("/media/newadmin/master/POPW/datasets/industreal/recordings") / args.split
    all_pairs = []
    for rec_dir in sorted(val_root.iterdir()):
        ar_csv = rec_dir / "AR_labels.csv"
        if ar_csv.exists():
            all_pairs.extend((rec_dir, p) for p in build_clip_dataset(ar_csv, val_root))
    print(f"Clips: {len(all_pairs)} in {args.split}/")

    if args.max_segments and len(all_pairs) > args.max_segments:
        all_pairs = all_pairs[:args.max_segments]
    print(f"Evaluating {len(all_pairs)} clips...\n")

    corr75 = corr69 = total = 0
    with torch.no_grad():
        for i, (rec_dir, (rid, aid, indices)) in enumerate(all_pairs):
            rgb_dir = rec_dir / "rgb"
            frames = [transform(Image.open(rgb_dir / f"{f:06d}.jpg").convert("RGB"))
                      for f in indices]
            clip = torch.stack(frames).permute(1, 0, 2, 3).unsqueeze(0).to(device)
            logits = model(clip)
            probs = F.softmax(logits, dim=-1).squeeze().cpu().numpy()

            p75 = int(probs.argmax())
            p69 = int(np.bincount(id_to_group, weights=probs).argmax())
            g75 = aid
            g69 = id_to_group[g75] if g75 < len(id_to_group) else 0

            corr75 += int(p75 == g75)
            corr69 += int(p69 == g69)
            total += 1
            if (i + 1) % 50 == 0 or i == 0:
                print(f"  [{i+1}/{len(all_pairs)}] 75: {corr75/total*100:.1f}% 69: {corr69/total*100:.1f}%")

    print(f"\n{'='*60}")
    print(f"T3: MViTv2-S (Meccano) on {args.split}")
    print(f"{'='*60}")
    print(f"Clips: {total}")
    print(f"75-class Top-1: {corr75/total*100:.2f}%")
    print(f"69-class Top-1: {corr69/total*100:.2f}%")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump({
        "model": "MViTv2-S (SlowFast arch, WACV 2024 Meccano)",
        "total_clips": total,
        "top1_75": round(corr75 / total, 4) if total else 0,
        "top1_69": round(corr69 / total, 4) if total else 0,
    }, open(out_path, "w"), indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
