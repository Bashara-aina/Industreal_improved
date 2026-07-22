"""YOLOv8-style detection head for MTL model.

Replaces our anchor-based 3x3 detection head with YOLOv8's anchor-free DFL head.

Differences from YOLOv8 native:
- Accepts our MTL backbone's 256-channel FPN features (vs YOLOv8's 192/384/576)
- Channel adapter: 256 → 192/384/576 for P3/P4/P5
- Decoupled head: cv2 (reg) + cv3 (cls) like YOLOv8
- DFL (Distribution Focal Loss) for box regression
- Anchor-free decoding at inference

This drops in to replace our existing det_head while keeping the rest of the
MTL model (backbone, activity/pose/PSR heads) intact.
"""
import math
import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class ConvBN(nn.Module):
    """YOLOv8-style Conv2d + BN + SiLU."""

    def __init__(self, in_ch: int, out_ch: int, k: int = 1, s: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, k // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_ch, eps=0.001, momentum=0.03)

    def forward(self, x):
        return F.silu(self.bn(self.conv(x)))


class DFL(nn.Module):
    """Distribution Focal Loss decoder.

    Converts a learned distribution over discrete values to a single value.
    Used by YOLOv8 for box regression.
    """

    def __init__(self, reg_max: int = 16):
        super().__init__()
        self.reg_max = reg_max
        # Initialize with evenly spaced values 0..reg_max-1
        # Conv: [reg_max, 1, 1, 1] -> single value (weighted sum)
        self.conv = nn.Conv2d(reg_max, 1, 1, bias=False)
        proj = torch.arange(reg_max, dtype=torch.float32).reshape(1, reg_max, 1, 1)
        self.conv.weight.data.copy_(proj)
        self.conv.weight.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 4*reg_max, H, W] -> [B, 4, H, W]

        Computes expected value over discrete distribution for each of 4 sides.
        """
        b, c, h, w = x.shape
        # Reshape to [B, 4, reg_max, H, W]
        x = x.view(b, 4, self.reg_max, h, w)
        # Apply softmax over reg_max to get distribution
        x = x.softmax(dim=2)  # [B, 4, reg_max, H, W]

        # Apply DFL conv per side. We need to apply conv over reg_max dim.
        # Reshape to [B*4, reg_max, H, W] for conv
        x = x.reshape(b * 4, self.reg_max, h, w)
        x = self.conv(x)  # [B*4, 1, H, W]
        x = x.reshape(b, 4, h, w)
        return x


class YOLOv8DetectHead(nn.Module):
    """YOLOv8-style anchor-free DFL detection head.

    Inputs: 3 FPN levels (P3, P4, P5) at 256 channels
    Outputs: For each level, 88-channel feature map (24 classes + 4*16 reg_max)

    Architecture mirrors YOLOv8 Detect head:
    - cv2 (reg): 2 conv layers per level (256 → mid → 4*reg_max)
    - cv3 (cls): 2 conv layers per level (256 → mid → num_classes)
    - dfl: convert distribution to box coordinates
    """

    def __init__(
        self,
        in_channels: int = 256,
        num_classes: int = 24,
        reg_max: int = 16,
        channels_list: Optional[List[Tuple[int, int, int]]] = None,
    ):
        """channels_list: [(p3_in, p4_in, p5_in)] - intermediate channels per level.
        Defaults match YOLOv8m: (192, 384, 576)."""
        super().__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.no = num_classes + 4 * reg_max  # output channels per anchor
        self.stride = torch.tensor([8.0, 16.0, 32.0])  # P3/P4/P5 strides

        # Intermediate channels per level (matches YOLOv8m)
        if channels_list is None:
            self.c2 = 64   # reg head intermediate
            self.c3 = 192  # cls head intermediate
            in_p3, in_p4, in_p5 = 192, 384, 576
        else:
            in_p3, in_p4, in_p5 = channels_list

        # Channel adapters from MTL's 256-channel FPN to YOLOv8's expected dims
        self.input_adapter_p3 = ConvBN(in_channels, in_p3, k=1)
        self.input_adapter_p4 = ConvBN(in_channels, in_p4, k=1)
        self.input_adapter_p5 = ConvBN(in_channels, in_p5, k=1)

        # Reg heads (cv2): 2 conv layers per level -> 4*reg_max channels
        self.cv2 = nn.ModuleList([
            nn.Sequential(
                ConvBN(in_p3, self.c2, k=3),
                ConvBN(self.c2, self.c2, k=3),
                nn.Conv2d(self.c2, 4 * reg_max, 1),
            ),
            nn.Sequential(
                ConvBN(in_p4, self.c2, k=3),
                ConvBN(self.c2, self.c2, k=3),
                nn.Conv2d(self.c2, 4 * reg_max, 1),
            ),
            nn.Sequential(
                ConvBN(in_p5, self.c2, k=3),
                ConvBN(self.c2, self.c2, k=3),
                nn.Conv2d(self.c2, 4 * reg_max, 1),
            ),
        ])

        # Cls heads (cv3): 2 conv layers per level -> num_classes channels
        self.cv3 = nn.ModuleList([
            nn.Sequential(
                ConvBN(in_p3, self.c3, k=3),
                ConvBN(self.c3, self.c3, k=3),
                nn.Conv2d(self.c3, num_classes, 1),
            ),
            nn.Sequential(
                ConvBN(in_p4, self.c3, k=3),
                ConvBN(self.c3, self.c3, k=3),
                nn.Conv2d(self.c3, num_classes, 1),
            ),
            nn.Sequential(
                ConvBN(in_p5, self.c3, k=3),
                ConvBN(self.c3, self.c3, k=3),
                nn.Conv2d(self.c3, num_classes, 1),
            ),
        ])

        # DFL decoder (shared across levels)
        self.dfl = DFL(reg_max)

        self.strides = torch.tensor([8.0, 16.0, 32.0])

    def forward(self, feats: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass: returns dict per level with raw cls/reg outputs."""
        # feats: list of [B, 256, H, W] for P3, P4, P5
        if len(feats) != 3:
            # Handle case where only some levels are available
            raise ValueError(f'YOLOv8DetectHead expects 3 FPN levels, got {len(feats)}')

        # Adapt channels from MTL's 256 to YOLOv8's expected dims
        p3 = self.input_adapter_p3(feats[0])
        p4 = self.input_adapter_p4(feats[1])
        p5 = self.input_adapter_p5(feats[2])

        outs = {}
        for i, (cv2, cv3, feat) in enumerate(zip(self.cv2, self.cv3, [p3, p4, p5])):
            reg_out = cv2(feat)  # [B, 4*reg_max, H, W]
            cls_out = cv3(feat)  # [B, num_classes, H, W]
            outs[f'P{i+3}'] = {'cls_logits': cls_out, 'reg_preds': reg_out}
        return outs

    def decode(self, outs: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        """Decode outputs to box coordinates using DFL.

        Returns:
            List of [N, 4] boxes (x1, y1, x2, y2) in pixel coords per level
            List of [N, num_classes] class scores per level
        """
        all_boxes = []
        all_scores = []
        all_level_ids = []

        for level_name, level_out in outs.items():
            stride = float(self.strides[int(level_name[1]) - 3])
            cls_logits = level_out['cls_logits']  # [B, 24, H, W]
            reg_preds = level_out['reg_preds']    # [B, 4*16, H, W]

            B = cls_logits.shape[0]
            H, W = cls_logits.shape[2], cls_logits.shape[3]

            # Decode box offsets via DFL
            box_dfl = self.dfl(reg_preds)  # [B, 4, H, W] (l, t, r, b)
            l, t, r, b = box_dfl[:, 0], box_dfl[:, 1], box_dfl[:, 2], box_dfl[:, 3]

            # Generate grid of anchor centers
            ys = (torch.arange(H, device=cls_logits.device) + 0.5) * stride
            xs = (torch.arange(W, device=cls_logits.device) + 0.5) * stride

            # box_xyxy = center + offsets
            x1 = xs[None, None, :] - l * stride  # [B, 1, W]
            y1 = ys[None, :, None] - t * stride  # [B, H, 1]
            x2 = xs[None, None, :] + r * stride  # [B, 1, W]
            y2 = ys[None, :, None] + b * stride  # [B, H, 1]

            # Broadcast to [B, H, W]
            x1 = x1.expand(B, H, W)
            x2 = x2.expand(B, H, W)
            y1 = y1.expand(B, H, W)
            y2 = y2.expand(B, H, W)

            boxes = torch.stack([x1, y1, x2, y2], dim=1)  # [B, 4, H, W]
            boxes = boxes.permute(0, 2, 3, 1).reshape(B, -1, 4)  # [B, H*W, 4]

            # Sigmoid scores
            scores = cls_logits.sigmoid()  # [B, 24, H, W]
            scores = scores.permute(0, 2, 3, 1).reshape(B, -1, 24)  # [B, H*W, 24]

            all_boxes.append(boxes)
            all_scores.append(scores)

        return all_boxes, all_scores


def init_from_yolov8_weights(
    head: YOLOv8DetectHead,
    yolov8_weights_path: str = '/home/newadmin/swarm-bot/master/POPW/datasets/industreal/assembly_state_detection_model_weights/asd_best_IndustRealandSynthetic.pt',
):
    """Initialize YOLOv8DetectHead weights from a trained YOLOv8 model.

    Maps YOLOv8's cv2/cv3/dfl weights to our head structure.
    """
    from ultralytics import YOLO
    import torch.nn as nn

    model = YOLO(yolov8_weights_path)
    raw_model = model.model
    detect = raw_model.model[-1]

    logger.info(f'Initializing YOLOv8DetectHead from {yolov8_weights_path}')

    # YOLOv8 stride-aligned channels: 192 (P3), 384 (P4), 576 (P5)
    in_channels = [192, 384, 576]
    c2 = head.c2  # 64
    c3 = head.c3  # 192

    with torch.no_grad():
        # cv2 (reg head)
        for i, (our_cv2, yolo_cv2) in enumerate(zip(head.cv2, detect.cv2)):
            # our_cv2[0] is ConvBN, our_cv2[1] is ConvBN, our_cv2[2] is nn.Conv2d
            # yolo_cv2[0] is Conv, yolo_cv2[1] is Conv, yolo_cv2[2] is Conv2d
            # Match input channels from adapter output
            # Our cv2[0]: ConvBN(in_pN, c2) expects in_channels[i]
            # But our conv is already initialized at random
            # We need to map YOLOv8's cv2[0].conv.weight (in_channels[i] -> c2)
            our_cv2[0].conv.weight.data.copy_(
                yolo_cv2[0].conv.weight.data.clone()
            )
            our_cv2[0].bn.weight.data.copy_(yolo_cv2[0].bn.weight.data.clone())
            our_cv2[0].bn.bias.data.copy_(yolo_cv2[0].bn.bias.data.clone())
            our_cv2[0].bn.running_mean.data.copy_(yolo_cv2[0].bn.running_mean.data.clone())
            our_cv2[0].bn.running_var.data.copy_(yolo_cv2[0].bn.running_var.data.clone())

            our_cv2[1].conv.weight.data.copy_(yolo_cv2[1].conv.weight.data.clone())
            our_cv2[1].bn.weight.data.copy_(yolo_cv2[1].bn.weight.data.clone())
            our_cv2[1].bn.bias.data.copy_(yolo_cv2[1].bn.bias.data.clone())
            our_cv2[1].bn.running_mean.data.copy_(yolo_cv2[1].bn.running_mean.data.clone())
            our_cv2[1].bn.running_var.data.copy_(yolo_cv2[1].bn.running_var.data.clone())

            our_cv2[2].weight.data.copy_(yolo_cv2[2].weight.data.clone())
            our_cv2[2].bias.data.copy_(yolo_cv2[2].bias.data.clone())

        # cv3 (cls head)
        for i, (our_cv3, yolo_cv3) in enumerate(zip(head.cv3, detect.cv3)):
            our_cv3[0].conv.weight.data.copy_(yolo_cv3[0].conv.weight.data.clone())
            our_cv3[0].bn.weight.data.copy_(yolo_cv3[0].bn.weight.data.clone())
            our_cv3[0].bn.bias.data.copy_(yolo_cv3[0].bn.bias.data.clone())
            our_cv3[0].bn.running_mean.data.copy_(yolo_cv3[0].bn.running_mean.data.clone())
            our_cv3[0].bn.running_var.data.copy_(yolo_cv3[0].bn.running_var.data.clone())

            our_cv3[1].conv.weight.data.copy_(yolo_cv3[1].conv.weight.data.clone())
            our_cv3[1].bn.weight.data.copy_(yolo_cv3[1].bn.weight.data.clone())
            our_cv3[1].bn.bias.data.copy_(yolo_cv3[1].bn.bias.data.clone())
            our_cv3[1].bn.running_mean.data.copy_(yolo_cv3[1].bn.running_mean.data.clone())
            our_cv3[1].bn.running_var.data.copy_(yolo_cv3[1].bn.running_var.data.clone())

            our_cv3[2].weight.data.copy_(yolo_cv3[2].weight.data.clone())
            our_cv3[2].bias.data.copy_(yolo_cv3[2].bias.data.clone())

        # DFL is already initialized in __init__ (not trainable)

    n_params = sum(p.numel() for p in head.parameters())
    logger.info(f'YOLOv8DetectHead initialized: {n_params:,} params')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    head = YOLOv8DetectHead()
    print(f'Head created: {sum(p.numel() for p in head.parameters()):,} params')

    # Test forward
    feats = [
        torch.randn(2, 256, 45, 80),  # P3
        torch.randn(2, 256, 23, 40),  # P4
        torch.randn(2, 256, 12, 20),  # P5
    ]
    out = head(feats)
    for k, v in out.items():
        print(f'  {k}: cls={v["cls_logits"].shape}, reg={v["reg_preds"].shape}')

    # Test init from YOLOv8
    init_from_yolov8_weights(head)

    # Test decode
    boxes, scores = head.decode(out)
    print(f'Decoded: boxes[0].shape={boxes[0].shape}, scores[0].shape={scores[0].shape}')