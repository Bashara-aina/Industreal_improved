"""
TrainingMonitor — Unified logging, visualization, and progress tracking for POPW
=================================================================================

Covers ALL metrics from BENCHMARK_TABLES.md (Tables 1-5):

  TABLE 1 — Activity Recognition (IKEA ASM):
    - Top-1 (RGB front view)        Target: >60.4%   (P3D)
    - Top-1 (RGB+pose front)        Target: >64.15%  (I3D RGB+pose)
    - Top-1 (all views, most relev)  Target: >80.2%   (PC3D)      [multiview]
    - Activity mcAP (csv)           Target: >84.47%  (PTMA)
    - Activity mcAP (cs)             Target: >86.99%  (PTMA)
    - Temporal Localization mAP@0.5  Target: >21.77%  (Gated SRM)
    - Phase Classification Acc@1.0   Target: >37.02%  (STEPs)
    - Temporal Order Kendall's Tau   Target: >0.91   (STEPs)
    - Top-1 (all views)             Target: >47.0%   (I3D)       [single-view fallback]

  TABLE 2 — Object Detection (IKEA ASM):
    - Object Segmentation AP@0.5    Target: >85.3%  (ResNeXt-101-FPN)
    - Object Segmentation AP (COCO) Target: >65.9%  (ResNeXt-101-FPN)
    - Object Segmentation AP@0.5    Target: >78.9%  (Mask R-CNN) [secondary]

  TABLE 3 — Pose Estimation (IKEA ASM):
    - Pose PCK@10px                 Target: >64.3%  (MaskRCNN-ft)
    - Pose PCK@0.2                  Target: >88.0%  (MaskRCNN-ft)

  TABLE 4 — IndustReal (if evaluated):
    - ASD Detection mAP@0.5        Target: >83.8%  (YOLOv8m)
    - Activity Top-1                Target: >66.45% (MViTv2 Kinetics)
    - Activity Top-5                Target: >88.43% (MViTv2 Kinetics)
    - PSR F1                        Target: >0.901  (STORM-PSR)
    - PSR POS                       Target: >0.812  (STORM-PSR)

  TABLE 5 — Efficiency:
    - Parameters (M)
    - GFLOPs
    - FPS (throughput)
    - GPU Memory (GB)

Visualization outputs every N epochs:
  - annotated_samples.png   — images with activity class + keypoints + detection bboxes
  - metrics_dashboard.png  — epoch-by-epoch scorecard vs benchmark targets
  - metrics_history.png     — all training + validation curves (train & val on same axes)
  - task_breakdown.png     — per-task loss components + bar chart
  - confusion_matrix.png   — activity classification confusion matrix
  - lr_schedule.png        — learning rate over time
  - efficiency_chart.png   — FPS, GPU memory, GFLOPs
  - per_metric_panels.png  — 18 individual metric panels with target lines

Usage:
  from training_monitor import TrainingMonitor
  monitor = TrainingMonitor(run_dir)
  monitor.log_train_step(loss_dict, lr)
  monitor.log_val_epoch(train_metrics, val_metrics, temporal_results, phase_results)
  monitor.save_viz_samples(images, targets, outputs)
  monitor.close()
"""

import json
import math
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

# ── Optional dependencies ──────────────────────────────────────────────────
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.table import Table
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark targets (from BENCHMARK_TABLES.md)
# ─────────────────────────────────────────────────────────────────────────────

IKEA_TARGETS: Dict[str, float] = {
    # Activity (Table 1)
    'act_top1_rgb_front':        0.604,   # P3D
    'act_top1_rgbpose_front':    0.6415,  # I3D RGB+pose
    'act_top1_all_views':        0.470,   # I3D (single-view fallback)
    'act_top1_multiview':        0.802,   # PC3D (multiview)
    'act_mcAP_csv':             0.8447,  # PTMA
    'act_mcAP_cs':              0.8699,  # PTMA
    'temporal_mAP50':           0.2177,  # Gated SRM
    'phase_acc_at_1':           0.3702,  # STEPs self-supervised
    'kendall_tau':              0.91,    # STEPs
    # Detection (Table 2)
    'det_AP50':                 0.853,   # ResNeXt-101-FPN
    'det_mAP_coco':             0.659,   # ResNeXt-101-FPN
    'det_AP50_maskrcnn':        0.789,   # Mask R-CNN (secondary)
    # Pose (Table 3)
    'pck_at_10px':              0.643,   # MaskRCNN-ft
    'pck_at_02':                0.880,  # MaskRCNN-ft
}

INDUSTREAL_TARGETS: Dict[str, float] = {
    'asd_mAP50':       0.838,   # YOLOv8m COCO+synth+real
    'act_top1_ind':    0.6645,  # MViTv2 Kinetics
    'act_top5_ind':    0.8843,  # MViTv2 Kinetics
    'psr_f1':          0.901,   # STORM-PSR
    'psr_pos':         0.812,   # STORM-PSR
}

EFFICIENCY_TARGETS = {
    'params_M':     12.9,  # PTMA
    'gflops':        1.96,  # PTMA
    'fps':           291.0,  # PTMA
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _safe(v: Any, fallback: float = 0.0) -> float:
    if isinstance(v, (float, np.floating)) and math.isfinite(v):
        return float(v)
    return fallback


def _make_dirs(base: Path) -> None:
    for sub in ['images', 'curves', 'confusion']:
        (base / sub).mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Class names (populated from config at init)
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_ACT_NAMES = [
    'NA', 'align_leg_screw', 'align_side_panel_holes', 'attach_drawer_back',
    'attach_drawer_side', 'attach_shelf_to_table', 'flip_shelf', 'flip_table',
    'flip_table_top', 'insert_drawer_pin', 'lay_down_back_panel',
    'lay_down_bottom_panel', 'lay_down_front_panel', 'lay_down_leg',
    'lay_down_pick_up_pin', 'lay_down_shelf', 'lay_down_side_panel',
    'lay_down_table_top', 'other', 'pick_up_back_panel', 'pick_up_bottom_panel',
    'pick_up_front_panel', 'pick_up_leg', 'pick_up_pin', 'pick_up_shelf',
    'pick_up_side_panel', 'pick_up_table_top', 'position_drawer_right_side_up',
    'push_table', 'push_table_top', 'rotate_table', 'slide_bottom_of_drawer',
    'spin_leg', 'tighten_leg',
]

KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
]

DET_CLASS_NAMES = {
    1: 'table_top', 2: 'leg', 3: 'shelf', 4: 'side_panel',
    5: 'front_panel', 6: 'bottom_panel', 7: 'rear_panel',
}


# ─────────────────────────────────────────────────────────────────────────────
# Main Monitor Class
# ─────────────────────────────────────────────────────────────────────────────

class TrainingMonitor:
    """
    Unified training monitor. Logs to:
      run_dir/
        metrics.jsonl
        tensorboard/
        visualizations/
          epoch_XXXX/
            annotated_samples.png
            metrics_dashboard.png
            metrics_history.png
            task_breakdown.png
            confusion_matrix.png
            lr_schedule.png
            efficiency_chart.png
            per_metric_panels.png
        latest_annotated_samples.png
        latest_metrics_dashboard.png
        latest_metrics_history.png
    """

    def __init__(
        self,
        run_dir: str | Path,
        log_interval: int = 10,
        save_viz_epochs: int = 5,
        num_viz_samples: int = 8,
        activity_class_names: Optional[List[str]] = None,
        enabled: bool = True,
    ):
        self.run_dir = Path(run_dir)
        self.log_interval = log_interval
        self.save_viz_epochs = save_viz_epochs
        self.num_viz_samples = num_viz_samples
        self.enabled = enabled
        self.act_class_names = activity_class_names or DEFAULT_ACT_NAMES

        if not enabled:
            return

        _make_dirs(self.run_dir)
        self.viz_dir = self.run_dir / 'visualizations'
        _make_dirs(self.viz_dir)

        # TensorBoard
        self._writer: Optional[Any] = None
        if TENSORBOARD_AVAILABLE:
            self._writer = SummaryWriter(log_dir=str(self.run_dir / 'tensorboard'))
            print(f'[TrainingMonitor] TensorBoard → {self.run_dir / "tensorboard"}')

        # History for plotting
        self._h: Dict[str, List[float]] = {
            'epoch': [],
            # Train losses
            'train_loss_total':    [],
            'train_loss_det':      [],
            'train_loss_pose':     [],
            'train_loss_act':      [],
            'train_loss_ordering': [],
            'train_loss_tma_kl':   [],
            'train_epoch_time':    [],
            # Detection
            'val_det_AP50':       [],
            'val_det_mAP_coco':   [],
            # Pose
            'val_pck_10px':       [],
            'val_pck_02':         [],
            'val_pck_005':        [],
            'val_pck_01':         [],
            # Activity
            'val_act_top1':       [],
            'val_act_top5':       [],
            'val_act_mcAP_csv':   [],
            'val_act_mcAP_cs':    [],
            'val_act_f1_present': [],
            # Temporal
            'val_kendall_tau':    [],
            'val_temporal_mAP50':  [],
            'val_phase_acc_1':    [],
            # Efficiency
            'val_fps':            [],
            'val_gflops':         [],
            'val_params_M':       [],
            'val_gpu_mem_gb':     [],
            # LR
            'lr':                 [],
        }

        self._step_count = 0
        self._epoch_count = 0
        self._last_confusion_matrix: Optional[np.ndarray] = None

        # Per-epoch sample cache (for annotated images)
        self._cached_images: Optional[torch.Tensor] = None
        self._cached_targets: Optional[Dict] = None
        self._cached_outputs: Optional[Dict] = None

    # ── Step-level logging ───────────────────────────────────────────────────

    def log_train_step(
        self,
        loss_dict: Dict[str, float],
        lr: Optional[float] = None,
        global_step: Optional[int] = None,
    ) -> None:
        if not self.enabled:
            return
        step = global_step if global_step is not None else self._step_count
        if self._writer:
            for k, v in loss_dict.items():
                self._writer.add_scalar(f'train/{k}', _safe(v), step)
            if lr is not None:
                self._writer.add_scalar('train/lr', lr, step)
        self._step_count += 1

    def cache_visualization_batch(
        self,
        images: torch.Tensor,
        targets: Dict,
        outputs: Dict,
    ) -> None:
        """Cache batch for annotated image generation at epoch end."""
        self._cached_images = images.detach().cpu()
        self._cached_targets = {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
                               for k, v in targets.items()}
        self._cached_outputs = {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
                                 for k, v in outputs.items()}

    # ── Epoch-level logging ──────────────────────────────────────────────────

    def log_val_epoch(
        self,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        temporal_results: Optional[Dict[str, float]] = None,
        phase_results: Optional[Dict[str, float]] = None,
        efficiency_results: Optional[Dict[str, float]] = None,
        lr: Optional[float] = None,
        confusion_matrix: Optional[np.ndarray] = None,
        multiview_results: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Main entry point — called once per epoch after validation.

        All metrics from BENCHMARK_TABLES.md are extracted and logged.
        """
        if not self.enabled:
            return

        self._epoch_count += 1
        epoch = self._epoch_count

        # ── Extract all metrics ─────────────────────────────────────────────
        # Train
        tr_loss_total    = _safe(train_metrics.get('total', 0))
        tr_loss_det      = _safe(train_metrics.get('det', 0))
        tr_loss_pose     = _safe(train_metrics.get('pose', 0))
        tr_loss_act      = _safe(train_metrics.get('activity', 0))
        tr_loss_order    = _safe(train_metrics.get('temporal_ordering', 0))
        tr_loss_tma_kl   = _safe(train_metrics.get('tma_kl', 0))
        tr_epoch_time    = _safe(train_metrics.get('epoch_time', 0))

        # Detection
        val_det_AP50     = _safe(val_metrics.get('det_mAP50', 0))
        val_det_mAP_coco = _safe(val_metrics.get('det_mAP', 0))

        # Pose
        val_pck_10px     = _safe(val_metrics.get('pck_at_10px', 0))
        val_pck_02       = _safe(val_metrics.get('pck_at_02', 0))
        val_pck_005      = _safe(val_metrics.get('pck_at_005', 0))
        val_pck_01       = _safe(val_metrics.get('pck_at_01', 0))

        # Activity
        val_act_top1     = _safe(val_metrics.get('act_accuracy', 0))
        val_act_top5     = _safe(val_metrics.get('act_top5', 0))
        val_act_mcAP_csv = _safe(val_metrics.get('act_mcAP_csv', val_metrics.get('act_mcAP', 0)))
        val_act_mcAP_cs  = _safe(val_metrics.get('act_mcAP_cs', 0))
        val_act_f1       = _safe(val_metrics.get('act_macro_f1_present', 0))
        val_act_top1_mv  = _safe(
            multiview_results.get('act_accuracy', 0) if multiview_results else 0
        )
        val_act_top5_mv  = _safe(
            multiview_results.get('act_top5', 0) if multiview_results else 0
        )

        # Temporal
        val_tau          = _safe(
            temporal_results.get('temporal_order_kendall_tau', 0)
            if temporal_results else val_metrics.get('temporal_order_kendall_tau', 0)
        )
        val_tmAP50       = _safe(
            temporal_results.get('temporal_mAP50', 0)
            if temporal_results else val_metrics.get('temporal_mAP50', 0)
        )

        # Phase
        val_phase_acc    = _safe(
            phase_results.get('phase_acc_at_1', 0)
            if phase_results else val_metrics.get('phase_acc_at_1', 0)
        )

        # Efficiency
        val_fps          = _safe(
            efficiency_results.get('fps', 0) if efficiency_results
            else val_metrics.get('fps', 0)
        )
        val_gflops       = _safe(
            efficiency_results.get('gflops', 0) if efficiency_results
            else val_metrics.get('gflops', 0)
        )
        val_params_M     = _safe(
            efficiency_results.get('params_M', 0) if efficiency_results
            else val_metrics.get('params_M', 0)
        )
        val_gpu_mem       = _safe(
            efficiency_results.get('gpu_mem_gb', 0) if efficiency_results
            else val_metrics.get('gpu_mem_gb', 0)
        )

        # ── Record history ─────────────────────────────────────────────────
        self._h['epoch'].append(epoch)
        self._h['lr'].append(_safe(lr, 0))
        self._h['train_loss_total'].append(tr_loss_total)
        self._h['train_loss_det'].append(tr_loss_det)
        self._h['train_loss_pose'].append(tr_loss_pose)
        self._h['train_loss_act'].append(tr_loss_act)
        self._h['train_loss_ordering'].append(tr_loss_order)
        self._h['train_loss_tma_kl'].append(tr_loss_tma_kl)
        self._h['train_epoch_time'].append(tr_epoch_time)
        self._h['val_det_AP50'].append(val_det_AP50)
        self._h['val_det_mAP_coco'].append(val_det_mAP_coco)
        self._h['val_pck_10px'].append(val_pck_10px)
        self._h['val_pck_02'].append(val_pck_02)
        self._h['val_pck_005'].append(val_pck_005)
        self._h['val_pck_01'].append(val_pck_01)
        self._h['val_act_top1'].append(val_act_top1)
        self._h['val_act_top5'].append(val_act_top5)
        self._h['val_act_mcAP_csv'].append(val_act_mcAP_csv)
        self._h['val_act_mcAP_cs'].append(val_act_mcAP_cs)
        self._h['val_act_f1_present'].append(val_act_f1)
        self._h['val_kendall_tau'].append(val_tau)
        self._h['val_temporal_mAP50'].append(val_tmAP50)
        self._h['val_phase_acc_1'].append(val_phase_acc)
        self._h['val_fps'].append(val_fps)
        self._h['val_gflops'].append(val_gflops)
        self._h['val_params_M'].append(val_params_M)
        self._h['val_gpu_mem_gb'].append(val_gpu_mem)

        if confusion_matrix is not None:
            self._last_confusion_matrix = confusion_matrix

        # ── TensorBoard ─────────────────────────────────────────────────────
        if self._writer:
            self._writer.add_scalar('epoch', epoch, epoch)
            self._writer.add_scalar('train/loss_total', tr_loss_total, epoch)
            self._writer.add_scalar('train/loss_det', tr_loss_det, epoch)
            self._writer.add_scalar('train/loss_pose', tr_loss_pose, epoch)
            self._writer.add_scalar('train/loss_act', tr_loss_act, epoch)
            self._writer.add_scalar('train/loss_ordering', tr_loss_order, epoch)
            self._writer.add_scalar('train/loss_tma_kl', tr_loss_tma_kl, epoch)
            if lr is not None:
                self._writer.add_scalar('train/lr', lr, epoch)
            # Val metrics
            self._writer.add_scalar('val/det_AP50', val_det_AP50, epoch)
            self._writer.add_scalar('val/det_mAP_coco', val_det_mAP_coco, epoch)
            self._writer.add_scalar('val/pck_10px', val_pck_10px, epoch)
            self._writer.add_scalar('val/pck_02', val_pck_02, epoch)
            self._writer.add_scalar('val/pck_005', val_pck_005, epoch)
            self._writer.add_scalar('val/act_top1', val_act_top1, epoch)
            self._writer.add_scalar('val/act_top5', val_act_top5, epoch)
            self._writer.add_scalar('val/act_mcAP_csv', val_act_mcAP_csv, epoch)
            self._writer.add_scalar('val/act_mcAP_cs', val_act_mcAP_cs, epoch)
            self._writer.add_scalar('val/act_f1_present', val_act_f1, epoch)
            self._writer.add_scalar('val/kendall_tau', val_tau, epoch)
            self._writer.add_scalar('val/temporal_mAP50', val_tmAP50, epoch)
            self._writer.add_scalar('val/phase_acc_1', val_phase_acc, epoch)
            self._writer.add_scalar('val/fps', val_fps, epoch)

        # ── JSONL record ─────────────────────────────────────────────────────
        record = {
            'epoch': epoch,
            'lr': lr,
            'train': {
                'loss_total':    tr_loss_total,
                'loss_det':      tr_loss_det,
                'loss_pose':     tr_loss_pose,
                'loss_act':      tr_loss_act,
                'loss_ordering': tr_loss_order,
                'loss_tma_kl':   tr_loss_tma_kl,
                'epoch_time_s':  tr_epoch_time,
            },
            'val': {
                'det_AP50':       val_det_AP50,
                'det_mAP_coco':   val_det_mAP_coco,
                'pck_10px':       val_pck_10px,
                'pck_02':         val_pck_02,
                'pck_005':        val_pck_005,
                'act_top1':       val_act_top1,
                'act_top5':       val_act_top5,
                'act_mcAP_csv':   val_act_mcAP_csv,
                'act_mcAP_cs':    val_act_mcAP_cs,
                'act_f1_present': val_act_f1,
                'kendall_tau':    val_tau,
                'temporal_mAP50': val_tmAP50,
                'phase_acc_1':   val_phase_acc,
                'fps':            val_fps,
                'gflops':         val_gflops,
                'params_M':       val_params_M,
                'gpu_mem_gb':     val_gpu_mem,
            },
        }
        self._write_jsonl(record)

    def _write_jsonl(self, record: Dict) -> None:
        path = self.run_dir / 'metrics.jsonl'
        with open(path, 'a') as f:
            f.write(json.dumps(record, default=str) + '\n')

    # ── Per-epoch visualization ─────────────────────────────────────────────

    def save_epoch_visualizations(self, epoch: int) -> None:
        """Generate all visualization charts for this epoch."""
        if not self.enabled or not MATPLOTLIB_AVAILABLE:
            return

        viz_sub = self.viz_dir / f'epoch_{epoch:04d}'
        viz_sub.mkdir(exist_ok=True)

        # 1. Metrics dashboard (benchmark scorecard)
        self._plot_dashboard(viz_sub, epoch)

        # 2. All metric curves (train + val where applicable)
        self._plot_metrics_history(viz_sub, epoch)

        # 3. Task loss breakdown
        self._plot_task_breakdown(viz_sub, epoch)

        # 4. LR schedule
        self._plot_lr_schedule(viz_sub, epoch)

        # 5. Efficiency
        self._plot_efficiency(viz_sub, epoch)

        # 6. Per-metric panels (18 panels covering all benchmark metrics)
        self._plot_per_metric_panels(viz_sub, epoch)

        # 7. Annotated sample images
        self._plot_annotated_samples(viz_sub, epoch)

        # 8. Confusion matrix (if available)
        if self._last_confusion_matrix is not None:
            self._plot_confusion_matrix(viz_sub, epoch, self._last_confusion_matrix)

        # Copy latest to root
        for fname in [
            'annotated_samples.png', 'metrics_dashboard.png',
            'metrics_history.png', 'task_breakdown.png',
            'per_metric_panels.png',
        ]:
            src = viz_sub / fname
            if src.exists():
                shutil.copy(src, self.viz_dir.parent / f'latest_{fname}')

        print(f'  [Monitor] Saved all visualizations → {viz_sub}')

    # ── Dashboard (benchmark scorecard) ─────────────────────────────────────

    def _plot_dashboard(self, viz_dir: Path, epoch: int) -> None:
        """Benchmark scorecard: every metric vs its target."""
        rows = []
        targets_map = {**IKEA_TARGETS, **EFFICIENCY_TARGETS}

        def make_row(label, value, target, unit=''):
            if target is None:
                status = '—'
                gap = '—'
            else:
                gap = value - target
                status = '✅' if gap >= 0 else '❌'
            return [label, f'{value:.4f}' if isinstance(value, float) else str(value),
                    f'{target:.4f}' if target is not None else '—',
                    f'{gap:+.4f}' if isinstance(gap, float) else str(gap), status]

        epoch_arr = self._h['epoch']
        latest = len(epoch_arr) - 1 if epoch_arr else 0

        def last(key):
            arr = self._h.get(key, [])
            return arr[latest] if len(arr) > latest else 0.0

        rows += [
            make_row('det_AP50',         last('val_det_AP50'),       0.853, ''),
            make_row('det_mAP_coco',    last('val_det_mAP_coco'),   0.659, ''),
            make_row('pck_10px',        last('val_pck_10px'),       0.643, ''),
            make_row('pck_02',          last('val_pck_02'),        0.880, ''),
            make_row('act_top1_rgb',     last('val_act_top1'),       0.604, ''),
            make_row('act_top1_rgbpose', last('val_act_top1'),      0.6415, ''),
            make_row('act_mcAP_csv',     last('val_act_mcAP_csv'),   0.8447, ''),
            make_row('act_mcAP_cs',      last('val_act_mcAP_cs'),    0.8699, ''),
            make_row('kendall_tau',      last('val_kendall_tau'),   0.91, ''),
            make_row('temporal_mAP50',  last('val_temporal_mAP50'), 0.2177, ''),
            make_row('phase_acc_1',      last('val_phase_acc_1'),  0.3702, ''),
            make_row('fps',              last('val_fps'),            291.0, ''),
            make_row('params_M',         last('val_params_M'),       12.9, ''),
            make_row('gflops',           last('val_gflops'),         1.96, ''),
        ]

        fig, ax = plt.subplots(figsize=(16, max(8, len(rows) * 0.6 + 2)))
        ax.axis('off')
        table = Table(ax, bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(10)

        n_cols = 5
        table_data = [['Metric', 'POPW', 'Target', 'Gap', 'Status']] + rows
        for r_idx, row in enumerate(table_data):
            for c_idx, cell in enumerate(row):
                table[r_idx, c_idx].text = cell
                if r_idx == 0:
                    table[r_idx, c_idx].set_facecolor('#37474F')
                    table[r_idx, c_idx].set_text_props(color='white', fontweight='bold')
                elif r_idx % 2 == 0:
                    table[r_idx, c_idx].set_facecolor('#ECEFF1')

        ax.add_table(table)
        ax.set_title(
            f'POPW Benchmark Scorecard — Epoch {epoch}\n'
            f'{len(epoch_arr)} epochs completed | Popwatch Multi-Task IKEA ASM',
            fontsize=13, fontweight='bold', pad=20
        )
        plt.tight_layout()
        plt.savefig(viz_dir / 'metrics_dashboard.png', dpi=130, bbox_inches='tight')
        plt.close()

    # ── Metrics history (all curves) ─────────────────────────────────────

    def _plot_metrics_history(self, viz_dir: Path, epoch: int) -> None:
        """All train + val curves on one figure."""
        epochs = self._h['epoch']
        if len(epochs) < 2:
            return

        metric_groups = [
            ('Loss', [
                ('train_loss_total',   'Train Total',   '#2196F3'),
                ('train_loss_det',     'Train Det',     '#F44336'),
                ('train_loss_pose',    'Train Pose',    '#00BCD4'),
                ('train_loss_act',     'Train Act',     '#FFC107'),
                ('train_loss_ordering','Train Ordering','#9C27B0'),
            ]),
            ('Detection', [
                ('val_det_AP50',      'AP@0.5',       '#F44336'),
                ('val_det_mAP_coco',  'mAP COCO',     '#E91E63'),
            ]),
            ('Pose', [
                ('val_pck_10px',  'PCK@10px',  '#00BCD4'),
                ('val_pck_02',    'PCK@0.2',    '#009688'),
            ]),
            ('Activity', [
                ('val_act_top1',      'Top-1 Acc',    '#FFC107'),
                ('val_act_top5',      'Top-5 Acc',    '#FF9800'),
                ('val_act_f1_present','Macro-F1',     '#FF5722'),
            ]),
            ('Temporal & Phase', [
                ('val_kendall_tau',     "Kendall's Tau", '#9C27B0'),
                ('val_temporal_mAP50', 'Temporal mAP', '#3F51B5'),
                ('val_phase_acc_1',    'Phase Acc@1',  '#009688'),
            ]),
        ]

        n_groups = len(metric_groups)
        fig, axes = plt.subplots(n_groups, 1, figsize=(16, 4 * n_groups))
        if n_groups == 1:
            axes = [axes]
        axes = list(axes)

        for ax, (group_name, metrics) in zip(axes, metric_groups):
            for key, label, color in metrics:
                vals = self._h.get(key, [])
                if vals and len(vals) == len(epochs):
                    ax.plot(epochs, vals, color=color, linewidth=2, label=label)
            ax.set_xlabel('Epoch')
            ax.set_ylabel(group_name)
            ax.set_title(group_name)
            ax.legend(fontsize=8, loc='best')
            ax.grid(True, alpha=0.3)

        plt.suptitle(f'POPW Training Curves — Epoch {epoch}', fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig(viz_dir / 'metrics_history.png', dpi=120, bbox_inches='tight')
        plt.close()

    # ── Task loss breakdown ─────────────────────────────────────────────────

    def _plot_task_breakdown(self, viz_dir: Path, epoch: int) -> None:
        epochs = self._h['epoch']
        if len(epochs) < 2:
            return

        loss_specs = [
            ('train_loss_total',    'Total',    '#9C27B0'),
            ('train_loss_det',      'Detection','#F44336'),
            ('train_loss_pose',     'Pose',     '#00BCD4'),
            ('train_loss_act',      'Activity', '#FFC107'),
            ('train_loss_ordering', 'Ordering', '#3F51B5'),
        ]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax = axes[0]
        for key, label, color in loss_specs:
            vals = self._h.get(key, [])
            if vals and len(vals) == len(epochs):
                ax.plot(epochs, vals, color=color, linewidth=2, label=label)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss Components')
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1]
        latest_vals = {label: self._h[k][-1] if self._h.get(k) else 0
                       for k, label, _ in loss_specs}
        labels = list(latest_vals.keys())
        values = list(latest_vals.values())
        colors = [c for _, _, c in loss_specs]
        bars = ax.bar(labels, values, color=colors)
        ax.set_ylabel('Loss')
        ax.set_title(f'Loss Breakdown (Epoch {epoch})')
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(viz_dir / 'task_breakdown.png', dpi=120, bbox_inches='tight')
        plt.close()

    # ── LR schedule ─────────────────────────────────────────────────────────

    def _plot_lr_schedule(self, viz_dir: Path, epoch: int) -> None:
        lr_hist = self._h.get('lr', [])
        epochs = self._h['epoch']
        if len(lr_hist) < 2:
            return
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(epochs, lr_hist, color='#2196F3', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(viz_dir / 'lr_schedule.png', dpi=120, bbox_inches='tight')
        plt.close()

    # ── Efficiency chart ───────────────────────────────────────────────────

    def _plot_efficiency(self, viz_dir: Path, epoch: int) -> None:
        epochs = self._h['epoch']
        if len(epochs) < 2:
            return

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for ax, key, label, color, target in [
            (axes[0], 'val_fps',       'FPS',             '#4CAF50', 291.0),
            (axes[1], 'val_gflops',    'GFLOPs',          '#FF5722', 1.96),
            (axes[2], 'val_params_M',  'Parameters (M)',  '#2196F3', 12.9),
        ]:
            vals = self._h.get(key, [])
            if vals and len(vals) == len(epochs):
                ax.plot(epochs, vals, color=color, linewidth=2)
                if target:
                    ax.axhline(y=target, color='red', linestyle='--', alpha=0.6, label=f'Target {target}')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(label)
            ax.set_title(label)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)

        plt.suptitle(f'Efficiency Metrics — Epoch {epoch}', fontsize=13)
        plt.tight_layout()
        plt.savefig(viz_dir / 'efficiency_chart.png', dpi=120, bbox_inches='tight')
        plt.close()

    # ── Per-metric panels (18 benchmark metrics) ───────────────────────────

    def _plot_per_metric_panels(self, viz_dir: Path, epoch: int) -> None:
        """18 individual panels, one per benchmark metric, with target line."""
        panels = [
            # (key, target, title, ylabel)
            ('val_det_AP50',       0.853,  'Detection AP@0.5',         'AP'),
            ('val_det_mAP_coco',   0.659,  'Detection mAP (COCO)',      'mAP'),
            ('val_pck_10px',       0.643,  'Pose PCK@10px',            'PCK'),
            ('val_pck_02',         0.880,  'Pose PCK@0.2',             'PCK'),
            ('val_pck_005',        None,   'Pose PCK@0.05',           'PCK'),
            ('val_pck_01',         None,   'Pose PCK@0.1',            'PCK'),
            ('val_act_top1',       0.6415, 'Activity Top-1 Acc',       'Acc'),
            ('val_act_top5',       None,   'Activity Top-5 Acc',      'Acc'),
            ('val_act_mcAP_csv',   0.8447, 'Activity mcAP (csv)',      'mAP'),
            ('val_act_mcAP_cs',    0.8699, 'Activity mcAP (cs)',       'mAP'),
            ('val_act_f1_present', None,   'Activity Macro-F1',       'F1'),
            ('val_kendall_tau',    0.91,   "Kendall's Tau (Ordering)", 'Tau'),
            ('val_temporal_mAP50', 0.2177, 'Temporal mAP@0.5',         'mAP'),
            ('val_phase_acc_1',   0.3702, 'Phase Classification Acc',  'Acc'),
            ('val_fps',            291.0,  'Throughput FPS',           'FPS'),
            ('val_gflops',         1.96,   'GFLOPs',                   'GFLOPs'),
            ('val_params_M',       12.9,   'Parameters (M)',           'M'),
            ('val_gpu_mem_gb',     None,   'GPU Memory (GB)',           'GB'),
        ]

        epochs = self._h['epoch']
        if len(epochs) < 2:
            return

        nrows = 6
        ncols = 3
        fig, axes = plt.subplots(nrows, ncols, figsize=(18, 22))
        axes = axes.flatten()

        for idx, (key, target, title, ylabel) in enumerate(panels):
            ax = axes[idx]
            vals = self._h.get(key, [])
            if not vals or len(vals) != len(epochs):
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(title, fontsize=9)
                ax.axis('off')
                continue

            ax.plot(epochs, vals, color='#2196F3', linewidth=2)
            ax.fill_between(epochs, vals, alpha=0.1, color='#2196F3')

            if target is not None:
                ax.axhline(y=target, color='red', linestyle='--', linewidth=1.5,
                           alpha=0.8, label=f'Target: {target:.4f}')

            ax.set_xlabel('Epoch', fontsize=8)
            ax.set_ylabel(ylabel, fontsize=8)
            ax.set_title(title, fontsize=9, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7, loc='best')

            # Latest value annotation
            latest = vals[-1]
            ax.annotate(f'{latest:.4f}', xy=(epochs[-1], latest),
                        xytext=(4, 4), textcoords='offset points', fontsize=8,
                        color='#1565C0')

            # Gap to target
            if target is not None:
                gap = latest - target
                color = '#4CAF50' if gap >= 0 else '#F44336'
                ax.annotate(f'gap: {gap:+.4f}', xy=(epochs[-1], latest),
                            xytext=(4, -12), textcoords='offset points', fontsize=7,
                            color=color)

        # Turn off extra axes
        for idx in range(len(panels), len(axes)):
            axes[idx].axis('off')

        plt.suptitle(
            f'POPW — All Benchmark Metrics (vs Targets) — Epoch {epoch}',
            fontsize=15, fontweight='bold'
        )
        plt.tight_layout(rect=[0, 0, 1, 0.98])
        plt.savefig(viz_dir / 'per_metric_panels.png', dpi=120, bbox_inches='tight')
        plt.close()

    # ── Annotated sample images ─────────────────────────────────────────────

    def _plot_annotated_samples(self, viz_dir: Path, epoch: int) -> None:
        """Draw activity class, keypoints, and detection bboxes on sample images."""
        if self._cached_images is None or not MATPLOTLIB_AVAILABLE:
            return

        try:
            from matplotlib.patches import FancyBboxPatch, Rectangle
        except Exception:
            return

        imgs = self._cached_images
        if imgs.ndim == 5:  # [B, T, C, H, W]
            imgs = imgs[:, 0]
        if imgs.ndim == 4 and imgs.shape[1] == 3:  # [B, C, H, W]
            imgs = imgs.permute(0, 2, 3, 1).numpy()
        elif imgs.ndim == 4 and imgs.shape[3] == 3:  # [B, H, W, C]
            pass
        else:
            return

        mean = np.array([0.485, 0.456, 0.406])
        std  = np.array([0.229, 0.224, 0.225])
        imgs = imgs * std + mean
        imgs = np.clip(imgs, 0, 1)

        n_samples = min(len(imgs), self.num_viz_samples)
        ncols = 4
        nrows = (n_samples + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows))
        if nrows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()

        outputs = self._cached_outputs or {}
        targets = self._cached_targets or {}

        for i in range(n_samples):
            ax = axes[i]
            img = imgs[i]
            if img.ndim == 3 and img.shape[-1] in [1, 3]:
                ax.imshow(img)
            else:
                ax.imshow(img, cmap='gray')
            ax.axis('off')

            # ── Activity class label ────────────────────────────────────────
            act_logits = outputs.get('act_logits', [])
            if isinstance(act_logits, torch.Tensor):
                act_logits = act_logits.cpu().numpy()
            if act_logits is not None and act_logits.size > 0:
                try:
                    if act_logits.ndim == 2:
                        probs_i = act_logits[i] if i < len(act_logits) else act_logits[0]
                    else:
                        probs_i = act_logits.flatten()
                    top1 = int(np.argmax(probs_i))
                    conf = float(probs_i[top1])
                    name = self.act_class_names[top1] if top1 < len(self.act_class_names) else f'C{top1}'
                    ax.set_title(f'{name}\nConf: {conf:.2f}', fontsize=8, loc='left',
                                color='#FFC107', fontweight='bold')
                except Exception:
                    pass
            else:
                act_lbl = targets.get('activity_labels_seq') or targets.get('activity')
                if act_lbl is not None:
                    try:
                        lbl_i = act_lbl[i].item() if hasattr(act_lbl[i], 'item') else act_lbl[i]
                        ax.set_title(f'GT: {self.act_class_names[int(lbl_i)]}',
                                   fontsize=8, loc='left', color='#9E9E9E')
                    except Exception:
                        pass

            # ── Keypoints (COCO 17) ─────────────────────────────────────────
            kp_out = outputs.get('keypoints', [])
            if isinstance(kp_out, torch.Tensor) and kp_out.numel() > 0:
                try:
                    kp = kp_out[i % kp_out.shape[0]].cpu().numpy()
                    H, W = img.shape[:2]
                    kp_x = kp[:, 0] * W
                    kp_y = kp[:, 1] * H
                    for j, (x, y) in enumerate(zip(kp_x, kp_y)):
                        if 0 <= x < W and 0 <= y < H:
                            ax.plot(x, y, 'o', color='#00E5FF', markersize=4, alpha=0.85)
                            if j == 0:  # nose
                                ax.annotate(KEYPOINT_NAMES[j], (x, y), color='#00BCD4',
                                          fontsize=5, xytext=(3, 3), textcoords='offset points')
                except Exception:
                    pass

            # ── Detection bboxes ───────────────────────────────────────────
            try:
                cls_preds = outputs.get('cls_preds', [])
                reg_preds = outputs.get('reg_preds', [])
                anchors   = outputs.get('anchors', [])

                if isinstance(cls_preds, torch.Tensor) and cls_preds.numel() > 0:
                    sig = torch.sigmoid(cls_preds[i % cls_preds.shape[0]]).cpu().numpy()
                    reg = reg_preds[i % reg_preds.shape[0]].cpu().numpy() if isinstance(reg_preds, torch.Tensor) else None
                    anch = anchors[i % anchors.shape[0]].cpu().numpy() if isinstance(anchors, torch.Tensor) else None

                    if reg is not None and anch is not None:
                        for cls_id in range(1, 7):
                            sc = sig[:, cls_id]
                            mask = sc > 0.25
                            if mask.sum() == 0:
                                continue
                            top_idx = np.argsort(sc[mask])[-2:]
                            for idx in top_idx:
                                real_idx = np.where(mask)[0][idx]
                                dx, dy, dw, dh = reg[real_idx]
                                ax1, ay1, ax2, ay2 = anch[real_idx]
                                bw = ax2 - ax1
                                bh = ay2 - ay1
                                if bw > 5 and bh > 5:
                                    rect = patches.Rectangle(
                                        (ax1, ay1), bw, bh,
                                        linewidth=1.2,
                                        edgecolor=list(DET_CLASS_NAMES.get(cls_id + 1, '#999'))[1:],
                                        facecolor='none',
                                    )
                                    ax.add_patch(rect)
            except Exception:
                pass

        for i in range(n_samples, len(axes)):
            axes[i].axis('off')

        plt.suptitle(
            f'Annotated Samples — Epoch {epoch} | '
            f'Yellow: Activity class, Cyan: Keypoints, Colored: BBoxes',
            fontsize=12, fontweight='bold'
        )
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig(viz_dir / 'annotated_samples.png', dpi=120, bbox_inches='tight')
        plt.close()

    # ── Confusion matrix ─────────────────────────────────────────────────────

    def _plot_confusion_matrix(
        self,
        viz_dir: Path,
        epoch: int,
        cm: np.ndarray,
    ) -> None:
        n = cm.shape[0]
        labels = self.act_class_names[:n]

        fig, ax = plt.subplots(figsize=(max(8, n * 0.55), max(6, n * 0.55)))
        im = ax.imshow(cm, cmap='Blues', vmin=0, vmax=1)
        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=5)
        ax.set_yticklabels(labels, fontsize=5)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(f'Activity Confusion Matrix — Epoch {epoch}')

        for i in range(n):
            for j in range(n):
                val = cm[i, j]
                color = 'white' if val > 0.5 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        color=color, fontsize=6)

        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig(viz_dir / 'confusion_matrix.png', dpi=120, bbox_inches='tight')
        plt.close()

    # ── Finalize ─────────────────────────────────────────────────────────────

    def close(self) -> None:
        if self._writer:
            self._writer.close()
            print('[TrainingMonitor] TensorBoard closed.')
        self._save_history_json()

    def _save_history_json(self) -> None:
        path = self.run_dir / 'metrics_history.json'
        with open(path, 'w') as f:
            json.dump(self._h, f, indent=2, default=str)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()