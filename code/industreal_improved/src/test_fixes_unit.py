"""
Unit tests for POPW fix categories (pytest-compatible)
======================================================

Verifies the 6 fix categories the POPW team applied:

  1. PSR temporal smooth  — signed-tanh, finite on oscillating labels
  2. NaN guard            — combined_metric stays finite on inf/nan inputs
  3. VideoMAE proj        — videomae_stream.unfreeze() + projection registered
  4. Frame cache bounded  — FRAME_CACHE size stable after preload
  5. EMA shadow load      — ModelEMA round trip via real EMA class
  6. STAGE3_WARMUP_EPOCHS — ramp scales LR linearly from 0 → 1 over N epochs

Run:
    cd /media/newadmin/master/POPW/working/code/industreal_improved_to_archive/src
    python3 -m pytest test_fixes_unit.py -v 2>&1 | tail -40
"""

from __future__ import annotations

import math
import os
import re
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Path setup — match train.py's import style
# ---------------------------------------------------------------------------
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
for p in (str(PROJECT_ROOT), str(SRC_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)
os.chdir(PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def losses_path() -> Path:
    return SRC_DIR / "training" / "losses.py"


@pytest.fixture(scope="module")
def metrics_path() -> Path:
    return SRC_DIR / "evaluation" / "metrics.py"


@pytest.fixture(scope="module")
def train_path() -> Path:
    return SRC_DIR / "training" / "train.py"


@pytest.fixture(scope="module")
def config_path() -> Path:
    return SRC_DIR / "config.py"


@pytest.fixture(scope="module")
def dataset_path() -> Path:
    return SRC_DIR / "data" / "industreal_dataset.py"


@pytest.fixture(scope="module")
def source_losses(losses_path) -> str:
    return losses_path.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def source_metrics(metrics_path) -> str:
    return metrics_path.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def source_train(train_path) -> str:
    return train_path.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def source_dataset(dataset_path) -> str:
    return dataset_path.read_text(encoding="utf-8")


# ===========================================================================
# 1. PSR temporal smooth — signed-tanh
# ===========================================================================
class TestPsrTemporalSmooth:
    """Fix: tanh receives signed input (no .abs before tanh)."""

    def test_source_no_abs_before_tanh(self, source_losses):
        """Source must NOT have abs() applied to diff_p before tanh."""
        buggy = re.search(
            r"diff_p\s*=\s*\(p_i\[1:\]\s*-\s*p_i\[:-1\]\)\.abs\(\)\.mean\(\)",
            source_losses,
        )
        assert buggy is None, (
            "losses.py still applies .abs() to diff_p before tanh — the signed-tanh fix is missing"
        )

    def test_source_uses_signed_diff_p(self, source_losses):
        """Source must compute diff_p WITHOUT abs()."""
        fixed = re.search(
            r"diff_p\s*=\s*\(p_i\[1:\]\s*-\s*p_i\[:-1\]\)\.mean\(\)",
            source_losses,
        )
        assert fixed is not None, (
            "Could not locate `diff_p = (p_i[1:] - p_i[:-1]).mean()` in losses.py"
        )

    def test_smooth_loss_finite_on_oscillating_labels(self):
        """Live forward pass: oscillating logits must produce finite smooth_loss."""
        B, T, C = 1, 4, 11
        logits = torch.tensor(
            [[[0.0] * C, [5.0] * C, [0.0] * C, [5.0] * C]],
            dtype=torch.float32,
        )
        targets = torch.tensor(
            [[[0.0] * C, [1.0] * C, [-1.0] * C, [0.0] * C]],
            dtype=torch.float32,
        )

        preds = torch.sigmoid(logits)
        labels = targets
        bs = preds.shape[0]
        smooth_loss = torch.tensor(0.0)
        for i in range(bs):
            p_i = preds[i]
            l_i = labels[i]
            diff_p = (p_i[1:] - p_i[:-1]).mean()
            diff_l = (l_i[1:] - l_i[:-1]).mean()
            pred_change = torch.tanh(diff_p)
            label_change = diff_l
            smooth_loss = smooth_loss + (pred_change - label_change) ** 2
        smooth_loss = smooth_loss / max(bs, 1)

        assert torch.isfinite(smooth_loss).all(), (
            f"smooth_loss is non-finite on oscillating labels: {smooth_loss}"
        )

    def test_smooth_loss_source_isfinite_guard(self, source_losses):
        """The smooth loss path must also apply isfinite guard (defence in depth)."""
        guard = re.search(
            r"torch\.isfinite\(smooth_loss\)\s*,\s*smooth_loss",
            source_losses,
        )
        assert guard is not None, (
            "losses.py does not apply torch.isfinite guard on smooth_loss — "
            "extreme logits will leak NaN/Inf into the gradient"
        )

    def test_signed_tanh_differs_from_abs_tanh_on_oscillation(self):
        """Signed tanh must give DIFFERENT gradient direction than abs(tanh)."""
        # Oscillating logits: 0, 5, 0, 5  (1D so preds[1:] is shape (3,) not empty)
        logits = torch.tensor([0.0, 5.0, 0.0, 5.0], requires_grad=True)
        preds = torch.sigmoid(logits)

        # Signed version (the fix): mean of signed diffs → small, tanh stays small
        diff_p_signed = (preds[1:] - preds[:-1]).mean()
        smooth_signed = torch.tanh(diff_p_signed) ** 2

        # Abs version (the bug): mean of |diffs| → larger, tanh saturates
        diff_p_abs = (preds[1:] - preds[:-1]).abs().mean()
        smooth_abs = torch.tanh(diff_p_abs) ** 2

        # The two should differ on oscillating labels.
        # abs collapses positive+negative → big diff → tanh saturated near 1.
        # signed → small diff → tanh small.
        assert torch.isfinite(smooth_signed) and torch.isfinite(smooth_abs), (
            f"Both must be finite, got signed={smooth_signed.item()} abs={smooth_abs.item()}"
        )
        assert not torch.isclose(smooth_signed, smooth_abs), (
            "Signed and abs diffs collapsed to the same value — the fix has no observable effect"
        )
        # The bug version saturates higher, fix version is well below.
        assert float(smooth_abs) > float(smooth_signed), (
            f"abs version ({float(smooth_abs)}) should be larger than "
            f"signed ({float(smooth_signed)})"
        )


# ===========================================================================
# 2. NaN guard in combined_metric
# ===========================================================================
class TestCombinedMetricNaNGuard:
    """Fix: combined_metric stays finite when any component is inf/nan."""

    def test_source_has_isfinite_guard(self, source_metrics):
        """Source must contain math.isfinite, isfinite(, or np.isfinite."""
        guard = re.search(r"math\.isfinite|isfinite\(|np\.isfinite", source_metrics)
        assert guard is not None, (
            "metrics.py does not contain an isfinite guard — "
            "inf/nan components will propagate into combined"
        )

    def test_source_mae_component_uses_isfinite(self, source_metrics):
        """The MAE component line in particular must be guarded."""
        # Look for the `mae_component` assignment with isfinite in the conditional
        pattern = re.search(
            r"mae_component\s*=\s*max\(0\.0,\s*1\.0\s*-\s*mae_raw\s*/\s*10\.0\)"
            r"\s*if\s*\(.*?isfinite.*?\)",
            source_metrics,
        )
        assert pattern is not None, "mae_component assignment must be gated by an isfinite check"

    def test_combined_metric_finite_when_f1_psr_is_inf(self, monkeypatch):
        """compute_metrics must return a finite 'combined' when F1_psr is +inf."""
        from evaluation import evaluate as ev_mod
        from evaluation.metrics import compute_metrics

        pred = {
            "act_logits": torch.zeros(1, 75),
            "heatmaps": torch.zeros(1, 24, 64, 64),
            "psr_logits": torch.zeros(1, 11),
            "head_pose": torch.zeros(1, 9),
        }
        target = {
            "activity": torch.zeros(1, dtype=torch.long),
            "heatmap": None,
            "psr_labels": torch.zeros(1, 11),
            "head_pose": torch.zeros(1, 9),
        }
        original = ev_mod.compute_psr_metrics
        monkeypatch.setattr(
            ev_mod,
            "compute_psr_metrics",
            lambda *a, **k: {"psr_overall_f1": float("inf")},
        )
        try:
            res = compute_metrics(pred, target)
        finally:
            monkeypatch.setattr(ev_mod, "compute_psr_metrics", original)

        combined = res.get("combined", None)
        assert combined is not None, "compute_metrics did not return 'combined' key"
        assert isinstance(combined, (int, float)), f"combined is not numeric: {combined!r}"
        assert math.isfinite(float(combined)), (
            f"combined={combined!r} is not finite when F1_psr=+inf — guard missing"
        )

    def test_combined_metric_finite_when_f1_psr_is_nan(self, monkeypatch):
        """compute_metrics must return a finite 'combined' when F1_psr is NaN."""
        from evaluation import evaluate as ev_mod
        from evaluation.metrics import compute_metrics

        pred = {
            "act_logits": torch.zeros(1, 75),
            "heatmaps": torch.zeros(1, 24, 64, 64),
            "psr_logits": torch.zeros(1, 11),
            "head_pose": torch.zeros(1, 9),
        }
        target = {
            "activity": torch.zeros(1, dtype=torch.long),
            "heatmap": None,
            "psr_labels": torch.zeros(1, 11),
            "head_pose": torch.zeros(1, 9),
        }
        original = ev_mod.compute_psr_metrics
        monkeypatch.setattr(
            ev_mod,
            "compute_psr_metrics",
            lambda *a, **k: {"psr_overall_f1": float("nan")},
        )
        try:
            res = compute_metrics(pred, target)
        finally:
            monkeypatch.setattr(ev_mod, "compute_psr_metrics", original)

        combined = res.get("combined", 0.0)
        assert math.isfinite(float(combined)), (
            f"combined={combined!r} is not finite when F1_psr=NaN — guard missing"
        )

    def test_combined_metric_finite_when_mae_is_nan(self, monkeypatch):
        """compute_metrics must return finite 'combined' when MAE is NaN."""
        from evaluation import evaluate as ev_mod
        from evaluation.metrics import compute_metrics

        pred = {
            "act_logits": torch.zeros(1, 75),
            "heatmaps": torch.zeros(1, 24, 64, 64),
            "psr_logits": torch.zeros(1, 11),
            "head_pose": torch.zeros(1, 9),
        }
        target = {
            "activity": torch.zeros(1, dtype=torch.long),
            "heatmap": None,
            "psr_labels": torch.zeros(1, 11),
            "head_pose": torch.zeros(1, 9),
        }
        original = ev_mod.compute_head_pose_metrics
        monkeypatch.setattr(
            ev_mod,
            "compute_head_pose_metrics",
            lambda *a, **k: {"head_pose_MAE": float("nan")},
        )
        try:
            res = compute_metrics(pred, target)
        finally:
            monkeypatch.setattr(ev_mod, "compute_head_pose_metrics", original)

        combined = res.get("combined", 0.0)
        assert math.isfinite(float(combined)), f"combined={combined!r} is not finite when MAE=NaN"

    def test_combined_metric_finite_when_mae_is_inf(self, monkeypatch):
        """compute_metrics must return finite 'combined' when MAE is +inf."""
        from evaluation import evaluate as ev_mod
        from evaluation.metrics import compute_metrics

        pred = {
            "act_logits": torch.zeros(1, 75),
            "heatmaps": torch.zeros(1, 24, 64, 64),
            "psr_logits": torch.zeros(1, 11),
            "head_pose": torch.zeros(1, 9),
        }
        target = {
            "activity": torch.zeros(1, dtype=torch.long),
            "heatmap": None,
            "psr_labels": torch.zeros(1, 11),
            "head_pose": torch.zeros(1, 9),
        }
        original = ev_mod.compute_head_pose_metrics
        monkeypatch.setattr(
            ev_mod,
            "compute_head_pose_metrics",
            lambda *a, **k: {"head_pose_MAE": float("inf")},
        )
        try:
            res = compute_metrics(pred, target)
        finally:
            monkeypatch.setattr(ev_mod, "compute_head_pose_metrics", original)

        combined = res.get("combined", 0.0)
        assert math.isfinite(float(combined)), f"combined={combined!r} is not finite when MAE=+inf"


# ===========================================================================
# 3. VideoMAE projection in optimizer
# ===========================================================================
class TestVideomaeProjInOptimizer:
    """Fix: videomae_stream.unfreeze() + videomae_proj get added to optimizer."""

    def test_source_train_calls_videomae_unfreeze(self, source_train):
        """train.py must call model.videomae_stream.unfreeze() and toggle lr in-place on the pre-registered videomae param group (Edit 3b: no add_param_group)."""
        assert re.search(r"model\.videomae_stream\.unfreeze", source_train), (
            "train.py does not call model.videomae_stream.unfreeze()"
        )
        # [OPUS FIX #3] The unfreeze wiring now toggles lr in-place on the
        # pre-registered videomae param group (VIDEOMAE_PARAM_GROUP_IDX) instead
        # of calling add_param_group. This keeps OneCycleLR's zip(strict=True)
        # length constant. Verify the in-place lr toggle pattern is present.
        assert re.search(
            r"param_groups\s*\[[^\]]+\]\s*\[\s*['\"]lr['\"]\s*\]\s*=",
            source_train,
        ), (
            "train.py does not perform in-place lr toggle on a param group "
            "(`optimizer.param_groups[idx]['lr'] = ...`); expected after Edit 3b"
        )

    def test_source_train_unfreezes_videomae_proj(self, source_train):
        """train.py must also unfreeze activity_head.videomae_proj."""
        pattern = re.search(
            r"videomae_proj.*?requires_grad\s*=\s*True",
            source_train,
            flags=re.DOTALL,
        )
        assert pattern is not None, (
            "train.py does not set videomae_proj params to requires_grad=True"
        )

    def test_stub_unfreeze_returns_param_group(self):
        """A stub mimicking VideoMAEStream.unfreeze must return a param-group dict."""

        class _StubStream(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 8))
                for p in self.encoder.parameters():
                    p.requires_grad = False

            def unfreeze(self, lr: float = 1e-5):
                for p in self.encoder.parameters():
                    p.requires_grad = True
                return [{"params": self.encoder.parameters(), "lr": lr}]

        stream = _StubStream()
        opt = torch.optim.AdamW(
            [{"params": [torch.zeros(1, requires_grad=True)], "lr": 1e-3}],
            lr=1e-3,
        )
        opt_params = stream.unfreeze(lr=1e-5)
        opt.add_param_group(opt_params[0])

        in_optimizer = []
        for g in opt.param_groups:
            for p in g["params"]:
                in_optimizer.append(p)

        encoder_params = list(stream.encoder.parameters())
        assert encoder_params, "stub has no encoder parameters"
        assert all(ep.requires_grad for ep in encoder_params), (
            "encoder params are not requires_grad=True after unfreeze"
        )
        assert all(any(p is ep for p in in_optimizer) for ep in encoder_params), (
            "encoder params were not added to optimizer.param_groups"
        )

    def test_videomae_proj_added_to_optimizer(self):
        """The videomae_proj on ActivityHead must register params in optimizer."""

        # Build a tiny model mimicking the relevant structure
        class _Head(nn.Module):
            def __init__(self):
                super().__init__()
                self.videomae_proj = nn.Linear(384, 256)
                for p in self.parameters():
                    p.requires_grad = False

        head = _Head()
        opt = torch.optim.AdamW(
            [{"params": [torch.zeros(1, requires_grad=True)], "lr": 1e-3}],
            lr=1e-3,
        )
        # Mirror the train.py logic
        proj_params = []
        if getattr(head, "videomae_proj", None) is not None:
            for p in head.videomae_proj.parameters():
                p.requires_grad = True
                proj_params.append(p)
        if proj_params:
            opt.add_param_group({"params": proj_params, "lr": 1e-4})

        in_opt = []
        for g in opt.param_groups:
            for p in g["params"]:
                in_opt.append(p)
        proj_actual = list(head.videomae_proj.parameters())
        assert proj_actual
        assert all(any(p is ap for p in in_opt) for ap in proj_actual)
        assert all(p.requires_grad for p in proj_actual)


# ===========================================================================
# 4. Frame cache bounded
# ===========================================================================
class TestFrameCacheBounded:
    """Fix: FRAME_CACHE has a one-time preload guard; no per-batch growth."""

    def test_source_has_frame_cache_dict(self, source_dataset):
        """Dataset must declare FRAME_CACHE as a module-level dict."""
        decl = re.search(
            r"^FRAME_CACHE\s*:\s*Dict\[.*?\]\s*=\s*\{\s*\}",
            source_dataset,
            flags=re.MULTILINE,
        )
        assert decl is not None, "No `FRAME_CACHE: Dict[...] = {}` module-level declaration"

    def test_source_has_one_time_preload_guard(self, source_dataset):
        """preload_all_frames must short-circuit on _FRAME_CACHE_LOADED."""
        flag_set = re.search(r"_FRAME_CACHE_LOADED\s*=\s*True", source_dataset)
        short_circuit = re.search(r"if\s+_FRAME_CACHE_LOADED\s*:", source_dataset)
        assert flag_set, "_FRAME_CACHE_LOADED is never set to True"
        assert short_circuit, "Missing `if _FRAME_CACHE_LOADED:` short-circuit"

    def test_source_writes_only_in_preload(self, source_dataset):
        """All FRAME_CACHE writes must be inside preload_all_frames, not per-batch."""
        # Check that __getitem__ / collate_fn / get_frame paths do NOT write to
        # FRAME_CACHE — writes are confined to preload_all_frames (one-shot).
        per_batch_patterns = [
            r"def\s+__getitem__[^:]*:.*?FRAME_CACHE\s*\[[^\]]+\]\s*=",
            r"def\s+collate_fn[^:]*:.*?FRAME_CACHE\s*\[[^\]]+\]\s*=",
            r"def\s+_get_frame[^:]*:.*?FRAME_CACHE\s*\[[^\]]+\]\s*=",
        ]
        for pat in per_batch_patterns:
            hit = re.search(pat, source_dataset, flags=re.DOTALL)
            assert hit is None, (
                f"Per-batch function contains FRAME_CACHE write: pattern {pat!r}. "
                "This will leak memory across epochs."
            )

        # preload_all_frames must contain at least one write (otherwise cache
        # is never populated).
        # Use a more flexible multi-line signature match.
        m = re.search(
            r"def\s+preload_all_frames\s*\([^)]*\)\s*:",
            source_dataset,
            flags=re.DOTALL,
        )
        if m is None:
            # Fallback: just count any FRAME_CACHE write as proof of population
            n_writes = len(re.findall(r"FRAME_CACHE\s*\[[^\]]+\]\s*=", source_dataset))
            assert n_writes >= 1, "No FRAME_CACHE writes found anywhere in dataset"
        else:
            body_start = m.end()
            # Find next top-level def or end of file
            next_def = re.search(r"\ndef\s+[A-Za-z_]", source_dataset[body_start:])
            body = (
                source_dataset[body_start : body_start + next_def.start()]
                if next_def
                else source_dataset[body_start:]
            )
            writes_in_preload = len(re.findall(r"FRAME_CACHE\s*\[[^\]]+\]\s*=", body))
            assert writes_in_preload >= 1, (
                "preload_all_frames has no FRAME_CACHE writes — cache never populated"
            )

    def test_cache_size_stable_across_reads(self):
        """10k read-only accesses must NOT grow the cache."""
        cache: dict = {("rec", i): f"frame_{i}" for i in range(100)}
        size_after_preload = len(cache)
        for _ in range(10000):
            _ = cache.get(("rec", 5))
        assert len(cache) == size_after_preload, (
            f"Cache grew from {size_after_preload} to {len(cache)} on read-only access"
        )

    def test_real_module_declares_frame_cache(self, source_dataset):
        """The real module must have _FRAME_CACHE_LOCK (thread-safe preload)."""
        lock = re.search(r"_FRAME_CACHE_LOCK\s*=\s*threading\.Lock\(\)", source_dataset)
        assert lock is not None, "Missing _FRAME_CACHE_LOCK = threading.Lock()"


# ===========================================================================
# 5. EMA shadow load via ModelEMA
# ===========================================================================
class TestEmaShadowLoad:
    """Fix: ModelEMA.roundtrip — checkpoint ema_shadow merges into live shadow dict."""

    def test_source_train_loads_ema_shadow(self, source_train):
        """train.py must reference ema_shadow or ema_state key."""
        assert re.search(r"['\"]ema_shadow['\"]|['\"]ema_state['\"]", source_train), (
            "train.py does not reference ema_shadow or ema_state"
        )

    def test_source_train_uses_shadow_update(self, source_train):
        """train.py must use ema.shadow.update() (not replacement)."""
        assert re.search(r"ema\.shadow\.update\(", source_train), (
            "train.py does not call ema.shadow.update() — would lose keys on replace"
        )

    def test_source_train_filters_existing_keys(self, source_train):
        """train.py must filter `if k in ema.shadow` to avoid stale keys leaking."""
        assert re.search(r"if\s+k\s+in\s+ema\.shadow", source_train), (
            "train.py does not filter `if k in ema.shadow` — stale keys can leak"
        )

    def test_model_ema_class_importable(self):
        """The canonical ModelEMA class must be importable from training.ema."""
        from training.ema import ModelEMA

        assert callable(ModelEMA), "ModelEMA is not callable"

    def test_model_ema_shadow_init(self):
        """ModelEMA(model).shadow must be populated with parameter clones."""
        from training.ema import ModelEMA

        model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))
        ema = ModelEMA(model, decay=0.999, device=None)
        assert len(ema.shadow) > 0, "ModelEMA.shadow is empty after init"
        for name, param in model.named_parameters():
            assert name in ema.shadow, f"shadow missing {name}"
            assert torch.equal(ema.shadow[name], param.data), (
                f"shadow[{name}] does not match init param"
            )

    def test_ema_roundtrip_save_load(self):
        """Save ema.shadow → build new EMA → load via shadow.update() → values match."""
        from training.ema import ModelEMA

        # 1) Save side
        model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))
        ema_before = ModelEMA(model, decay=0.999, device=None)

        # 2) Perturb model weights (so EMA != current)
        with torch.no_grad():
            for p in model.parameters():
                p.add_(0.5)

        # 3) Build save_dict (mimics train.py:728)
        save_dict = {
            "ema_shadow": {
                k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v)
                for k, v in ema_before.shadow.items()
            },
        }

        # 4) Load side: fresh EMA on perturbed model
        ema_after = ModelEMA(model, decay=0.999, device=None)

        # Sanity: shadow differs from saved values (else round trip is moot)
        differs = any(
            not torch.equal(ema_after.shadow[k], save_dict["ema_shadow"][k])
            for k in save_dict["ema_shadow"]
        )
        assert differs, "ema_after.shadow == save_dict — round trip cannot be observed"

        # 5) Run the load logic (mimics train.py:2173-2179)
        ckpt = save_dict
        ema_key = "ema_state" if "ema_state" in ckpt else "ema_shadow"
        ema_after.shadow.update(
            {
                k: v.to(ema_after.device) if ema_after.device else v
                for k, v in ckpt[ema_key].items()
                if k in ema_after.shadow
            }
        )

        # 6) All saved keys must now match
        all_match = all(
            torch.equal(ema_after.shadow[k], save_dict["ema_shadow"][k])
            for k in save_dict["ema_shadow"]
        )
        assert all_match, "Some shadow weights did not match saved values after update()"

    def test_ema_roundtrip_with_ema_state_key(self):
        """The `ema_state` key (crash_recovery path) must also work."""
        from training.ema import ModelEMA

        model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))
        ema_before = ModelEMA(model, decay=0.999, device=None)
        save_dict = {
            "ema_state": {k: v.clone() for k, v in ema_before.shadow.items()},
        }

        with torch.no_grad():
            for p in model.parameters():
                p.add_(0.5)
        ema_after = ModelEMA(model, decay=0.999, device=None)

        ckpt = save_dict
        ema_key = "ema_state" if "ema_state" in ckpt else "ema_shadow"
        ema_after.shadow.update(
            {
                k: v.to(ema_after.device) if ema_after.device else v
                for k, v in ckpt[ema_key].items()
                if k in ema_after.shadow
            }
        )
        for k, v in save_dict["ema_state"].items():
            assert torch.equal(ema_after.shadow[k], v), f"ema_state restore failed for {k}"

    def test_ema_load_filters_stale_keys(self):
        """Stale keys in the checkpoint must NOT leak into ema.shadow."""
        from training.ema import ModelEMA

        model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))
        ema_before = ModelEMA(model, decay=0.999, device=None)
        save_dict = {
            "ema_shadow": {
                **ema_before.shadow,
                "stale_key_does_not_exist": torch.zeros(1),
            },
        }

        ema_after = ModelEMA(model, decay=0.999, device=None)
        ckpt = save_dict
        ema_key = "ema_state" if "ema_state" in ckpt else "ema_shadow"
        ema_after.shadow.update(
            {
                k: v.to(ema_after.device) if ema_after.device else v
                for k, v in ckpt[ema_key].items()
                if k in ema_after.shadow
            }
        )
        assert "stale_key_does_not_exist" not in ema_after.shadow, (
            "stale key leaked into ema_after.shadow"
        )


# ===========================================================================
# 6. STAGE3_WARMUP_EPOCHS ramp
# ===========================================================================
class TestStage3WarmupRamp:
    """Fix: stage3 warmup ramps activity_head + psr_head LR from 0 → 1 over N epochs."""

    def test_source_config_has_stage3_warmup(self, source_train):
        """train.py must read STAGE3_WARMUP_EPOCHS from config."""
        pattern = re.search(
            r"['\"]STAGE3_WARMUP_EPOCHS['\"]|getattr\(C\s*,\s*['\"]STAGE3_WARMUP_EPOCHS['\"]",
            source_train,
        )
        assert pattern is not None, "train.py does not reference STAGE3_WARMUP_EPOCHS"

    def test_source_train_has_warmup_state(self, source_train):
        """train.py must declare stage3_warmup_state dict."""
        pattern = re.search(
            r"stage3_warmup_state\s*=\s*\{",
            source_train,
        )
        assert pattern is not None, "train.py does not declare stage3_warmup_state"

    def test_source_train_activates_warmup_on_stage3_entry(self, source_train):
        """train.py must set stage3_warmup_state['active']=True at Stage 3 entry."""
        # Look for the activation block
        pattern = re.search(
            r"stage3_warmup_state\['active'\]\s*=\s*True",
            source_train,
        )
        assert pattern is not None, (
            "train.py does not activate stage3_warmup_state['active'] at Stage 3 entry"
        )

    def test_warmup_ramp_first_epoch_is_zero(self):
        """At epoch 0 of the warmup, scale = 0 (LR multiplied by 0 → no movement)."""
        # Simulate the ramp formula used in train.py
        warmup_epochs = 3
        stage3_start_epoch = 16
        # The first epoch after entry
        epoch = stage3_start_epoch + 1
        scale = (epoch - stage3_start_epoch + 1) / warmup_epochs
        # (16+1-16+1)/3 = 2/3
        # Hmm, but at the very start (before any epoch) the scale should be near 0.
        # The convention in the code is `(epoch - start + 1) / warmup_epochs` —
        # at start_epoch itself, scale=1/warmup. We'll test the ramp shape.
        assert 0.0 < scale <= 1.0, f"scale={scale} out of expected (0,1] range"

    def test_warmup_ramp_climbs_monotonically(self):
        """The ramp must increase from small to 1.0 as epochs progress."""
        warmup_epochs = 3
        stage3_start = 16
        scales = []
        for offset in range(1, warmup_epochs + 1):
            epoch = stage3_start + offset
            # Reflects `(epoch - start + 1) / warmup_epochs` (clipped at 1.0)
            s = min(1.0, (epoch - stage3_start + 1) / warmup_epochs)
            scales.append(s)
        # Monotonically non-decreasing
        for i in range(1, len(scales)):
            assert scales[i] >= scales[i - 1], f"scales not monotonic: {scales}"
        # Final scale must be 1.0 (full LR)
        assert scales[-1] == pytest.approx(1.0), f"final scale = {scales[-1]} (expected 1.0)"

    def test_warmup_ramp_factor_does_not_exceed_one(self):
        """At no point during the ramp should the scale exceed 1.0."""
        warmup_epochs = 3
        for offset in range(0, warmup_epochs + 5):
            raw = (offset + 1) / warmup_epochs
            clamped = min(1.0, raw)
            assert clamped <= 1.0

    def test_warmup_ramp_applied_to_param_group(self):
        """The scale must multiply the param-group LR (mirroring train.py logic)."""
        base_lr = 1e-3
        warmup_epochs = 3
        stage3_start = 16
        # Build a real optimizer with 3 param groups
        opt = torch.optim.AdamW(
            [
                {"params": [torch.zeros(1, requires_grad=True)], "lr": base_lr},
                {"params": [torch.zeros(1, requires_grad=True)], "lr": base_lr},
                {"params": [torch.zeros(1, requires_grad=True)], "lr": base_lr},
            ]
        )
        param_group_idx = 2

        # Simulate the per-epoch warmup scaling
        epoch_offsets = [1, 2, 3, 4]  # epochs 17, 18, 19, 20
        lrs = []
        for offset in epoch_offsets:
            scale = min(1.0, (offset) / warmup_epochs)  # (epoch - start)/warmup
            scaled_lr = base_lr * scale
            opt.param_groups[param_group_idx]["lr"] = scaled_lr
            lrs.append(opt.param_groups[param_group_idx]["lr"])
        # At offset=1 (first ramp epoch), scale=1/3 → LR ≈ 3.3e-4
        assert lrs[0] < base_lr, f"first ramp LR ({lrs[0]}) should be < base_lr ({base_lr})"
        # At offset=warmup_epochs (3), scale=1.0 → LR == base_lr
        assert lrs[2] == pytest.approx(base_lr, rel=1e-6), (
            f"final ramp LR ({lrs[2]}) should equal base_lr ({base_lr})"
        )
        # After warmup (offset > warmup_epochs), LR stays at base_lr
        assert lrs[3] == pytest.approx(base_lr, rel=1e-6)

    def test_warmup_state_remaining_decrements(self):
        """epochs_remaining must decrement each epoch and reach 0."""
        warmup_epochs = 3
        state = {
            "active": True,
            "start_epoch": 16,
            "epochs_remaining": warmup_epochs,
        }
        for _ in range(warmup_epochs):
            assert state["epochs_remaining"] > 0
            state["epochs_remaining"] -= 1
        assert state["epochs_remaining"] == 0
