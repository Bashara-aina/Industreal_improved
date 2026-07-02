"""Regression tests for the 2026-07-02 Fable RF4 consultation fixes (F1-F18).

Each test pins one of the fixes documented in
analyses/consult_2026_06_10/96-FABLE-RF4-CONSULTATION-ANSWER.md so it cannot
silently regress. Tests follow the repo's existing conventions
(test_loss_kendall.py): functional where the behavior is importable without
the dataset/GPU, source-level assertions where the fix lives inline in
train_one_epoch/main.
"""
import os
import sys

import pytest
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, os.path.join(_ROOT, 'src')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

TRAIN_PY = os.path.join(_ROOT, 'src', 'training', 'train.py')
LOSSES_PY = os.path.join(_ROOT, 'src', 'training', 'losses.py')


# ---------------------------------------------------------------------------
# F1 — seq-batch grad wipe must never blank accumulated backbone/FPN grads
# ---------------------------------------------------------------------------
class TestF1SeqBatchGradWipe:
    def test_destructive_wipe_removed(self):
        src = open(TRAIN_PY).read()
        # The old block set .grad = None for backbone/fpn params right after
        # the seq backward, erasing accumulated non-seq gradients.
        assert 'Zero backbone + FPN gradients on seq batches' not in src, (
            'F1 regression: the destructive seq-batch grad wipe is back'
        )
        assert '_bbfpn_grad_snapshot' in src, (
            'F1 regression: snapshot-restore path for DETACH_PSR_FPN=False missing'
        )

    def test_snapshot_before_backward(self):
        src = open(TRAIN_PY).read()
        snap = src.index('_bbfpn_grad_snapshot = {}')
        bwd = src.index('scaler.scale(loss_seq).backward()')
        assert snap < bwd, 'F1: snapshot must be taken BEFORE the seq backward'


# ---------------------------------------------------------------------------
# F3/F3b — lv_psr gets gradient only from batches with a real PSR loss
# ---------------------------------------------------------------------------
class TestF3PsrLogVarGuard:
    def _make_criterion(self):
        from src import config as C
        C.USE_KENDALL = True
        C.KENDALL_FIXED_WEIGHTS = False
        C.STAGED_TRAINING = False
        C.USE_PSR_TRANSITION = True
        from src.training.losses import MultiTaskLoss
        crit = MultiTaskLoss()
        crit.train_det = False
        crit.train_pose = False
        crit.train_act = False
        crit.train_psr = True
        crit.set_epoch(3)
        return crit

    def test_perframe_batch_gives_lv_psr_no_gradient(self):
        crit = self._make_criterion()
        out = {'psr_logits': torch.randn(4, 11, requires_grad=True)}
        tgt = {'psr_labels': torch.randint(0, 2, (4, 11)).float()}
        total, loss_dict = crit(out, tgt)
        crit.zero_grad()
        if total.requires_grad:
            total.backward()
        g = crit.log_var_psr.grad
        assert g is None or abs(g.item()) < 1e-12, (
            'F3 regression: lv_psr received gradient on a structurally-zero '
            'per-frame PSR batch (the +lv_psr term leaked back in)'
        )
        # F3b: the sensitivity penalty must not leak either — displayed psr
        # loss stays at the logging epsilon, not a real value.
        assert loss_dict.get('psr', 0.0) < 1e-4, (
            'F3b regression: per-frame PSR loss is non-trivially nonzero under '
            'the transition objective (sensitivity penalty leaked through)'
        )

    def test_sequence_batch_gives_lv_psr_gradient(self):
        crit = self._make_criterion()
        out = {'psr_logits': torch.randn(2, 8, 11, requires_grad=True)}
        tgt = {'psr_labels': torch.randint(0, 2, (2, 8, 11)).float()}
        total, _ = crit(out, tgt)
        crit.zero_grad()
        total.backward()
        g = crit.log_var_psr.grad
        assert g is not None and abs(g.item()) > 1e-12, (
            'F3 over-fix: lv_psr no longer learns from live sequence batches'
        )


# ---------------------------------------------------------------------------
# F4 — OneCycle peak factor is config-driven; F4b — re-applied after resume
# ---------------------------------------------------------------------------
class TestF4OneCyclePeak:
    def test_peak_factor_config_driven(self):
        src = open(TRAIN_PY).read()
        assert "getattr(C, 'ONE_CYCLE_PEAK_FACTOR'" in src, (
            'F4 regression: OneCycle peak factor hardcoded again'
        )

    def test_resume_reapplies_config_max_lr(self):
        src = open(TRAIN_PY).read()
        assert '_one_cycle_max_lr_cfg' in src and 'F4b' in src, (
            'F4b regression: resume path no longer re-applies config max_lr '
            '(optimizer.load_state_dict silently restores checkpoint max_lr)'
        )

    def test_torch_restores_max_lr_on_load(self):
        """Documents the gotcha F4b exists for: load_state_dict brings back
        the OLD max_lr hyperparams."""
        p = torch.nn.Parameter(torch.zeros(3))
        opt = torch.optim.AdamW([{'params': [p], 'lr': 1e-3}])
        torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=[1e-3], total_steps=100, pct_start=0.1)
        state = opt.state_dict()
        opt2 = torch.optim.AdamW([{'params': [p], 'lr': 1e-3}])
        torch.optim.lr_scheduler.OneCycleLR(
            opt2, max_lr=[2e-3], total_steps=100, pct_start=0.1)
        assert abs(opt2.param_groups[0]['max_lr'] - 2e-3) < 1e-12
        opt2.load_state_dict(state)
        assert abs(opt2.param_groups[0]['max_lr'] - 1e-3) < 1e-12, (
            'torch behavior changed: load_state_dict no longer restores '
            'max_lr — F4b re-application may be obsolete (review it)'
        )


# ---------------------------------------------------------------------------
# F6 — BF16 AMP helpers
# ---------------------------------------------------------------------------
class TestF6AmpHelpers:
    def test_amp_dtype_and_scaler_gating(self):
        try:
            from src.training import train as T
        except ImportError:
            pytest.skip('train.py deps unavailable in this environment')
        from src import config as C
        old_dtype, old_mp = getattr(C, 'AMP_DTYPE', 'bf16'), C.MIXED_PRECISION
        try:
            C.AMP_DTYPE = 'bf16'
            assert T._amp_dtype() is torch.bfloat16
            C.MIXED_PRECISION = True
            assert T._amp_scaler_enabled() is False, 'no GradScaler for bf16'
            C.AMP_DTYPE = 'fp16'
            assert T._amp_dtype() is torch.float16
            assert T._amp_scaler_enabled() is True, 'GradScaler required for fp16'
            C.MIXED_PRECISION = False
            assert T._amp_scaler_enabled() is False
        finally:
            C.AMP_DTYPE, C.MIXED_PRECISION = old_dtype, old_mp

    def test_no_bare_autocast_left(self):
        src = open(TRAIN_PY).read()
        assert "autocast('cuda', enabled=C.MIXED_PRECISION)" not in src, (
            'F6 regression: an autocast site lost its dtype=_amp_dtype() argument'
        )


# ---------------------------------------------------------------------------
# F13 — sentinels must fire on non-seq steps (odd offset)
# ---------------------------------------------------------------------------
class TestF13SentinelParity:
    def test_kendall_sentinel_fires_on_odd_offset(self):
        try:
            from src.training import train as T
        except ImportError:
            pytest.skip('train.py deps unavailable in this environment')

        class _FakeLV:
            def __init__(self):
                self._t = torch.zeros(1)
                self.grad = None
            def item(self):
                return 0.0

        class _FakeCrit:
            log_var_det = _FakeLV()
            log_var_pose = _FakeLV()
            log_var_act = _FakeLV()
            log_var_psr = _FakeLV()

        fired = []
        real_info = T.logger.info
        T.logger.info = lambda *a, **k: fired.append(a)
        try:
            T._log_kendall_gradient_sentinel(_FakeCrit(), 500, 500)
            assert not fired, (
                'F13: sentinel fired at step % interval == 0 — with an even '
                'seq cadence these steps are ALL seq batches and the function '
                'is never reached there in real training'
            )
            T._log_kendall_gradient_sentinel(_FakeCrit(), 501, 500)
            assert fired, 'F13 regression: sentinel no longer fires at offset 1'
        finally:
            T.logger.info = real_info

    def test_grad_norm_probe_trigger_is_odd(self):
        src = open(TRAIN_PY).read()
        fn = src[src.index('def _log_per_head_grad_norm'):]
        fn = fn[:fn.index('\ndef ')] if '\ndef ' in fn else fn
        assert 'step_idx % log_interval != 1' in fn, (
            'F13 regression: per-head grad-norm probe trigger reverted to the '
            'structurally-dead == 0 parity'
        )


# ---------------------------------------------------------------------------
# F14 — Kendall log_vars excluded from weight decay
# ---------------------------------------------------------------------------
class TestF14LogVarWeightDecay:
    def test_loss_param_group_has_zero_wd(self):
        src = open(TRAIN_PY).read()
        assert src.count(
            "{'params': loss_params, 'lr': head_lr, 'weight_decay': 0.0}"
        ) == 2, (
            'F14 regression: the Kendall log_var param group lost its '
            'weight_decay=0.0 in the Lion and/or AdamW branch'
        )


# ---------------------------------------------------------------------------
# F16 — ablation presets exist and apply with the intended flags
# ---------------------------------------------------------------------------
class TestF16AblationPresets:
    @pytest.mark.parametrize('preset,flags', [
        ('ablation_det_only',  dict(TRAIN_DET=True,  TRAIN_ACT=False, TRAIN_PSR=False, TRAIN_HEAD_POSE=False)),
        ('ablation_act_only',  dict(TRAIN_DET=False, TRAIN_ACT=True,  TRAIN_PSR=False, TRAIN_HEAD_POSE=False)),
        ('ablation_psr_only',  dict(TRAIN_DET=False, TRAIN_ACT=False, TRAIN_PSR=True,  TRAIN_HEAD_POSE=False)),
        ('ablation_pose_only', dict(TRAIN_DET=False, TRAIN_ACT=False, TRAIN_PSR=False, TRAIN_HEAD_POSE=True)),
    ])
    def test_single_task_presets(self, preset, flags):
        from src import config as C
        C.apply_preset(preset)
        for k, v in flags.items():
            assert getattr(C, k) == v, f'{preset}: {k} != {v}'
        # identical data distribution + hyperparams across the whole matrix
        assert abs(C.DET_GT_FRAME_FRACTION - 0.4) < 1e-9
        assert C.EFFECTIVE_BATCH == 24
        assert C.STAGED_TRAINING is False

    def test_act_only_zeroes_det_conf(self):
        from src import config as C
        C.apply_preset('ablation_act_only')
        assert C.ZERO_DET_CONF_FOR_RECOVERY is True, (
            'act-only ablation must not feed untrained det confidences into '
            'the activity input'
        )


# ---------------------------------------------------------------------------
# F17 — data package re-exports (missing __init__.py broke fresh clones)
# ---------------------------------------------------------------------------
class TestF17DataPackage:
    def test_data_package_reexports(self):
        try:
            import data as ds
        except ImportError:
            pytest.skip('data package deps unavailable')
        for name in ('IndustRealMultiTaskDataset', 'collate_fn',
                     'collate_fn_sequences', 'clear_frame_cache'):
            assert hasattr(ds, name), (
                f'F17 regression: data package no longer re-exports {name} — '
                'train.py getattr calls will fail on a fresh clone'
            )


# ---------------------------------------------------------------------------
# F18 — activity ramp applied exactly once (was ramp^2: loss-level AND
# Kendall-precision-level during warmup)
# ---------------------------------------------------------------------------
class TestF18SingleActivityRamp:
    def _total_at(self, counter, ramp_epochs=4):
        from src import config as C
        C.USE_KENDALL = True
        C.KENDALL_FIXED_WEIGHTS = False
        C.STAGED_TRAINING = False
        C.ACT_RAMP_EPOCHS = ramp_epochs
        from src.training.losses import MultiTaskLoss
        crit = MultiTaskLoss(train_det=False, train_pose=False,
                             train_act=True, train_psr=False)
        crit.act_loss_fn = torch.nn.CrossEntropyLoss()  # dataset-independent
        crit._act_epoch_counter = counter
        crit._act_warmup_epochs = ramp_epochs
        crit._current_epoch = counter
        torch.manual_seed(0)
        out = {'act_logits': torch.randn(8, 10)}
        tgt = {'activity': torch.randint(0, 10, (8,))}
        total, _ = crit(out, tgt)
        return float(total.detach())

    def test_ramp_is_linear_not_squared(self):
        ratio = self._total_at(0) / self._total_at(10)
        # single application: (0+1)/4 = 0.25; the historical double-ramp bug
        # (loss * ramp AND prec_act * ramp) would give 0.0625.
        assert abs(ratio - 0.25) < 1e-3, (
            f'F18 regression: activity warmup contribution ratio {ratio:.4f} '
            f'!= 0.25 — ramp is being applied twice (or not at all)'
        )

    def test_prec_act_ramp_removed_from_kendall_block(self):
        src = open(LOSSES_PY).read()
        assert 'prec_act = prec_act * ((self._act_epoch_counter' not in src, (
            'F18 regression: Kendall-precision activity ramp is back — '
            'warmup supervision becomes ramp^2 again'
        )
        assert src.count('prec_act = prec_act * act_ramp') == 0, (
            'F18 regression: staged-path prec_act ramp is back'
        )
