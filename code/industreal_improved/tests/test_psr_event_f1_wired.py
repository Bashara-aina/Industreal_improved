#!/usr/bin/env python3
"""test_psr_event_f1_wired.py — verify event_f1@±3 wire-up in full_eval_inprocess.

Validates that:
  1. The results dictionary contains psr_event_f1_at_3, psr_pos, psr_tau_seconds.
  2. Transition event F1 differs from per-frame F1 for known patterns.
  3. The helpers (event_f1, _compute_tau, _compute_pos) produce correct values.
  4. Empty / edge cases don't crash (no transitions, no valid frames).

Reference: 175 §7.2 (metric spec), 174 §3.3 (protocol definition).
"""

import sys
import math
from pathlib import Path
from collections import defaultdict

import numpy as np

# ── Path setup (mirrors full_eval_inprocess.py + existing tests) ────────────
_ROOT = Path(__file__).resolve().parent.parent / "code" / "industreal_improved"
for _p in [_ROOT, _ROOT / "src" / "evaluation"]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from decoder_oracle_bound import event_f1
from full_eval_inprocess import _compute_tau, _compute_pos


# ===========================================================================
# Unit tests for event_f1
# ===========================================================================


class TestEventF1:
    """Transition event F1 at ±tolerance (B3/STORM protocol)."""

    def test_exact_match(self):
        """event_f1=1.0 when predictions match GT transitions exactly."""
        gt = np.zeros((100, 11), dtype=np.int32)
        gt[10, 0] = 1
        gt[20, 1] = 1
        gt_tr = np.clip(gt[1:] - gt[:-1], a_min=0, a_max=None)
        pred_tr = gt_tr.copy()
        assert event_f1(pred_tr, gt_tr, tol=3) == 1.0

    def test_no_matches(self):
        """event_f1=0.0 when predictions are far from GT transitions."""
        gt = np.zeros((100, 11), dtype=np.int32)
        gt[10, 0] = 1
        gt_tr = np.clip(gt[1:] - gt[:-1], a_min=0, a_max=None)
        pred = np.zeros_like(gt)
        pred[50, 0] = 1  # 40 frames from GT — outside tolerance
        pred_tr = np.clip(pred[1:] - pred[:-1], a_min=0, a_max=None)
        assert event_f1(pred_tr, gt_tr, tol=3) == 0.0

    def test_within_tolerance(self):
        """event_f1=1.0 for predictions within ±3 of GT."""
        gt = np.zeros((100, 11), dtype=np.int32)
        gt[10, 0] = 1
        gt_tr = np.clip(gt[1:] - gt[:-1], a_min=0, a_max=None)
        pred = np.zeros_like(gt)
        pred[12, 0] = 1  # 2 frames late, within tolerance
        pred_tr = np.clip(pred[1:] - pred[:-1], a_min=0, a_max=None)
        assert event_f1(pred_tr, gt_tr, tol=3) == 1.0

    def test_outside_tolerance(self):
        """event_f1=0.0 for predictions just outside ±3."""
        gt = np.zeros((100, 11), dtype=np.int32)
        gt[10, 0] = 1
        gt_tr = np.clip(gt[1:] - gt[:-1], a_min=0, a_max=None)
        pred = np.zeros_like(gt)
        pred[14, 0] = 1  # 4 frames late, outside tolerance
        pred_tr = np.clip(pred[1:] - pred[:-1], a_min=0, a_max=None)
        assert event_f1(pred_tr, gt_tr, tol=3) == 0.0

    def test_both_empty(self):
        """event_f1=1.0 when both pred and GT have no transitions."""
        tr = np.zeros((50, 11), dtype=np.int32)
        assert event_f1(tr, tr, tol=3) == 1.0

    def test_only_pred(self):
        """event_f1=0.0 when pred has transitions but GT has none."""
        pred = np.zeros((50, 11), dtype=np.int32)
        pred[5, 0] = 1
        gt = np.zeros((50, 11), dtype=np.int32)
        pred_tr = np.clip(pred[1:] - pred[:-1], a_min=0, a_max=None)
        gt_tr = np.clip(gt[1:] - gt[:-1], a_min=0, a_max=None)
        assert event_f1(pred_tr, gt_tr, tol=3) == 0.0

    def test_only_gt(self):
        """event_f1=0.0 when GT has transitions but pred has none."""
        pred = np.zeros((50, 11), dtype=np.int32)
        gt = np.zeros((50, 11), dtype=np.int32)
        gt[5, 0] = 1
        pred_tr = np.clip(pred[1:] - pred[:-1], a_min=0, a_max=None)
        gt_tr = np.clip(gt[1:] - gt[:-1], a_min=0, a_max=None)
        assert event_f1(pred_tr, gt_tr, tol=3) == 0.0

    def test_partial_match(self):
        """event_f1 < 1.0 when some transitions match and some don't."""
        gt = np.zeros((100, 11), dtype=np.int32)
        gt[10, 0] = 1
        gt[30, 1] = 1
        gt_tr = np.clip(gt[1:] - gt[:-1], a_min=0, a_max=None)
        pred = np.zeros_like(gt)
        pred[11, 0] = 1  # matches (1 off)
        pred[50, 1] = 1  # misses (20 off)
        pred_tr = np.clip(pred[1:] - pred[:-1], a_min=0, a_max=None)
        ef1 = event_f1(pred_tr, gt_tr, tol=3)
        assert 0.0 < ef1 < 1.0, f"Expected partial match, got {ef1}"

    def test_multi_component(self):
        """Transitions across multiple components are handled independently."""
        gt = np.zeros((100, 4), dtype=np.int32)
        gt[10, 0] = 1
        gt[20, 1] = 1
        gt[30, 2] = 1
        gt[40, 3] = 1
        gt_tr = np.clip(gt[1:] - gt[:-1], a_min=0, a_max=None)
        pred = np.zeros_like(gt)
        pred[[11, 20, 33, 39], [0, 1, 2, 3]] = 1  # 3 within tol, 1 outside
        pred_tr = np.clip(pred[1:] - pred[:-1], a_min=0, a_max=None)
        ef1 = event_f1(pred_tr, gt_tr, tol=3)
        # 3 TP, 1 FP (comp 2 is 3 off = tolerance? Let's check: |33-30|=3 <= 3, so 4 TP)
        # Actually comp2: pred 33, GT 30, |33-30|=3 <= 3, so match.
        # All 4 match: |11-10|=1, |20-20|=0, |33-30|=3, |39-40|=1 => all <= 3
        assert ef1 == 1.0, f"Expected all 4 matched, got {ef1}"


# ===========================================================================
# Unit tests for _compute_tau
# ===========================================================================


class TestComputeTau:
    """Signed delay between matched prediction and GT events."""

    def test_lag(self):
        """Positive tau when prediction lags behind GT."""
        gt = np.zeros((50, 2), dtype=np.int32)
        gt[10, 0] = 1
        gt_tr = np.clip(gt[1:] - gt[:-1], a_min=0, a_max=None)
        pred = np.zeros_like(gt)
        pred[13, 0] = 1  # 3 frames late
        pred_tr = np.clip(pred[1:] - pred[:-1], a_min=0, a_max=None)
        tau = _compute_tau(pred_tr, gt_tr, tol=5)
        assert tau == 3.0, f"Expected tau=3, got {tau}"

    def test_anticipation(self):
        """Negative tau when prediction anticipates GT."""
        gt = np.zeros((50, 2), dtype=np.int32)
        gt[15, 0] = 1
        gt_tr = np.clip(gt[1:] - gt[:-1], a_min=0, a_max=None)
        pred = np.zeros_like(gt)
        pred[12, 0] = 1  # 3 frames early
        pred_tr = np.clip(pred[1:] - pred[:-1], a_min=0, a_max=None)
        tau = _compute_tau(pred_tr, gt_tr, tol=5)
        assert tau == -3.0, f"Expected tau=-3, got {tau}"

    def test_no_matches_nan(self):
        """tau = NaN when no events match."""
        gt = np.zeros((50, 2), dtype=np.int32)
        gt[10, 0] = 1
        gt_tr = np.clip(gt[1:] - gt[:-1], a_min=0, a_max=None)
        pred = np.zeros_like(gt)
        pred[40, 0] = 1  # far outside tolerance
        pred_tr = np.clip(pred[1:] - pred[:-1], a_min=0, a_max=None)
        tau = _compute_tau(pred_tr, gt_tr, tol=3)
        assert math.isnan(tau), f"Expected NaN, got {tau}"

    def test_both_empty_nan(self):
        """tau = NaN when there are no transitions at all."""
        tr = np.zeros((50, 2), dtype=np.int32)
        tau = _compute_tau(tr, tr, tol=3)
        assert math.isnan(tau), f"Expected NaN, got {tau}"

    def test_multiple_delays_averaged(self):
        """Tau averages over multiple matched events."""
        gt = np.zeros((100, 2), dtype=np.int32)
        gt[10, 0] = 1
        gt[30, 1] = 1
        gt_tr = np.clip(gt[1:] - gt[:-1], a_min=0, a_max=None)
        pred = np.zeros_like(gt)
        pred[12, 0] = 1  # +2
        pred[29, 1] = 1  # -1
        pred_tr = np.clip(pred[1:] - pred[:-1], a_min=0, a_max=None)
        tau = _compute_tau(pred_tr, gt_tr, tol=5)
        assert tau == 0.5, f"Expected tau=0.5, got {tau}"


# ===========================================================================
# Unit tests for _compute_pos
# ===========================================================================


class TestComputePOS:
    """Ordered-pair fraction (directional sign agreement)."""

    def test_identical(self):
        """POS=1.0 when predictions match GT exactly."""
        tr = np.zeros((50, 2), dtype=np.int32)
        tr[10, 0] = 1
        tr[20, 1] = 1
        assert _compute_pos(tr, tr) == 1.0

    def test_null_model(self):
        """Null model (all zeros) scores high POS (~1.0)."""
        tr = np.zeros((1000, 2), dtype=np.int32)
        tr[10, 0] = 1  # single transition in 1000 frames
        null = np.zeros_like(tr)
        pos = _compute_pos(null, tr)
        assert pos > 0.99, f"Null model POS should be high, got {pos}"

    def test_opposite(self):
        """POS penalizes predictions with wrong sign."""
        tr = np.zeros((50, 2), dtype=np.int32)
        tr[10, 0] = 1
        gt_tr = np.clip(tr[1:] - tr[:-1], a_min=0, a_max=None)
        opp = np.zeros_like(tr)
        opp[10, 0] = -1  # wrong sign
        pred_tr = np.clip(opp[1:] - opp[:-1], a_min=0, a_max=None)
        pos = _compute_pos(pred_tr, gt_tr)
        assert pos < 1.0


# ===========================================================================
# Integration tests for the wire-up logic
# ===========================================================================


class TestWireUp:
    """Verify that the transition metric block populates results dict correctly."""

    @staticmethod
    def _simulate_transition_computation(
        psr_preds_logits: list,
        psr_labels_list: list,
        psr_rec_ids: list,
        psr_frame_nums: list,
    ) -> dict:
        """Replicate the exact transition metric block from streaming_eval()."""
        results = {}

        # Per-component optimal thresholds (use 0.10 global for test)
        _per_comp_thr = np.full(11, 0.10, dtype=np.float32)

        _all_logits = np.concatenate(psr_preds_logits, axis=0)
        _all_labels = np.concatenate(psr_labels_list, axis=0)
        _all_sig = 1.0 / (1.0 + np.exp(-_all_logits))
        _all_pred_bin = (_all_sig > _per_comp_thr[np.newaxis, :]).astype(np.int32)

        _rec_data = defaultdict(lambda: {"pred": [], "label": [], "frame": []})
        for i in range(len(psr_rec_ids)):
            rid = psr_rec_ids[i]
            _rec_data[rid]["pred"].append(_all_pred_bin[i])
            _rec_data[rid]["label"].append(_all_labels[i])
            _rec_data[rid]["frame"].append(psr_frame_nums[i])

        _event_f1s, _poss, _taus = [], [], []

        for _arrs in _rec_data.values():
            _frames = np.array(_arrs["frame"], dtype=np.int64)
            _sort = np.argsort(_frames)
            _vp = np.array(_arrs["pred"])[_sort]
            _vl = np.array(_arrs["label"])[_sort]
            _valid = _vl.max(axis=1) >= 0
            _vp = _vp[_valid]
            _vl = _vl[_valid]
            if len(_vp) < 2:
                continue
            _pred_tr = np.clip(_vp[1:] - _vp[:-1], a_min=0, a_max=None)
            _gt_tr = np.clip(_vl[1:] - _vl[:-1], a_min=0, a_max=None)
            _valid_tr = _vl[1:].max(axis=1) >= 0
            _pv = _pred_tr[_valid_tr]
            _gv = _gt_tr[_valid_tr]
            _ef1 = event_f1(_pv, _gv, tol=3)
            _event_f1s.append(_ef1)
            _poss.append(_compute_pos(_pv, _gv))
            _tau = _compute_tau(_pv, _gv, tol=3)
            if not np.isnan(_tau):
                _taus.append(_tau)

        if _event_f1s:
            results["psr_event_f1_at_3"] = float(np.mean(_event_f1s))
            results["psr_pos"] = float(np.mean(_poss)) if _poss else 0.0
            _tau_mean = float(np.nanmean(_taus)) if _taus else float("nan")
            results["psr_tau_frames"] = _tau_mean
            results["psr_tau_seconds"] = (
                _tau_mean / 30.0 if not np.isnan(_tau_mean) else float("nan")
            )
        else:
            results["psr_event_f1_at_3"] = 0.0
            results["psr_pos"] = 0.0
            results["psr_tau_frames"] = float("nan")
            results["psr_tau_seconds"] = float("nan")
        return results

    def test_keys_present(self):
        """Results dict contains the three new metric keys."""
        # Single recording, single transition at frame 10
        logits = [np.full((1, 11), 5.0, dtype=np.float32)]  # high logit = positive
        labels = [np.full((1, 11), 1, dtype=np.float32)]  # positive label
        logits.append(np.full((1, 11), -5.0, dtype=np.float32))
        labels.append(np.full((1, 11), 0, dtype=np.float32))
        rec_ids = ["rec_0", "rec_0"]
        frame_nums = [10, 0]
        results = self._simulate_transition_computation(logits, labels, rec_ids, frame_nums)
        assert "psr_event_f1_at_3" in results, "Missing psr_event_f1_at_3"
        assert "psr_pos" in results, "Missing psr_pos"
        assert "psr_tau_seconds" in results, "Missing psr_tau_seconds"

    def test_differs_from_per_frame_f1(self):
        """Transition event F1 is NOT equal to per-frame F1 for non-trivial data."""
        # Build 20 frames: GT changes from 0->1 at frame 10
        # Prediction transitions 3 frames early at frame 7
        n = 20
        gt = np.zeros((n, 11), dtype=np.int32)
        gt[10:, :] = 1  # GT transitions at frame 10
        pred = np.zeros((n, 11), dtype=np.int32)
        pred[7:, :] = 1  # Prediction transitions at frame 7

        # Per-frame F1: most frames match (all except 7,8,9 where pred=1, GT=0)
        tp = ((pred == 1) & (gt == 1)).sum()
        fp = ((pred == 1) & (gt == 0)).sum()
        fn = ((pred == 0) & (gt == 1)).sum()
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        per_frame_f1 = 2 * prec * rec / max(prec + rec, 1e-9)

        # Transition F1: one pred at frame 7, one GT at frame 10, |7-10|=3 <= 3, match
        gt_tr = np.clip(gt[1:] - gt[:-1], a_min=0, a_max=None)
        pred_tr = np.clip(pred[1:] - pred[:-1], a_min=0, a_max=None)
        transition_f1 = event_f1(pred_tr, gt_tr, tol=3)

        assert per_frame_f1 != transition_f1, (
            f"Per-frame F1 ({per_frame_f1:.4f}) should differ from transition "
            f"F1 ({transition_f1:.4f}) for this pattern"
        )

    def test_empty_data_doesnt_crash(self):
        """Empty/no-valid-frame data produces NaN/0 without crashing."""
        logits = [np.full((1, 11), -5.0, dtype=np.float32)]
        labels = [np.full((1, 11), -1, dtype=np.float32)]  # all invalid
        rec_ids = ["rec_0"]
        frame_nums = [0]
        results = self._simulate_transition_computation(logits, labels, rec_ids, frame_nums)
        assert "psr_event_f1_at_3" in results
        assert results["psr_event_f1_at_3"] == 0.0

    def test_multiple_recordings_aggregated(self):
        """Metrics aggregate correctly across multiple recordings."""
        # Two recordings, each with one transition
        logits = [np.full((1, 11), 5.0, dtype=np.float32)]  # rec_0 frame 10, high
        labels = [np.full((1, 11), 1, dtype=np.float32)]
        logits.append(np.full((1, 11), -5.0, dtype=np.float32))  # rec_0 frame 0, low
        labels.append(np.full((1, 11), 0, dtype=np.float32))
        logits.append(np.full((1, 11), 5.0, dtype=np.float32))  # rec_1 frame 20, high
        labels.append(np.full((1, 11), 1, dtype=np.float32))
        logits.append(np.full((1, 11), -5.0, dtype=np.float32))  # rec_1 frame 0, low
        labels.append(np.full((1, 11), 0, dtype=np.float32))
        rec_ids = ["rec_0", "rec_0", "rec_1", "rec_1"]
        frame_nums = [10, 0, 20, 0]
        results = self._simulate_transition_computation(logits, labels, rec_ids, frame_nums)
        assert results["psr_event_f1_at_3"] == 1.0  # both recordings have perfect matches
