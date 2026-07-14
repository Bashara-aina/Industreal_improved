#!/usr/bin/env python3
"""Unit test for the authors' PSR evaluation pipeline (Options 2 & 3).

Tests the following functions in isolation:
  - _load_psr_labels
  - _convert_states_to_steps
  - _apply_psr_naive_pass
  - _apply_psr_accumulated_pass
  - _determine_performance
  - _load_psr_raw_states
  - compute_authors_psr_metrics (end-to-end with fake recordings)
  - compute_authors_psr_state_accuracy (end-to-end with fake recordings)

Run:  python3 -m pytest src/test_authors_psr_pipeline.py -v -p no:cacheprovider
Or:   python3 src/test_authors_psr_pipeline.py
"""

"""Unit test for authors' PSR evaluation pipeline.

Run from project root:
  python3 -m pytest src/test_authors_psr_pipeline.py -v -p no:cacheprovider
Or:
  cd src && python3 test_authors_psr_pipeline.py
"""

import sys
import os
import tempfile
import numpy as np
from pathlib import Path

# When running from src/ directly, project root is parent
_PROJ = Path(__file__).resolve().parent.parent
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))

# ---------------------------------------------------------------------------
# Monkey-patch config flags needed by the eval module BEFORE importing it.
# The eval module does `from src import config as C` at import time, so we
# need to set these before the import.
# ---------------------------------------------------------------------------
import types as _types

_C = _types.ModuleType("config_stub")
_C.RECORDINGS_ROOT = Path("/tmp/psr_test_fake_root")
_C.PSR_AUTHORS_METHOD = "naive"
_C.PSR_AUTHORS_CONF_THRESHOLD = 0.6
_C.PSR_AUTHORS_CUM_THRESHOLD = 8.0
_C.PSR_AUTHORS_CUM_DECAY = 0.75
_C.SKIP_DET_METRICS_EVAL = True
_C.SKIP_SEGMENT_METRICS_EVAL = True
# Minimally satisfy other config lookups
_C.IMG_WIDTH = 1280
_C.IMG_HEIGHT = 720
_C.NUM_DET_CLASSES = 24
_C.EVAL_MAX_BATCHES = 20
_C.DET_EVAL_SCORE_THRESH = 0.5
_C.DET_EVAL_MAX_PER_IMAGE = 300

import src.config as C

# Inject flags not in the real config yet
for _key in (
    "USE_AUTHORS_PSR_EVAL",
    "PSR_AUTHORS_METHOD",
    "PSR_AUTHORS_CONF_THRESHOLD",
    "PSR_AUTHORS_CUM_THRESHOLD",
    "PSR_AUTHORS_CUM_DECAY",
    "USE_AUTHORS_PSR_STATE_ACCURACY",
    "SKIP_DET_METRICS_EVAL",
    "SKIP_SEGMENT_METRICS_EVAL",
):
    setattr(C, _key, getattr(_C, _key, None))

# Now import the eval functions
from src.evaluation.evaluate import (
    compute_authors_psr_metrics,
    compute_authors_psr_state_accuracy,
    _load_psr_labels,
    _load_psr_raw_states,
    _convert_states_to_steps,
    _apply_psr_naive_pass,
    _apply_psr_accumulated_pass,
    _determine_performance,
)

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _make_entry(frame: int, action_id: int, conf: int = 1) -> dict:
    return {"frame": frame, "id": action_id, "conf": conf}


def _make_fake_recording(tmpdir: str, rec_id: str, split: str,
                         events: list, raw_frames: list | None = None):
    """Create a fake recording directory with PSR_labels.csv and optionally
    PSR_labels_raw.csv.

    Args:
        events: list of (frame_num, action_id) tuples for PSR_labels.csv
        raw_frames: list of (frame_num, [11 floats]) for PSR_labels_raw.csv
    """
    rec_dir = Path(tmpdir) / split / rec_id
    rec_dir.mkdir(parents=True, exist_ok=True)

    # PSR_labels.csv (step events)
    if events:
        csv_path = rec_dir / "PSR_labels.csv"
        with open(csv_path, "w") as f:
            for frame, aid in events:
                f.write(f"{frame:06d}.jpg,{aid},test_action_{aid}\n")

    # PSR_labels_raw.csv (per-component states)
    if raw_frames:
        raw_path = rec_dir / "PSR_labels_raw.csv"
        with open(raw_path, "w") as f:
            for frame, comps in raw_frames:
                comps_str = ",".join(str(int(v)) for v in comps)
                f.write(f"{frame:06d}.jpg,{comps_str}\n")


# ===================================================================
# Tests
# ===================================================================

def test_load_psr_labels():
    """Parse PSR_labels.csv correctly."""
    with tempfile.TemporaryDirectory() as td:
        fp = Path(td) / "PSR_labels.csv"
        fp.write_text(
            "000521.jpg,3,Install front chassis\n"
            "000783.jpg,9,Install rear chassis\n"
            "002233.jpg,17,Remove front rear chassis pin\n"
        )
        events = _load_psr_labels(str(fp))
        assert len(events) == 3
        assert events[0]["frame"] == 521
        assert events[0]["id"] == 3
        assert events[1]["id"] == 9
        assert events[2]["frame"] == 2233
        assert events[2]["id"] == 17


def test_load_psr_raw_states():
    """Parse and fill-forward PSR_labels_raw.csv."""
    with tempfile.TemporaryDirectory() as td:
        fp = Path(td) / "PSR_labels_raw.csv"
        fp.write_text(
            "000000.jpg,0,0,0,0,0,0,0,0,0,0,0\n"
            "000521.jpg,1,1,0,0,0,0,0,0,0,0,0\n"
            "000783.jpg,1,1,1,0,0,1,0,0,0,0,0\n"
        )
        dense = _load_psr_raw_states(str(fp), 1000)
        assert dense.shape == (1000, 11)
        # Frame 0: all zeros
        assert np.allclose(dense[0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        # Frame 521: comp0=1, comp1=1
        assert np.allclose(dense[521], [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        # Frame 783: comp0=1, comp1=1, comp2=1, comp5=1
        assert np.allclose(dense[783], [1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0])
        # Frame 800: fill-forward = same as frame 783
        assert np.allclose(dense[800], dense[783])


def test_load_psr_raw_states_empty_file():
    """Return zeros when PSR_labels_raw.csv doesn't exist."""
    dense = _load_psr_raw_states("/nonexistent/path.csv", 100)
    assert dense.shape == (100, 11)
    assert np.allclose(dense, 0.0)


def test_load_psr_raw_states_handles_neg_one():
    """-1 values should NOT be carried forward (error transients)."""
    with tempfile.TemporaryDirectory() as td:
        fp = Path(td) / "PSR_labels_raw.csv"
        fp.write_text(
            "000000.jpg,0,0,0,0,0,0,0,0,0,0,0\n"
            "000100.jpg,1,-1,0,0,0,0,0,0,0,0,0\n"
        )
        dense = _load_psr_raw_states(str(fp), 200)
        # Frame 100: comp0=1, comp1 stays 0 (because -1 is error, keep last valid)
        assert dense[100, 0] == 1.0
        assert dense[100, 1] == 0.0  # NOT carried forward


def test_convert_states_to_steps():
    """Detect install/remove from consecutive binary states."""
    prev = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    curr = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    events = _convert_states_to_steps(prev, curr, 100)
    assert len(events) == 1
    assert events[0]["frame"] == 100
    assert events[0]["id"] == 0  # k=0, install

    # remove
    prev = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    curr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    events = _convert_states_to_steps(prev, curr, 200)
    assert len(events) == 1
    assert events[0]["id"] == 2  # k=0, remove

    # multiple changes
    prev = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    curr = [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    events = _convert_states_to_steps(prev, curr, 300)
    assert len(events) == 2
    assert events[0]["id"] == 0  # comp0 install
    assert events[1]["id"] == 6  # comp2 install (2*3+0)


def test_naive_pass_accepts_confident_transitions():
    """NaivePSR: only accept 0→1 when sigmoid > threshold."""
    pred_bin = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ], dtype=np.int32)

    pred_logits = np.array([
        [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        [0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],  # comp0=0.9 > 0.6 ✓
        [0.9, 0.7, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],  # comp1=0.7 > 0.6 ✓
    ])

    frame_nums = np.array([0, 100, 200])
    events = _apply_psr_naive_pass(pred_bin, pred_logits, frame_nums, conf_threshold=0.6)
    assert len(events) == 2
    assert events[0]["frame"] == 100
    assert events[0]["id"] == 0   # comp0 install
    assert events[1]["frame"] == 200
    assert events[1]["id"] == 3   # comp1 install


def test_naive_pass_rejects_low_confidence():
    """NaivePSR: reject 0→1 when sigmoid <= threshold."""
    pred_bin = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ], dtype=np.int32)
    pred_logits = np.array([
        [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        [0.55, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],  # 0.55 <= 0.6 → reject
    ])
    frame_nums = np.array([0, 100])
    events = _apply_psr_naive_pass(pred_bin, pred_logits, frame_nums, conf_threshold=0.6)
    assert len(events) == 0, f"Expected 0 events, got {len(events)}"


def test_naive_pass_accepts_remove():
    """NaivePSR: accept 1→0 when sigmoid < 1 - threshold."""
    pred_bin = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ], dtype=np.int32)
    pred_logits = np.array([
        [0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        [0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],  # 0.3 < 0.4 ✓
    ])
    frame_nums = np.array([0, 150])
    events = _apply_psr_naive_pass(pred_bin, pred_logits, frame_nums, conf_threshold=0.6)
    assert len(events) == 1
    assert events[0]["id"] == 2  # comp0 remove


def test_accumulated_pass():
    """AccumulatedConfidencePSR: accumulate and emit at threshold."""
    pred_bin = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ], dtype=np.int32)
    # |0.9 - 0.5| = 0.4 per action, need cum > 8.0
    # With 1 frame contribution, total = 0.4 → not enough
    pred_logits = np.array([
        [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        [0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
    ])
    frame_nums = np.array([0, 100])
    events = _apply_psr_accumulated_pass(pred_bin, pred_logits, frame_nums,
                                          cum_threshold=0.3, decay=0.75)
    # With cum_threshold=0.3 and 1 contribution of 0.4 > 0.3
    assert len(events) >= 1, "Expected at least 1 event with low threshold"


def test_accumulated_pass_requires_accumulation():
    """AccumulatedConfidencePSR: need enough accumulated confidence."""
    pred_bin = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ], dtype=np.int32)
    pred_logits = np.array([
        [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        [0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
    ])
    frame_nums = np.array([0, 100])
    # cum_threshold=1.0, single contribution |0.9-0.5|=0.4 < 1.0
    events = _apply_psr_accumulated_pass(pred_bin, pred_logits, frame_nums,
                                          cum_threshold=1.0, decay=0.75)
    assert len(events) == 0, f"Expected 0 events, got {len(events)}"


def test_determine_performance_perfect_match():
    """Perfect F1=1.0, POS=1.0, delay=0."""
    import math
    gt = [
        {"frame": 100, "id": 0, "description": "Install base"},
        {"frame": 200, "id": 3, "description": "Install front chassis"},
    ]
    pred = [
        _make_entry(100, 0),
        _make_entry(200, 3),
    ]
    m = _determine_performance(gt, pred)
    assert math.isclose(m["f1"], 1.0, abs_tol=1e-5), f"F1={m['f1']}"
    assert m["pos"] == 1.0, f"POS={m['pos']}"
    assert m["avg_delay"] == 0.0, f"delay={m['avg_delay']}"
    assert m["system_TPs"] == 2


def test_determine_performance_no_pred():
    """No predictions = F1=0, POS=0."""
    gt = [{"frame": 100, "id": 0, "description": "Install base"}]
    pred = []
    m = _determine_performance(gt, pred)
    assert m["f1"] == 0.0
    assert m["pos"] == 0.0
    assert m["system_FNs"] == 1


def test_determine_performance_wrong_order():
    """Wrong order penalizes POS but may still have correct matches."""
    import math
    gt = [
        {"frame": 100, "id": 0, "description": "Install base"},
        {"frame": 200, "id": 3, "description": "Install front chassis"},
    ]
    pred = [
        _make_entry(200, 3),  # reversed order
        _make_entry(100, 0),
    ]
    m = _determine_performance(gt, pred)
    # F1 can still be 1.0 (all match), but POS should be < 1.0 (order wrong)
    assert math.isclose(m["f1"], 1.0, abs_tol=1e-5), f"F1={m['f1']}"
    assert m["pos"] < 1.0, f"POS={m['pos']} (expected < 1.0)"
    assert m["avg_delay"] == 0.0


# ===================================================================
# End-to-end tests with fake recordings
# ===================================================================

def test_compute_authors_psr_metrics_naive():
    """End-to-end: NaivePSR detects events and scores correctly."""
    import math
    with tempfile.TemporaryDirectory() as td:
        # Create a fake recording
        gt_events = [
            (100, 0),   # Install base at frame 100
            (200, 3),   # Install front chassis at frame 200
        ]
        _make_fake_recording(td, "test_rec_01", "val", gt_events)

        # Create synthetic predictions: comp0 goes 0→1 at frame 100 (sigmoid=0.9),
        # comp1 goes 0→1 at frame 200 (sigmoid=0.9)
        psr_preds_logits = [
            np.array([
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                [0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                [0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            ])
        ]
        psr_rec_ids = ["test_rec_01", "test_rec_01", "test_rec_01"]
        psr_frame_nums = [0, 100, 200]

        metrics = compute_authors_psr_metrics(
            psr_preds_logits, psr_rec_ids, psr_frame_nums,
            recordings_root=td, split="val",
        )
        assert metrics["authors_psr_recordings"] == 1, \
            f"Expected 1 recording, got {metrics['authors_psr_recordings']}"
        assert math.isclose(metrics["authors_psr_f1"], 1.0, abs_tol=1e-5), \
            f"Expected F1=1.0, got {metrics['authors_psr_f1']}"
        assert metrics["authors_psr_pos"] == 1.0, \
            f"Expected POS=1.0, got {metrics['authors_psr_pos']}"
        assert metrics["authors_psr_delay"] == 0.0, \
            f"Expected delay=0, got {metrics['authors_psr_delay']}"


def test_compute_authors_psr_metrics_no_recording_match():
    """When rec_id doesn't match any directory, returns zeros."""
    with tempfile.TemporaryDirectory() as td:
        gt_events = [(100, 0)]
        _make_fake_recording(td, "real_recording", "val", gt_events)

        psr_preds_logits = [np.array([[0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]])]
        psr_rec_ids = ["non_existent_rec"]
        psr_frame_nums = [100]

        metrics = compute_authors_psr_metrics(
            psr_preds_logits, psr_rec_ids, psr_frame_nums,
            recordings_root=td, split="val",
        )
        assert metrics["authors_psr_recordings"] == 0
        assert metrics["authors_psr_f1"] == 0.0


def test_compute_authors_psr_metrics_accumulated():
    """End-to-end with AccumulatedConfidencePSR."""
    import math
    import src.config as _C2
    _C2.PSR_AUTHORS_METHOD = "accumulated"
    _C2.PSR_AUTHORS_CUM_THRESHOLD = 0.3  # low to trigger with small data
    _C2.PSR_AUTHORS_CUM_DECAY = 0.75

    try:
        with tempfile.TemporaryDirectory() as td:
            gt_events = [(100, 0)]
            _make_fake_recording(td, "test_rec_acc", "val", gt_events)

            psr_preds_logits = [
                np.array([
                    [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                    [0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                ])
            ]
            psr_rec_ids = ["test_rec_acc", "test_rec_acc"]
            psr_frame_nums = [0, 100]

            metrics = compute_authors_psr_metrics(
                psr_preds_logits, psr_rec_ids, psr_frame_nums,
                recordings_root=td, split="val",
            )
            assert metrics["authors_psr_recordings"] == 1, \
                f"Expected 1 recording, got {metrics['authors_psr_recordings']}"
            assert math.isclose(metrics["authors_psr_f1"], 1.0, abs_tol=1e-5), \
                f"F1={metrics['authors_psr_f1']}"
    finally:
        _C2.PSR_AUTHORS_METHOD = "naive"
        _C2.PSR_AUTHORS_CUM_THRESHOLD = 8.0
        _C2.PSR_AUTHORS_CUM_DECAY = 0.75


def test_compute_authors_psr_state_accuracy():
    """End-to-end state accuracy with fake PSR_labels_raw.csv."""
    import math
    with tempfile.TemporaryDirectory() as td:
        # All 11 components go 0→1 at frame 1 so each has activity — otherwise
        # inactive components score 0/0=0.0 and drag macro-accuracy down.
        raw_frames = [
            (0, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            (1, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
            (2, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        ]
        _make_fake_recording(td, "test_rec_sa", "val", [], raw_frames=raw_frames)

        # Perfect predictions: exactly matches GT
        psr_preds_logits = [
            np.array([
                [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9],
                [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9],
            ])
        ]
        psr_rec_ids = ["test_rec_sa"] * 3
        psr_frame_nums = [0, 1, 2]

        metrics = compute_authors_psr_state_accuracy(
            psr_preds_logits, psr_rec_ids, psr_frame_nums,
            recordings_root=td, split="val",
        )
        assert metrics["state_recordings"] == 1, \
            f"Expected 1 rec, got {metrics['state_recordings']}"
        assert math.isclose(metrics["state_macro_accuracy"], 1.0, abs_tol=1e-5), \
            f"acc={metrics['state_macro_accuracy']}"
        assert math.isclose(metrics["state_macro_f1"], 1.0, abs_tol=1e-5), \
            f"F1={metrics['state_macro_f1']}"


def test_compute_authors_psr_state_accuracy_all_wrong():
    """State accuracy: all predictions wrong → macro F1 = 0."""
    with tempfile.TemporaryDirectory() as td:
        raw_frames = [
            (0,   [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        ]
        _make_fake_recording(td, "test_rec_sb", "val", [], raw_frames=raw_frames)

        # Predict all zeros, GT has comp0=1 → all FNs
        psr_preds_logits = [np.array([[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]])]
        psr_rec_ids = ["test_rec_sb"]
        psr_frame_nums = [0]

        metrics = compute_authors_psr_state_accuracy(
            psr_preds_logits, psr_rec_ids, psr_frame_nums,
            recordings_root=td, split="val",
        )
        assert metrics["state_recordings"] == 1
        # comp0: TP=0, FP=0, FN=1 → F1=0, acc=0
        # comps1-10: TP=0, FP=0, FN=0 → F1=0, acc=0 (total=0 -> 0.0)
        # macro-F1 = 0/11 = 0
        assert metrics["state_macro_f1"] == 0.0, \
            f"macro_F1={metrics['state_macro_f1']}"
        assert metrics["state_macro_accuracy"] == 0.0, \
            f"acc={metrics['state_macro_accuracy']}"


def test_no_pred_events_returns_zero_f1():
    """When no events are predicted (all sigmoids flat at 0.5), returns zeros."""
    with tempfile.TemporaryDirectory() as td:
        gt_events = [(100, 0)]
        _make_fake_recording(td, "flat_rec", "val", gt_events)

        # sigmoids all at 0.51 (barely above 0.5, below 0.6 threshold)
        psr_preds_logits = [
            np.array([[0.51] * 11, [0.51] * 11])
        ]
        psr_rec_ids = ["flat_rec", "flat_rec"]
        psr_frame_nums = [0, 100]

        metrics = compute_authors_psr_metrics(
            psr_preds_logits, psr_rec_ids, psr_frame_nums,
            recordings_root=td, split="val",
        )
        assert metrics["authors_psr_recordings"] == 0, \
            f"Expected 0 recs (flat sigmoids), got {metrics['authors_psr_recordings']}"
        assert metrics["authors_psr_f1"] == 0.0


# ===================================================================
# Main
# ===================================================================

def _run_all():
    """Run all test functions and print results."""
    tests = [
        ("test_load_psr_labels", test_load_psr_labels),
        ("test_load_psr_raw_states", test_load_psr_raw_states),
        ("test_load_psr_raw_states_empty_file", test_load_psr_raw_states_empty_file),
        ("test_load_psr_raw_states_handles_neg_one", test_load_psr_raw_states_handles_neg_one),
        ("test_convert_states_to_steps", test_convert_states_to_steps),
        ("test_naive_pass_accepts_confident_transitions", test_naive_pass_accepts_confident_transitions),
        ("test_naive_pass_rejects_low_confidence", test_naive_pass_rejects_low_confidence),
        ("test_naive_pass_accepts_remove", test_naive_pass_accepts_remove),
        ("test_accumulated_pass", test_accumulated_pass),
        ("test_accumulated_pass_requires_accumulation", test_accumulated_pass_requires_accumulation),
        ("test_determine_performance_perfect_match", test_determine_performance_perfect_match),
        ("test_determine_performance_no_pred", test_determine_performance_no_pred),
        ("test_determine_performance_wrong_order", test_determine_performance_wrong_order),
        ("test_compute_authors_psr_metrics_naive", test_compute_authors_psr_metrics_naive),
        ("test_compute_authors_psr_metrics_no_recording_match", test_compute_authors_psr_metrics_no_recording_match),
        ("test_compute_authors_psr_metrics_accumulated", test_compute_authors_psr_metrics_accumulated),
        ("test_compute_authors_psr_state_accuracy", test_compute_authors_psr_state_accuracy),
        ("test_compute_authors_psr_state_accuracy_all_wrong", test_compute_authors_psr_state_accuracy_all_wrong),
        ("test_no_pred_events_returns_zero_f1", test_no_pred_events_returns_zero_f1),
    ]

    passed = 0
    failed = 0
    for name, func in tests:
        try:
            func()
            print(f"  ✓ {name}")
            passed += 1
        except Exception as e:
            print(f"  ✗ {name}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*50}")
    print(f"  {passed} passed, {failed} failed out of {len(tests)}")
    return failed == 0


if __name__ == "__main__":
    success = _run_all()
    sys.exit(0 if success else 1)
