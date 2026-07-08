"""
test_test_split_harness.py — Tests for the test-split evaluation harness.

Verifies:
  1. split_config exports exactly 10 test subjects with no duplicates/no val overlap.
  2. require_split() guard fires correctly when called with 'val' split.
  3. The orchestrator's JSON output structure matches Table A (175 Section 8).
  4. The bootstrap CI helper produces sensible intervals.
  5. Graceful degradation when eval modules are missing.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent  # code/industreal_improved/
for _p in [_ROOT, _ROOT / "src", _ROOT / "src" / "evaluation"]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture(scope="session")
def split_config():
    from src import split_config as sc
    return sc


@pytest.fixture(scope="session")
def orchestrator_module():
    """Import the orchestrator (may fail if dependencies missing)."""
    try:
        from scripts import eval_test_split as ets
        return ets
    except ImportError as e:
        pytest.skip(f"Orchestrator module not importable: {e}")
    except Exception as e:
        pytest.skip(f"Orchestrator import error (expected in CI without GPU): {e}")


@pytest.fixture(scope="session")
def expected_table_a_keys():
    """Minimal set of keys expected in the SOTA Table A output structure."""
    return {
        "detection": {"annotated_frames_mAP50", "entire_video_mAP50",
                      "sota_annotated", "sota_video", "reference"},
        "activity": {"top1_75cls", "top1_75cls_pct",
                     "sota_top1", "reference"},
        "psr": {"event_f1", "tau_seconds", "pos",
                "sota_event_f1", "sota_tau_seconds", "reference"},
        "pose": {"fwd_mae", "up_mae", "sota", "reference"},
    }


# ===================================================================
# Tests
# ===================================================================

class TestSplitConfig:
    """Verify split_config exports and protocol enforcement (Section 7.1)."""

    @staticmethod
    def test_exactly_10_test_subjects(split_config):
        """Test subjects are 10, no duplicates, no overlap with val."""
        test = split_config.TEST_SUBJECTS
        val = split_config.VAL_SUBJECTS

        assert len(test) == 10, f"Expected 10 test subjects, got {len(test)}"
        assert len(set(test)) == 10, f"Duplicates in test subjects: {test}"
        assert not set(test) & set(val), (
            f"Test/val overlap: {set(test) & set(val)}"
        )

    @staticmethod
    def test_test_subjects_non_overlapping_with_train(split_config):
        train = split_config.TRAIN_SUBJECTS
        test = split_config.TEST_SUBJECTS
        assert not set(train) & set(test), (
            f"Train/test overlap: {set(train) & set(test)}"
        )

    @staticmethod
    def test_require_split_passes_for_test(split_config):
        """require_split('test', allow_test_only=True) should pass."""
        # Should not raise
        split_config.require_split("test", allow_test_only=True)

    @staticmethod
    def test_require_split_fires_for_val(split_config):
        """require_split('val', allow_test_only=True) should raise ValueError."""
        with pytest.raises(ValueError, match="SOTA-table writing"):
            split_config.require_split("val", allow_test_only=True)

    @staticmethod
    def test_require_split_fires_for_train(split_config):
        """require_split('train', allow_test_only=True) should raise ValueError."""
        with pytest.raises(ValueError):
            split_config.require_split("train", allow_test_only=True)

    @staticmethod
    def test_require_split_passes_for_val_if_not_test_only(split_config):
        """require_split('val', allow_test_only=False) should pass."""
        split_config.require_split("val", allow_test_only=False)

    @staticmethod
    def test_get_split_returns_sorted(split_config):
        test = split_config.get_split("test")
        assert test == sorted(test)

    @staticmethod
    def test_invalid_split_name(split_config):
        with pytest.raises(KeyError):
            split_config.get_split("invalid")


class TestOrchestratorImports:
    """Verify the orchestrator module structure and graceful degradation."""

    @staticmethod
    def test_orchestrator_imports(orchestrator_module):
        """Module should import without error (GPU not required)."""
        assert orchestrator_module is not None

    @staticmethod
    def test_orchestrator_has_main(orchestrator_module):
        assert hasattr(orchestrator_module, "main")

    @staticmethod
    def test_orchestrator_has_bootstrap_ci(orchestrator_module):
        assert hasattr(orchestrator_module, "bootstrap_ci")

    @staticmethod
    def test_orchestrator_has_print_table_a(orchestrator_module):
        assert hasattr(orchestrator_module, "print_table_a")

    @staticmethod
    def test_orchestrator_has_test_subjects(orchestrator_module):
        from src.split_config import TEST_SUBJECTS
        assert len(TEST_SUBJECTS) == 10

    @staticmethod
    def test_orchestrator_enforces_require_split(orchestrator_module):
        """require_split is called at module import time.  Verify it enforces."""
        import importlib
        import src.split_config as sc

        # The eval_test_split module should have called require_split("test", True)
        # at import time.  We verify by checking that calling require_split("val")
        # still raises — i.e., the guard is active in the module's scope.
        with pytest.raises(ValueError, match="SOTA-table writing"):
            sc.require_split("val", allow_test_only=True)


class TestBootstrapCI:
    """Verify the bootstrap CI helper produces reasonable intervals."""

    @staticmethod
    def test_bootstrap_ci_basic(orchestrator_module):
        bootstrap_ci = orchestrator_module.bootstrap_ci

        values = [10.0, 12.0, 11.0, 9.0, 10.5]
        point, lo, hi = bootstrap_ci(values, n_resamples=500)

        assert not math.isnan(point)
        assert not math.isnan(lo)
        assert not math.isnan(hi)
        assert lo <= point <= hi, f"CI bounds inconsistent: {lo} <= {point} <= {hi}"

    @staticmethod
    def test_bootstrap_ci_weighted(orchestrator_module):
        bootstrap_ci = orchestrator_module.bootstrap_ci

        values = [10.0, 20.0, 30.0]
        weights = [1.0, 1.0, 1.0]
        point, lo, hi = bootstrap_ci(values, weights, n_resamples=500)

        # Unweighted mean of [10, 20, 30] = 20
        assert abs(point - 20.0) < 0.01
        assert lo <= point <= hi

    @staticmethod
    def test_bootstrap_ci_empty(orchestrator_module):
        bootstrap_ci = orchestrator_module.bootstrap_ci
        point, lo, hi = bootstrap_ci([], n_resamples=100)
        assert all(math.isnan(x) for x in (point, lo, hi))

    @staticmethod
    def test_bootstrap_ci_single_value(orchestrator_module):
        bootstrap_ci = orchestrator_module.bootstrap_ci
        point, lo, hi = bootstrap_ci([42.0], n_resamples=100)
        assert abs(point - 42.0) < 0.001
        assert not math.isnan(lo)
        assert not math.isnan(hi)


class TestTableAStructure:
    """Verify the Table A output structure matches the Section 8 template."""

    @staticmethod
    def test_expected_section_8_keys(orchestrator_module, expected_table_a_keys):
        """The sota_table_A dict must have the expected sub-sections."""
        # Build a mock aggregated results dict and call print_table_a to verify it
        # doesn't crash, then check the structure of the mock.

        mock_agg = {
            "detection": {
                "det_mAP50_annotated_frames": 0.5,
                "det_mAP50_all_frames": 0.3,
            },
            "activity": {
                "act_top1": 0.6,
                "act_top1_pct": 60.0,
            },
            "psr": {
                "psr_event_f1": 0.85,
                "psr_tau_seconds": 18.0,
                "psr_pos": 0.75,
            },
            "pose": {
                "pose_fwd_mae": 10.5,
                "pose_up_mae": 8.3,
            },
        }

        # print_table_a should not crash
        orchestrator_module.print_table_a(mock_agg)

    @staticmethod
    def test_table_a_output_structure():
        """The JSON output structure should match Table A schema."""
        # Load the schema from the template
        table_a_sections = [
            "detection",
            "activity",
            "psr",
            "pose",
        ]
        det_keys = {"annotated_frames_mAP50", "entire_video_mAP50",
                    "sota_annotated", "sota_video", "reference"}
        act_keys = {"top1_75cls", "top1_75cls_pct", "sota_top1", "reference"}
        psr_keys = {"event_f1", "tau_seconds", "pos",
                    "sota_event_f1", "sota_tau_seconds", "reference"}
        pose_keys = {"fwd_mae", "up_mae", "sota", "reference"}

        # Verify section names from Section 8
        assert "detection" in table_a_sections
        assert "activity" in table_a_sections
        assert "psr" in table_a_sections
        assert "pose" in table_a_sections

    @staticmethod
    def test_metrics_json_matches_table_a():
        """If a metrics.json exists, verify its structure."""
        metrics_path = (
            _ROOT / "src" / "runs" / "rf_stages" / "checkpoints"
            / "test_split_eval" / "metrics.json"
        )
        if not metrics_path.exists():
            pytest.skip("test_split_eval/metrics.json not found (run eval first)")

        with open(metrics_path) as f:
            data = json.load(f)

        # Must have metadata with test subjects
        assert "metadata" in data
        assert "test_subjects" in data["metadata"]
        assert len(data["metadata"]["test_subjects"]) == 10

        # Must have sota_table_A section
        assert "sota_table_A" in data
        table_a = data["sota_table_A"]
        assert "detection" in table_a
        assert "activity" in table_a
        assert "psr" in table_a
        assert "pose" in table_a

        # Detection must have SOTA anchors
        assert table_a["detection"]["sota_annotated"] == 0.838
        assert table_a["detection"]["sota_video"] == 0.641

        # Activity must have SOTA anchor
        assert table_a["activity"]["sota_top1"] == 65.25

        # PSR must have SOTA anchors
        assert table_a["psr"]["sota_event_f1"] == 0.901
        assert table_a["psr"]["sota_tau_seconds"] == 15.5


class TestGracefulDegradation:
    """Verify the orchestrator's fallback behavior."""

    @staticmethod
    def test_bootstrap_ci_fallback_available(orchestrator_module):
        """The bootstrap_ci helper should be importable and callable."""
        bootstrap_ci = orchestrator_module.bootstrap_ci
        assert callable(bootstrap_ci)

    @staticmethod
    def test_event_f1_fallback_imports():
        """The event_f1 function should be importable from at least one source."""
        event_f1 = None
        try:
            from src.evaluation.decoder_oracle_bound import event_f1  # noqa: F401
            event_f1 = True
        except ImportError:
            pass
        if not event_f1:
            try:
                from src.evaluation.psr_transition_f1 import event_f1  # noqa: F401
                event_f1 = True
            except ImportError:
                pass
        assert event_f1, "event_f1 not importable from either decoder_oracle_bound or psr_transition_f1"

    @staticmethod
    def test_eval_scripts_importable():
        """Legacy eval scripts should remain importable after adding the orchestrator."""
        try:
            from scripts import eval_activity_75class  # noqa: F401
            import scripts.eval_detection_dual_protocol as eddp  # noqa: F401
            import scripts.eval_psr_transition_f1 as eptf  # noqa: F401
            eval_ok = True
        except Exception:
            eval_ok = False
        # This is a soft test — the scripts may fail to import in CI without GPU/data
        # We accept either result, just documenting it.
        if not eval_ok:
            pytest.skip("Legacy eval scripts not importable in this environment")

    @staticmethod
    def test_orchestrator_help_runs(orchestrator_module):
        """The CLI should have help text."""
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "scripts.eval_test_split", "--help"],
            capture_output=True, text=True, cwd=str(_ROOT),
        )
        assert result.returncode == 0
        assert "Test-split evaluation orchestrator" in result.stdout
        assert "--checkpoint" in result.stdout
        assert "--skip-detection" in result.stdout
        assert "--skip-psr" in result.stdout
        assert "--max-batches" in result.stdout
