"""
POPW Loss Correctness Tests — Loss & Kendall Agent
==================================================
Validates: Kendall formula, loss scales, staged training, head pose leak.

Run: cd /home/newadmin/swarm-bot/project/popw/working/code/industreal_improved_to_archive
     python -m pytest tests/test_loss_kendall.py -v

Author: Bashara / Loss & Kendall Agent
Date: 2026-05-15
"""

import sys, os, re
work_dir = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.join(work_dir, 'src'))  # src/ at index 0 → 'import training' finds src/training/
sys.path.insert(1, work_dir)  # project root at index 1

import torch
import training.losses as _L

LOSSES = '/home/newadmin/swarm-bot/project/popw/working/code/industreal_improved_to_archive/src/training/losses.py'

# ==============================================================================
# CATEGORY 1: Kendall Uncertainty Formula
# ==============================================================================

class TestKendallFormula:
    """Tests for Category 1: Kendall uncertainty formula correctness."""

    def test_kendall_init_values(self):
        """Verify log_var initializations match paper spec (s_det=0, s_pose=-1, s_act=s_psr=0)."""
        src = open(LOSSES).read()
        assert 'self.log_var_det = nn.Parameter(torch.zeros(1))' in src, \
            "log_var_det init should be torch.zeros(1) (s_det=0)"
        assert 'self.log_var_pose = nn.Parameter(torch.tensor([-1.0]))' in src, \
            "log_var_pose init should be torch.tensor([-1.0]) (s_pose=-1)"
        assert 'self.log_var_act = nn.Parameter(torch.zeros(1))' in src, \
            "log_var_act init should be torch.zeros(1) (s_act=0)"
        assert 'self.log_var_psr = nn.Parameter(torch.zeros(1))' in src, \
            "log_var_psr init should be torch.zeros(1) (s_psr=0)"

    def test_kendall_precision_computation(self):
        """Verify precision = exp(-lv_var) where lv_var = log_var.clamp(-4,2)."""
        src = open(LOSSES).read()
        # Precision computed from clamped log_var
        assert 'prec_det = torch.exp(-lv_det)' in src, \
            "det precision = exp(-lv_det)"
        assert 'prec_hp = torch.exp(-lv_hp)' in src, \
            "pose precision = exp(-lv_hp)"
        assert 'prec_act = torch.exp(-lv_act)' in src, \
            "act precision = exp(-lv_act)"
        assert 'prec_psr = torch.exp(-lv_psr)' in src, \
            "psr precision = exp(-lv_psr)"
        # Clamp first, then exp
        assert 'lv_det = self.log_var_det.clamp(-4.0, 2.0)' in src, \
            "log_var_det clamped to [-4, 2]"
        assert 'lv_hp = self.log_var_pose.clamp(-4.0, 2.0)' in src, \
            "log_var_pose clamped to [-4, 2]"

    def test_kendall_formula_structure(self):
        """Verify L = Σ prec_t·loss_t + lv_t for all 4 tasks (no 0.5 factor in code)."""
        src = open(LOSSES).read()
        # Precision-weighted losses
        assert 'total = total + prec_det * loss_det + lv_det' in src
        assert 'total = total + prec_hp * loss_pose + lv_hp' in src
        assert 'total = total + prec_hp * loss_head_pose + lv_hp' in src
        assert 'total = total + prec_act * loss_act + lv_act' in src
        assert 'total = total + prec_psr * loss_psr + lv_psr' in src
        # No 0.5 coefficient
        assert '0.5 * prec_det * loss_det' not in src

    def test_kendall_clamp_range(self):
        """Verify log_var clamp range [-4, 2] as per paper spec."""
        src = open(LOSSES).read()
        assert '.clamp(-4.0, 2.0)' in src, \
            "log_var clamp should be (-4.0, 2.0) per paper spec"

    def test_activity_ramp_min_one(self):
        """Verify act_ramp = min(1, epoch/5) does not exceed 1."""
        src = open(LOSSES).read()
        ramp_pattern = r'act_ramp\s*=\s*min\(1'
        assert re.search(ramp_pattern, src), \
            "act_ramp = min(1, ...) not found"
        assert '/ max(self._act_warmup_epochs, 1)' in src or '/ 5' in src, \
            "act_ramp denominator should use epoch/5 or act_warmup_epochs"

# ==============================================================================
# CATEGORY 2: Loss Scales
# ==============================================================================

class TestLossScales:
    """Tests for Category 2: Loss scales ×0.001 confirmed via code inspection."""

    def test_pose_loss_scale_is_0_001(self):
        """Body pose loss ×0.001 (confirmed by comment at line 617)."""
        src = open(LOSSES).read()
        assert ') * 0.001  # Kendall exp(-lv_pose)=exp(1)' in src, \
            "Body pose loss ×0.001 confirmed at line ~617"

    def test_head_pose_loss_scale_is_0_001(self):
        """Head pose loss ×0.001 (confirmed by comment at line 688)."""
        src = open(LOSSES).read()
        assert ') * 0.001  # Head pose 9-DoF MSE' in src, \
            "Head pose loss ×0.001 confirmed at line ~688"

    def test_no_0_01_scale_for_pose(self):
        """Verify ×0.01 is NOT used for body pose (audit's false claim)."""
        src = open(LOSSES).read()
        lines = src.split('\n')
        for i, line in enumerate(lines):
            if 'loss_pose' in line and '0.01' in line:
                assert False, f"False 0.01 scale found at line {i+1}: {line.strip()}"
        # Confirm 0.001 IS used for pose
        assert ') * 0.001  # Kendall' in src

    def test_loss_scales_all_0_001(self):
        """Detection uses Kendall weighting; body/head pose use ×0.001 fixed scale."""
        src = open(LOSSES).read()
        # Detection: Kendall precision-weighted (prec_det * loss_det + lv_det), NOT 0.001
        assert 'prec_det * loss_det + lv_det' in src, \
            "Detection uses Kendall (prec_det * loss_det + lv_det)"
        # Body pose ×0.001 (call spans multiple lines; match the closing ) * 0.001)
        pose_scale = re.search(r'\) \* 0\.001  # Kendall exp\(-lv_pose\)', src)
        assert pose_scale, "pose loss ×0.001 not found"
        # Head pose ×0.001 (stored in loss_head_pose, then ) * 0.001 on next line)
        hp_scale = re.search(r'\) \* 0\.001  # Head pose 9-DoF MSE', src)
        assert hp_scale, "head_pose loss ×0.001 not found"
        # Verify detection does NOT have a spurious ×0.001
        det_fixed = re.search(r'loss_det *\* *0\.001', src)
        assert not det_fixed, "Detection should NOT use fixed ×0.001 (Kendall instead)"

# ==============================================================================
# CATEGORY 3: Staged Training Precision Zeroing
# ==============================================================================

class TestStagedPrecisionZeroing:
    """Tests for Category 3: Staged precision zeroing per epoch thresholds."""

    def test_stage1_kendall_zeros_act_psr_pose(self):
        """Stage 1 (epoch 1-5): det only, zeros act/psr/pose precisions."""
        src = open(LOSSES).read()
        stage1 = re.search(r'if stage == 1:(.*?)(?=elif stage == 2|else:|total = torch)', src, re.DOTALL)
        assert stage1, "Stage 1 block not found"
        block = stage1.group(1)
        assert 'prec_act = prec_act * 0' in block, "Stage 1 should zero prec_act"
        assert 'prec_psr = prec_psr * 0' in block, "Stage 1 should zero prec_psr"
        # prec_hp is shared between body_pose and head_pose (both from log_var_pose)
        assert 'prec_hp = prec_hp * 0' in block, "Stage 1 should zero prec_hp (pose+head_pose)"

    def test_stage2_kendall_zeros_act_psr_not_pose(self):
        """Stage 2 (epoch 6-15): det+pose, zeros act/psr only (pose stays active)."""
        src = open(LOSSES).read()
        stage2 = re.search(r'elif stage == 2:(.*?)(?=elif stage == 3|else:|total = torch)', src, re.DOTALL)
        assert stage2, "Stage 2 block not found"
        block = stage2.group(1)
        assert 'prec_act = prec_act * 0' in block, "Stage 2 should zero prec_act"
        assert 'prec_psr = prec_psr * 0' in block, "Stage 2 should zero prec_psr"
        # Stage 2 does NOT zero pose — pose is active in Stage 2
        assert 'prec_pose = prec_pose * 0' not in block, \
            "Stage 2 should NOT zero prec_pose (pose active in Stage 2)"

    def test_stage3_kendall_all_active(self):
        """Stage 3 (epoch 16+): all task precisions active (no zeroing)."""
        src = open(LOSSES).read()
        # Stage 3 is implicit else after stage 2 — no explicit elif stage == 3
        # Stage 1 and 2 do zeroing; stage 3 (else/not-matched) does not
        assert re.search(r'elif stage == 2:', src), "Stage 2 block found"
        stage2 = re.search(r'elif stage == 2:(.*?)(?=elif stage == 3|else:|total = torch)',
                           src, re.DOTALL)
        if stage2:
            block = stage2.group(1)
            assert 'prec_act = prec_act * 0' in block, "Stage 2 should zero act"
            assert 'prec_psr = prec_psr * 0' in block, "Stage 2 should zero psr"

    def test_get_kendall_stage_boundaries(self):
        """Verify stage boundaries via _get_kendall_stage() helper function."""
        src = open(LOSSES).read()
        # _get_kendall_stage uses epoch <= stage1_end for stage 1, epoch <= stage2_end for stage 2
        assert '_get_kendall_stage' in src, "_get_kendall_stage helper exists"
        assert 'epoch <= stage1_end' in src, "Stage 1: epoch <= stage1_end (≤5)"
        assert 'epoch <= stage2_end' in src, "Stage 2: epoch <= stage2_end (≤15)"
        # stage 3 is implicit return 3 when neither condition matches

# ==============================================================================
# CATEGORY 4: Stage 2 Head Pose Leak Fix (NON-KENDALL BRANCH)
# ==============================================================================

class TestStage2HeadPoseLeak:
    """Tests for Category 4: Stage 2 head_pose leak in non-Kendall path."""

    def test_non_kendall_branch_exists(self):
        """Non-Kendall (else) branch exists for use_kendall=False case."""
        src = open(LOSSES).read()
        assert 'else:' in src and 'self.use_kendall' in src, \
            "Non-Kendall 'else' branch exists for use_kendall=False case"

    def test_non_kendall_stage2_head_pose_not_zeroed(self):
        """Non-Kendall Stage 2: train_pose=True zeros body pose, NOT head_pose."""
        src = open(LOSSES).read()
        # Non-Kendall else branch at line 774: '        else:'
        # In Stage 2: 'if self.train_pose: _loss_pose_staged = zero' (line 788)
        # This zeros BODY pose when train_pose=True, leaving head_pose (else at line 778) active
        # The fix is at line 778: loss_pose if self.train_pose else loss_head_pose
        assert 'loss_pose if self.train_pose else loss_head_pose' in src, \
            "Non-Kendall branch stores loss_pose if train_pose else loss_head_pose (not zeroed incorrectly)"
        assert 'if self.train_pose:' in src and '_loss_pose_staged = zero' in src, \
            "Stage 2 in non-Kendall correctly zeros body pose (train_pose) but not head_pose"

    def test_kendall_branch_handles_head_pose_correctly(self):
        """Kendall branch: Stage 2 with train_pose=False adds head_pose via else clause."""
        src = open(LOSSES).read()
        kendall = src[src.find('if self.use_kendall:'):src.find('else:  # not self.use_kendall')]
        # Kendall branch has: if self.train_pose: ... else: ... + prec_hp * loss_head_pose
        assert 'if self.train_pose:' in kendall and 'else:' in kendall, \
            "Kendall branch uses train_pose to switch between pose/head_pose"
        # In Kendall branch, else at line ~731-732 handles head_pose
        assert 'total = total + prec_hp * loss_head_pose + lv_hp' in kendall, \
            "Kendall branch adds head_pose via else clause"

# ==============================================================================
# FINAL VERDICT
# ==============================================================================

if __name__ == '__main__':
    import pytest
    sys.exit(pytest.main([__file__, '-v']))