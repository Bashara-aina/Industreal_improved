# 159 — 2x2 Ablation Matrix: Per-Head Multi-Task vs Single-Task Comparison

## Section 1. The 4-Head 2x2 Matrix

| Head | Multi-Task (current) | Multi-Task (with all 9 fixes) | Single-Task (with same fixes) | Best |
|---|---|---|---|---|
| Detection (mAP) | 0.00009 (impl bug) | ? (target 0.1-0.5) | ? (target 0.5-0.7) | ? |
| Activity (top-1) | 0.0236 (class collapse) | ? (target 0.05-0.10) | ? (target 0.05-0.10) | ? |
| PSR (F1) | 0.7018 (GELU) | ? (V3 target 0.78+) | ? (target 0.65-0.75) | ? |
| Pose (MAE) | 9.14 deg fwd | ? (9.14 deg) | ? (target 5-7 deg) | ? |

## Section 2. The Decision Tree

### If multi-task (with all 9 fixes) >= 0.9 x single-task:
- Multi-task is fine
- Keep multi-task for paper
- Contribution: fix path (9 fixes)

### If multi-task (with all 9 fixes) < 0.5 x single-task:
- Multi-task is fundamentally wrong
- Switch to single-task
- Contribution: failed multi-task story

### If multi-task is between 0.5-0.9 x single-task:
- Mixed: some heads work, some don't
- Per-head decision
- Contribution: which heads work in multi-task

## Section 3. The Training Schedule (12 weeks)

| Week | Action | Expected Result |
|---|---|---|
| 1-2 | V3 PSR + single-task det | 2 of 4 baselines |
| 3-4 | 4 single-task baselines | All 4 baselines |
| 5-6 | MViTv2-S fine-tune | Activity baseline |
| 7-8 | Multi-task V4 (all 9 fixes) | Multi-task reference |
| 9-10 | 4x4 comparison | Final matrix |
| 11-12 | Paper | Submission |

## Section 4. The Per-Head Hypothesis

| Head | Hypothesis | Test |
|---|---|---|
| Detection | Multi-task hurts (impl bug is dominant) | Single-task > 5 x multi-task |
| Activity | Backbone wrong (ImageNet vs Kinetics) | MViTv2-S single > 0.4 |
| PSR | Multi-task helps (transfer from pose/det) | Multi-task with fix > single |
| Pose | Multi-task is fine (spatial task) | Multi-task approx equal single |

## Section 5. The Decision Matrix

For each head, the paper should:
- Report the final 2x2 result
- Note which was the dominant cause
- Suggest the fix path

## Section 6. File Locations

- /media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/analyses/consult_2026_06_10/AAIML/150-156: all strategy docs
- /media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/src/runs/rf_stages/checkpoints/: all evidence
- /tmp/train_*.log: training logs (workstation-only)
