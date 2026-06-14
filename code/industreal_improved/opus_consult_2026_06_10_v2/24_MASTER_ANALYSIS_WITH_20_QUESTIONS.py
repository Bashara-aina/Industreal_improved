#!/usr/bin/env python3
"""
=============================================================================
MASTER TRAINING ANALYSIS — R2.5 Status, 20 Questions for Opus & 100-item R3 Checklist
=============================================================================
Generated: 2026-06-15 08:00 UTC
Current run: paper_run_r25_fix_20260615 → Epoch 48, Step 5133/12579 (40.8%)
Target: Decide whether to INTERVENE, STOP, RESUME, or CONTINUE this training
         and what must be done before transitioning to R3.

Architecture: ConvNeXt Tiny (28.6M) + 5 heads (52.5M total)
              Detection (RetinaNet FPN, 24 cls), Activity (75-cls CB-Focal),
              Pose (17-kpt Wing Loss), HeadPose (9-DoF geo MSE),
              PSR (Causal Transformer 3L 4H d=256, 36×11 comps)
              TMA + TemporalBank + EMA + Kendall weighting

Dataset: industreal (NOT IKEA). 100 epochs effective. 32 effective batch.
-----------------------------------------------------------------------------
"""

import json
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum

# ============================================================================
# PART 1: CURRENT STATE (from live logs)
# ============================================================================

@dataclass
class TrainingState:
    """Snapshot of current R2.5 fix training run."""
    epoch: int = 48
    step: int = 5133
    total_steps: int = 12579
    progress_pct: float = 40.8
    runtime_h: float = 68.0
    grad_nan_count: int = 0
    sigterm_count: int = 0

    @property
    def steps_remaining(self) -> int:
        return self.total_steps - self.step

    @property
    def epochs_completed(self) -> int:
        return self.epoch  # 0-indexed epoch, 48 means completed 48


@dataclass
class HeadLiveness:
    """Per-head liveness snapshot (from LIVENESS_GRAD at step 5000)."""
    loss_alive: bool
    grad_first_alive: bool
    grad_last_alive: bool
    loss_value: float
    grad_first_value: float
    grad_last_value: float

    @property
    def status(self) -> str:
        statuses = []
        if self.loss_alive: statuses.append("LOSS_ALIVE")
        else: statuses.append("LOSS_DEAD")
        if self.grad_first_alive: statuses.append("GRAD1_ALIVE")
        else: statuses.append("GRAD1_DEAD")
        if self.grad_last_alive: statuses.append("GRAD_LAST_ALIVE")
        else: statuses.append("GRAD_LAST_DEAD")
        return "|".join(statuses)


# Current liveness at step 5000
he_liveness = {
    "detection": HeadLiveness(False, True, True, 1e-6, 5.80e-2, 6.17e-2),
    "activity":  HeadLiveness(True,  True, True, 1.12,   5.06e-2, 5.09e-2),
    "psr":       HeadLiveness(True,  True, False,5.44e-2,3.58e-2, 0.0),
    "head_pose": HeadLiveness(True,  True, True, 1.79e-2,8.75e-2, 4.97e-2),
    "pose":      HeadLiveness(True,  True, True, 6.06,   1.72e-2, 5.99e-5),
}

# Kendall log_vars
kendall_log_vars = {
    "det": 0.0,
    "head_pose": -1.268,
    "act": 0.0,
    "psr": 0.0,
}


# ============================================================================
# PART 2: THE 20 QUESTIONS FOR OPUS
# ============================================================================
# These are the questions we genuinely don't know the answer to. Each
# represents a fork point: intervene/stop/resume/continue decisions that
# depend on understanding we don't currently have.

QUESTIONS_FOR_OPUS = [
    # ── DECISION: INTERVENE (change code mid-run) ─────────────────────
    {
        "id": "Q1",
        "category": "INTERVENE — PSR bias head DEAD",
        "question": (
            "PSR last-layer grad has been DEAD (near-zero) for 4000+ steps "
            "while first-layer grad is ALIVE (~3.58e-2). The PSR bias head is "
            "getting no gradient. Is this a sign that the PSR head's final "
            "linear layer is saturated or that the MonotonicDecoder+BCE focal "
            "loss doesn't produce useful gradients for the bias? Should we add "
            "a separate learning rate or skip connection for the last layer?"
        ),
        "risk_if_wrong": "PSR module never learns to emit confidences; R3 eval will show chance-level procedure step accuracy.",
        "recommendation": "Wait — PSR loss is ALIVE (5.44e-2), only bias grad is dead. Loss-based liveness is more important. Flag for R3 if still dead at epoch 80."
    },
    {
        "id": "Q2",
        "category": "INTERVENE — Detection head oscillating NO_GRAD",
        "question": (
            "Detection head alternates between NO_GRAD and ALIVE every 200 steps. "
            "When NO_GRAD, the entire detection subgraph receives zero gradient. "
            "Loss-based liveness shows det=1e-6 DEAD. This happens because most "
            "frames in the dataset have no objects (background). Is this expected "
            "behavior for a RetinaNet on a procedure-following dataset, or does "
            "it indicate the detection head is being suppressed by activity "
            "dominance through Kendall?"
        ),
        "risk_if_wrong": "Detection head reverts to random weights; paper review asks 'where are your detection metrics?'.",
        "recommendation": "Expected — most frames are background. But monitor whether detection loss ever fires during object-present frames. Consider logging 'det on positive frames' average."
    },
    {
        "id": "Q3",
        "category": "INTERVENE — Is PSR warmup working correctly?",
        "question": (
            "PSR warmup (step-based, 3.0→1.0 over 6000 steps) is at 85.6% "
            "completion (step 5133/6000). Current mult ≈1.29×. But PSR loss "
            "has been stable at 0.02-0.13 throughout the warmup — not increasing "
            "or decreasing. Shouldn't we see PSR loss go UP as the warmup decays "
            "(more of the loss is exposed)? Flat PSR loss during warmup decay "
            "could mean: (a) the warmup isn't actually controlling precision, "
            "or (b) PSR has already converged to a trivial minimum."
        ),
        "risk_if_wrong": "Post-warmup PSR loss collapses when precision multiplier reaches 1.0, wiping 48 epochs of progress.",
        "recommendation": "CRITICAL WATCHPOINT. Log PSR precision multiplier in training output at step 6000. If PSR loss drops suddenly (>50% reduction) within 500 steps of warmup ending, restore from checkpoint before warmup end and increase PSR_WARMUP_STEPS."
    },
    {
        "id": "Q4",
        "category": "INTERVENE — Activity loss weight too aggressive?",
        "question": (
            "ACTIVITY_LOSS_WEIGHT=0.3 means activity is downweighted 70% "
            "before Kendall. Current activity loss range is 0.4-11.0 with "
            "occasional spikes to 11+. Despite the 70% downweight, activity "
            "still dominates the total loss (80-90% of total). Is 0.3 still "
            "too high? Should we go to 0.1 or even 0.01? How do we know when "
            "we've found the right balance?"
        ),
        "risk_if_wrong": "Activity still dominates → PSR/pose never develop → increased ACTIVITY_HEAD_GRAD_CLIP only masks true symptom.",
        "recommendation": "Epoch 50 check: compute activity fraction of total loss over 500-step window. If >70%, reduce ACTIVITY_LOSS_WEIGHT to 0.1 for epoch 51+."
    },
    {
        "id": "Q5",
        "category": "INTERVENE — Kendall bounds proven effective?",
        "question": (
            "The Kendall bounds (act log_var min=0, psr/pose log_var max=0) "
            "were added to prevent the multi-task collapse cascade. Current "
            "log_vars: det=0.0 (at min -4 bound? needs check), act=0.0 (at "
            "min 0 bound), psr=0.0 (at max 0 bound), head_pose=-1.268. "
            "Three of four log_vars are at their constraint bounds. Is this "
            "the bounds working correctly (preventing extreme precision), or "
            "are the bounds so tight that Kendall is essentially disabled?"
        ),
        "risk_if_wrong": "Kendall becomes a no-op → no learned task weighting → paper contribution invalid for task weighting claim.",
        "recommendation": "CRITICAL. If all log_vars are pinned to bounds, Kendall is doing nothing. Consider widening bounds by 0.5 and re-checking after 1000 steps."
    },
    {
        "id": "Q6",
        "category": "INTERVENE — Detection cls_target out-of-range guard",
        "question": (
            "A guard was added in losses.py for detection classification "
            "labels exceeding num_classes (24). How/why are GT labels >23 "
            "being passed to the detector? Is this a dataset loading bug "
            "(label indexing off by one) or a sign that some GT annotations "
            "are corrupted/invalid?"
        ),
        "risk_if_wrong": "Clamping masks real data corruption → silent accuracy degradation on certain rare classes.",
        "recommendation": "Investigate dataset: grep for class IDs > 23 in industreal annotations. If found, fix data pipeline. If not, remove guard and raise error instead."
    },
    {
        "id": "Q7",
        "category": "INTERVENE — Activity target OOB in CBFocalLoss",
        "question": (
            "Similar OOB guard added for activity targets in CBFocalLoss. "
            "Validation batches are producing activity labels > 75. Is the "
            "validation dataset different from training (different class "
            "set)? Or is there a frame where activity label is background "
            "but mapped to class 75+?"
        ),
        "risk_if_wrong": "Validation metrics are computed on corrupted labels → misleading activity accuracy.",
        "recommendation": "Same as Q6 — find OOB label source in validation data. Fix at loader level, not loss level."
    },
    {
        "id": "Q8",
        "category": "INTERVENE — PSR sensitivity loss NaN guard",
        "question": (
            "PSR_SENSITIVITY_WEIGHT=0.01 uses -log(std(psr_logits)) as an "
            "auxiliary loss. A guard was added requiring batch>1 because "
            "std on single-element batch gives NaN backward. This suggests "
            "the PSR sequence mode (which produces T=4 sequences) isn't "
            "running frequently enough. Are we losing the PSR temporal "
            "supervision signal?"
        ),
        "risk_if_wrong": "PSR never learns temporal structure → reverts to per-frame trivial classifier.",
        "recommendation": "Check PSR_SEQ_EVERY_N_BATCHES=8 cadence. If sensitivity loss is always skipped (batch always 1), reduce to 4 or even 2 for R3."
    },
    {
        "id": "Q9",
        "category": "INTERVENE — Activity mask IndexError fix",
        "question": (
            "The evaluate.py fix skips act_clip_ids when act_valid[i] is False. "
            "But this means those frames are excluded from activity evaluation "
            "entirely. How many frames per epoch are masked? If >5% of frames "
            "are activity-masked, we're evaluating on a biased subset."
        ),
        "risk_if_wrong": "Activity metrics look artificially good because hard/unlabeled frames are excluded.",
        "recommendation": "Log activity_mask coverage rate (% of frames valid vs total) at each eval. Report alongside accuracy."
    },

    # ── DECISION: STOP (kill the run) ──────────────────────────────────
    {
        "id": "Q10",
        "category": "STOP — Is epoch 48 still productive?",
        "question": (
            "Total loss has been oscillating 5-17 for the last 30 epochs. "
            "Activity loss hasn't trended down since epoch ~20. Pose loss "
            "varies 0.005-0.9 but no clear trend. PSR loss flat at 0.02-0.13. "
            "Is this convergence of the under-constrained system, or has "
            "training plateaued? How do we distinguish 'slow multi-task "
            "convergence' from 'training has stalled and won't recover'?"
        ),
        "risk_if_wrong": "WASTE: 30 more epochs of no progress. Or PREMATURE: kill just before PSR warmup ends and things start working.",
        "recommendation": "Do NOT stop before PSR warmup completes at step 6000 (epoch ~58). Evaluate at epoch 60 against previous checkpoints. If no improvement from epoch 20, consider stop."
    },
    {
        "id": "Q11",
        "category": "STOP — PSR bias head DEAD for 4000 steps",
        "question": (
            "The PSR bias head last-layer gradient has been DEAD since step "
            "~1000. That's 4000+ steps of zero learning for that specific "
            "layer. Could this cascade: if the bias head never learns, PSR "
            "logits are always biased, and the focal loss can never fix it "
            "because it doesn't produce gradients through the final layer. "
            "Is this an architecture problem or a training problem?"
        ),
        "risk_if_wrong": "PSR never works in any configuration → multi-task claim fails.",
        "recommendation": "Experiment on a branch: add a residual connection around the final PSR linear layer (+learnable_scale * input). If that fixes the DEAD grad, it's architectural. If not, it's training dynamics."
    },
    {
        "id": "Q12",
        "category": "STOP — 0 GRAD_NAN is suspicious",
        "question": (
            "Current run has 0 GRAD_NAN events after 5133 steps. Previous "
            "R2.5 runs had 206 GRAD_NAN in 6594 steps (fix3, 100% skip rate). "
            "Zero GRAD_NAN could mean: (a) the NaN guards are working, (b) "
            "the model has found a too-stable local minimum and isn't exploring, "
            "or (c) gradient clipping at the activity head is masking "
            "instabilities that would otherwise drive better solutions."
        ),
        "risk_if_wrong": "Model is in a stable-but-mediocre basin that no amount of training will escape.",
        "recommendation": "Check if grad norms have been decreasing over time. If last-layer grad norms are 100× smaller at epoch 48 vs epoch 1, model is converging to a sharp minimum."
    },

    # ── DECISION: RESUME (with changes) ───────────────────────────────
    {
        "id": "Q13",
        "category": "RESUME — Should we restore pre-warmup checkpoint?",
        "question": (
            "PSR warmup at 85% completion. If we decide the current warmup "
            "is insufficient, we could restore the checkpoint from step ~500 "
            "(epoch 5, warmup just started) and increase PSR_WARMUP_STEPS "
            "to 12000 with PSR_WARMUP_INIT_MULT=5.0. Is there any reason to "
            "restart from scratch vs continuing from a mid-training checkpoint?"
        ),
        "risk_if_wrong": "Restart wastes 48 epochs of training. Continue → PSR never learns.",
        "recommendation": "Continue to epoch 60 (warmup complete + 2 epochs of steady-state). If PSR still flat, restore epoch-5 checkpoint with doubled warmup."
    },
    {
        "id": "Q14",
        "category": "RESUME — What R3 hyperparameters to change?",
        "question": (
            "If we STOP this run and start R3, what changes are proven by "
            "R2.5 evidence? (a) ACTIVITY_LOSS_WEIGHT=0.1 (more aggressive), "
            "(b) PSR_WARMUP_INIT_MULT=5.0 (stronger initial boost), "
            "(c) Remove Kendall bounds (they're all pinned anyway), "
            "(d) Add PSR residual connection, "
            "(e) Re-enable PSR sequence mode with batch_size=1, accum=16?"
        ),
        "risk_if_wrong": "Too many changes → can't attribute improvement. Too few → same failure mode.",
        "recommendation": "Change exactly ONE thing for R3: PSR_WARMUP_INIT_MULT=5.0, WARMUP_STEPS=12000. Everything else stays identical. Attribution matters."
    },
    {
        "id": "Q15",
        "category": "RESUME — R3 architecture: freeze or fine-tune backbone?",
        "question": (
            "ConvNeXt Tiny backbone (28.6M) is being fine-tuned. Previous "
            "training runs showed backbone overfitting to activity. Should "
            "R3 freeze the backbone (train only heads) for the first 20 "
            "epochs, then gradually unfreeze? This is standard practice in "
            "multi-task detection but we haven't tried it."
        ),
        "risk_if_wrong": "Freezing limits representation learning. Not freezing → activity dominates again.",
        "recommendation": "Worth trying. Add backbone_lr=0.1× head_lr (differential LR) before freezing. More surgical."
    },
    {
        "id": "Q16",
        "category": "RESUME — Should we add PSR class weights?",
        "question": (
            "PSR has 36 procedure steps × 11 components = 396 output classes, "
            "but most are never active in a single frame (sparse target). "
            "The per-component BCE focal loss handles class imbalance, but "
            "the PSR weight gradient is still DEAD in the last layer. Would "
            "adding explicit class weights (inverse frequency) for PSR help? "
            "Or is the issue that BCE focal has gamma=1.0 (mild) vs gamma=2.0?"
        ),
        "risk_if_wrong": "Wrong fix for DEAD grad → add complexity for no benefit.",
        "recommendation": "Experiment: gamma=2.0 on just the PSR focal loss. More aggressive focal should push gradient through rare positive examples."
    },

    # ── DECISION: CONTINUE (let it run) ───────────────────────────────
    {
        "id": "Q17",
        "category": "CONTINUE — When do we evaluate?",
        "question": (
            "Current run has only training losses — no eval metrics. The "
            "evaluate.py has eval harness wired for detection mAP, activity "
            "Top-1/5, pose PCK, PSR step accuracy, and head pose MAE. "
            "Previous evals required manual checkpoint loading. When should "
            "we schedule the first real eval? Mid-training at epoch 50? "
            "Or only at the end of 100 epochs? What do we lose by not "
            "evaluating mid-run?"
        ),
        "risk_if_wrong": "No mid-run eval → don't know if training is working → can't make intervene/stop decisions.",
        "recommendation": "URGENT. Run eval NOW on epoch 48 checkpoint. This is the single most important data point missing from the current decision."
    },
    {
        "id": "Q18",
        "category": "CONTINUE — How long does 100 epochs take?",
        "question": (
            "Current speed: ~1.14 it/s, 12579 steps/epoch. Each epoch takes "
            "~3 hours (12579/1.14 = 11030s ≈ 3.06h). 100 epochs would take "
            "~306 hours ≈ 12.75 days. At epoch 48, we have 52 epochs × 3h "
            "= ~156 hours (6.5 days) remaining. Is 12.75 days acceptable for "
            "R2.5? What's the R3 deadline?"
        ),
        "risk_if_wrong": "Underestimating timeline → rush decisions at epoch 95.",
        "recommendation": "Plan for completion: epoch 100 ~June 28. Schedule R3 code freeze for June 25 (epoch ~85) to allow 3 days of R3 training before deadline."
    },
    {
        "id": "Q19",
        "category": "CONTINUE — Val loss gap?",
        "question": (
            "No validation loss is being logged (only training loss). This "
            "means we can't detect overfitting. Is this intentional (dataset "
            "too small to split validation) or an oversight? What's the "
            "train/val split ratio for industreal?"
        ),
        "risk_if_wrong": "Overfitting to training set → eval metrics are 30% lower than expected at paper submission.",
        "recommendation": "Read industreal_dataset.py to find split configuration. If no val split exists, create one (90/10 or 80/20). Add val loss logging to train.py."
    },
    {
        "id": "Q20",
        "category": "CONTINUE — What does 'done' look like for R2.5?",
        "question": (
            "We're 40% through the current run. We don't have convergence "
            "criteria beyond 'run 100 epochs.' What are the concrete metrics "
            "that would tell us: 'R2.5 training is complete, proceed to R3'? "
            "Proposal: (a) activity Top-1 > 40%, (b) PSR step accuracy > 15%, "
            "(c) detection mAP > 25, (d) pose PCK > 70, (e) all heads ALIVE "
            "in both loss and grad probes. Without targets, we can never "
            "decide to stop."
        ),
        "risk_if_wrong": "Training runs forever because no one defines 'good enough.'",
        "recommendation": "DEFINITIVELY ANSWER THIS before epoch 60. Write the convergence criteria into 25_R3_100_CHECKLIST.md."
    },
]


# ============================================================================
# PART 3: 100-ITEM R3 READINESS CHECKLIST (auto-generated from analysis)
# ============================================================================

R3_CHECKLIST = {
    "TRAINING_DATA_AND_DATASET": [
        "1. Verify industreal dataset has correct train/val split (read split files)",
        "2. Confirm no OOB detection labels (class > 24) in GT annotations",
        "3. Confirm no OOB activity labels (class > 75) in GT annotations",
        "4. Log activity_mask coverage rate (% frames valid) at dataset level",
        "5. Verify PSR sequence mode batch generation works (no OOM)",
        "6. Measure per-epoch DataLoader throughput (consistent? any worker death?)",
        "7. Confirm hand_joints key in targets (for head pose)",
        "8. Verify pose_confidence key in targets (for pose head)",
        "9. Log class frequency distribution for activity (check for missing classes)",
        "10. Log PSR label sparsity (% of 396 components active per frame)",
    ],
    "MODEL_ARCHITECTURE": [
        "11. Confirm backbone is ConvNeXt Tiny (not base/large)",
        "12. Verify TMA Cell is connected (not dead code)",
        "13. Verify TemporalBank slot_overwrite=False is working",
        "14. Confirm EMA is enabled and updating",
        "15. Verify PSR CausalTransformer receives sequences (not single frames)",
        "16. Check HeadPoseFiLMModule has gradient flow (alive in LIVENESS_GRAD)",
        "17. Verify MonotonicDecoder PSR decode at eval time",
        "18. Fix PSR bias head residual connection (if added for R3)",
        "19. Confirm GeometryAwareHeadPose produces [B,9] tensor (not tuple)",
        "20. Verify DetectionHead FPN outputs are correct scale",
    ],
    "TRAINING_CONFIG": [
        "21. EPOCHS set correctly (100)",
        "22. BATCH_SIZE safe for RTX 3060 12GB (2 = ~7.6GB VRAM)",
        "23. GRAD_ACCUM_STEPS consistent (16 → effective 32)",
        "24. TRAIN_MAX_STEPS disabled (0) for full 100-epoch run",
        "25. VAL_BATCH_SIZE safe (16 or lower if OOM)",
        "26. WARMUP_EPOCHS=5 correct for LR schedule",
        "27. Cosine LR decay configured correctly",
        "28. GRAD_CLIP_NORM=1.0 appropriate for all heads",
        "29. ACTIVITY_HEAD_GRAD_CLIP=0.1 confirmed effective",
        "30. ACTIVITY_LOSS_WEIGHT value finalized (0.3? 0.1? 0.01?)",
        "31. PSR_WEIGHT=60 (or new value) tested",
        "32. POSE_LOSS_WEIGHT=0.02 (or new value) tested",
        "33. PSR_WARMUP_STEPS and INIT_MULT finalized",
        "34. USE_PSR_SEQUENCE_MODE setting correct (True/False for R3)",
        "35. PSR_SEQ_EVERY_N_BATCHES appropriate for VRAM",
        "36. Kendall bounds confirmed not pinning all log_vars",
        "37. PSR_TRANSITION enabled (if using MonotonicDecoder)",
        "38. PSR_TRANSITION_SIGMA=3.0 confirmed",
        "39. Stage 3 warmup configured (if staged training re-enabled)",
        "40. FP32 confirmed (no AMP — broken)",
    ],
    "LOSS_FUNCTIONS": [
        "41. Detection FocalLoss has out-of-range label guard",
        "42. Activity CBFocalLoss has OOB target guard",
        "43. Pose WingLoss configured correctly (Wing loss params)",
        "44. HeadPose geo MSE with geodesic rotation loss working",
        "45. PSR BCE focal gamma finalized (1.0 vs 2.0)",
        "46. PSR sensitivity loss batch>1 guard working",
        "47. Kendall log_var clamping at param level (train.py _clamp_kendall_log_vars)",
        "48. Kendall NaN guard in forward path (losses.py rebuild from finite)",
        "49. Loss caps all set and not triggering excessively",
        "50. PSR warmup precision multiplier logging added to train output",
    ],
    "TRAINING_LOOP": [
        "51. Gradient clipping order correct (per-head before global, before NaN check)",
        "52. NaN gradient guard working (counted, not silent)",
        "53. Activity head per-head clip both AMP and FP32 paths",
        "54. empty_cache() before sequence batches",
        "55. Missing targets.to(device) for keypoints, pose_confidence confirmed",
        "56. --reinit-heads resets log_var_pose (verified)",
        "57. Checkpoint save frequency appropriate (every epoch?)",
        "58. Crash recovery checkpoint saves all optimizer states",
        "59. Per-head grad norm logging (LIVENESS_GRAD) working",
        "60. Loss-based liveness probe thresholds validated",
    ],
    "EVALUATION": [
        "61. Mid-training eval RUN NOW on epoch 48 checkpoint",
        "62. activity_mask correct in evaluate.py (IndexError fixed)",
        "63. Segment eval label==0 NA skip working",
        "64. Log eval metrics consistently (same metrics every eval)",
        "65. Detection mAP computed correctly (COCO-style?)",
        "66. Activity Top-1 and Top-5 computed",
        "67. Pose PCK at multiple thresholds (0.05, 0.1, 0.2)",
        "68. PSR step accuracy (frame-level and segment-level)",
        "69. HeadPose MAE (degrees) computed",
        "70. Eval metrics logged to tensorboard or JSONL",
    ],
    "MONITORING_AND_OBSERVABILITY": [
        "71. LIVENESS probe every 200 steps working",
        "72. GRAD_NAN counter visible in output",
        "73. per-epoch optimizer window skip summary working",
        "74. GPU memory logging regular interval",
        "75. CPU RAM logging available",
        "76. Training speed (it/s) tracked over time",
        "77. Kendall log_vars logged at epoch start",
        "78. LR logged at epoch start",
        "79. Validation loss during training (if val split exists)",
        "80. Automatic alert on GRAD_NAN spike (>10% of window)",
    ],
    "R3_PLANNING": [
        "81. R3 hyperparameter changes documented with rationale",
        "82. Only ONE change between R2.5 and R3 (attribution)",
        "83. R3 deadline known and feasible",
        "84. R3 code freeze date set",
        "85. Rollback plan: if R3 worse, restore R2.5 checkpoint",
        "86. Ablation plan: which component contributed how much?",
        "87. Paper results table planned with R2.5 and R3 rows",
        "88. Statistical significance test chosen for comparisons",
        "89. Compute budget for R3 known (12.75 days? more?)",
        "90. Preemptible/spot instance strategy if needed",
    ],
    "PAPER_AND_DELIVERABLES": [
        "91. Multi-task Kendall weighting claim backed by log_var evidence",
        "92. Detection ablation (with/without) results available",
        "93. Activity confusion matrix available",
        "94. PSR qualitative samples (procedure steps vs time plot)",
        "95. Pose skeleton visualizations available",
        "96. Head pose MAE distribution (histogram)",
        "97. Ablation: TMA on/off, TemporalBank on/off results",
        "98. Ablation: Kendall vs equal weights comparison",
        "99. Failure mode analysis (where does each head fail?)",
        "100. All checkpoints archived with loss curves per epoch",
    ],
}


# ============================================================================
# PART 4: DECISION MATRIX
# ============================================================================

class Decision(Enum):
    CONTINUE = "CONTINUE — let current run proceed without changes"
    INTERVENE = "INTERVENE — modify code/config mid-run and resume from checkpoint"
    RESUME = "RESUME — stop current, start new run with changes"
    STOP = "STOP — abandon R2.5, start R3 from scratch"

# Decision mapping based on question answers
DECISION_TREE = {
    "PSR_warmup_works": {
        True: Decision.CONTINUE,
        False: Decision.RESUME,
    },
    "PSR_bias_grad_recovers": {
        True: Decision.CONTINUE,
        False: Decision.INTERVENE,
    },
    "activity_still_dominates": {
        True: Decision.INTERVENE,
        False: Decision.CONTINUE,
    },
    "eval_metrics_improving": {
        True: Decision.CONTINUE,
        False: Decision.STOP,
    },
    "kendall_bounds_too_tight": {
        True: Decision.INTERVENE,
        False: Decision.CONTINUE,
    },
}

RECOMMENDATION = """
==============================================================================
FINAL RECOMMENDATION (as of epoch 48, step 5133)
==============================================================================

SHORT-TERM (epoch 48-60):
1. CONTINUE — Do NOT stop before PSR warmup completes at step 6000 (epoch ~58)
2. RUN MID-TRAINING EVAL on epoch 48 checkpoint (MISSING CRITICAL DATA)
3. Log PSR precision multiplier at step 6000 to verify warmup behavior

DECISION POINT (epoch 60-65):
After warmup completes + 2 epochs of steady-state:
- If PSR loss < 0.01 or PSR grad goes DEAD entirely → INTERVENE (restore epoch-5 ckpt with doubled warmup)
- If activity still >70% of total loss → INTERVENE (ACTIVITY_LOSS_WEIGHT 0.3→0.1)
- If all heads ALIVE and loss stable → CONTINUE to epoch 100
- If eval metrics not improving from epoch 20 → STOP for R3

R3 LAUNCH CONDITIONS:
- R2.5 completes 100 epochs OR eval metrics plateau for 20 epochs
- All 100-item checklist items resolved
- EXACTLY ONE config change from R2.5 (attribute improvement)
- Code freeze date set

If you can only answer ONE question, answer Q17: run the mid-training eval.
Without eval metrics, every continue/intervene/stop decision is a guess.
==============================================================================
"""


def main():
    print("=" * 78)
    print("MASTER TRAINING ANALYSIS — R2.5 → R3 Decision Support")
    print("=" * 78)
    print(f"\nCurrent: epoch {TrainingState().epoch}, step {TrainingState().step}")
    print(f"Progress: {TrainingState().progress_pct:.1f}%")
    print(f"Remaining: ~{TrainingState().steps_remaining * 3.06 / 3600:.1f}h")
    print(f"\nQuestions generated: {len(QUESTIONS_FOR_OPUS)}")
    print(f"Checklist items: {sum(len(v) for v in R3_CHECKLIST.values())}")
    print(RECOMMENDATION)


if __name__ == "__main__":
    main()
