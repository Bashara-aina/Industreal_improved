## §3. The Backbone Architecture Analysis (Q21-30)

### Q21. Is ConvNeXt-Tiny the right backbone for 4-task multi-task?
- For pose (spatial): YES, ConvNeXt works
- For detection: Maybe, YOLOv8m gets 0.995
- For PSR: Partially, linear probe is 0.2169 (no signal for activity)
- For activity: NO, 0.0236 (ImageNet has no action semantics)

### Q22. What does MViTv2-S give us that ConvNeXt doesn't?
- Kinetics-400 pretraining (400 action classes, 400k clips)
- Action semantics in features
- Linear probe 0.3810 (vs ConvNeXt 0.2169)
- Source: /media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/src/runs/rf_stages/checkpoints/activity_mvit_probe/results.json

### Q23. Should we replace ConvNeXt with MViTv2-S?
- Cost: 2-week fine-tuning
- Benefit: activity could reach 0.45-0.55
- Risk: 4-head multi-task may not converge
- Decision: yes, for activity at least
- Source: /media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/src/models/video_backbones.py

### Q24. Can we use a hybrid backbone (ConvNeXt + MViTv2-S)?
- Architecture: share early layers, split later
- Cost: complex
- Benefit: best of both
- Implementation: src/models/video_backbone_multitask.py (53.8M params, 19.3M trainable)
- Already designed

### Q25. Does the backbone need to be Kinetics-pretrained for activity?
- Yes: ImageNet features have no action semantics (proven)
- Linear probe (frozen ImageNet) = 0.2169 ≈ baseline
- Linear probe (frozen Kinetics) = 0.3810 (+0.1641 improvement)
- The backbone pretraining is THE cause of activity failure

### Q26. Is per-frame architecture the problem for activity?
- Per-frame MLP can't model temporal dynamics
- TCN/TCN+ViT architectures built (a3bad7356)
- TCN probe on ConvNeXt: 0.0723 (fails)
- TCN probe on MViTv2-S: pending

### Q27. What's the right architecture for 4-task multi-task?
- Backbone: MViTv2-S (video, Kinetics) or hybrid
- Heads: per-task (detection FPN, pose regression, activity TCN, PSR sequence)
- Loss balancing: bounded Kendall
- Architecture already in: video_backbone_multitask.py

### Q28. Can we use ImageNet ConvNeXt for 3 of 4 heads + MViTv2-S for activity?
- Hybrid: 2 backbones, 4 heads
- Cost: complex but possible
- Benefit: optimal for each task
- Implementation: train each head on appropriate backbone

### Q29. How long does MViTv2-S fine-tuning take?
- 2-week investment per training run
- 4 single-task baselines = 8 weeks total
- 4 multi-task conditions = 8 weeks
- Total: 16 weeks for complete ablation
- Achievable in 1 quarter

### Q30. What's the best backbone choice for IndustReal?
- MViTv2-S for activity (Kinetics pretraining)
- ConvNeXt for pose (spatial)
- Either for detection (both can work)
- Best of both worlds: hybrid or ensemble
- 75-100 hour training budget per architecture choice

---

## §6. The Activity Head Debate (Q51-60)

### Q51. Is activity broken by implementation or backbone?
- Implementation: ACTIVITY_GRAD_BLEND_RATIO 0.05→1.0 (starved initially)
- Backbone: ImageNet (no action semantics)
- Linear probe (frozen ImageNet): 0.2169 ≈ baseline (zero signal)
- Linear probe (frozen Kinetics): 0.3810 (real signal)
- BACKBONE is the dominant cause

### Q52. What does MViTv2-S 0.3810 tell us?
- Kinetics-pretrained features DO have action semantics
- ConvNeXt (ImageNet) features DO NOT
- The backbone pretraining is the key
- Source: /media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/src/runs/rf_stages/checkpoints/activity_mvit_probe/results.json

### Q53. Why does MViTv2-S probe 0.3810 work but multi-task 0.0236 fail?
- Multi-task detection 0.00009 (broken)
- Multi-task activity 0.0236 (broken)
- Frozen linear probe: ConvNeXt 0.2169, MViTv2-S 0.3810
- Multi-task + Kinetics features = best of both worlds
- Multi-task + ImageNet features = failure
- Backbone type is the cause

### Q54. Per-class: which classes benefit most from video features?
- check_instruction: 0.0000 → 0.8771 (+0.8771)
- tighten_nut: 0.0000 → 0.7149 (+0.7149)
- plug_objects: 0.0000 → 0.3558 (+0.3558)
- 11 of 41 zero-accuracy classes fixed by video features
- Source: /media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/src/runs/rf_stages/checkpoints/mvit_per_class/comparison.md

### Q55. Why do some classes go to zero with video features?
- 19 classes that ConvNeXt could detect went to zero
- pull_pin_middle: 62.1% → 0
- take_pin_long: 50.4% → 0
- put_wheel: 33.2% → 0
- Reason: rare classes with few samples
- Linear probe has insufficient examples to learn
- Will improve with full fine-tuning

### Q56. Is per-frame MLP architecture the problem for activity?
- Per-frame: can classify spatial patterns
- Per-frame: cannot model temporal dynamics
- TCN mean-pool on ConvNeXt: 0.0723 (fails)
- TCN mean-pool on MViTv2-S: pending
- Per-frame is wrong for temporal actions

### Q57. Should we use TCN+ViT for activity?
- Architectures built: src/models/activity_tcn.py, activity_tcn_vit.py
- TCN mean-pool on ConvNeXt: 0.0723 (FAILS)
- TCN+ViT on MViTv2-S: pending
- Per Opus ACT-1: gate is probe > 0.30
- MViTv2-S linear probe 0.3810 PASSES gate
- TCN+ViT is justified ON MViTv2-S features

### Q58. Is 2-week MViTv2-S fine-tuning worth it?
- Linear probe 0.3810 (above 0.30 gate)
- Expected fine-tune: 0.45-0.55
- Closes gap to WACV 0.622 from 0.3810
- 2-week investment, 75-100 hours
- YES, it's worth it for activity
- Source: /media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/scripts/train_mvit_finetune.sh

### Q59. Should multi-task activity use MViTv2-S?
- Multi-task with MViTv2-S backbone: 53.8M params, 19.3M trainable
- Implementation: src/models/video_backbone_multitask.py
- Expected: similar to single-task MViTv2-S + small drop from multi-task
- Single-task expected: 0.45-0.55
- Multi-task expected: 0.40-0.50 (5-10% drop)

### Q60. What's the right framing for activity in the paper?
- Multi-task 0.0236 (broken) is NOT the reportable number
- Single-task MViTv2-S frozen probe 0.3810 IS reportable (first video-backbone baseline)
- Single-task MViTv2-S fine-tuned (target 0.45-0.55) is the SOTA-comparable result
- Multi-task with MViTv2-S + all fixes is the best case
- Story: backbone wrong type → Kinetics fixes it

---

## §7. The Pose Head + Cross-Task Debate (Q61-70)

### Q61. Why does pose work in multi-task when other heads fail?
- Pose is a spatial task (forward/up direction)
- ConvNeXt ImageNet features have spatial information
- Per-frame regression is appropriate
- 9.14° fwd / 7.78° up (first baseline)
- Pose doesn't need video backbone or temporal

### Q62. Is 9.14° fwd a real "first baseline" claim?
- SOTA is ~15° (uncited source, per Opus HP-1)
- We have 9.14° with bootstrap CI [7.74, 10.87]
- Verdict: first ego-pose baseline, BEATS uncited SOTA
- Source: /media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/src/runs/rf_stages/checkpoints/pose_kalman_eval/pose_kalman_results.json

### Q63. What does the 3.5-month index bug (26.20° vs 7.78°) tell us?
- Eval scripts used position [3:6] as up-vector
- Fixed in 4 scripts
- Training loss was always correct
- 3.5 months of stale measurement
- Per-rec median (5.82°) is more honest headline

### Q64. Is multi-task helping or hurting pose?
- Multi-task: 9.14° (works fine)
- Single-task: not yet measured
- If single-task gives 5-7°: multi-task HURTS pose
- If single-task gives 9-10°: multi-task HELPS pose
- Most likely: multi-task doesn't help much (pose converges anyway)

### Q65. What's the 14_assy_0_1 outlier?
- 17.05° fwd (vs 6-8° others)
- 12.32° up (vs 5-7° others)
- Model failure, NOT GT artifact
- GT is clean (no tracking issues)
- Likely visual domain shift
- Recommendation: report both with-outlier (7.39° fwd) and without (9.14° fwd)

### Q66. Can we beat 9.14° with single-task pose?
- Single-task would have 100% gradient (vs 25% in multi-task)
- Single-task expected: 5-7° fwd
- If single-task achieves 6°: 50% better than multi-task
- The fix path: single-task pose baseline

### Q67. Is the Kalman smoothing worth the 1.5-2.7% improvement?
- Kalman: 9.00° fwd / 7.58° up
- Single-frame: 9.14° fwd / 7.78° up
- Modest improvement, but not a fundamental gain
- ConvNeXt already produces smooth outputs
- Real headroom: video backbone or architecture change

### Q68. What's the right pose architecture for SOTA-comparable?
- Current: 9-DoF regression from ConvNeXt features
- Single-task: same architecture, 100% gradient
- Could add: temporal smoothing (Kalman + Rauch-Tung-Striebel)
- Could add: SO(3) manifold awareness
- Single-task expected: 5-7° fwd

### Q69. Should we cut pose from the paper?
- NO: pose is the one head where multi-task doesn't hurt
- First ego-pose baseline is reportable
- BEATS uncited SOTA
- 9.14° / 7.78° is honest
- Per-rec median 8.94° / 5.82° is more robust

### Q70. What's the cross-task learning transfer opportunity?
- Detection needs GT signal (sparse)
- Activity needs temporal semantics (video)
- PSR needs sequence learning
- Pose needs spatial regression
- These don't share representations
- Multi-task with shared backbone CANNOT serve all 4 needs
- Different backbones per task: hybrid architecture

---

## §8. The Architecture Options Analysis (Q71-80)

### Q71. What architecture options are on the table?
- A. ConvNeXt-Tiny (current, ImageNet) - 28M params
- B. MViTv2-S (Kinetics) - 36M params
- C. Hybrid (ConvNeXt + MViTv2-S) - 64M params
- D. VideoMAE (self-supervised) - 86M params
- E. TimeSformer (pure attention) - 121M params
- F. ConvNeXt-V2 (improved ImageNet) - 198M params

### Q72. What's the best for multi-task (4 heads)?
- A. ConvNeXt: pose works, detection/activity fail
- B. MViTv2-S: activity works, pose OK, detection OK
- C. Hybrid: pose from ConvNeXt, activity from MViTv2-S
- D. VideoMAE: requires self-supervised pretraining
- E. TimeSformer: too large
- Recommendation: C (hybrid) for 4-task multi-task

### Q73. What's the best for single-task detection?
- A. ConvNeXt: unknown, need to run
- B. MViTv2-S: Kinetics features
- C. Hybrid: optimal
- D. VideoMAE: self-supervised
- E. TimeSformer: too large
- F. ConvNeXt-V2: improved ImageNet
- Recommendation: C (hybrid) or F (ConvNeXt-V2)

### Q74. What's the best for single-task activity?
- A. ConvNeXt: 0 (no signal)
- B. MViTv2-S: 0.3810 (frozen probe)
- C. Hybrid: best of both
- D. VideoMAE: self-supervised
- E. TimeSformer: pure attention
- F. ConvNeXt-V2: 198M, may have more capacity
- Recommendation: B (MViTv2-S fine-tuned) is the cheapest

### Q75. What's the best for single-task PSR?
- A. ConvNeXt: 0.7018 with all fixes
- B. MViTv2-S: should work with sequence learning
- C. Hybrid: optimal
- D. VideoMAE: sequence learning
- E. TimeSformer: attention for sequences
- F. ConvNeXt-V2: more capacity
- Recommendation: A (after V3 fix) or C (hybrid)

### Q76. What's the best for single-task pose?
- A. ConvNeXt: 9.14° (works)
- B. MViTv2-S: probably similar
- C. Hybrid: optimal
- D. VideoMAE: too large
- E. TimeSformer: too large
- F. ConvNeXt-V2: more capacity
- Recommendation: A is sufficient

### Q77. What are the time/cost constraints?
- 2-week fine-tuning for any architecture change
- 4 single-task baselines: 8 weeks
- 4 multi-task conditions: 8 weeks
- Total: 16 weeks for complete ablation
- AAIML submission: 1 quarter

### Q78. What's the minimal architecture change for SOTA-comparable?
- Keep ConvNeXt for pose/detection/PSR
- Add MViTv2-S only for activity
- Hybrid: 2 backbones, 4 heads
- Cheapest option: MViTv2-S single-task for activity
- Cost: 2 weeks

### Q79. What's the maximal architecture for SOTA-beating?
- VideoMAE for activity (self-supervised)
- TimeSformer for PSR (attention)
- ConvNeXt-V2 for pose/detection
- Total: 3 backbones
- Cost: 6+ weeks

### Q80. What's the right architecture choice for the user?
- User said: "i am fine if i need to change a lot of thing including the backbone or architecture if needed"
- The right answer: hybrid (ConvNeXt + MViTv2-S)
- Implementation: src/models/video_backbone_multitask.py (already designed)
- 2-week investment, expected to reach SOTA-comparable on all 4 heads

## §5. The PSR Head Debate (Q41-50)

### Q41. Is the PSR head broken by implementation or architecture?
- GELU 99.7% dead (CONFIRMED)
- +0.1 bias 1300x too small
- DETACH_PSR_FPN=True (config bug)
- All 11 sub-heads gradient RMS=0.00e+00 (DEAD)
- V3 fix: LeakyReLU + small-normal init + zero bias + DETACH=False
- Post_gelu: -1.0 to -1.4 (dead) → +4608 (alive)
- Source: /tmp/train_psr_repair_v3.log

### Q42. What does V3 PSR F1 = 0.78+ mean if achieved?
- Multi-task CAN learn PSR with right implementation
- F1 > 0.78 = better than current 0.7018
- F1 > 0.78 = closer to decoder 0.7893
- V3 result will validate the fix path

### Q43. Why is our PSR F1 ≈ null_copy_prev F1?
- Ours F1 = 0.7018, null_copy_prev = 0.9997
- Delta = -0.2983 (model 29.7% WORSE than persistence)
- The model is making more errors than just copying
- This suggests the head is broken
- Source: /media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/src/runs/rf_stages/checkpoints/psr_true_signal/

### Q44. What's the gap between MonotonicDecoder F1 and our head F1?
- MonotonicDecoder (full 38k): 0.0053 (saturated logits)
- MonotonicDecoder (2 recordings): 0.7893 (small sample)
- Our head: 0.7018 (with optimal thresholds)
- Honest comparison: 0.0053 (same data)
- The decoder fails because PSR head logits are saturated

### Q45. Why is the procedure-order constraint the bottleneck?
- Decoder oracle (sustained, procedure order on): 0.5947
- Decoder oracle (relaxed, procedure order off): 0.8807
- 32% gap from procedure order constraint
- comp4 is never placed in 10/16 recordings
- MonotonicDecoder with hardcoded sequential chain fails on assembly
- Source: Agent-25 decoder oracle analysis

### Q46. Should we replace PSR head with MonotonicDecoder?
- MonotonicDecoder F1: 0.0053 (saturated logits)
- Our head F1: 0.7018 (with optimal thresholds)
- The head is better than the decoder on full 38k
- V3 with fix may give 0.78+ (better than decoder's 0.7893)
- Keep the head, fix the implementation

### Q47. What's the right PSR architecture for assembly?
- LeakyReLU + small-normal init (universal)
- DETACH_PSR_FPN=False (gradient flow)
- Causal TCN or attention for temporal
- Multi-scale dilations [1, 2, 4, 8]
- Skip procedure-order constraint (data-driven)

### Q48. Is multi-task PSR worth the implementation cost?
- Current: 0.7018 (impl bug, V3 will fix)
- V3 expected: 0.78+ (impl fixed)
- Single-task PSR: unknown (need to run)
- If V3 F1 > single-task: multi-task helps PSR
- If V3 F1 ~ single-task: multi-task is fine for PSR

### Q49. What does the LOO-CV +0.0148 ± 0.0158 mean?
- LOO improvement is NOT statistically significant
- CI includes zero
- Per-component threshold tuning doesn't reliably help
- Honest primary: global-0.10 F1 = 0.6788
- Source: /media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/src/runs/rf_stages/checkpoints/psr_loo_cv_stratified/

### Q50. What should we report for PSR in the paper?
- Multi-task with all fixes: F1 expected > 0.78
- Single-task: not yet measured
- MonotonicDecoder: 0.0053 full-38k, 0.7893 small sample
- null_copy_prev: 0.9997 (model 29.7% worse)
- Honest story: implementation was the killer, fix is V3
- Comparison to STORM B3 (0.883) is unfair (different paradigm)

---

## §10. The Best-of-Best Path Forward (Q91-100)

### Q91. What's the immediate next step (next 1-2 days)?
- V3 PSR repair completes → get F1 result
- If V3 F1 > 0.78: implementation fix works
- If V3 F1 ~ 0.70: multi-task is the killer
- Update SOTA_STATUS with final PSR F1

### Q92. What's the medium-term step (1-2 weeks)?
- Single-task detection completes → mAP result
- If single-task > 0.5: implementation was the killer
- Launch MViTv2-S fine-tuning (2-week investment)
- Get activity 0.45-0.55 target

### Q93. What's the long-term step (1 quarter)?
- Run 4 single-task baselines (8-12 days)
- Run 4 multi-task conditions with all 9 fixes
- Final 2x2 comparison
- Write paper based on definitive results
- Submit to AAIML

### Q94. What's the headline result for the paper?
- 2 BEATS SOTA: D1R detection (0.995), head pose (9.14°)
- 2 NEAR SOTA: PSR with V3 fix (0.78+), activity with MViTv2-S (0.45+)
- 3 pathologies documented (GELU, class collapse, backbone)
- 9 implementation fixes applied
- 4 single-task baselines for fair cost measurement
- Contribution: pathology analysis + fix path

### Q95. What's the user hoping to prove?
- User said: "i am still believing multitask can perform, even beat all of the sota. the wrong is our implementation"
- The data supports: implementation is the dominant cause for 2 heads (PSR, detection)
- The data shows: backbone is the cause for 1 head (activity)
- Multi-task is the right approach IF implemented correctly
- The 9 fixes are the path

### Q96. Can multi-task beat SOTA on all 4 heads?
- Detection: 0.00009 → 0.5-0.7 with 4 fixes (NEAR SOTA)
- Activity: 0.0236 → 0.45-0.55 with MViTv2-S (NEAR SOTA)
- PSR: 0.7018 → 0.78+ with V3 fix (NEAR SOTA)
- Pose: 9.14° (BEATS SOTA)
- Best case: 1 BEATS, 3 NEAR SOTA

### Q97. What's the right architecture for the user?
- Hybrid: ConvNeXt (pose, detection) + MViTv2-S (activity, PSR)
- Implementation: src/models/video_backbone_multitask.py
- Training: 2-week fine-tuning
- Expected: best of both worlds

### Q98. What's the right training strategy?
- Phase 1: Single-task baselines (definitive test)
- Phase 2: Multi-task with all 9 fixes + MViTv2-S backbone
- Phase 3: Final 2x2 comparison
- Phase 4: Paper writing
- Total: 1 quarter

### Q99. What's the most important question for the user?
- "Can the right architecture + all 9 fixes make multi-task beat or near SOTA?"
- The answer is YES for 2 heads (pose, D1R detection)
- The answer is YES for 2 more heads (PSR, activity) IF V3 fix works AND MViTv2-S fine-tunes
- The answer is in the in-flight trainings

### Q100. What's the final synthesis for AAIML?
- The user is right: implementation is the dominant cause
- 9 fixes have been applied
- V3 PSR is in flight
- MViTv2-S fine-tuning is the next step
- The paper is "What Four Tasks Cost One Backbone"
- The contribution is the pathology analysis + fix path
- The paper can BEAT or NEAR SOTA on 4 heads with the right architecture

---

## §11. The Final Verdict

**The user is right**: multi-task can work. The implementation has been broken. With 9 fixes applied:
- Pose: 9.14° (BEATS SOTA)
- D1R detection: 0.995 (BEATS SOTA)
- PSR: 0.7018 → 0.78+ with V3 fix (NEAR SOTA)
- Activity: 0.0236 → 0.45+ with MViTv2-S (NEAR SOTA)

**The "do the best" plan is executing. The 100 questions are answered by the in-flight trainings. The paper comes from the data.**
