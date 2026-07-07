# 156 — 100 Deep Questions: Multi-Task Theory Defense, Diagnosis, and Resolution

## §1. The Multi-Task Theory Defense (Q1-10)

### Q1. Is multi-task learning a valid theoretical framework?
- Multi-task is well-established in ML literature (Caruana 1997, Ruder 2017)
- Standard practice in NLP (BERT, T5), vision (Mask R-CNN), and audio
- The theory is NOT the problem in our case
- Source: /media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/analyses/consult_2026_06_10/AAIML/153_MULTI_TASK_DEBATE.md

### Q2. Can multi-task help when tasks share representations?
- Detection, head pose, activity, PSR all need image-based features
- They DO share low-level vision features
- Multi-task transfer is REAL when task correlations are positive
- The question is which architecture supports all 4

### Q3. What does Kendall uncertainty weighting guarantee?
- Theoretically: auto-balancing between tasks
- In practice: can collapse on imbalanced tasks
- BUT: bounded-Kendall (HP_PREC_CAP, KENDALL_FIXED_WEIGHTS) prevents collapse
- Source: src/config.py (HP_PREC_CAP, KENDALL_FIXED_WEIGHTS)

### Q4. Does the user's belief that "multi-task is fine" have literature support?
- Multi-task works when implemented correctly
- Standard results: +1-5% from multi-task
- The negative transfer is usually from bug, not theory
- User is right in principle

### Q5. Is the V3 PSR repair evidence supporting multi-task theory?
- PSR head activations +4608 (was -1.0 to -1.4 dead)
- The head CAN produce non-constant output
- The repair + DETACH_PSR_FPN=False fixes the implementation
- Multi-task theory is sound
- Source: /tmp/train_psr_repair_v3.log

### Q6. Can the same ConvNeXt serve 4 different heads?
- Yes, if the right heads are designed
- ConvNeXt features support spatial tasks (pose)
- Temporal tasks need additional aggregation (TCN/attention)
- The architecture is the bottleneck, not the backbone sharing

### Q7. Does the cascade table prove multi-task theory is wrong?
- NO. The cascade shows implementation failures
- Each broken head has a specific implementation cause
- GELU dead = arch bug, not theory bug
- 5 classes never predicted = training bug, not theory bug
- Source: /media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/src/runs/rf_stages/checkpoints/multi_task_cascade/cascade_table.md

### Q8. Can multi-task work with the right architecture?
- Yes, with the right backbone (Kinetics for activity)
- With the right heads (TCN for temporal)
- With the right loss balancing (bounded Kendall)
- With the right loss formulations (per-class for imbalanced)

### Q9. What's the strongest evidence FOR multi-task being correct?
- Pose works in multi-task (9.14 deg fwd)
- Linear probe (frozen) shows backbone has signal
- D1R single-task BEATS SOTA (multi-task CAN work)
- The user is right that implementation is the dominant cause

### Q10. Should the paper claim multi-task helps?
- If V3 PSR F1 > 0.78: yes, multi-task helps with fixes
- If single-task detection > 0.5: yes, with fixes
- If MViTv2-S fine-tuning activity > 0.45: yes, with right backbone
- Otherwise: multi-task is broken, single-task is the answer
