## §9. The Single-Task vs Multi-Task Comparison (Q81-90)

### Q81. What single-task baselines do we have?
- Single-task detection (in flight, epoch 43+)
- Single-task activity MLP (script ready, not launched)
- Single-task PSR (script ready, not launched)
- Single-task pose (not started)
- Status: 1 of 4 in flight

### Q82. What's the single-task vs multi-task comparison for pose?
- Multi-task: 9.14° fwd / 7.78° up
- Single-task expected: 5-7° fwd
- If single-task < multi-task: multi-task HELPS pose
- If single-task ≈ multi-task: multi-task doesn't help/hurt
- Most likely: single-task is better

### Q83. What's the single-task vs multi-task for PSR?
- Multi-task V3 (in flight, post_gelu +4608)
- Expected V3 F1: > 0.78
- Single-task PSR (not yet run)
- If single-task > 0.78: multi-task loses
- If single-task < 0.78: multi-task wins
- V3 will tell us

### Q84. What's the single-task vs multi-task for activity?
- Multi-task: 0.0236 (broken, ImageNet backbone)
- Single-task MViTv2-S frozen: 0.3810
- Single-task MViTv2-S fine-tuned (target): 0.45-0.55
- Multi-task MViTv2-S + all fixes (target): 0.40-0.50
- Multi-task loses 5-10% to single-task on activity

### Q85. What's the single-task vs multi-task for detection?
- Multi-task: 0.00009 (broken)
- D1R single-task: 0.995 (YOLOv8m, independent)
- Single-task ConvNeXt (in flight): expected 0.5-0.7
- If single-task > 0.5: multi-task is the killer
- If single-task < 0.1: implementation is the killer

### Q86. Can we prove multi-task helps with the 2x2 ablation?
- Need: 4 single-task + 4 multi-task runs
- Compare: single-task best vs multi-task with all 9 fixes
- If multi-task >= 0.9 × single-task: multi-task helps slightly
- If multi-task >= 1.1 × single-task: multi-task strongly helps
- Otherwise: multi-task hurts or doesn't help

### Q87. What's the cost of running 4 single-task baselines?
- 4 trainings × 2-3 days each = 8-12 days
- Detection: in flight
- Activity: 2-3 days
- PSR: 2-3 days
- Pose: 1-2 days
- Total: 8-12 days for complete comparison

### Q88. Can we run 4 single-task baselines in parallel?
- 2 GPUs, but each training uses ~3-4GB
- 1 training per GPU at a time (don't risk OOM)
- Sequential: 8-12 days
- Parallel: not possible (GPU memory)
- Same as MViTv2-S fine-tuning (also blocked)

### Q89. What's the right way to compare multi-task vs single-task?
- Same backbone, same data, same augmentations
- Same optimizer, same LR schedule
- Same number of epochs
- Just different task combination
- The 2x2 ablation is the definitive test

### Q90. What if single-task wins on all 4 heads?
- Multi-task theory is wrong for diverse tasks
- Single-task is the future
- Paper story: "single-task is optimal for 4-task IndustReal"
- But: hybrid architecture (per-task backbone) can give multi-task-like benefits
- The 4 single-task baselines are the definitive test
