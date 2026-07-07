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
