# 156 — 100 Deep Questions for Opus

**Date:** 2026-07-07
**Purpose:** Comprehensive deep-dive Q&A spanning detection, PSR, activity, head pose, implementation, and strategic direction.

---

## §4. The Detection Head Debate (Q31-40)

### Q31. Why does D3 multi-task detection get 0.00009 mAP?
- 91.9% of frames have zero GT
- 5 classes NEVER predicted across 38k frames
- Class 12 is "default catch-all" for 7 different states
- Box regression: mean IoU 0.234 (below 0.5)
- Source: Agent-55 root cause analysis

### Q32. Is detection broken by multi-task or implementation?
- 4 fixes applied: GT-balanced sampler, gamma_neg 2.0, anchor audit, class verify
- D1R single-task (YOLOv8m) = 0.995 (BEATS SOTA)
- Pending: single-task ConvNeXt detection (in flight, ~3.4 days)
- Likely: implementation is dominant cause

### Q33. Will the 4 detection fixes make D3 work?
- GT-balanced sampler: 100% batches have GT (was 8%)
- gamma_neg 2.0: harder negative mining
- Anchor audit: not the issue
- Class index: not the issue
- Likely: D3 mAP improves from 0.00009 to 0.1-0.5

### Q34. Can multi-task detection beat single-task?
- With 4 fixes: maybe 80-90% of single-task
- Without fixes: 0.1% of single-task
- The 91.9% empty frames is structural (not multi-task)
- Multi-task doesn't cause 91.9% empty frames
- Multi-task just doesn't help with sparse GT

### Q35. Is D1R the right comparison for multi-task?
- D1R is YOLOv8m (different architecture)
- Multi-task uses ConvNeXt (different backbone)
- Fair comparison: single-task ConvNeXt vs multi-task ConvNeXt
- Both with same fixes
- Currently: single-task ConvNeXt training in flight

### Q36. What does D4+D1R = 0.6364 tell us?
- Decoder F1 = 0.6364 with D1R weights (dense detection)
- vs 0.347 with pretrained YOLOv8m (sparse detection)
- 83% relative improvement from dense detection
- Confirms: detection density is the binding constraint
- Multi-task decoder transfer works

### Q37. Why does D4 default = 0.000?
- Q48 hysteresis thresholds: hi=0.5, lo=0.3, sustain=3
- Sparse YOLOv8m detections don't meet hi=0.5
- Re-tuned: hi=0.3, lo=0.1, sustain=2 → 0.347
- Multi-task with D1R + re-tuned → 0.6364
- Default thresholds starve sparse detection

### Q38. Should we report detection per-class?
- 24 classes total
- 6 zero-GT classes (channels 1, 2, 3, 14, 15, 23)
- 18 present classes
- Per-class AP varies widely
- Per-class detection rate: most 0%, some 3-4%
- Source: /media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/src/runs/rf_stages/checkpoints/detection_root_cause/analysis.md

### Q39. What's the true multi-task detection ceiling?
- After 4 fixes + V3 PSR repair: unknown
- Estimated: 0.05-0.15 mAP (vs single-task 0.5-0.7)
- Multi-task still loses to single-task for detection
- Reason: 91.9% empty frames is structural
- Multi-task is wrong for detection in this setup

### Q40. Should we cut detection from the paper?
- Multi-task 0.00009 is not reportable
- Single-task D1R 0.995 IS reportable (independent model)
- Recommendation: cut multi-task detection, report single-task D1R
- Show multi-task as "limitation, see pathology analysis"
- The fix path is single-task with same backbone
