# TODO: PSR Backbone Swap Experiment

**Date added:** 2026-07-04
**Priority:** Pre-submission (before final paper numbers)
**Estimated effort:** 2-4 hours (offline, no training needed)
**Runs on:** Idle RTX 3060 (while main training continues on 5060 Ti)

---

## Problem

Our PSR F1 (0.144) cannot be directly compared to the B3 SOTA (0.883) because our detection backbone is much weaker (mAP 0.317 vs YOLOv8m's 0.838). Reviewers will ask: *"Is the PSR head weak, or is detection the bottleneck?"*

## Experiment

1. Download the IndustReal authors' YOLOv8m weights (public, COCO→Real+Synth trained) from:
   https://github.com/TimSchoonbeek/IndustReal

2. Run inference on the IndustReal validation set to produce ASD state predictions

3. Feed those predictions through our MonotonicDecoder (F22 fix active)

4. Report PSR F1@±3, Edit, POS

## Expected Outcomes

| Scenario | PSR F1 | Conclusion |
|---|---|---|
| F1 > 0.70 | PSR head is fine — detection is the bottleneck |
| F1 0.30–0.70 | Both contribute — moderate PSR head quality |
| F1 < 0.20 | PSR head itself needs work (architecture change) |

## Follow-up (if time permits)

After the swap experiment, add procedural knowledge constraints to match B3's setup (mask impossible state transitions). Expected gain: +0.10–0.15 F1.

---

## References

- SOTA checkpoints: https://github.com/TimSchoonbeek/IndustReal
- Our PSR decoder: `src/models/psr_transition.py` (MonotonicDecoder)
- Our PSR eval pipeline: `src/evaluation/evaluate.py` (F22 fix at line ~324)
- Benchmark reference: `AAIML/industreal-sota-benchmarks.md`
