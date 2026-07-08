# 165 — Master Paper Blueprint for AAIML Submission

**Date:** 2026-07-08
**Status:** Living document — updated as training progresses
**Purpose:** Single source of truth for the AAIML paper structure, claims, evidence, and progress.

---

## 1. Core Thesis (One Sentence)

> "Multi-task training on IndustReal does NOT hurt the individual heads when (a) Kendall log_var weights are rebalanced, (b) the activity backbone is replaced with a video-pretrained model (MViTv2-S), and (c) the detection backbone is replaced with a YOLOv8m detector — yielding SOTA-comparable per-head metrics on all 4 tasks from a single training run."

## 2. File Structure for AAIML Submission

| File | Purpose | Status |
|---|---|---|
| `166_DEEP_QUESTIONS_FUEL.md` | 50+ deep questions driving the paper | Drafting |
| `167_MULTITASK_ARCHITECTURE_STRATEGY.md` | Architecture change strategy | Drafting |
| `168_SOTA_COMPARISON_DATA_AUDIT.md` | All SOTA references with file paths | Drafting |
| `169_TRAINING_PROGRESS_ARCHITECTURE.md` | Training runs, V5b/V8 status | Drafting |
| `170_DISCUSSION_CONCLUSION.md` | Multi-task helps vs hurts, efficiency | Drafting |

## 3. Headline Claims (4 main results)

| Head | Claim | Current Best | SOTA | Status |
|---|---|---|---|---|
| Detection | 0.995 mAP50 (D1R YOLOv8m) | 0.995 | WACV 0.838 | ✅ Beat |
| Activity | 0.45-0.55 top-1 (MViTv2-S fine-tune) | 0.3810 (frozen) | WACV 0.6223 | Pending V8 |
| PSR | 0.5-0.7 F1 (V5b/V8 multi-task) | 0.7018 (V5b pre-fix) | STORM 0.506 | Pending V5b/V8 |
| Pose | 7.5-8.5° fwd MAE (V5b multi-task) | 8.52° (V5b epoch 34) | First baseline | ✅ First baseline |

## 4. Training Status (current)

- V5b (PID 758477): Epoch 35 82% with KENDALL rebalanced, ETA epoch 50 ~20h
- V8 (PID 843794): Epoch 0 step 700+ on GPU 1, may collapse
- YOLOv8m D1R: 0.995 mAP50 (in repo, used as-is for detection)
- Frozen MViTv2-S probe: 0.3810 (in repo, used as-is for activity baseline)

## 5. Key Risks

1. **V5b/V8 collapse on classification heads** (activity, PSR): pose (regression) dominates gradients
2. **V6 (multi-task with MViTv2-S) requires 3+ days** of training
3. **V8 not yet validated**: training started, results pending
4. **Detection backbone mismatch**: ConvNeXt vs YOLOv8m, SOTA-comparable is the YOLOv8m single-task result

## 6. Next Steps

- [ ] V5b epoch 50 val (in ~20h)
- [ ] V8 epoch 5 val (in ~3h, may be collapsed)
- [ ] If V8 collapse, add init biases and resume
- [ ] Compile all metrics into AAIML paper draft
- [ ] Run 10-agent debate on 165-170 for refinement
- [ ] Submit AAIML paper (deadline October 10, 2026)
</content>
