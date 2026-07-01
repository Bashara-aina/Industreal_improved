# ICHCIIS-26 — Day-by-Day Execution Plan (97 Days)

**Paper Deadline**: October 4, 2026
**Start Date**: June 29, 2026
**Total**: 97 days

---

## Phase 0: Foundation (Jun 29 — Jul 6) — 8 days

### Week 1

| Day | Task | Output | GPU Needed? |
|-----|------|--------|-------------|
| Jun 29 | Run efficiency measurement | Params, GFLOPs, FPS | 1h |
| Jun 29 | Generate per-class diagnostic | Detection per-class AP | 1h |
| Jun 30 | Run confusion matrix | 24x24 matrix PNG | 1h |
| Jun 30 | Plot Kendall weight curves | log_var_* figures | No |
| Jul 1 | Plot per-task validation curves | All task curves | No |
| Jul 1 | Write limitations section | Section 7 draft | No |
| Jul 2 | Write ethical framework draft | Section 6 draft | No |
| Jul 2 | Write conclusion draft | Section 8 draft | No |
| Jul 3 | Compile reference list | 30+ citations | No |
| Jul 4 | Check rf2 progress | Detection + head pose numbers | Check only |
| Jul 5-6 | Complete Phase 0 documentation | All free numbers done | No |

**Milestone**: All free numbers (efficiency, diagnostics) completed. Draft prose started.

---

## Phase 1: Core Training (Jul 7 — Aug 15) — 40 days

### Week 2-3: Let rf2 finish + full evaluation

| Task | Duration | Notes |
|------|----------|-------|
| Let rf2 reach epoch 36 | ~13 GPU-h | Detection + head pose |
| Full test evaluation | 1h | mAP50, mAP50_pc, head pose MAE |

### Week 3-4: PSR go/no-go

| Task | Duration | Notes |
|------|----------|-------|
| Overfit PSR on 50 sequences | 1h | If f1 > 0.3 → keep. If 0.0 → drop. |

### Week 4-6: Activity training (rf3)

| Task | Duration | Notes |
|------|----------|-------|
| Advance to rf3 | ~11 GPU-h | activity_top1 expected 10-30% |
| Full evaluation | 1h | Top-1, Top-5, confusion matrix |

### Week 6-8: Ablation A (single-task vs multi-task)

| Task | Duration | Notes |
|------|----------|-------|
| Recovery_det_only (single-task det) | ~8 GPU-h | Compare vs rf2 detection |
| Evaluate both | 1h | Δ mAP50_pc = the result |

### Week 8-10: Ablation B (FiLM) — if activity works

| Task | Duration | Notes |
|------|----------|-------|
| No-FiLM training | 3h | Activity only |
| PoseFiLM-only | 3h | |
| Both FiLM | 3h | |

**Milestone**: All training done, all numbers collected.

---

## Phase 2: Paper Writing (Aug 16 — Sep 15) — 30 days

### Week 11-12: First draft

| Section | Deadline | Notes |
|---------|----------|-------|
| Abstract | Aug 16 | 200-250 words |
| Introduction | Aug 18 | Human problem first |
| Background | Aug 21 | 30+ citations |
| System Design | Aug 25 | Architecture diagram |
| Results | Aug 28 | All tables filled |
| Blockchain | Sep 1 | x402 protocol |
| Ethical Framework | Sep 5 | IEEE 7005-2021 deep dive |
| Discussion | Sep 8 | Limitations + future work |
| Conclusion | Sep 10 | Summary paragraph |
| References | Sep 12 | Format check |

**Milestone**: Complete first draft by September 12.

---

## Phase 3: Polish and Submit (Sep 16 — Oct 4) — 19 days

### Week 13-14: Revision

| Task | Deadline |
|------|----------|
| Self-review | Sep 16-18 |
| Check all format requirements | Sep 19 |
| Proofread for language | Sep 20-22 |
| Generate all figures | Sep 23-25 |
| Final formatting (6-8 pages, double-column) | Sep 26-28 |

### Week 15: Submit

| Task | Deadline |
|------|----------|
| Final PDF generation | Sep 29 |
| Submit to APSTE system | **Oct 4** |
| Register (early bird) | Sep 24 |

---

## What If You Have Only 30 Days?

If you start late, here is the compressed plan:

| Priority | Task | Time |
|----------|------|------|
| 1 | Efficiency measurement (free) | 1h |
| 2 | Let rf2 finish | 13 GPU-h |
| 3 | Ablation A (single-task det) | 8 GPU-h |
| 4 | Write paper with existing numbers | 5 days |
| 5 | Submit | — |

The irreducible minimum paper: detection (reframed) + head pose + efficiency + Ablation A + ethical framework. This is a complete, publishable paper.
