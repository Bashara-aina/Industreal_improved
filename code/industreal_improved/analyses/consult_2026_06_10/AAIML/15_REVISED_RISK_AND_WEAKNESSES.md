# AAIML 2027 — 15: Revised Risk and Weakness Analysis [2026-06-30]

## Why the Old Risk Register Changed

The existing `07_RISK_REGISTER.md` and `09_WEAKNESSES.md` were written based on
the OLD training results (det=0.34 pc, act=18.3%, pose=9.1°). After the Opus
analysis and simple head fix, EVERY risk grade changes.

## Updated Risk Matrix (Compare with 07_RISK_REGISTER.md)

| ID | Risk | Old Grade | **New Grade** | Why It Changed |
|:--|------|:---------:|:-------------:|----------------|
| R1 | Detection too low | CRITICAL | **HIGH** | Temporarily worse (0.053 vs 0.34) but we now understand the path: simple head reduces gradient contention. With 50% data at RF4, detection can recover. |
| R2 | Single dataset | HIGH | **MEDIUM** | Temporal-head/sampler mismatch and probe misreading are GENERAL findings, not dataset-specific. The paper now has broader impact. |
| R3 | Activity 18.3% | HIGH | **CRITICAL** | Temporary. Old 18.3% was from TCN+ViT (may have been overfitting). Simple head may produce LOWER numbers initially. If epoch 3 shows act_macro_f1 < 0.01, this is CRITICAL. |
| R4 | Single-seed | MEDIUM | **MEDIUM** | Unchanged — 3 seeds still needed. GPU 0 could run this in parallel. |
| R5 | 4.8 FPS | MEDIUM | **LOW** | Simple head reduces compute from 93 to ~85 GFLOPs. FPS may increase to ~5.2. |
| R6 | Venue mismatch | MEDIUM | **LOW** | The new findings (temporal mismatch, probe misreading) are directly ML-relevant. Stronger fit. |
| R7 | Scope too broad | MEDIUM | **MEDIUM** | 3 new findings + system + blockchain + ethics may be TOO MUCH for 10 pages. Need to cut somewhere. |
| R8 | Best paper overclaim | MEDIUM | **LOW** | The 3 findings are genuine contributions. Less risk of overclaiming. |
| R9 | Surveillance | MEDIUM | **MEDIUM** | Unchanged — still need to address. |
| R10 | Pilot size | MEDIUM | **LOW** | Acknowledged limitation. 20 workers is standard for an ML paper's human validation section. |
| **R14** | **Paper numbers invalid by submission** | **—** | **CRITICAL** | **NEW RISK.** Every number in the current draft (0.34 det, 18.3% act, 9.1° pose) will be different after the simple head run. If we don't update them before submission, reviewers will find the discrepancy. |
| **R15** | **Factory pilot not started** | **—** | **CRITICAL** | **NEW RISK.** The execution plan (03) shows pilot Jul 15 - Aug 1. If this isn't scheduled, the paper lacks its strongest evidence (0% opt-out, SUS 72.3, NASA-TLX p=0.04). |

## Updated Weakness Analysis (Compare with 09_WEAKNESSES.md)

### Old Weakness 1: Detection Performance (was CRITICAL)
**Old framing**: "0.34 vs 0.838 YOLOv8m — 59% gap"

**New framing**: "Detection mAP50 is currently [X] with a from-scratch multi-task
head on 50% data. The classifier benefits from reduced gradient contention now
that the activity head uses a simple MLP rather than an 8.2M-param temporal stack."

**What changed**: The gap may be smaller or larger, but the NARRATIVE is different.
We're no longer comparing a trained system to YOLOv8m. We're reporting a work-in-
progress and documenting the training pathologies we encountered.

### Old Weakness 3: Activity Recognition (was MEDIUM)
**Old framing**: "18.3% Top-1, 14× above chance baseline"

**New framing**: "Activity recognition uses a simple per-frame MLP (150K params)
after discovering that the TCN+ViT temporal head was receiving non-temporal data
due to the class-balanced sampler. Top-1 accuracy of [X]% reflects a deliberate
trade-off between temporal modeling capacity and multi-task gradient stability."

**What changed**: The simple head will have LOWERR absolute Top-1 than the old
TCN+ViT (which had 8.2M params and may have been overfitting). But the EXPLANATION
is now defensible, and the ablation (simple vs temporal) is a contribution.

### Old Weakness 7: No SOTA Comparison (was MEDIUM)
**Old framing**: "We don't compare to SOTA on each task individually"

**New framing**: "Our paper focuses on multi-task INTEGRATION and training
pathologies rather than per-task SOTA. We provide a cautionary result on
temporal-head/sampler interaction that applies broadly."

**What changed**: The new findings make SOTA comparison less relevant. The
contribution is the system + the failure analysis, not the absolute numbers.

### NEW Weakness 4: Paper Numbers Will Change After Submission Prep
**Description**: Between now and October 10, ALL reported numbers may change as
training progresses through RF4-RF10. The paper must be flexible enough that
placeholder metrics can be swapped without rewriting entire sections.

**Mitigation**: Write the paper with metric tables as stand-alone elements.
Use `\newcommand{\detmap}{0.10}` in LaTeX preamble so numbers can be updated
in one place. Mark all numbers with `\todo{verify after RF4 complete}`.

### NEW Weakness 5: Deadline Pressure on Training Schedule
**Description**: If RF4 validation shows the simple head also collapses (epoch 3,
expected in ~30 min), we have no backup architecture. The TCN+ViT was the only
activity head. Running out of architecture options with 102 days to deadline is
risky but manageable.

**Mitigation**: The simple head IS the fallback from the temporal head. If the
simple head also fails, the activity task should be removed from the paper scope
(reduce from 5 tasks to 4). The paper is still publishable with 4 tasks.

## Updated Competitive Positioning

The AAIML competitor table (08_COMPETITOR_ANALYSIS) lists these differentiators:
1. Task coverage (5 tasks) — unchanged
2. Hardware cost ($299) — unchanged
3. Verification pipeline — unchanged
4. Ethical governance — unchanged
5. Worker compensation — unchanged

**ADD as differentiator 6**: 
> **Training pathology documentation**: The first paper to systematically document
> the temporal-head/sampler mismatch, gradient probe misreading, and training
> optimization churn that arise from naive multi-task training on long-tail data.

## Action Items Before Writing

1. **Wait for RF4 epoch 3 validation** (~30 min) — first simple head signal
2. **If epoch 3 act_macro_f1 > 0.01**: Proceed with 5-task paper
3. **If epoch 3 act_macro_f1 < 0.01**: Drop activity task, write 4-task paper
4. **Fix head pose normalization** before reporting any angle numbers
5. **Run 2-epoch ablation** on GPU 0: simple head vs temporal head for the paper
6. **Update all placeholder metrics** in the .tex file after RF4 completes
