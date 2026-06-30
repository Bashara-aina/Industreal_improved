# AAIML 2027 — 14: Opus Master Prompt for AAIML [2026-06-30]

## Who We Are

Training a multi-task assembly verification model on IndustReal for an
**AAIML 2027 paper (IEEE Xplore, Tokyo, October 10 2026 deadline)**.
We have 102 days. After 10 days of training pathology documented with Opus's help
(files 56-63), we now have:
- Simple MLP head replacing broken TCN+ViT (ACTIVITY_HEAD_SIMPLE=True)
- 3 verified findings (temporal-head/sampler mismatch, gradient probe misreading,
  head pose annotation artifact)
- All infrastructure fixes (NUM_WORKERS=0, RAM cache, watchdog, pre-val checkpoint)

## The AAIML Paper

**Venue**: IEEE AAIML 2027, Tokyo, March 29-31, 2027
**Deadline**: October 10, 2026 (102 days)
**Format**: IEEE 2-column, 6-10 pages
**Current state**: 10 strategy files in `analyses/consult_2026_06_10/AAIML/`
  (00-10) covering win strategy, reviewer defense, section-by-section, execution
  plan, tables/figures, risk register, competitor analysis, weaknesses, checklist.
  Plus a LaTeX draft: `popw_aaiml2027.tex`

**Current paper draft claims:**
- Detection mAP50_pc = 0.34 (from old RF2 run, 3 heads only)
- Activity Top-1 = 18.3% (from old RF3 run, TCN+ViT now replaced)
- Head pose MAE = 9.1° (un-normalized GT)
- 53M params, 93 GFLOPs, 4.8 FPS

**ALL of these numbers are from pre-fix runs and will change.**

## What's Running Now (PID 3618126)

```
Stage: RF4 (50% data, simple head, all 5 heads)
Epoch: 3/23
Expected first simple head validation within 30 min
```

## Three New Findings to Validate with Opus

### Finding 1: Temporal-Head/Sampler Mismatch (PUBLISHABLE)
- WeightedRandomSampler + FeatureBank = shuffled "temporal" windows
- TCN+ViT (8.2M params) learns noise → 3.7k frames overfit → majority collapse
- Fix: simple 150K MLP bypasses the temporal stack
- **Question for Opus**: Is this finding novel enough for AAIML? The "balanced sampler
  defeats temporal head" mechanism seems obvious in retrospect. Is there prior art
  documenting this specific failure mode?

### Finding 2: Gradient Probe Misreading (PUBLISHABLE)
- `_log_per_head_grad_norm` logs first/last param only, not head totals
- We spent 10 days on non-existent 312x gap
- **Question for Opus**: Does this merit its own subsection, or is it a paragraph
  in the Discussion section? Is the finding strong enough to survive review?

### Finding 3: Head Pose Annotation Artifact (DATA CONTRIBUTION)
- pose.csv forward vectors un-normalized (norm 0.014-0.030)
- Eval normalizes → angular MAE valid, training MSE suboptimal
- **Question for Opus**: Is this worth a paragraph, a footnote, or a data appendix?
  Should we contact the IndustReal dataset authors?

## What We Need from Opus: Final Decisions

### Architecture (1-3)
1. **Simple MLP head architecture**: The current design is
   LayerNorm→Linear(512→256)→GELU→Dropout→Linear(256→75). Is this optimal for
   per-frame 74-class activity on 3.7k frames? Should we try deeper (2 hidden layers)
   or wider (512→512)? The head only has 150K params so experimentation is cheap.

2. **Should we keep the old TCN+ViT code in the repo?** For the ablation paper claim
   ("simple head vs temporal head"), we need to compare. But the temporal head requires
   feature bank which needs the recording_id keyed state. If we remove the temporal head,
   we can't reproduce the negative result. Should we keep it as `ACTIVITY_HEAD_SIMPLE=False`
   for the ablation experiment?

3. **Detection with less gradient competition**: With the simple head (150K vs 8.2M params),
   detection faces less gradient contention. Should we raise DET_LR_MULTIPLIER from 1.0
   to 2.0 or 3.0 to accelerate detection? Or keep 1.0 and let the natural gradient
   flow improve detection automatically?

### Paper Structure (4-6)
4. **How should we structure the "Lessons from Multi-Task Training" section?**
   Three findings (temporal mismatch, probe misreading, pose annotation) — are they
   strong enough as separate subsections, or should they be combined into one
   "Lessons Learned" section?

5. **The ethics + blockchain section**: AAIML is an ML/AI conference, not ethics.
   The current draft has blockchain (§5) and ethics (§7) taking 1.75 pages.
   Should we reduce to:
   - 0.5 page blockchain (just the x402 pipeline diagram + latency)
   - 0.25 page ethics (IEEE 7005 table only)
   - Move remaining space to architecture + experiments
   Or does the blockchain section make the paper stand out at AAIML?

6. **Page budget**: Currently:
   - Introduction: 1 page
   - Related Work: 1.5 pages
   - Architecture: 1.5 pages
   - Experiments: 1.5 pages
   - Blockchain: 0.75 page
   - Factory Pilot: 1 page
   - Ethics: 0.5 page
   - Conclusion: 0.5 page
   Total: 8.25 pages (within 10-page limit with $140 extra page fees).
   With the new findings, we need:
   - "Lessons from Multi-Task Training" (~0.75 page)
   Where does it go? Squeeze Related Work to 1.0 page and Ethics to 0.25 page?

### Training Strategy (7-10)
7. **When does the simple head's first validation tell us "it works"?**
   We need a clear go/no-go criterion. If epoch 3 shows:
   - act_macro_f1 > 0.01 = works (diverse predictions)
   - act_macro_f1 < 0.001 = still collapsing (need further fix)
   - What threshold should we use?

8. **Head pose normalization**: Should we normalize at data-load time
   (industreal_dataset.py) or at loss time (losses.py)? Data-load is cleaner
   but requires re-running the entire pipeline. Loss-time normalization
   is surgical but affects gradient computation differently.

9. **RF10 schedule**: We have 102 days until deadline but only ~3 days of
   continuous training needed. Should we aim to complete RF10 by July 3
   (3 days) and spend the remaining 99 days writing, or run longer ablations
   (3 seeds, more data aug, etc.) first?

10. **GPU 0 (RTX 3060)**: Still completely idle. With the simple head reducing
    model size, could we run a 3-seed ablation on GPU 0 in parallel with
    main training on GPU 1? Even 2 epochs × 3 seeds ≈ 5 hours of GPU 0 time
    would give us variance estimates for the paper.

## What We've Committed

All code changes pushed to `main`:
```
18e0160 fix: apply Opus-recommended simple_classifier init (logit bias=-0.5)
8207632 fix: bypass feature bank in non-staged mode
ea325d2 chore: raise RAM_CACHE_MAX_IMAGES to 8000
```

All analysis files committed:
- 56-61: Original consult (activity collapse, gradient imbalance, infrastructure)
- 62-63: Opus response + verified analysis (DEBUNKED our gradient analysis)
- 64-68: Follow-up consult (subprocess eval, head pose fix, paper reframe, roadmap)
- AAIML/00-10: AAIML strategy files
- AAIML/11-14: This update set

## File Guide for Opus

| File | Topic | What We Need |
|------|-------|-------------|
| AAIML/11 | Numbers update | Understand which paper numbers changed |
| AAIML/12 | New contributions | Validate 3 findings for paper inclusion |
| AAIML/13 | Architecture rewrite | Review the simplified activity head section |
| **AAIML/14 (this)** | Master prompt | Answers to 10 questions above |
