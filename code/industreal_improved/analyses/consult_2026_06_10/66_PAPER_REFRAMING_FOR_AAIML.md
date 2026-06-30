# 66: Paper Reframing for AAIML 2027 Tokyo [2026-06-30]

## Opus's Core Finding (Venue-Agnostic)

From 63 §PART 4 — verified against every line of code:

> **Do not frame the paper as "we match SOTA on five tasks."** That framing forces
> gaps you cannot close. The defensible paper is an applied ML contribution combining
> (a) a working multi-task system on consumer GPUs and (b) rigorous analysis of the
> optimization pathologies encountered.

File 62 §5:

> A single shared-backbone model performing multi-task assembly verification in
> real time on consumer GPUs; we report multi-task trade-offs, establish the first
> consumer-hardware multi-task baseline on IndustReal, demonstrate head-pose tracking
> at parity with SOTA, and provide a rigorous analysis of joint-training failure modes
> under severe class imbalance and limited annotation — including a cautionary result on
> how a per-frame sampler silently defeats a temporal head, and how a per-parameter
> liveness probe can be misread as gradient starvation.

**This last sentence is the key.** It turns 10 days of "failure" into a genuine
methods contribution. Files 56-60 are the raw material.

## Why AAIML Is the Right Venue (Not AHFE)

| Factor | AHFE 2026 Hawaii | AAIML 2027 Tokyo |
|--------|:----------------:|:----------------:|
| Deadline | Passed (or imminent) | **Oct 10, 2026 (102 days)** |
| Primary audience | Human factors | **ML/AI researchers** |
| Best fit for our content | Ethics + pilot | **Architecture + training analysis** |
| Our 3 new findings | Too technical | **Perfect fit** |
| Travel | Hawaii (expensive) | **Tokyo (domestic, Chiyoda)** |
| Publisher | Conference proceedings | **IEEE Xplore** |
| Page limit | 10 pages | **6-10 pages (IEEE format)** |

AAIML's CFP explicitly covers: Deep Learning Architectures, Transfer Learning,
Multimodal Learning, Hybrid AI Systems, AI in Manufacturing, Explainable AI.
Every topic maps to our paper.

## The New Framing for AAIML

### Title Suggestion
> **POPW: A Multi-Task Deep Learning Framework for Assembly Verification on Consumer GPUs**

### Abstract (draft, 200 words)
```
We present POPW, a single-model multi-task system for real-time assembly verification
on consumer GPUs (RTX 5060 Ti, 4.8 FPS). The system jointly predicts assembly state
detection, action recognition, procedure step recognition, head pose, and body pose
from egocentric video using a ConvNeXt-Tiny backbone with shared FPN and two-stage
FiLM conditioning. With 46M trainable parameters and 85 GFLOPs per frame—less than
a single specialist model—POPW achieves head pose accuracy within 1° of dedicated SOTA
while covering five tasks that would otherwise require 3-5 separate models costing
$12K-$55K.

The paper's primary contribution is a rigorous analysis of the optimization pathologies
that arise when training five heterogeneous tasks on a single backbone under severe
class imbalance (46/72 classes with <1% annotation) and limited data (3.7k frames).
We document three verified failure modes:

1. Temporal-head/sampler mismatch: a per-frame class-balanced sampler feeds
   non-consecutive frames into a TCN+ViT temporal stack, eliminating temporal signal
   and inducing majority-class collapse.
2. Multi-task gradient dynamics under Kendall uncertainty weighting.
3. Per-parameter liveness probe misreading that wasted 10 days of optimization.

All code and model weights are open-source.
```

### Contribution Claims (ranked for AAIML reviewers)

| # | Claim | Evidence | Strength |
|:-|-------|----------|:--------:|
| 1 | Training pathology analysis (3 findings) | Opus-verified, reproducible | ★★★★★ |
| 2 | Temporal-head/sampler mismatch documented | Code + ablation | ★★★★★ |
| 3 | Gradient probe misreading identified | 10 days of wasted tuning | ★★★★★ |
| 4 | Head pose at SOTA on consumer GPU | [X]° angular MAE | ★★★★☆ |
| 5 | First multi-task baseline on IndustReal | 5 tasks from 1 model | ★★★★☆ |
| 6 | 97% cost reduction vs multi-model | $299 vs $12K-$55K | ★★★★☆ |
| 7 | Per-frame MLP vs temporal head ablation | Head-to-head comparison | ★★★☆☆ |

**For AAIML, claims 1-3 are the headline.** The failure analysis is more novel than
the absolute performance numbers. ML/AI reviewers will value the training pathology
insights more than system deployment details.

## Paper Structure for AAIML

| Section | Pages | Focus |
|---------|-------|-------|
| 1. Introduction | 1 | Multi-task on consumer GPU, cost motivation, 3 pathology findings |
| 2. Related Work | 1 | Multi-task learning, assembly understanding, consumer GPU systems |
| 3. Architecture | 1.5 | ConvNeXt-T + FPN + 5 heads + FiLM + simple MLP activity head |
| 4. Experiments | 2.5 | Dataset, primary results, ablations, efficiency |
| 5. Training Pathologies | 2 | **Core contribution** — temporal mismatch, gradient dynamics, probe misreading |
| 6. Conclusion | 0.5 | Summary, limitations, code release |
| **Total** | **~8.5 pages** | (within 10-page IEEE limit) |

### Section 5: Training Pathologies (2 pages — the novel contribution)

**5.1 Temporal-Head/Sampler Mismatch** (0.75 page)
- Balanced WeightedRandomSampler + recording_id FeatureBank = non-temporal sequences
- 8.2M TCN+ViT learns noise → majority collapse on 3.7k frames
- Fix: 150K simple MLP bypasses temporal stack
- Ablation results: simple head vs temporal head

**5.2 Multi-Task Gradient Dynamics** (0.5 page)
- Kendall uncertainty weighting precision dynamics
- log_var evolution showing task competition
- KENDALL_LOG_VAR bounds prevent complete suppression

**5.3 Gradient Probe Misreading** (0.5 page)
- `_log_per_head_grad_norm` logs first/last param only, not head totals
- 10 days wasted on non-existent "312x gap"
- Lesson: per-parameter norms ≠ head-level magnitudes

**5.4 Head Pose Annotation Artifact** (0.25 page)
- pose.csv forward vectors un-normalized (norm 0.014-0.030)
- Data contribution: documented and corrected for the community

### What to Remove from AHFE Version

| AHFE Content | AAIML Action | Reason |
|-------------|-------------|--------|
| Ethics as primary contribution | Reduce to 1 paragraph | AAIML is ML/AI, not ethics |
| IEEE 7005-2021 detailed mapping | Remove or 1 table | Not core to ML audience |
| Factory pilot (20 workers) | Remove or 1 sentence | Not replicable, no space |
| Worker surveillance analysis | Remove | Out of scope for AAIML |
| Blockchain motivations debate | Keep but shorten | Interesting hybrid AI angle |

### What to Add for AAIML

| New Content | Reason |
|-------------|--------|
| Detailed architecture diagram | ML audience expects it |
| Gradient norm comparison table | Supports claim 3 (probe misreading) |
| Simple head vs temporal head ablation | Supports claim 2 |
| log_var evolution over epochs | Supports claim 1 (Kendall dynamics) |
| Efficiency table (params, GFLOPs, FPS) | Supports consumer GPU claim |

## Key Differences from AHFE Framing

| Aspect | AHFE Version | AAIML Version |
|--------|-------------|---------------|
| Primary contribution | Ethics framework + pilot | **Training pathology analysis** |
| Secondary contribution | System deployment | **Multi-task architecture** |
| Tertiary contribution | Worker privacy | **Consumer GPU efficiency** |
| Blockchain role | Worker compensation | **Hybrid AI system** |
| Ethics content | 2+ pages | 0-1 paragraph |
| Failure mode analysis | 0.5 page | **2 pages (core section)** |
| Architecture detail | Moderate | **High** |
| Ablation experiments | Standard | **Extended** |

## Questions for Opus

1. **Does the AAIML-focused abstract above correctly balance the 3 training pathology
   findings with the system contribution?** Since AAIML is an ML/AI venue, the failure
   analysis is the novelty. But we don't want to hide the system either. Is the split
   (2 pages pathologies + 2.5 pages experiments) appropriate?

2. **Should the paper title include "Training Pathologies"?** Options:
   - "POPW: Multi-Task Assembly Verification on Consumer GPUs"
   - "Multi-Task Training Pathologies in Assembly Verification: A Case Study"
   - "Learning to Verify Assembly on Consumer GPUs: System and Training Insights"

3. **The ethics + blockchain content:** Opus originally said the ethics+blockchain
   made the AHFE paper stand out. For AAIML (ML/AI venue), does this content add
   value or dilute the ML contribution? Should we keep it as 0.5 page of "Hybrid AI
   System" framing, or cut it entirely?

4. **Head pose normalization:** The paper draft (AAIML/11) flags that 8.71° may change
   after GT normalization. If it degrades to 12-15°, is that still competitive enough
   for the "head pose at SOTA" claim? Or should we drop the SOTA comparison?

5. **Probe misreading (Claim 3):** For an ML/AI venue, this is the most directly
   relevant finding. But is a 10-day optimization mistake publishable? Or do reviewers
   see this as "you should have known better"? How do we frame it as a community lesson
   rather than an individual error?

6. **AAIML's best paper criteria:** The strategy files target Best Paper. With the
   training pathology findings as the core contribution, do we have a realistic shot?
   The competition at AAIML is likely single-task benchmark papers. A multi-task system
   + failure analysis is unusual. Does that help or hurt?

7. **Paper length:** 8.5 pages with 4 figures. AAIML allows 6 pages free + up to 4 at
   $70/page. Should we aim for 6 tight pages (free) or pay for 8-10? Is the extra
   content worth $140-$280?
