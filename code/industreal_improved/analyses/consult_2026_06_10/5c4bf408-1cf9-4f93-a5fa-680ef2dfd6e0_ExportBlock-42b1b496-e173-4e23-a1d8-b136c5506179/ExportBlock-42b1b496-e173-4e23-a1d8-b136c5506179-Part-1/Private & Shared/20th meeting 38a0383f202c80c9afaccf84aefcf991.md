# 20th meeting

![image.png](20th%20meeting/image.png)

---

![image.png](20th%20meeting/image%201.png)

## Paper 1, POPW: A Unified Multi-Task Architecture for Egocentric Assembly Understanding (Method)

**Target:** WACV 2027 R2 (Aug 28) - [https://wacv.thecvf.com/](https://wacv.thecvf.com/)

**Status:** Architecture complete, all results are placeholders pending training.

### Highlights

- First unified 5-task architecture for industrial assembly video (ASD detection + body pose + head pose + activity recognition + PSR) in a single forward pass, no prior work covers this specific task combination
- Shared ConvNeXt-Tiny (28M params) + FPN neck, 5 task heads, 53.42M trainable / 76.16M total params, replaces 3-4 separate models totaling ~75.4M
- Two-stage FiLM conditioning (PoseFiLM → HeadPoseFiLM) is the key architectural novelty. Body keypoints (51-dim) → γ,β with 1+tanh activation → modulate C5 features. Then 9-DoF head pose → γ,β with stop-gradient. Enables cross-task information flow without gradient interference
- Kendall homoscedastic uncertainty weighting for stable joint optimization of heterogeneous losses (focal + GIoU, Wing, MSE, CE, binary focal). Log-variances clamped [-4, 2] to prevent saturation. Compared against UW-SO (single temperature parameter) in discussion
- Staged training curriculum: rf1 (det only, 20ep) → rf2 (+pose+hp, 30ep) → rf3 (+act, 15ep). Prevents early-task competition
- Validated on two datasets: IKEA ASM (third-person, 371 videos, 33 actions) + IndustReal (egocentric, 74 actions, 24 ASD classes, 11-component PSR)
- Wing Loss parameters (ω=0.05, ε=0.005) justified relative to original paper (ω=10, ε=2), smaller ω due to reduced heatmap resolution 180×320 vs 256×256

---

## Paper 2, POPW-Assembly: An Empirical Study of Multi-Task Learning for Industrial Assembly Video (Empirical)

**Status:** All results are \popwres placeholders[https://eval.ai/web/challenges/challenge-page/1623/leaderboard/3910](https://eval.ai/web/challenges/challenge-page/1623/leaderboard/3910)

### Highlights

- Comprehensive benchmark comparing single multi-task model against 4 dedicated baselines (YOLOv8m, MViTv2, STORM-PSR, Mask R-CNN) on 2 datasets
- 4 research questions: (1) accuracy gap vs dedicated models, (2) which design choices matter, (3) when does MTL help vs hurt, (4) how do Kendall weights evolve
- 5 ablations isolating each design choice:
    1. **Backbone**: ResNet-50 vs ConvNeXt-Tiny
    2. **Task heads**: task dropout (det only → +HP → +Act → all)
    3. **FiLM**: none vs PoseFiLM only vs HeadPoseFiLM only vs both
    4. **MTL weighting**: equal vs Kendall no staging vs Kendall+staged (ours)
    5. **Temporal**: spatial only → +feature bank → +TCN → +2×ViT → +VideoMAE
- Per-component PSR analysis: prevalence from 0.95 (base plate) to 0.28 (wheels), F1 correlates with prevalence
- Failure mode taxonomy: (1) heavy occlusion → detection drops, FiLM gets zero confidence, (2) rare actions (<15 examples), (3) PSR temporal lag across action boundaries
- Efficiency: 53.3 replaces ~75.4M across 3 separate models, single forward pass

"we establish the evaluation protocol, baselines, and leaderboard for assembly video MTL.”

[https://cocodataset.org/#detection-leaderboard](https://cocodataset.org/#detection-leaderboard)

[https://codalab.lisn.upsaclay.fr/competitions/5256#results](https://codalab.lisn.upsaclay.fr/competitions/5256#results)

[https://eval.ai/web/challenges/challenge-page/1623/leaderboard/3910](https://eval.ai/web/challenges/challenge-page/1623/leaderboard/3910)

[https://ryenhails.github.io/IKEA-Bench/](https://ryenhails.github.io/IKEA-Bench/)

---

## Paper 3, POPW-Sys: From CCTV to Wallet, A Production-Grade Multi-Task Assembly Verification System with Blockchain Micropayments (System)

**Status:** System design complete, all placeholders. Blockchain integration at protocol level.

### Highlights

- First end-to-end system connecting CCTV-based multi-task assembly verification to Solana blockchain micropayments, fills genuine gap (no existing academic paper combines CV verification + x402 payment)
- 4-layer architecture: Vision (\popw{} model) → Verification (rule-based state machine) → Payment (Solana x402 smart contract) → Monitoring (web dashboard)
- **WorkerNet**: custom CCTV dataset for beverage bottle assembly, 5 detection classes (person, bottle, barcode, cap_close, cap_open), 3 activities (checking, rotating, storing), 17 keypoints, 10 yen/bottle payment structure
- **x402 race condition mitigation** (new systems contribution): temporal deduplication (cooldown window), state machine grounding (forward-only transitions)
- Context: Konnex PoPW raised $15M for proof-of-physical-work, directly validates premise. iFactory reports 374% ROI through automated inspection. Renault Cleon factory reduced defect rates 30% with AI monitoring

- How to handle multi-camera overlap (two cameras see same worker, double payment)?
- How to deal with network partitions (Solana RPC goes down mid-shift)?
- How to handle false negatives (system misses a task, worker loses money

---

## Cross-Paper Coordination

### Citation Flow

- **Paper 1** → foundational architecture
- **Paper 2** → cites Paper 1 as the method reference. Must have 70% new content beyond Paper 1 (protocol, ablations, dynamics, per-class analysis). Primary results table belongs in Paper 1; all hyperparameter tuning, detailed experiments, extended analysis go in Paper 2
- **Paper 3** → cites Paper 1 for architecture. Independent systems contribution with blockchain integration

### Shared Elements

- **Architecture diagram**: Paper 1 primary, reused in Paper 3. Paper 2 needs a more comprehensive version
- **Dataset config table**: Paper 2 primary (detailed), summarized in Paper 1
- **Main results table**: Headline numbers in Paper 1 (compact conference format). Full breakdown with ablations in Paper 2

### Sequencing

1. **Paper 1 first**, most material already exists, architecture sections complete
2. **Paper 3 second**, dataset collection + blockchain integration are independent workstreams
3. **Paper 2 last**, longest, requires all training numbers stable. Joins results from Paper 1 training + Paper 3 deployment

### Thesis Structure

The 3 papers together form: "we propose a method (Paper 1), evaluate it rigorously (Paper 2), and demonstrate real-world feasibility with blockchain integration (Paper 3)"

### Risk Management

- Papers 1 and 2 overlap on main results → resolved by: primary results in Paper 1 (compact), full analysis in Paper 2 (70% new content). Journal version (Paper 2 if submitted) should be self-contained but cite Paper 1 for architecture details
- Experiment 3 (Paper 3 blockchain) is a completely different track, no overlap risk
- Training not yet complete → all papers use \popwres placeholders. Paper outlines and contribution narratives are complete and ready for number insertion