# 136 — Activity: 50 Deep Questions for Opus

**Date:** 2026-07-06
**Purpose:** 50 deep questions on the Activity head, integrating all current progress with file location references. Designed to fuel the activity architecture decision (TCN+ViT vs MViTv2-S vs cut), expose every weakness a reviewer could attack, and direct the temporal probe fix.
**References:** Questions cite specific file paths so Opus can verify every assertion without hunting.

---

## Section 0. Evidence Inventory

### Files referenced (all paths absolute)

| File | Content |
|------|---------|
| `.../src/runs/rf_stages/checkpoints/SOTA_STATUS.md` | Consolidated SOTA table; activity linear probe 0.2169; clip-level 0.028; T3 match 0.6223 |
| `.../src/evaluation/activity_linear_probe.py` | Fixed linear probe (pre-extract features, filter -1 labels, gradient clipping) |
| `.../src/evaluation/activity_linear_probe_fixed.py` | Original buggy probe (NaN from ignore_index=-1 with all -1 batches) |
| `.../src/evaluation/eval_activity_clip.py` | Clip-level eval with stride=8 majority vote; produces activity_clip.json |
| `.../src/evaluation/eval_activity_seq.py` | Sequence-mode eval (16-frame clips, proper TCN+ViT temporal context) |
| `.../src/evaluation/activity_temporal_probe.py` | Temporal probe script — crashed with "Built 0 clips" error (see /tmp/temporal_probe.log) |
| `.../src/runs/rf_stages/checkpoints/activity_confusion_matrix.md` | Confusion matrix: 350 verb-antonym errors (1.3%), but class-imbalance collapse dominates |
| `.../src/runs/rf_stages/checkpoints/activity_confusion_matrix.png` | Confusion matrix figure |
| `.../src/runs/rf_stages/checkpoints/activity_take_put_confusion.png` | Focused take/put confusion |
| `.../src/runs/rf_stages/checkpoints/activity_clip_ep18/activity_clip.json` | Clip-level eval: 0.028 top-1 (4436 clips) — 37/66 per-class accuracies are exactly 0.0 |
| `.../src/runs/rf_stages/checkpoints/t3_full_eval.json` | T3 protocol verification: 0.6223 (matches WACV 0.622) |
| `.../src/runs/rf_stages/checkpoints/t3_mecanno_eval.json` | Meccanno eval on same protocol: 0.18 / 0.04 — catastrophic drop |
| `.../analyses/consult_2026_06_10/AAIML/133_OPUS_COMPLETE_ANSWERS.md` | Opus verdicts on ACT-1 through ACT-7 (§3); cross-document contradictions (§0) |
| `/tmp/temporal_probe.log` | In-flight temporal probe log — crashed at ClipDataset._build_index (0 clips built) |

### Key numbers at a glance

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Per-frame MLP top-1 | 0.0236 | Broken — class-imbalance collapse |
| 16-frame clip majority top-1 | 0.028 | Slightly above per-frame; no temporal reasoning |
| Linear probe (frozen ConvNeXt GAP C5) | 0.2169 | Weak signal at frame level |
| Majority-class baseline | 0.2217 | Always-predict-take_short_brace |
| ACT-1 gate (probe > 0.10 → TCN+ViT) | 0.2169 > 0.10 → PASS | TCN+ViT justified per gate |
| Temporal probe result | CRASHED (0 clips) | Bug in ClipDataset._build_index — see §A-1 |
| T3 protocol verification | 0.6223 | Matches WACV 0.622 |
| MViTv2-S SOTA | 0.622 | Full video architecture |

---

## Section 1. Per-Frame MLP Ceiling (10 questions)

### ACT-MLP-1: Is 0.0236 statistically distinguishable from random (1/69 = 0.0145)?

Why it matters: If 0.0236 is within noise of random, the per-frame MLP learned nothing — not even the majority class prior. The 0.028 clip-level number becomes equally meaningless. A binomial significance test on 28,665 labeled frames would give a p-value. The paper's honesty hinges on reporting whether 0.0236 is above chance or not.

Evidence: `activity_clip_ep18/activity_clip.json` — 0.028 over 4436 clips. Per-frame accuracy from `activity_confusion_matrix.md`: 0.0236 on 28,665 frames. Majority class is take_short_brace (0.2217 baseline from `activity_linear_probe.py` computation).

Missing: Binomial test p-value for 0.0236 vs 1/69 at n=28,665. Expected: p < 0.001 (n is large enough that even 0.0236 may be significant). But what matters is whether it's practically distinguishable.

### ACT-MLP-2: Why does the MLP collapse to take_short_brace specifically?

Why it matters: `activity_confusion_matrix.md` shows 8 of the top-10 confused pairs predict into take_short_brace. If take_short_brace dominates the training set (like "background" class in detection), the MLP learned prevalence, not action. Understanding this collapse mechanism dictates the fix.

Evidence: Confusion matrix §1 "Top-20 Confused Pairs" — check_instruction → take_short_brace (67.0%), take_partial_model → take_short_brace (39.2%), fit_long_brace → take_short_brace (48.5%). The collapse is systematic.

Missing: Class frequency distribution for all 69 classes in both train and val splits. The dataset AR_labels.csv should have this. Take_short_brace's proportion in val (from majority baseline: 0.2217) suggests it's 22%+ of labeled val frames.

### ACT-MLP-3: Does softmax temperature scaling salvage the per-frame MLP?

Why it matters: The MLP outputs are logits from `Linear(768, 69)`. If logits are poorly calibrated (too sharp or too flat), temperature scaling (learned on a held-out subset) could improve accuracy even without retraining. This is a trivial experiment (2 minutes on cached logits) that could recover 1-5% top-1 — and if it doesn't, confirms the MLP is fundamentally underpowered.

Evidence: `activity_linear_probe_fixed.py` trains `LinearProbeHead` with standard CrossEntropyLoss (no temperature). The probe's training accuracy (0.6267) suggests logits can separate train classes — val collapse may be a calibration issue on unseen distributions.

Missing: Temperature-scaled evaluation on cached val features. The cached features at `activity_linear_probe.json` could be reused with `T=2.0, 5.0, 10.0`.

### ACT-MLP-4: What is the MLP's top-5 and top-10 accuracy?

Why it matters: Top-1 accuracy for 69-way classification is an extremely harsh metric. If top-5 is 0.30-0.40, the network is confused but not random — it identifies the right *object* but wrong *verb*. This would strengthen the verb-antonym argument and help decide whether temporal aggregation is likely to help.

Evidence: `eval_activity_clip.py` and `eval_activity_seq.py` compute only argmax accuracy. No top-k metrics exist in any eval script.

Missing: Top-5 and top-10 accuracy from the cached predictions (checkpoint_35000frames.pkl). A 10-line analysis script.

### ACT-MLP-5: Does the MLP differentiate objects even if it can't differentiate actions?

Why it matters: If the MLP's 768→69 projection splits the data by object class but not verb (e.g., correctly predicts `take_short_brace` when object is `short_brace`, regardless of verb), then the backbone encodes object identity but not action dynamics. This would mean temporal modeling is *required* for action, which is the key claim that justifies TCN+ViT.

Evidence: Confusion matrix shows same-object confusions dominate (take_X → put_X, plug_X → pull_X). `activity_confusion_matrix.md` §2: "Among same-object confusions, verb-antonym pairs account for 20.4%." The remaining 79.6% of same-object errors are probably different objects predicted as the same object.

Missing: Hierarchical confusion breakdown. For each of the ~15 objects in the dataset: what fraction of predictions get the object right but verb wrong? This directly measures "backbone encodes objects" vs "backbone encodes nothing."

### ACT-MLP-6: Does class-balanced training change the MLP's behavior?

Why it matters: The training set is heavily imbalanced (take_short_brace dominating). The linear probe used `train_dataset.get_sampler()` (balanced sampler) — yet val accuracy is still 0.2169 ≈ majority baseline. If class-balanced training doesn't help, the issue is **feature quality**, not sampling.

Evidence: `activity_linear_probe_fixed.py` line 290: `train_sampler = train_dataset.get_sampler()` followed by `DataLoader(..., sampler=train_sampler)`. Despite balanced sampling, val accuracy is exactly at majority baseline.

Missing: What would majority-baseline-balanced accuracy be? (Weighted version where each class gets equal weight.) If balanced accuracy is also ~0.02, the model is truly random at class level.

### ACT-MLP-7: Is 0.0236 computed correctly? Could the metric be bugged?

Why it matters: The confusion matrix at `activity_confusion_matrix.md` says 0.0236 on 28,665 labeled frames. `activity_clip_ep18/activity_clip.json` says 0.028 on 4436 clips. The linear probe gave 0.2169 on 31,217 val samples. These three numbers (0.0236, 0.028, 0.2169) come from different scripts, different pipelines, and different label filtering. Are they comparable?

Evidence: The confusion matrix used `checkpoint_35000frames.pkl` (35k frames, 28,665 labeled). The clip eval used `best.pth` and the full model with per-frame MLP head. The linear probe used a completely separate pipeline (frozen backbone + linear layer). These are three different models/heads.

Missing: A single unified evaluation script that computes all three metrics from the same predictions. Without it, contradictions between 0.0236 and 0.2169 are apples-to-oranges.

### ACT-MLP-8: Does the per-frame MLP head have sufficient capacity?

Why it matters: The head is `Linear(768, 69)` — a single matrix multiply. This is 768 × 69 = 52,992 parameters for 69-way classification. The feature space (768-dim GAP-pooled C5) may not be linearly separable for fine-grained actions. A 2-layer MLP (768 → 256 → 69) would test whether non-linearity helps.

Evidence: `activity_linear_probe_fixed.py` uses `LinearProbeHead(nn.Module)` with one Linear layer. The temporal probe script (`activity_temporal_probe.py`) also uses a single Linear layer on pooled features — same capacity limit.

Missing: Two-layer probe result. Could be trained on cached features in 30 seconds.

### ACT-MLP-9: Is the 0.0236 accuracy an artifact of -1 sentinel labels?

Why it matters: The confusion matrix filtered -1 labels (valid frames only). But `activity_confusion_matrix.md` reports 28,665 labeled frames out of 35,000 total — meaning ~18% of frames have -1 sentinel labels. If -1 labels cluster at action boundaries, the eval is only on "easy" interior frames, and true per-frame accuracy may be even lower.

Evidence: Linear probe code filters -1 labels: `valid = labels >= 0` then `train_features[valid]` and `val_features[valid]`. The activity_clip eval has a different filtering path: `valid_l = clip_l[clip_l >= 0]` then majority vote. The confusion matrix uses checkpoint_35000frames.pkl.

Missing: Analysis of where -1 labels occur. Are they at action boundaries (most likely) or random? If boundaries: the 0.0236 number already overestimates performance because it excludes the hardest frames.

### ACT-MLP-10: Could per-frame accuracy ever exceed 0.30 with a per-frame architecture?

Why it matters: If the theoretical ceiling of per-frame ConvNeXt-Tiny on this dataset is < 0.30 (as the linear probe 0.2169 ≈ baseline suggests), then temporal architecture is non-optional. But if per-frame could reach 0.40-0.50 with better heads (2-layer MLP, attention pooling), a different fix is possible.

Evidence: Linear probe: 0.2169 (frozen features, linear head). Per-frame MLP: 0.0236 (trained end-to-end in multi-task setting). The MLP *should* outperform the probe (backbone can adapt), but it's 10× worse — suggesting multi-task interference or training dynamics kill it.

Missing: Per-frame MLP trained in *isolation* (single-task activity only). This would reveal whether multi-task interference is damaging activity, or whether the MLP head is inherently limited.

---

## Section 2. Linear Probe — Does the Backbone Have Signal? (10 questions)

### ACT-LP-1: Why is 0.2169 almost exactly at the majority baseline (0.2217)?

Why it matters: `SOTA_STATUS.md` says 0.2169 "approximately at the majority-class baseline." The difference is 0.0048 — about 150 frames out of 31,217. A 95% confidence interval for the baseline would be ±0.0046 (sqrt(p(1-p)/n) × 1.96). This means 0.2169 and 0.2217 are **statistically indistinguishable**. The probe may have learned literally nothing.

Evidence: `activity_linear_probe_fixed.py` computes `majority_baseline, majority_class = compute_majority_class_baseline(...)` and compares. The best_val_top1 is 0.2169, majority is 0.2217. The verdict is "BACKBONE HAS SIGNAL" because 0.2169 > 0.05 threshold.

Missing: Confidence interval for the probe accuracy. Also: paired comparison (does the probe's per-class accuracy differ from majority on a per-class basis?).

### ACT-LP-2: Is the 0.2169 coming from the probe learning the class prior, not from visual signal?

Why it matters: If the linear probe just learns class frequencies (predict the most common classes more often), its accuracy could match the majority baseline trivially. The gate decision (probe > 0.10 → TCN+ViT justified) only tests whether accuracy exceeds an arbitrary threshold, not whether the signal is visual.

Evidence: Linear probe trains on GAP-pooled C5 features, no image augmentation. Training accuracy hits 0.6267 (epoch 4), suggesting features do separate classes in the training distribution. But val accuracy is baseline.

Missing: Permutation test. Shuffle labels, re-run probe. If shuffled-label accuracy is also ~0.22, the probe is just matching the prior. If shuffled accuracy drops to 0.05, there is genuine visual signal.

### ACT-LP-3: The probe's train accuracy (0.6267) vs val (0.2169) is a 3× gap. Is overfitting fixable?

Why it matters: The probe has 52,992 parameters and trains on ~26,000 samples — a 2:1 parameter-to-sample ratio. Massive overfitting is expected. But if overfitting is driven by spurious correlations (background, lighting) rather than genuine label noise, stronger regularization (dropout, weight decay ×10, early stopping) could bridge the gap.

Evidence: `activity_linear_probe_fixed.py`: weight_decay=1e-4, no dropout, 5 epochs. Training accuracy peaks at 0.6267 by epoch 4. Val accuracy flat at ~0.22 throughout (never overfits to val, just never generalizes).

Missing: Probe with L2 ×10, dropout 0.5, and early stopping. Also: does the train-val gap exist per-class or is it driven by a few highly-overfit classes?

### ACT-LP-4: Would a k-NN probe on backbone features confirm or contradict the linear probe?

Why it matters: k-NN (nearest neighbor) probes are invariance-based: if the backbone maps same-action frames to nearby points in feature space, k-NN accuracy should be high. k-NN doesn't overfit (no learned parameters) and directly measures feature space quality. If k-NN accuracy is also ~0.22, features truly have no signal. If k-NN is higher, the linear probe's linearity constraint is the bottleneck.

Evidence: Cached features exist at `.../activity_linear_probe.json` training path. k-NN (k=5, cosine distance) on these features is a 10-line script.

Missing: k-NN probe accuracy. Expected: if features don't cluster by action, k-NN ≈ 0.0145 (random). If features are linearly separable but not clustered, k-NN < 0.10.

### ACT-LP-5: Is C5 the right feature level for action recognition?

Why it matters: ConvNeXt-Tiny C5 is 7×7 spatial resolution (after 32× downsampling). GAP pooling removes all spatial structure. For fine-grained actions (hand manipulating a screw), spatial detail from C3 (56×56) or C4 (28×28) may be essential. The probe tested only C5 GAP.

Evidence: `extract_backbone_features` in `activity_linear_probe_fixed.py`: `F.adaptive_avg_pool2d(c5, 1).flatten(1)`. Only C5 is used. C3 (384 dim) and C4 (768 dim) are ignored.

Missing: C4 GAP probe, C3 GAP probe, and multi-scale (C3+C4+C5 concatenated) probe. These would reveal which feature level carries action signal.

### ACT-LP-6: Does the feature cache have all 18% of -1 labels filtered, biasing the val set?

Why it matters: The original (buggy) `activity_linear_probe_fixed.py` excluded -1 labels at 15% of val batches having ALL invalid labels. The fixed version filters per-sample (not per-batch). But filtered val features (31,217 samples from 38,036) are a **non-random subset** if -1 labels cluster by recording or by action boundary.

Evidence: `extract_features_and_labels` filters `valid = labels >= 0` and keeps only valid features. The proportion of -1 labels in the output is 0% — the filtered set is all action-labeled frames.

Missing: Recording-level validation accuracy. If the excluded 18% are concentrated in specific recordings, the 0.2169 may not generalize to those recordings.

### ACT-LP-7: Could feature normalization (L2, z-score) improve the linear probe?

Why it matters: `torch.nan_to_num` is applied as a safety net, but features are not normalized. If C5 feature magnitudes differ per class or per image (e.g., high-activation frames dominate linear classifier), normalization could help. Standard practice: L2-normalize features before linear classification.

Evidence: `extract_features_and_labels` line 214: `features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)`. No L2 normalization. The Linear layer sees raw features.

Missing: Probe with L2-normalized features. A controlled comparison (normalized vs raw) on the same cached features.

### ACT-LP-8: What is the per-class accuracy breakdown of the linear probe?

Why it matters: The overall 0.2169 could come from 3-4 well-separated classes (maybe background classes never in the data) while all others are at 0.0. This would radically change the interpretation from "weak backbone signal" to "the signal is entirely from trivial classes."

Evidence: `activity_linear_probe_fixed.py` saves only aggregate accuracy. The per-class accuracy requires debugging the output log (standard logger, no per-class breakdown).

Missing: Per-class accuracy from the probe. The confusion-matrix approach (used for the MLP at `activity_confusion_matrix.md`) should also be run for the probe's predictions.

### ACT-LP-9: If the probe passes the gate at 0.2169, does it also pass a stricter gate like "0.15 above baseline"?

Why it matters: The ACT-1 gate (0.10) was set before 0.2169 was known. With 0.2169 ≈ baseline 0.2217, the gate proved too permissive. Opus (133 §3 ACT-1) says "probe first, then decide on TCN+ViT." But if the probe is indistinguishable from baseline, the decision is already made: backbone has zero usable frame-level signal. The TCN+ViT justification rests entirely on the hope that temporal aggregation amplifies signal that is invisible at frame level.

Evidence: Opus 133 ACT-1 verdict: "Architectural ceiling claim requires the linear-probe result; probe first (1 day), then decide on TCN+ViT/MViTv2-S." The probe is done, and the result is ambiguous.

Missing: Explicit statement: given probe = baseline, does TCN+ViT remain justified? The temporal probe was designed to answer this — but it crashed.

### ACT-LP-10: Can the linear probe be salvaged with C5 spatial features (no GAP)?

Why it matters: GAP pooling discards all spatial information. ConvNeXt C5 is 7×7 spatial resolution. A linear probe on 7×7×768 = 37,632 features (with heavy regularization) could determine whether spatial layout matters for action. If spatial features improve accuracy, the TCN+ViT should use patch-level features, not GAP-pooled.

Evidence: Current approach uses GAP. No spatial probe exists.

Missing: Conv1×1 probe on C5 (7×7×768 → 69). A convolutional classifier with global average pooling at the end (not feature-level GAP) — this is literally the standard ImageNet classifier head applied to frozen features.

---

## Section 3. Confusion Matrix — Verb-Antonym vs Class Collapse (10 questions)

### ACT-CM-1: What is the full class distribution of the 69 activity classes?

Why it matters: `activity_confusion_matrix.md` shows 0.0236 accuracy, but doesn't show class frequencies. If the entire dataset is 50% take_short_brace and 50% everything else (200-way heavy tail), the class collapse pattern is caused by data imbalance more than model failure. Understanding the data distribution is prerequisite to any fix.

Evidence: SOTA_STATUS.md mentions majority class take_short_brace at 0.2217 baseline. Linear probe computes majority class as some class ID (not reported in confusion matrix).

Missing: Full class frequency table (train + val). Available from AR_labels.csv. Should be added to `activity_confusion_matrix.md`.

### ACT-CM-2: Why does take_partial_model → take_short_brace dominate at 39.2% of true class?

Why it matters: `activity_confusion_matrix.md` §1: rank 1 confusion pair is take_partial_model → take_short_brace (2436 frames, 39.2% of all take_partial_model frames). If 39% of a true class is misclassified as a single wrong class, the model found a specific visual shortcut — probably object similarity (partial model and short brace may look similar in egocentric view).

Evidence: Confusion matrix top-20 pairs. take_partial_model has 2436+774+601+336+253 = 4400+ errors in the top-5 pairs alone (out of ~28,000 total errors).

Missing: Visual inspection of the two objects. Are partial_model and short_brace visually similar? If yes, confusion is driven by object appearance, not action temporality.

### ACT-CM-3: What would the confusion matrix look like if we group by verb only (not verb+object)?

Why it matters: The current 69-class taxonomy is verb+object (hybrid grouping). If we collapse to pure verb (7-10 verbs: take, put, plug, pull, fit, tighten, loosen, check, etc.), per-verb accuracy may be much higher. This would prove the backbone encodes verb information correctly even if the specific object-verb pairing is noisy.

Evidence: `act_remap_75_to_69.json` maps 75 raw classes to 69 hybrid groups. A pure-verb remap would map to ~10 groups. The confusion matrix data (checkpoint_35000frames.pkl) has raw class predictions before remapping.

Missing: Verb-level confusion matrix. High verb-accuracy + low verb+object-accuracy = model gets the action right but the object wrong (which is fixable with detection features).

### ACT-CM-4: Are verb-antonym confusions (350 frames, 1.3%) time-localized to action boundaries?

Why it matters: Opus (133 ACT-4) says verb-antonym errors are "temporally ambiguous by construction — a frame at the transition between 'taking a screw' and 'putting a screw' is genuinely ambiguous." This is a strong claim that should be verified with annotation timestamps. If verb-antonym errors are uniformly distributed through the recording (not at boundaries), the model has a genuine semantic confusion.

Evidence: `activity_confusion_matrix.md` lists 350 verb-antonym errors (1.3% of total errors). Claim: these occur at action boundaries. But the confusion matrix is computed per-frame without temporal context — it doesn't know which frames are at boundaries.

Missing: Temporal heatmap of verb-antonym errors. Plot error density vs. distance from nearest action transition. If errors peak at ±1-2 frames of transitions, Opus's claim is verified. If uniform, the model has a real semantic confusion.

### ACT-CM-5: What is the inter-annotator agreement on action labels?

Why it matters: If human annotators disagree at action boundaries (e.g., annotator A labels frame 100 as "taking pin_short" while annotator B says "putting pin_short"), the model's 0.0236 accuracy on single frames is measuring noise, not failure. Publication-quality activity datasets measure and report inter-annotator agreement. If ours doesn't have it, the per-frame protocol itself is questionable.

Evidence: The IndustReal dataset paper (WACV 2024) likely has annotation details. Our annotation pipeline is in `src/data/industreal_dataset.py` which loads AR_labels.csv.

Missing: Inter-annotator agreement statistics. If two annotators only agree ~70% at frame level, the ceiling for any model is ~0.70 — and 0.0236 is less bad than it looks.

### ACT-CM-6: What is the temporal density of action transitions?

Why it matters: If most frames are interior to actions (far from transitions), per-frame accuracy should be high because interior frames are unambiguous. If most frames are near transitions, per-frame accuracy is inherently limited by annotation noise. The ratio of boundary frames to interior frames sets the theoretical ceiling.

Evidence: `activity_confusion_matrix.md` provides aggregate statistics but no temporal density metrics.

Missing: Distribution of frames by distance to nearest transition. Histogram: 0 frames from transition (boundary), 1 frame, 2 frames, ..., 30+ frames (interior). If 30%+ of frames are within 3 frames of a transition, per-frame protocol is fundamentally noisy.

### ACT-CM-7: Would a transition-aware evaluation (ignore boundary ±k frames) fundamentally change accuracy?

Why it matters: If we exclude ±3 frames around each action transition (standard practice in some activity datasets like Breakfast, 50Salads), per-frame accuracy would rise because the remaining frames are unambiguously single-action. This would give an upper bound on how much temporal ambiguity hurts accuracy.

Evidence: Current evaluation includes all frames equally. No transition-aware filtering is implemented.

Missing: Accuracy as a function of distance from transition. Compute: for frames at distance d from nearest transition, what is top-1 accuracy? If accuracy at d=0 is 0.01 and at d=30 is 0.40, temporal ambiguity is the dominant error source.

### ACT-CM-8: How many classes in the 69-way taxonomy have zero or near-zero training examples?

Why it matters: Pathological classes (1-2 training examples) are impossible to learn. If 20+ of 69 classes have <10 training frames, any per-frame accuracy above random is impressive — and 0.0236 may be state-of-the-art for that data regime.

Evidence: The paper draft (C-3 from 133 §0): "Pathology 2 uses '46/74 classes <1%'." But current eval uses 69-class grouping, not 75 or 47.

Missing: Class frequency table for 69 groups in training set. Which classes have <10, <100, <1000 training frames?

### ACT-CM-9: Is the confusion matrix symmetric? (take_X → put_X vs put_X → take_X)

Why it matters: The confusion matrix at `activity_confusion_matrix.md` §2 lists "take_pin_short → put_pin_short (210 frames)" but doesn't report the reverse (put_pin_short → take_pin_short). Symmetry would suggest temporal adjacency (transitions go both ways). Asymmetry would suggest a model bias (e.g., always predicts "take" near transitions).

Evidence: The confusion matrix shows asymmetric counts: take_pin_short → put_pin_short = 210 frames. The reverse confusion may be lower or higher. Asymmetry is informative about the direction of temporal ambiguity.

Missing: Symmetry analysis. Same-object confusion pairs should be reported both ways.

### ACT-CM-10: What drives the remaining ~98.7% of errors that are NOT verb-antonym?

Why it matters: Verb-antonym errors (1.3%) are the most discussable error type, but they are negligible in magnitude. The other 98.7% of errors (class-imbalance collapse) is the real problem. Understanding the non-antonym errors is essential: are they also temporally ambiguous (different action on same object) or are they completely unrelated (model seeing something that isn't there)?

Evidence: `activity_confusion_matrix.md` top-20 pairs. Most pairs predict into take_short_brace regardless of true class. This is collapse, not confusion. But ranks 4, 5, 10, 12, 15, 17, 19 show non-take_short_brace predictions (plug_wheel, fit_wing_beam, etc.) — these are genuine confusions, not collapse.

Missing: Error decomposition: collapse (predict majority class) vs confusion (predict wrong minority class). If 90%+ of errors are collapse, the fix is class imbalance (re-weighting, sampling). If 50%+ are genuine confusion, the fix is better features/architecture.

---

## Section 4. TCN+ViT vs MViTv2-S — What Architecture? (10 questions)

### ACT-ARCH-1: The temporal probe crashed. What is the root cause and fix?

Why it matters: `/tmp/temporal_probe.log` shows "Built 0 clips of length 16" for both train and val sets. The `ClipDataset._build_index()` method in `activity_temporal_probe.py` silently fails to group frames into clips. The except clause (`except Exception: meta = {}`) swallows the error, making every frame appear as its own unique recording. The dataset then has no recording with ≥16 frames, producing 0 clips.

Evidence: `activity_temporal_probe.py` lines 68-89: `_build_index` iterates the base dataset, extracts metadata, forms clips. The except clause (line 78) catches all exceptions. Most likely cause: `self.base[i]` returns a dict structure different from what the code expects (maybe `images, targets = self.base[i]` unpacking fails in the except block).

Missing: Error trace inside the except clause. The actual exception (likely AttributeError or TypeError from metadata structure mismatch) is swallowed. Fix: remove the bare except and log the actual error.

### ACT-ARCH-2: Does temporal pooling (mean over 16 frames) amplify the linear probe's 0.2169?

Why it matters: The temporal probe tests mean-pool over 16-frame clips followed by a linear classifier. If this reaches >0.27 (the gate threshold), TCN+ViT is justified. If mean pooling doesn't help (stays at ~0.22), the signal is not temporally consistent over short windows — the model sees different things on adjacent frames, suggesting frame-rate noise or annotation jitter dominates.

Evidence: No temporal probe result exists yet (crashed). The per-frame probe is 0.2169. The 16-frame clip eval (per-frame MLP + majority vote) is 0.028 — which is the MLP's per-frame predictions aggregated, not pooled features.

Missing: The fixed temporal probe result. Priority: fix `_build_index` bug and rerun.

### ACT-ARCH-3: Does attention-based temporal pooling outperform mean pooling?

Why it matters: If mean pooling over 16 frames barely helps (+0.02-0.03), the question is whether a learned temporal aggregator (lightweight self-attention across T frames) could capture the non-linear temporal structure. A simple 2-head, 2-layer transformer on per-frame features (768-dim, 16 frames) would test this.

Evidence: The temporal probe only tests mean and max pooling (cheap, no learned params). Attention-based pooling is not implemented.

Missing: Attention-pool temporal probe. Implementation: per-frame features → transformer encoder → cls token → linear classifier. Adds 1-2M parameters, 30 min training.

### ACT-ARCH-4: What is the minimum TCN+ViT that demonstrates temporal reasoning?

Why it matters: TCN+ViT is described as heavy (config.py:960 says "needs a fresh run, 2-3 days"). But a minimal temporal head (Conv1D with kernel_size=3, 2-3 layers, 256 channels) on frozen ConvNeXt features could be trained in 2-4 hours. Before committing to the full 2-3 day run, we should confirm that ANY temporal aggregation helps at all.

Evidence: `activity_temporal_probe.py` is the minimal test (pooling only). The actual TCN+ViT head lives in the model architecture (`src/models/model.py`, likely gated by ACTIVITY_HEAD_SIMPLE=False). Training it end-to-end requires a multi-task training run.

Missing: TCN-only ablation (no ViT). A 3-layer Conv1D(768 → 256 → 69) on 16-frame sequences, trained with activity loss only (no multi-task). Tests the temporal hypothesis at minimal cost.

### ACT-ARCH-5: Can frozen ConvNeXt + temporal head match MViTv2-S?

Why it matters: MViTv2-S achieves 0.622 with a full video architecture (3D convolutions + factorized attention). Our approach (frozen 2D backbone + temporal head) is strictly weaker on temporal modeling. The question is not whether we match 0.622, but whether our approach can reach 0.30-0.40 — which would be a "competitive" result under the PW-3 rubric.

Evidence: T3 full eval (`t3_full_eval.json`): MViTv2-S achieves 0.6223 on our dataset. No frozen-backbone+TCN result exists.

Missing: Frozen ConvNeXt + TCN benchmark. This is the P1.4 priority from Opus 130/132.

### ACT-ARCH-6: How does the TCN+ViT head in config.py work?

Why it matters: The TCN+ViT architecture in config.py may have design choices that limit performance (number of layers, hidden dim, whether it uses causal masking, what frame rate it expects). Understanding the existing implementation before training saves wasted runs.

Evidence: config.py (lines ~955-985): `ACTIVITY_HEAD_SIMPLE=True` enables per-frame MLP; `ACTIVITY_HEAD_SIMPLE=False` enables TCN+ViT. The TCN+ViT is already implemented and tested in `eval_activity_seq.py`.

Missing: Architecture diagram/description. How many TCN layers? ViT depth and heads? Input feature dimension? Receptive field in frames?

### ACT-ARCH-7: What frame rate (fps) does the temporal model operate at?

Why it matters: If frames are 30 fps and the model processes 16-frame clips (0.53 seconds), the temporal window may be too short for assembly actions (which can take 2-10 seconds). A 0.53-second window may capture only a fraction of "take pin" → "place pin" → "release pin." Increasing clip length to 32 or 64 frames (1-2 seconds) would capture more temporal structure at the cost of GPU memory.

Evidence: `eval_activity_clip.py` uses clip_length=16, stride=8. `eval_activity_seq.py` uses seq_length=16. `activity_temporal_probe.py` uses clip_len=16. All experiments use 16-frame windows.

Missing: Ablation over clip lengths (8, 16, 32, 64) to determine optimal temporal receptive field. If 64-frame clips outperform 16-frame, actions are longer than 0.5 seconds.

### ACT-ARCH-8: Does the temporal head benefit from multi-task training or should it be trained in isolation?

Why it matters: If the temporal head is trained as part of the multi-task model, gradient conflicts from PSR/detection/pose heads may degrade activity performance. A single-task temporal ablation (activity only, freeze other heads) would reveal whether multi-task training helps or hurts activity.

Evidence: Current training script (`train.py`) is multi-task. The PSR head has dead gradients (PSR-3 from 133). These dead gradients still produce backprop through shared backbone, potentially interfering with activity.

Missing: Single-task activity training with temporal head. Expected: higher activity accuracy, lower other-head accuracy.

### ACT-ARCH-9: Can we use VideoMAE pretrained weights for the temporal backbone instead of ConvNeXt?

Why it matters: VideoMAE (video masked autoencoder) provides spatiotemporal pretrained features. The model config already has `use_videomae` flag in `POPWMultiTaskModel`. If VideoMAE pretrained weights are available, the temporal backbone may already encode action-relevant features better than ConvNeXt-Tiny.

Evidence: `src/models/model.py` `POPWMultiTaskModel(..., use_videomae=False)`. The flag exists but is unused.

Missing: VideoMAE backbone comparison. One forward pass on val set with VideoMAE features → linear probe. If >0.30, the temporal pretraining itself (not architecture) is the key.

### ACT-ARCH-10: Is there a simpler temporal baseline (TSN, TRN, TSM) we should benchmark before TCN+ViT?

Why it matters: TCN+ViT is complex. Simpler temporal models (TSN = temporal segment networks, TRN = temporal relational network, TSM = temporal shift module) could achieve 80% of the gain at 20% of the implementation cost. Benchmarking these first would prevent over-investment in a complex architecture that a simple baseline could beat.

Evidence: No temporal baseline except per-frame MLP and linear probe exist. TCN+ViT in config.py is the only implemented temporal option.

Missing: TSN baseline (segment consensus: 16 frames → 8 segments → segment consensus → classification). Could reuse the cached ConvNeXt features (just need segment-level aggregation).

---

## Section 5. SOTA Comparison — MViTv2-S and Protocol (10 questions)

### ACT-SOTA-1: What does T3's 0.6223 "match" actually mean?

Why it matters: `t3_full_eval.json` reports 0.6223 — matching WACV's published 0.622. Opus (133 SOTA-4) says this is "protocol verification only, methods section." But what exactly does the match prove? If WACV's MViTv2-S gets 0.622 on the same clips, does it mean: (a) our clip-sampling protocol matches WACV's (so our numbers are on the same data), or (b) our model achieves the same accuracy?

Evidence: `t3_full_eval.json`: model=MViTv2-S, total_clips=916, top1_69=0.6223. This is WACV's model evaluated on our clips. The match confirms our clip extraction pipeline reproduces WACV's data.

Missing: Explicit statement: "T3 match confirms clip-level protocol consistency, not model capability." The 0.6223 is a baseline floor, not an achievement.

### ACT-SOTA-2: Meccanno eval (t3_mecanno_eval.json) shows 0.18/0.04 — why is it 20-30× worse than full T3?

Why it matters: `t3_mecanno_eval.json` reports top1_75=0.18, top1_69=0.04 on 100 clips. This is catastrophically worse than the full T3 0.6223 on 916 clips. The 100 clips may be disproportionately hard, the Meccanno model may be a different checkpoint, or something is wrong with the evaluation.

Evidence: `t3_mecanno_eval.json`: model="WACV 2024 MViTv2-S (Meccanno pretrained, 75-class)", split="val", total=100, top1_75=0.18, top1_69=0.04. By_class entries show most classes at 0.0.

Missing: Why 100 clips? Are these the same 100 clips every time? Is this a canonical subset? The massive gap (0.04 vs 0.622) needs explanation before any SOTA claim.

### ACT-SOTA-3: Is the gap from 0.028 to 0.622 entirely architectural, or is data mismatch also a factor?

Why it matters: If the 0.622 SOTA uses different training data (more recordings, different annotations, additional pretraining data), then 0.622 is not our ceiling — it's the ceiling with data we don't have. Understanding the gap decomposition (architecture vs data vs protocol) determines whether TCN+ViT can realistically close it.

Evidence: MViTv2-S is pretrained on Kinetics-400 (video classification, 400 classes, 300K videos). Our ConvNeXt-Tiny is pretrained on ImageNet-1K (image classification, 1000 classes, 1.2M images). The pretraining gap is massive: video data vs image data, 400-way vs 1000-way action vs object.

Missing: How much of the gap is the Kinetics-400 pretraining? A "backbone contribution analysis" would freeze the MViTv2-S backbone and train a linear probe — if >0.50, pretraining is decisive.

### ACT-SOTA-4: Can a slowfast or I3D baseline interpolate the gap?

Why it matters: Before committing to TCN+ViT (unknown cost/benefit), simple video baselines (slowfast, I3D, X3D) should be evaluated. These models are well-understood, have known compute requirements, and would provide a realistic upper bound for what video architectures can achieve on this dataset.

Evidence: Activity-clip eval achieves 0.028 with per-frame MLP. No slowfast/I3D/X3D baseline exists.

Missing: Slowfast or I3D baseline (can use torchvision models with Kinetics-400 pretrained weights). If these achieve 0.30-0.50 on our 69-class task, TCN+ViT is viable. If they achieve the same 0.02-0.03, the problem is deeper than architecture.

### ACT-SOTA-5: Is the 69-class grouping compatible with WACV's 75-class evaluation?

Why it matters: Opus ACT-3 (133 §3): "T3's top1_75 = top1_69 = 0.6223 is cited as evidence the grouping doesn't inflate baselines." But `t3_full_eval.json` shows both 75-class and 69-class accuracy are 0.6223 — identical to 4 decimal places. This suggests the 6 merged classes (75→69) had zero clips in the evaluation set, making the remapping moot.

Evidence: `t3_full_eval.json`: `top1_75: 0.6223, top1_69: 0.6223`. If the 6 merged classes had clips, top1_69 would differ from top1_75 (because merging changes the label space). The identity implies the evaluation set doesn't include any of the 6 merged classes.

Missing: How many test clips belong to the 6 merged classes? If zero, the identity is a data artifact, not evidence that grouping is benign.

### ACT-SOTA-6: What is the per-clip agreement between our per-frame MLP majority vote and the MViTv2-S?

Why it matters: If our per-frame MLP (0.028) agrees with MViTv2-S (0.622) on 60%+ of random clips, the MLP is partially correct on most clips but wrong on the majority vote — perhaps it has temporal confusion but correct action class. If agreement is low (5-10%), the MLP is making completely different errors.

Evidence: No per-clip comparison between MLP and MViTv2-S predictions exists.

Missing: Per-clip prediction comparison. The cached clip predictions from `activity_clip_ep18/activity_clip.json` could be compared to MViTv2-S outputs from `t3_full_eval.json` if clip indices are aligned.

### ACT-SOTA-7: What fraction of WACV's validation clips have a single-labels majority > 0.5?

Why it matters: MViTv2-S achieves 0.622 top-1. This means 62.2% of clips have the majority label correctly predicted. But some clips may have near-unanimous labels (28/30 frames = same action) while others are ambiguous (9/16/16 split). The distribution of clip-label purity affects how much improvement is possible.

Evidence: WACV's clip construction is stride-based (overlapping windows). Our eval (16-frame, stride=8) also produces overlapping clips. Label purity per clip can be computed from the annotations.

Missing: Label entropy per clip. If 40% of clips already have near-random labels (entropy > 2.0 bits), the SOTA ceiling is < 0.60 even for a perfect model.

### ACT-SOTA-8: If TCN+ViT achieves 0.30, is that publishable as "competitive"?

Why it matters: Under Opus PW-3 rubric: **"competitive"** = within 10% relative under identical protocol. If SOTA is 0.622, 10% relative = 0.560 — impossible for TCN+ViT. Under **"first baseline"** = no published prior after documented search — our per-frame MLP already qualifies. Under **"measured cost"** = ratio against self-established ceiling — detection qualifies. So what happens at 0.30?

Evidence: PW-3 rubric from 133 §9: "competitive" requires within 10% relative. 0.622 × 0.9 = 0.560. Even a 100% relative improvement would only reach 0.056 (2× 0.028).

Missing: Explicit activity target under PW-3. If we can't reach "competitive," the framing must shift to "first baseline" or "cost measurement" for activity.

### ACT-SOTA-9: Is the T3 MViTv2-S model fine-tuned on our training data or used off-the-shelf?

Why it matters: If MViTv2-S was fine-tuned on our training data (from Meccanno pre-trained weights), the 0.622 is a fine-tuned result and represents the ceiling for our data. If it was used off-the-shelf (Kinetics-400 only, zero-shot on our data), 0.622 is a lower bound and the ceiling is higher.

Evidence: `t3_full_eval.json` doesn't specify whether the model was fine-tuned. The question in WACV's meccanno_eval.json (t3_mecanno_eval.json: 0.18/0.04) suggests fine-tuning happened — but for the full 916-clip eval.

Missing: Fine-tuning status of the T3 model. If off-the-shelf, MViTv2-S achieves 0.622 zero-shot on our data — which would be remarkable and raise questions about test set leakage.

### ACT-SOTA-10: Should we report activity at all, given the numbers?

Why it matters: Opus (133 ACT-2): "Per-frame MLP is defensible only as a floor baseline within the multi-task probe framing, paired with the latency argument." If neither TCN+ViT nor MViTv2-S can be trained before deadline, the paper reports 0.0236/0.028 with no temporal improvement. Is that better than cutting activity entirely and presenting a "3-task + probe" paper?

Evidence: ACT-1 through ACT-7 (133 §3) all converge on: probe first, then decide. Probe is done (0.2169). Temporal probe crashed. No temporal result yet.

Missing: Go/no-go decision for activity in the paper after the temporal probe result. If temporal probe < 0.10 (suggesting temporal pooling doesn't help), activity should be cut to a supplementary footnote.

---

## Section 6. Adversarial Review (7 questions)

### ACT-ADV-1: "You report 0.0236 per-frame accuracy. Why should anyone care about an accuracy that's statistically indistinguishable from random?"

Why it matters: This is the first question every reviewer will ask. The answer (temporal ambiguity, latency advantage, multi-task framing) must be crisp and evidence-backed. If it's not, the paper is desk-rejected.

Evidence: `activity_confusion_matrix.md` — verb-antonym confusion (1.3% of errors) provides temporal ambiguity evidence. SOTA_STATUS.md explains that per-frame labels are "temporally ambiguous by construction."

Missing: One-paragraph defense of why 0.0236 is informative despite being near-baseline.

### ACT-ADV-2: "If activity in your system is at chance level, how can you claim the multi-task model doesn't have negative interference?"

Why it matters: Opus (133 AC-2) addresses this: the linear probe result (0.2169) is the interference measurement — if the probe partially works but the MLP doesn't, multi-task interference killed activity. But if both are at baseline, there's no evidence of interference either — the backbone might just lack action features.

Evidence: Linear probe: 0.2169 (frozen backbone, single-task). MLP: 0.0236 (trained end-to-end, multi-task). The 10× gap between them (0.2169 vs 0.0236) is itself evidence of severe multi-task interference.

Missing: Dedicated interference measurement: compare multi-task MLP accuracy to single-task MLP accuracy. If single-task is also 0.02-0.03, the interference claim is unsupported.

### ACT-ADV-3: "You mention 'temporally ambiguous by construction.' Can you prove that with annotation data?"

Why it matters: The claim that per-frame labels are ambiguous at boundaries is central to defending 0.0236. But it needs evidence: inter-annotator agreement, temporal density analysis, or annotation timestamps showing boundary frames are contested.

Evidence: Activity confusion matrix provides circumstantial evidence (verb-antonym errors at boundaries) but not direct annotation evidence.

Missing: Inter-annotator study on boundary frames. Even 100 frames with 3 annotators would support or refute the ambiguity claim.

### ACT-ADV-4: "Your SOTA comparison for activity is misleading. You're comparing per-frame MLP (0.028) to MViTv2-S (0.622). This is comparing a tricycle to a Ferrari."

Why it matters: Opus (133 SOTA-6): "Clean break, no MViTv2-S row anywhere in results." The comparison must be avoided entirely, or presented with such strong disclaimers that it doesn't mislead.

Evidence: `activity_clip_ep18/activity_clip.json` (0.028) vs `t3_full_eval.json` (0.622). A 22× gap.

Missing: Protocol-comparison paragraph for the paper. Something like: "Per-frame and clip-level protocols measure fundamentally different quantities. The WACV clip-level baseline (0.622) is provided for dataset protocol verification only."

### ACT-ADV-5: "How much of the SOTA gap (0.028 → 0.622) is due to Kinetics-400 pretraining? If I initialize your ConvNeXt with video-pretrained weights, do you get 0.30?"

Why it matters: The reviewer is asking for the ablation between data-pretraining (ImageNet vs Kinetics) and architecture (2D vs 3D). If a ConvNeXt initialized with VideoMAE weights achieves 0.30, pretraining > architecture. If not, architecture > pretraining.

Evidence: `use_videomae` flag exists in the model but hasn't been tested.

Missing: Video-pretrained backbone comparison. Could reuse the activity_linear_probe infrastructure with a video-pretrained ConvNeXt.

### ACT-ADV-6: "Your temporal probe crashed. What does that say about the maturity of your temporal reasoning approach?"

Why it matters: The temporal probe bug (`/tmp/temporal_probe.log`: "Built 0 clips") is embarrassing if the reviewer finds it — but more importantly, it means zero temporal evidence exists. The paper cannot claim "temporal modeling improves activity" without at least one working temporal experiment.

Evidence: The crash is caused by `ClipDataset._build_index` silently failing (bare `except Exception: meta = {}`). The fix is straightforward: log the actual error and handle the data structure correctly.

Missing: A patched temporal probe result before the paper freeze.

### ACT-ADV-7: "If temporal aggregation helps, why did your 16-frame clip majority vote (0.028) barely improve over per-frame (0.0236)?"

Why it matters: The clip-level eval (0.028) should theoretically be higher than per-frame (0.0236) because majority voting smooths random noise. It improved by only 0.0044 — suggesting predictions are not random (they consistently pick the same wrong class), which is consistent with class-imbalance collapse rather than temporal noise.

Evidence: `activity_clip_ep18/activity_clip.json` clip_top1=0.028. Per-frame accuracy from confusion matrix: 0.0236. Improvement: 0.0044.

Missing: What is the expected improvement from majority voting on random predictions? Expected: majority vote on 16 random draws from uniform distribution should be ~1/69 ≈ 0.0145. Majority vote on 16 draws from majority-class distribution (take_short_brace at 0.2217) should be ~0.2217. If majority vote matches 0.028 (slightly above per-frame 0.0236 but well below 0.2217), the model is making consistent but wrong predictions, not random ones.

---

## Section 7. Open Decisions for Opus

### Decision 1: Activity in or out of the paper?

If the temporal probe (after fixing the `_build_index` crash) shows mean-pooling over 16 frames achieves < 0.10 (no temporal benefit): cut activity to a supplementary note. The paper becomes "3-task + probe," which is honest.

If temporal probe achieves > 0.27 (temporal benefit confirmed): proceed to TCN+ViT (P1.4, 2-3 day training run).

If temporal probe is between 0.10 and 0.27 (ambiguous): gate on whether there is time for TCN+ViT before the paper freeze. If yes (≥ 2 weeks), proceed. If no, relegate activity to §5.4.

### Decision 2: Temporal probe bug fix priority

The `ClipDataset._build_index` bug blocks the central experiment. Priority: fix and rerun today. Expected effort: 30 minutes (remove bare except, handle metadata structure correctly, add logging). The temporal probe should be running overnight on RTX 3060.

### Decision 3: SOTA comparison strategy — Opus PW-3 rubric application

Options:
1. **First baseline** (lowest risk): claim first per-frame action baseline on IndustReal. Minimum bar: documented literature search showing no prior per-frame result.
2. **Measured cost** (medium risk): compute ratio of MLP (0.0236) to SOTA ceiling (0.622) = 3.8%. With the linear probe (0.2169), cost = 65% of ceiling. This pairs with detection's cost story.
3. **Competitive** (high risk, unattainable): require > 0.560 under same protocol. Not possible with current architecture.

### Decision 4: Multi-task interference — extract a standalone finding

The 10× gap between linear probe (0.2169) and multi-task MLP (0.0236) is the strongest multi-task interference signal in the entire system. Detection (0.995 separate, 0.358 multi-task) shows a similar pattern. This observation could unify the interference story across heads. Opus should confirm whether this framing is worth developing in §5.4.

### Decision 5: What does "BACKBONE HAS SIGNAL" actually mean?

The current verdict (from 0.2169 > 0.05 gate) is misleading: the signal is at chance level (indistinguishable from majority baseline). Opus should either: (a) tighten the gate to "probe > baseline + 0.05" (explicit margin over majority), or (b) add a second condition: "probe accuracy statistically > majority baseline at p < 0.05."

### Decision 6: Frame-rate argument — how many fps is needed?

If 16-frame clips at 30 fps capture 0.53 seconds, and assembly actions last 2-10 seconds, the temporal window is too short. Should the TCN+ViT architecture support variable-length clips (16, 32, 64 frames) to determine optimal temporal receptive field? This would require a single training run with multiple eval checkpoints.

### Decision 7: Activity confusion matrix figure for the paper

The confusion matrix at `activity_confusion_matrix.png` and focused take/put analysis at `activity_take_put_confusion.png` are the visual evidence for the temporal ambiguity claim. Opus should confirm: (a) whether these figures belong in §5.4, and (b) whether a per-class accuracy bar chart (from the linear probe) adds value.

---

## Appendix A. Temporal Probe Bug Report

### A-1. Crash details

**File:** `/media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/src/evaluation/activity_temporal_probe.py`  
**Log:** `/tmp/temporal_probe.log`  
**Error:** `ValueError: num_samples should be a positive integer value, but got num_samples=0`  
**Root cause:** `ClipDataset._build_index()` produced 0 clips because frame metadata extraction silently failed.

### A-2. Bug mechanism

The `_build_index` method in `activity_temporal_probe.py` (lines 68-89):
```python
def _build_index(self):
    from collections import defaultdict
    recs = defaultdict(list)
    for i in range(len(self.base)):
        try:
            meta = self.base[i][1].get('metadata', {}) if isinstance(self.base[i], tuple) else {}
        except Exception:
            meta = {}
        rec_id = meta.get('recording_id', f'unknown_{i}')
        frame_num = meta.get('frame_num', i)
        recs[rec_id].append((i, frame_num))
    ...
```

The bare `except Exception` catches any error in accessing the dataset item, which means metadata errors are silently converted to empty dicts. Every frame gets rec_id = f'unknown_{i}' (unique per frame), so no recording has >1 frame, and no clip of length 16 can be formed.

The most likely cause: `self.base[i]` returns a structure that doesn't match `(images, targets)` tuple pattern, or the metadata dict doesn't have 'recording_id' and 'frame_num' keys in the expected format (they may be nested or require special accessor methods).

### A-3. Fix

1. Remove the bare `except Exception` — log the actual exception.
2. Inspect the actual return structure of `IndustRealMultiTaskDataset.__getitem__`.
3. Extract metadata correctly (may need to use the dataset's internal annotation cache).
4. Use `recording_id` from the dataset's metadata rather than trying to extract from sample return.

**Priority:** Fix and rerun before any other activity experiment. The temporal probe result gates the entire activity architecture decision.

---

*End of file 136 — 50 deep questions + evidence inventory + bug report for Opus.*
