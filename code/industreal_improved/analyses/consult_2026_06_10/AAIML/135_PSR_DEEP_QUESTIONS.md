# 135 — PSR: 50 Deep Questions for Opus

**Date:** 2026-07-06
**Status:** In-flight training active — PSR_HEAD_REPAIR=1 + KENDALL_FIXED_WEIGHTS=1, RTX 5060 Ti, resumed from epoch 26, ~3 hours per epoch.
**Precedes:** 136 (Opus answers to these questions)
**Supersedes:** PSR questions in 127/128/129 (these are grounded in the post-133 evidence state)

---

## §0. Evidence Inventory — file paths and current numbers

All numbers below are the **post-133 evidence base**. Any PSR question not referencing these specific files and values is stale.

| File | What it contains | Key numbers |
|---|---|---|
| `SOTA_STATUS.md` | Full PSR results table, per-comp breakdown, null-POS experiment, D4 verdict | macro F1=0.7499 (per-comp), 0.7810 (5k subset), 0.7217 (global 0.10) |
| `psr_transition.py` lines 226-259 | Dead transition heads (ReLU+bias=-1.0) and repair (LeakyReLU+bias=0.0+Xavier) | L216-236: heads use `nn.ReLU(inplace=True)` + `nn.init.normal_(head[2].bias, -1.0)`; L247-253: repaired path |
| `psr_transition.py` lines 146-175 | Q48 hysteresis in MonotonicDecoder (sustain_hi, sustain_lo, sustain_min) | sustain_hi=0.5, sustain_lo=0.3, sustain_min=3 |
| `psr_transition_f1.py` | Per-frame F1 vs transition F1 side-by-side eval | event_f1() at L23-46 with tolerance ±3 frames |
| `psr_loo_cv.py` | LOO-CV threshold validation across 16 recordings | LOO improvement +0.0358 ± 0.0216 |
| `psr_null_delta_table.md` | Per-component null-delta over always-positive baseline | comp4: +0.097, comp10: +0.093 |
| `null_model_pos/null_model_pos.json` | Null POS experiment across 3 recordings | ours=0.9988, null_zeros=0.9995, null_copy=0.9984 |
| `d4_retuned/verdict.json` | D4 YOLOv8m→decoder threshold re-tune | F1=0.000 default → 0.347 retuned (hi=0.3, lo=0.1, min=2) |
| `full_eval_ep18_stream/metrics.json` | Full 38k-frame eval at global threshold 0.10 | macro_f1=0.6773 (note: up_angular_MAE=26.20 still buggy—pre-fix era) |
| `psr_optimal_thr/optimal_thresholds.json` | Per-comp optimal thresholds (10k frames) | optimal_macro_f1=0.7499, global_0.10_macro_f1=0.7217 |
| `run_psr_kendall_fixed.sh` | In-flight training launch script | PSR_HEAD_REPAIR=1, KENDALL_FIXED_WEIGHTS=1, bs=2, resumed from crash_recovery.pth epoch 26 |
| `133_OPUS_COMPLETE_ANSWERS.md` §PSR-1 through PSR-7 | Seven verdicts from 133 | PSR-3: head repair before Kendall ablation; PSR-5: LOO-CV mandatory; PSR-6: null-delta analysis confirms learned signal |

**In-flight training parameters:**
- PSR_HEAD_REPAIR=1, KENDALL_FIXED_WEIGHTS=1
- Resumed from epoch 26 (crash_recovery.pth from `full_multi_task_tma_tbank_benchmark`)
- Batch size 2 (avoids CUDA timeout on RTX 5060 Ti 16GB)
- ~3 hours per epoch
- Launch: `scripts/run_psr_kendall_fixed.sh`

---

## §1. PSR Dead Head Diagnosis and Repair (10 questions)

**Q1. Activation function choice — was the original ReLU the sole cause of head death?**

- **File/line:** `psr_transition.py` L233: `nn.ReLU(inplace=True) if use_repaired_head else nn.ReLU(inplace=True)`
- **Current answer:** The repair (L233, L247-253) replaces ReLU with LeakyReLU(0.01), but the dead-neuron diagnosis was never empirically confirmed — no activation histogram was collected from the live model before the repair was integrated.
- **What to verify:** The repair launches with PSR_HEAD_REPAIR=1, meaning the training currently running will never run the old ReLU path. We cannot know post-hoc whether the death was ReLU saturation (all heads), bias=-1.0 collapse (all heads), or both. The in-flight training will show the head-repair delta, but the question "which fix mattered more" is unanswerable without a factorial ablation (LeakyReLU only, bias=0 only, Xavier only). Is this ablation worth running, or do we accept the bundled repair as a single intervention?
- **Connection to in-flight:** Epoch 26 starting point has the old dead heads; resumed training has the new heads. If F1 jumps within 1-2 epochs, the repair was effective. If it stays flat, the death is elsewhere (transformer, input projection).

**Q2. Bias=-1.0 — what gradient does sigmoid(-1) receive?**

- **File/line:** `psr_transition.py` L258-259: `nn.init.constant_(head[2].bias, -1.0)` (old init); L253: `nn.init.zeros_(head[2].bias)` (repaired init)
- **Current answer:** Sigmoid(-1) ≈ 0.27, d_sigmoid(0.27) ≈ 0.27 * 0.73 ≈ 0.20. The gradient is not zero—it's merely attenuated. The real problem is that sigmoid(-1) outputs a constant 0.27, so the decoder sees almost-uniform "low transition probability" regardless of input. The bias=-1.0 doesn't kill gradients, it kills *dynamic range*.
- **What to verify:** Compute the actual sigmoid output distribution from a forward pass of the old checkpoint. If all 11 heads produce output ~0.27 ± 0.01 across all frames, the bias is the primary cause (class-prior fitting). If outputs are bimodal (some ~0.0, some ~0.5), ReLU death is the primary cause. This 10-minute diagnostic should precede any factorial ablation decision.

**Q3. Does the repair address all 11 heads symmetrically, or do low-prevalence heads need different init?**

- **File/line:** `psr_transition.py` L230-235 (identical Sequential for all 11 heads), `psr_null_delta_table.md` rows comp4 (p=0.142) and comp10 (p=0.183)
- **Current answer:** All 11 heads share identical architecture and init. The null-delta analysis shows comp4 (+0.097) and comp10 (+0.093) have the most "learned signal" under the old heads, while high-prevalence heads are near ceiling. With the repaired init (Xavier, bias=0.0), low-prevalence heads may now overfit to the majority class (always-negative) since their positive frames are rare (3.6% and 4.8% respectively).
- **What to verify:** Check per-comp F1 after the in-flight training converges. If low-prevalence heads show recall=0 (all predictions negative) despite the repair, they need class-imbalance weighting or a prevalence-proportional bias init (not zero). The head repair bundles identical init for all components, but the optimal init for a component with 90% positives differs from one with 14% positives.

**Q4. Is the causal transformer also dead, or are features reaching the per-component heads?**

- **File/line:** `psr_transition.py` L218-224 (transformer), L277-285 (forward: projection → causal mask → encoded)
- **Current answer:** The transformer feeds encoded features to all 11 heads. If the encoder output has collapsed variance (e.g., all tokens converge to the same vector after 3 layers), the heads receive no discriminative signal regardless of head repair. We only checked head outputs—not transformer hidden states.
- **What to verify:** During the in-flight training's first eval epoch, capture `encoded` (L285) and compute per-component variance across time and batch. If variance is <0.01, the transformer collapsed (attention sink or value-space degradation). If variance is healthy, the heads were the sole bottleneck. This diagnostic is zero-cost: one `print(encoded.std().item())` in the forward pass.

**Q5. Does return_states=True use the decoder's Q48 thresholds or the threshold argument default?**

- **File/line:** `psr_transition.py` L294-296: `transition_probs = torch.sigmoid(transition_logits)` then `result['states'] = self.decoder(transition_probs)`. `MonotonicDecoder.forward()` L110: `def forward(self, transition_logits, threshold=0.3)`; L146-148: reads config PSR_TRANSITION_THRESHOLD_HI/LO
- **Current answer:** The decoder's `forward()` signature has a `threshold` argument that defaults to 0.3, but lines 146-148 *override* it with config values (`PSR_TRANSITION_THRESHOLD_HI=0.5`, `PSR_TRANSITION_THRESHOLD_LO=0.3`). When called from `PSRTransitionPredictor.forward` at L296, no threshold argument is passed, so the config values are used. However, the eval scripts (`psr_transition_f1.py` L79) create a standalone `MonotonicDecoder(num_components=11)` and call it themselves with their own sigmoid thresholds—they don't use `return_states`. There are two separate threshold paths.
- **What to verify:** When computing PSR F1 from the in-flight checkpoint, ensure the eval path and the training path use identical thresholds. The `psr_transition_f1.py` script thresholds raw sigmoid at 0.5 (L114) before passing to the decoder, while the decoder's own Q48 hysteresis (sustain_hi=0.5, sustain_lo=0.3, sustain_min=3) adds an extra temporal filter. The metric path and the training loss path were already computing different quantities before the repair.

**Q6. What happens to the Gaussian transition target when sigma=3 but transitions are sparse (comp4: 14.2% prevalence, ~1-2 transitions per recording)?**

- **File/line:** `psr_transition.py` L34-73, `build_transition_targets()` with sigma=3. `psr_null_delta_table.md` showing comp4 prevalence=0.142.
- **Current answer:** For a low-prevalence component with ~500 positive frames across 38k total frames, there are approximately 1-2 transition events. The Gaussian kernel at sigma=3 spans ±9 frames (19-frame window). Each transition event fires ~19 non-zero target frames. The loss gradient for that component is computed over ~19-38 frames out of 38k — a 0.05-0.1% coverage. This makes the transition loss effectively zero for 99.9% of frames, and the head sees no learning signal except near transitions.
- **What to verify:** After the in-flight training, compute per-component loss contribution. If low-prevalence heads have near-zero loss (and therefore near-zero gradient), the Gaussian-smoothed target design may need adjustment: either sigma proportional to prevalence, or a focal loss that amplifies transition-adjacent frames for rare components.

**Q7. The monotonicity regularization (L332-339) penalizes 1→0 transitions. Does this interact pathologically with the Gaussian-smeared targets?**

- **File/line:** `psr_transition.py` L332-339: `reverse_mask = (psr_labels[:, 1:, :] - psr_labels[:, :-1, :]).clamp(max=0).abs() > 0.5` then `reg_loss = (probs[:, :-1, :][reverse_mask]).mean() * 0.1`
- **Current answer:** The regularization checks ground-truth labels for reverse transitions and penalizes the *predicted* probability at those frames. But the Gaussian-smeared target (L315) blurs transition boundaries: a transition at frame t creates target>0.01 for frames t-9 through t+9. The sigmoid loss computes against these blurred targets, while the regularization computes against hard binary labels. The two loss terms are computed on different target spaces with potentially conflicting gradients.
- **What to verify:** Track the ratio `reg_loss / transition_loss` over the in-flight training. If reg_loss dominates (>50% of total), the conflicting targets are harming learning. If reg_loss is negligible (<1%), the concern is theoretical only and no action needed.

**Q8. Input_dim=512 vs hidden_dim=256 — does the information bottleneck discard spatial information?**

- **File/line:** `psr_transition.py` L207: `hidden_dim: int = 256`, L216: `self.input_proj = nn.Linear(input_dim, hidden_dim)`. Input comes from ConvNeXt C5 features (768-dim GAP) or from the ROI head (512-dim as named).
- **Current answer:** The actual input to PSRTransitionPredictor from the model is 768-dimensional ConvNeXt C5 GAP features (`backbone_out` in `model.py`), not 512. The `input_dim=512` default in the constructor means the input projection L216 creates a Linear(512, 256) layer that receives data of shape [B, T, 768]. This dimension mismatch either (a) raises a runtime error, or (b) the model silently wraps PSR features differently. If the mismatch exists, the entire PSR head never processed a single valid forward pass in any training run.
- **What to verify:** Check `model.py` for how PSRTransitionPredictor is instantiated and what `input_dim` is passed. If the model was running with the wrong input dimension, the in-flight training's head repair is fixing the wrong problem. This is a blocking diagnostic: verify the input shape to `forward()` at L262 produces a valid projection.

**Q9. Does the seq_every_n_batches=4 schedule mean the PSR head trains on 25% of batches, and did the old head benefit from the other 75% at all?**

- **File/line:** `133_OPUS_COMPLETE_ANSWERS.md` PSR-7: "per-frame batches contribute psr=0.0000, so the head sees gradient only on 25% of steps"
- **Current answer:** Per 133 PSR-7, 75% of batches are single-frame (T=1) and set psr_loss=0. The transformer on T=1 degenerates to an MLP (L219-224: causal mask is trivial for T=1). The per-component heads get zero gradient from 75% of batches. After head repair, the in-flight training inherits the same SEQ_EVERY_N_BATCHES=4 schedule.
- **What to verify:** After the head repair stabilizes, consider a PSR-focused fine-tune with `PSR_SEQ_EVERY_N_BATCHES=1` (all batches are sequence-mode). Config.py line 1671 documents this option as env-overridable. If the head repair alone gives +0.05 F1, the seq-every-batch fine-tune could give another +0.02-0.03. Compare convergence speed: does the repaired head need all batches, or can it learn from 25%?

**Q10. Is the 3-layer causal transformer (L219-224) over-parameterized for 11-component binary prediction, and does its learned positional encoding cause overfitting?**

- **File/line:** `psr_transition.py` L219-224: 3-layer transformer, d_model=256, nhead=4, dim_feedforward=1024. Total PSR transformer parameters: 3 * (4*256*64*2 + 256*1024*2) ≈ 3 * (131K + 524K) ≈ 1.97M.
- **Current answer:** The transformer has ~2M parameters for a task with 11 binary outputs over a maximum sequence length of ~100 frames (sequence batches are typically 64-128 frames). With 25% batch coverage (Q9), this is a high parameter-to-data ratio. The transformer was likely underspecified in the old model (dead heads masked the overfitting); the repaired heads may now overfit to sequence patterns.
- **What to verify:** Monitor train vs val PSR F1 divergence in the in-flight training. If train F1 >> val F1 by epoch 30, add dropout (currently 0.1) or reduce num_layers to 1. Also compute the effective rank of encoded features: if rank < 11, the transformer compresses information below what the heads need.

---

## §2. Per-Component Thresholding and LOO-CV (10 questions)

**Q11. The full-set per-comp optimal thresholds (Q17 `optimal_thresholds.json`) vs the 5k-subset thresholds (SOTA_STATUS.md) differ. Are they from the same checkpoint?**

- **File/line:** `SOTA_STATUS.md` L30-44 (per-comp table, including best_thresh column); `psr_optimal_thr/optimal_thresholds.json` L17-28 (optimal_thresholds array)
- **Current answer:** SOTA_STATUS.md reports thresholds: [0.05, 0.20, 0.15, 0.85, 0.80, 0.50, 0.45, 0.90, 0.90, 0.05, 0.70]. `psr_optimal_thr/optimal_thresholds.json` reports: [0.05, 0.20, 0.15, 0.85, 0.80, 0.50, 0.45, 0.90, 0.90, 0.05, 0.70] — identical. But SOTA_STATUS also reports "5k subset" macro F1=0.7810, implying thresholds were recomputed on the subset and yielded higher F1 on that subset. The subset thresholds and F1 are not reported as a separate threshold array.
- **What to verify:** Compute per-comp optimal thresholds on the 5k subset and compare to the full 38k thresholds. If they differ by >0.1 for any component, the threshold is sensitive to frame selection and a single optimal threshold is not a stable configuration. If they are identical, the 0.7810 vs 0.7499 difference is just evaluation randomness (the subset was easier), not a better threshold configuration.

**Q12. LOO-CV improvement +0.0358 ± 0.0216 (mean ± std across 16 recordings). The std is 60% of the mean — which recordings benefit, which lose?**

- **File/line:** `psr_loo_cv.py` L216: `loo_improvement_std = float(np.std(...))`; the per-recording results are stored but not summarized in the print output (L198-206).
- **Current answer:** The mean +0.0358 says threshold optimization helps on average. The std ±0.0216 says some recordings improve by +0.06 while others barely change (or degrade). If a specific recording consistently loses under optimal thresholds, it suggests a distribution shift (different lighting, different worker, different assembly variant) that per-component thresholds don't generalize across.
- **What to verify:** Read `psr_loo_cv/loo_cv_results.json` (once created) and extract the per-recording improvement column. If any recording shows improvement < -0.01 (degradation), identify what makes it different. The LOO-CV F1 values reported in `psr_loo_cv.py` are per-component macro means—the distribution of improvements matters more than the mean.

**Q13. The LOO-CV uses sigmoid outputs from the old dead heads. After head repair, will the optimal thresholds change?**

- **File/line:** `psr_loo_cv.py` L93-129 (threshold search over sigmoid scores); the sigmoid scores come from the checkpoint's PSR head outputs.
- **Current answer:** The LOO-CV was conducted on epoch_18's sigmoid outputs, produced by the dead ReLU+bias=-1.0 heads. After head repair, the sigmoid distribution will shift: instead of clustering around 0.27 (sigmoid(-1)), outputs may spread across the full [0,1] range. The optimal thresholds computed on the old output distribution are *not* transferable to the new distribution.
- **What to verify:** After the in-flight training converges, re-run the LOO-CV on the new checkpoint. If the new thresholds differ systematically (e.g., comp4 optimal threshold drops from 0.80 to 0.30), it means the repaired head produces higher-confidence negative predictions for low-prevalence components, which changes the precision-recall tradeoff. The old thresholds may have been compensating for low-confidence outputs.

**Q14. The LOO-CV sweeps 19 thresholds (0.05 to 0.95 in 0.05 steps). Is the 0.05 granularity too coarse for components with optimal thresholds at the extremes (comp0: 0.05, comp7/8: 0.90)?**

- **File/line:** `psr_loo_cv.py` L93: `thresholds = np.arange(0.05, 1.0, 0.05)`. `psr_optimal_thr/optimal_thresholds.json`: comp0=0.05, comp7=0.90, comp8=0.90.
- **Current answer:** Comp0 (gt_pos_frac=1.000) has optimal threshold at the minimum of the sweep range (0.05). The true optimal could be 0.01, 0.02, or even 0.0 (always-positive classifier). Similarly, comp7 and comp8 are at the maximum (0.90). Without extending the sweep, we cannot know if comp0 benefits from a near-zero threshold or if comp7/8 want >0.95.
- **What to verify:** Re-run the threshold sweep for comp0 with search space [0.01, 0.05, 0.10, 0.15, 0.20] and comp7/8 with [0.85, 0.90, 0.95, 0.99]. If the extreme thresholds actually yield higher F1, the current values are bound by the sweep range. The impact is small for comp0 (already F1=1.0) but could matter for comp7/8 (currently F1=0.804/0.854).

**Q15. The macro F1 improvement from global 0.10 to per-comp optimal is +0.028 (0.7499 - 0.7217). How much of this improvement comes from adjusting thresholds for comp4 and comp10 alone?**

- **File/line:** `SOTA_STATUS.md` L32-43 (per-comp F1); `psr_optimal_thr/optimal_thresholds.json` (thresholds: comp4=0.80, comp10=0.70); global threshold=0.10
- **Current answer:** At global threshold 0.10, comp4 (gt_pos_frac=0.142) has precision ≈ 0.11 (from `full_eval_ep18_stream/metrics.json` L40-44: at global 0.10, comp4 recall=1.0 but precision=0.11, F1=0.198). At optimal threshold 0.80, F1=0.346. The improvement for comp4 alone is +0.148. Similarly comp10: at global 0.10, F1≈0.40; at optimal 0.70, F1≈0.40 (similar). Comp4 contributes disproportionately to the macro improvement because the global threshold was catastrophically bad for it (almost all predictions positive due to detection sparsity).
- **What to verify:** Compute macro F1 with per-comp optimal thresholds EXCEPT comp4 and comp10 set to global 0.10. If the improvement drops to ~0.010 (from +0.028), the entire threshold-tuning benefit comes from two components. If the improvement persists at ~0.025, the benefit is distributed across components. This tells us whether threshold tuning is a targeted fix for low-prevalence components or a broad improvement.

**Q16. The optimal thresholds for comp1 and comp2 are 0.20 and 0.15 respectively, yet both have gt_pos_frac=0.911. Why would two components with identical prevalence need different thresholds?**

- **File/line:** `SOTA_STATUS.md` L32-33 (comp1: p=0.911, thr=0.20, F1=0.963; comp2: p=0.911, thr=0.15, F1=0.958); `psr_null_delta_table.md` L9-10 (comp1 delta=+0.009, comp2 delta=+0.004)
- **Current answer:** Both have identical prevalence and similar F1. The 0.05 threshold difference may be noise (the LOO-CV would likely show comp1 and comp2 benefiting equally from either threshold). Alternatively, comp2 may have a different sigmoid distribution despite identical prevalence—perhaps comp2's transitions are more abrupt and the sigmoid scores are lower around transition boundaries.
- **What to verify:** Plot the sigmoid score distribution for comp1 vs comp2. If they are nearly identical, threshold them at the same value (say 0.18) and verify F1 doesn't change. The null-delta analysis already shows both are near-ceiling (delta=+0.009 and +0.004), so threshold optimization precision for these components doesn't matter for the bottom line.

**Q17. Comp9 (gt_pos_frac=0.527) has optimal threshold 0.05 — near the minimum of the sweep — and null-delta of -0.000 (at-null). Is comp9 an "always positive" component at any threshold below 0.527?**

- **File/line:** `SOTA_STATUS.md` L41 (comp9: p=0.527, thr=0.05, F1=0.690); `psr_null_delta_table.md` L22 (comp9 delta=-0.000)
- **Current answer:** Comp9 is at-null: the model adds zero learned signal beyond predicting positivity for all frames. The optimal threshold 0.05 (nearly all frames predicted positive) confirms the model's best strategy is to guess positive for everything. The F1=0.690 is purely prevalence-driven: F1_null = 2*0.527/(1+0.527) = 0.690. The threshold optimization didn't find signal—it found the threshold that maximizes an always-positive guess.
- **What to verify:** After head repair, comp9 is the critical test: if the repaired head learned signal for comp9 (delta > +0.02), then the old head was truly dead for all components. If comp9 remains at-null, comp9's features may genuinely not support prediction (e.g., the posture is visually identical to other postures), and the threshold should be set to always-negative (thr=0.95) to avoid false positives.

**Q18. The LOO-CV computes improvement as optimal_f1 - global_f1, but both are macro means over 11 components. Could the improvement come from components where both optimal and global thresholds give similar results, masking components that degrade?**

- **File/line:** `psr_loo_cv.py` L200-206: per-recording improvement = `np.mean(f1s_optimal_c) - np.mean(f1s_global_c)`. This is a macro difference of macro means.
- **Current answer:** A macro-mean improvement of +0.0358 could theoretically mask one component degrading by -0.10 while four components improve by +0.035 each (net: +0.04). The standard deviation ±0.0216 is across recordings, not across components. The cross-component variance is not reported.
- **What to verify:** Add per-component LOO-CV results to the analysis: for each component, what is the held-out improvement when thresholds are optimized on the other recordings? If any component consistently shows degradation (negative improvement) across recordings, that component's threshold should remain at global 0.10 regardless of what the sweep says.

**Q19. Global threshold 0.10 gives macro F1=0.7217. What is the single global threshold that maximizes macro F1?**

- **File/line:** `SOTA_STATUS.md` L21: "Global threshold 0.10 macro F1=0.7217". The choice of 0.10 is arbitrary from the LOO-CV script (L180: `binary = (sigs_v > 0.10)`).
- **Current answer:** The LOO-CV script uses 0.10 as the global threshold, but this is a default value, not the optimal global threshold. A sweep of global thresholds (0.05, 0.10, 0.15, ..., 0.50) against the full dataset would find the single threshold that maximizes macro F1. If the optimal global threshold is also 0.10, then 0.10 is justified. If a different global threshold gives higher F1, the improvement from "global 0.10 to per-comp optimal" is inflated by a suboptimal baseline.
- **What to verify:** Sweep global thresholds 0.05 to 0.50 in 0.05 steps on full dataset, compute macro F1 for each. Report the optimal global threshold's F1 alongside the per-comp optimal F1. This gives a fair "baseline improvement" that doesn't compare against an arbitrary default.

**Q20. The 16-recordings LOO-CV (psr_loo_cv.py) allocates thresholds per recording held-out. But does the training split (which recordings are in train vs val) affect the LOO results?**

- **File/line:** `psr_loo_cv.py` L140: `for held in recordings:` — leaves out one recording, thresholds on the other 15. But the original train/val split may have put some of the 16 recordings in the training set.
- **Current answer:** The LOO-CV treats all 16 recordings as independent evaluations, but the checkpoint was trained only on the training split. Some of the 16 recordings may have been in the training set (the model saw them during training) while others were in the validation set (held out during training). Thresholds optimized on recordings the model already saw are not truly "held-out." The LOO-CV should either be restricted to recordings that were in the original validation split, or the improvement should be reported separately for training-set recordings vs validation-set recordings.
- **What to verify:** Identify which of the 16 recordings are in the original train/val split. Report LOO improvement separately for each split. If validation-set recordings show lower improvement than training-set recordings, the threshold optimization partially relies on recording familiarity, not just prevalence.

---

## §3. POS Artifact and Null-Model Proof (10 questions)

**Q21. The null-zeros POS is 0.9995 — higher than ours (0.9988). Is this because predicting "all zeros" is actually correct for more than 99.9% of frame-pairs?**

- **File/line:** `null_model_pos/null_model_pos.json` L5-7: ours=0.9988, null_zeros=0.9995, null_copy=0.9984.
- **Current answer:** POS (ordered-pair fraction) measures what fraction of adjacent frame-pairs have the correct sign of change. For a component that transitions exactly once at frame t, 99% of frame-pairs show no change (sign=0). An all-zeros predictor matches all of those. The all-zeros model misses only the 1% of pairs that span the transition — hence POS=0.999 for sparse-transition sequences. This is a structural property of POS as a metric for sparse transitions: it rewards models that predict "nothing happened" because "nothing happened" is true 99% of the time.
- **What to verify:** Compute POS restricted to windows ±k frames around ground-truth transitions (POS@k). For k=3 (tolerance window), the null-zeros POS would drop to ~0.0 (it never predicts the transition), while a good model might achieve 0.3-0.5. This would simultaneously verify that POS is inflated (null models prove it) and show that there IS information in POS at shorter windows. The POS@tolerance variant proposed in 133 PSR-1.

**Q22. The null_copy_prev predictor copies the previous frame's state — POS=0.9984, almost identical to null_zeros. Why? Because fill-forward monotonic data has almost no frame-to-frame changes?**

- **File/line:** `null_model_pos/null_model_pos.json`. The copy-prev model for monotonic fill-forward labels: if the state only changes a few times across 1000+ frames, copying the previous frame is the same as all-zeros for 99%+ of frame-pairs.
- **Current answer:** For monotonic data, the only frame-pair where copy-prev differs from all-zeros is the transition frame itself: copy-prev predicts (1,1) for the post-transition pair, which has the correct sign (0) while all-zeros predicts (0,0) with sign 0 as well. Actually, they produce identical POS because both predict 0 change at every frame-pair, just from different constant states. The insight: POS cannot distinguish any constant prediction from any other for monotonic data because all constants have identical ordered-pair structure.
- **What to verify:** Confirm analytically: for monotonic binary sequences, POS(constant) = 1 - (num_transitions / (T-1)). For T=1000 and 3 transitions, POS(constant) = 0.997. Neither the sigmoid head nor the decoder can beat this structural ceiling by more than a few points. The only way to have POS significantly above the ceiling is to predict transitions that match GT timing, but the transition window is so narrow that the POS contribution is bounded by ~0.001 per correct transition. This formally proves POS is uninformative for this data.

**Q23. Does the POS paradox affect the D4 YOLOv8m→decoder transition F1=0.000 result?**

- **File/line:** `SOTA_STATUS.md` L25-26: D4 event F1=0.000, POS=0.999; `d4_retuned/verdict.json`: F1=0.000 default.
- **Current answer:** Yes — for D4 at default Q48 thresholds (hi=0.5, lo=0.3, min=3), the decoder predicts all-zeros (no transitions), which gives POS≈0.999 (same as null_zeros) but F1=0.000 (no transitions detected). The "F1=0 with POS=0.999" combination is the POS paradox in action: the metric that rewards all-zeros prediction gives a perfect score while the metric that measures actual transition detection gives zero. Removing POS from the D4 evaluation would change the narrative from "almost perfect" (0.999) to "completely failed" (0.000).
- **What to verify:** After retuning D4 (PSR-4 from 133), the F1=0.000 becomes F1=0.347. The corresponding POS at retuned thresholds should be reported. If POS drops (because the decoder now predicts some transitions, creating false positive ordered-pairs), the tradeoff becomes clear: POS and F1 are inversely correlated here.

**Q24. The null model experiment uses 3 recordings (5000 frames total). Are these recordings representative of all 16 val recordings?**

- **File/line:** `null_model_pos/null_model_pos.json` L9-27: recordings 14_main_2_2, 14_main_2_3, 20_assy_0_1 with 1404, 1679, 1917 frames respectively.
- **Current answer:** Three recordings, all from the "14_main" and "20_assy" scenarios. The POS scores vary across recordings: 14_main_2_2 (ours=0.9994), 14_main_2_3 (0.9991), 20_assy_0_1 (0.9979). Recording 20_assy_0_1 has the lowest POS (0.9979) and the largest gap between null_zeros and ours (0.9976 vs 0.9979). If we tested all 16 recordings, we might find some with lower POS (more transitions) where the null-copy model diverges from our model.
- **What to verify:** Run the null POS experiment on all 16 recordings (not just 3). This is a ~10-minute compute: cached logits for all recordings, then compute null_model_pos for each. If the POS gap (ours - null_copy) is consistently <0.001 across all recordings, the artifact is universal. If some recordings show a larger gap (e.g., >0.003), those recordings have enough transitions that POS might carry signal.

**Q25. The "null" models are always-positive and always-zero. What about a model that randomly predicts transitions at the empirical transition rate?**

- **File/line:** `null_model_pos/null_model_pos.json` — only tests all-zeros and copy-prev. Neither is a proper null for POS under transition sparsity.
- **Current answer:** The copy-prev and all-zeros models are degenerate special cases. A proper null model for POS would be: for each component, randomly sample transition frames at rate equal to the empirical transition rate (transitions per frame), then compare POS. This null would have POS lower than 0.998 because some random transitions would fall near real transitions (good for POS) but many would not (bad for POS). If OUR model's POS beats this transition-rate-matched null, there IS some ordering signal. If it doesn't, even the ordering signal is at chance.
- **What to verify:** Build a Monte Carlo null: for each component, sample transition locations uniformly at random with probability = observed transition rate. Compute POS over 1000 random samples and compare to our model's POS. This separates "ordering ability" from "transition detection ability" and is the fairest null for POS.

**Q26. The POS@tolerance variant (PSR-1 from 133) would evaluate ordering only within ±k frames of GT transitions. What is the expected POS@3 for a random predictor?**

- **File/line:** `133_OPUS_COMPLETE_ANSWERS.md` PSR-1: "POS@tolerance — ordering scored only within ±k-frame windows around GT transitions — is the only version of POS with informational content"
- **Current answer:** For a tolerance of k=3 and a random predictor that fires with probability p, the random POS@3 = 0.5 (it's binary ordering within a small window; random ordering is at chance). At k=3 and an average of 1 transition per component per recording, the evaluation window is 7 frames × 11 components = 77 frames per recording — still sparse but no longer structurally dominated by constant-state pairs.
- **What to verify:** Implement POS@k and compute for both our model and the null models. Report values for k=1, 3, 5. If our model's POS@3 is >0.6 while null models are at 0.5, there IS ordering signal. This would reconcile the POS paradox: raw POS is inflated, POS@k is informative.

**Q27. The null delta analysis (PSR-6) uses always-positive as null. Is always-positive the correct null for F1, or should we use a prevalence-calibrated random classifier?**

- **File/line:** `psr_null_delta_table.md` L20: "Always-positive classifier always predicts the majority class. For a component with prevalence p, F1_null = 2p/(1+p)."
- **Current answer:** Always-positive is the correct null for per-component F1 under monotonic labels because (a) it's the optimal constant prediction for binary F1 when p>0.5, and (b) for components with p<0.5 (none here), always-negative would be the null. However, always-positive is the "oracle constant" — no real model with our architecture could achieve it (since the model can't know p before evaluation). A more informative reference is: what F1 does a model that outputs the training-set prevalence achieve? If training prevalence differs from validation prevalence, the always-positive null overestimates the baseline.
- **What to verify:** Compute training-set prevalence for each component. If train prevalence differs from val prevalence for any component, compute F1_null using train prevalence instead of val prevalence. This addresses the concern (from 127's DS-8) that the null-delta might be inflated by train/val prevalence shift.

**Q28. The null-zeros POS=0.9995 is suspiciously close to 1.0000. Is there a numerical precision issue in the POS computation?**

- **File/line:** `psr_transition_f1.py` L173-176: `pos = float((np.sign(pred_pairs) == np.sign(gt_pairs)).mean())`
- **Current answer:** The POS computation uses `np.sign` on the difference of adjacent frames. For monotonic data with sparse transitions, most differences are 0, and `np.sign(0) = 0`. Both pred_pairs and gt_pairs are mostly zeros, so `np.sign(pred_pairs) == np.sign(gt_pairs)` is True for most entries. With 5000 frames and ~30 transitions across all components, the expected POS = (5000*11 - 30) / (5000*11) = 54970/55000 = 0.99945. The null-zeros value 0.9995 matches this. The computation is numerically correct — the metric is structurally inflated.
- **What to verify:** Check for edge cases: if a component has its only transition at the very first frame (frame 0→1), the difference (1 - 0) = 1, sign=1. A null-zeros model predicts diff=0, sign=0, and this frame-pair is counted as incorrect (sign 1 ≠ 0). This single pair out of 55000 changes POS by ~0.00002. The numerical precision concern is unfounded — the artifact is real.

**Q29. The null-model experiment claims POS is a "fill-forward artifact." But does the same argument apply to Edit score?**

- **File/line:** `psr_transition_f1.py` L178-200 (Edit score computation via Levenshtein distance on event strings). Edit scores are reported in the same summary as POS (L211).
- **Current answer:** Edit score measures string edit distance between predicted and ground-truth transition sequences. For a null-zeros model (no predicted transitions), the predicted event string is all zeros, and the edit distance equals the number of GT events (each deletion costs 1). Edit score = 1 - (num_deletions / max(len_pred, len_gt)). For monotonic data with few transitions, Edit ≈ 0.99 for null models — same structural inflation. The null-model experiment should be extended to Edit score.
- **What to verify:** Compute null-Edit and null-copy-Edit alongside null-POS. Report the three metrics (POS, Edit, F1) for both our model and null models as a 2×3 comparison table. This either confirms Edit is also inflated (and should be dropped) or reveals that Edit captures something POS misses.

**Q30. The null-model experiment was analyzed on best.pth (epoch 18). Would the results differ on the in-flight training's checkpoint (epoch 26, head repair)?**

- **File/line:** `run_psr_kendall_fixed.sh` — in-flight training from epoch 26; after repair, sigmoid outputs will not cluster at ~0.27.
- **Current answer:** The null POS analysis depends only on the monotonic structure of labels and the decoder's prediction pattern. After head repair, if the model produces more accurate transition predictions (higher sigmoid near transitions), the POS might increase slightly (more correct transition-pair signs) but the null models won't change. The conclusion "POS is structurally inflated" is independent of model quality — it's a property of the metric + data structure, not the model.
- **What to verify:** Re-run null-model POS on the in-flight checkpoint after convergence. If POS stays at ~0.999 and ours also stays at ~0.999, the artifact persists regardless of model quality. If our POS drops significantly (e.g., to 0.990), the repaired head is predicting more false positives (transitions that aren't there), which lowers POS but may improve F1 — revealing the POS-vs-F1 tradeoff formally.

---

## §4. D4 Backbone Swap and Threshold Re-tuning (10 questions)

**Q31. D4 default thresholds (hi=0.5, lo=0.3, min=3) gave F1=0.000. Retuned (hi=0.3, lo=0.1, min=2) gave F1=0.347. Is the improvement from threshold relaxation alone, or from better alignment with YOLOv8m's sparse output distribution?**

- **File/line:** `d4_retuned/verdict.json` L7-11: best_global_config = {sustain_hi=0.3, sustain_lo=0.1, sustain_min=2}; `SOTA_STATUS.md` L65-68.
- **Current answer:** YOLOv8m detects objects on <1% of frames. Its non-default logits are rare but high-confidence when they occur. The ConvNeXt-based PSR logits are dense (every frame has a non-zero prediction). The default Q48 thresholds (hi=0.5, lo=0.3, min=3) require sustained evidence over 3 consecutive frames — which works for dense ConvNeXt outputs but kills the sparse YOLOv8m outputs. The retuned thresholds (hi=0.3, lo=0.1, min=2) only require 2 frames of moderate confidence. The improvement comes from matching thresholds to YOLOv8m's output statistics, not from better thresholds per se.
- **What to verify:** Compute YOLOv8m logit statistics: what fraction of frames exceed sustain_lo=0.1? What is the average duration of above-lo runs? If the average run length is 1.2 frames, then even min=2 is challenging. The true optimal min for YOLOv8m might be 1 (no sustained requirement — fire on any single confident frame), which would change the F1 ceiling.

**Q32. After D4 retuning, F1=0.347. What is the theoretical maximum F1 achievable by any decoder on YOLOv8m's detections?**

- **File/line:** `SOTA_STATUS.md` L26: "YOLOv8m produces detections on <1% of frames; backbone detection density is the binding constraint."
- **Current answer:** The theoretical maximum F1 depends on YOLOv8m's detection coverage: if it detects the relevant object on 40% of frames where PSR transitions occur, the maximum recall is 0.40 (assuming perfect thresholding). If precision can be 1.0 (every detected transition is real), max F1 = 2*1.0*0.4/(1.0+0.4) = 0.57. The current 0.347 suggests either recall is ~0.25 or precision is ~0.55.
- **What to verify:** Compute the oracle-switch F1 for D4: for each ground-truth transition, check whether YOLOv8m produced a detection within ±3 frames. The fraction of transitions with a nearby detection is the maximum achievable recall. This sets a hard upper bound on D4 performance and tells us whether further threshold tuning can beat 0.347.

**Q33. Per-component optimal thresholds after retuning give F1=0.261, which is lower than the global best 0.347. Why does per-component optimization hurt?**

- **File/line:** `d4_retuned/verdict.json` L3: `"f1_at_t_retuned": 0.261` (per-component optimal). L2: `"f1_at_t_best_global": 0.347`.
- **Current answer:** Per-component optimal thresholds increased F1 from 0.000 to 0.261, but the global best (single threshold for all components) achieves 0.347. This is unusual — per-component optimization should always be >= global optimization. The likely explanation: the per-component thresholds were found by scanning each component independently, but the decoder's hysteresis (sustain_min, sustain_counter that resets) creates cross-component interactions. A component that fires early may prevent another component from ever reaching its sustain_lo threshold (because the state is now "component placed" and the decoder moves on). The independent per-component scan doesn't account for this ordering constraint.
- **What to verify:** Visualize the decoder state trajectory for the per-component vs global threshold configurations. If the ordering constraint (procedure_order graph in MonotonicDecoder) causes sequential components to block each other, per-component thresholds are inherently non-independent and must be optimized jointly. This means the 19^11 search space concern (PSR-5 from 133) is real for D4 even if it's not for the simpler per-frame F1 evaluation.

**Q34. The D4 YOLOv8m→decoder pipeline uses YOLOv8m detection logits (24 classes) as PSR features. How are 24 class logits mapped to 11 PSR components?**

- **File/line:** The D4 experiment setup is described in `SOTA_STATUS.md` §5.4 but the mapping logic is not in any of the referenced files.
- **Current answer:** The mapping from 24 ASD detection classes to 11 PSR components is a critical design decision not documented in the readable files. If the mapping is incorrect (e.g., components 4 and 10 are mapped to detection classes that YOLOv8m never activates), the F1=0.000 at default thresholds would be explained by a dataflow bug, not by decoder failure.
- **What to verify:** Find the mapping code (likely in `model.py` or a PSR evaluation bridge). Verify each PSR component maps to the correct detection class. If not, the D4 retuning was optimizing a broken pipeline. This is the D4 equivalent of the MonotonicDecoder variable-shadow bug (SOTA_STATUS.md win #3) and should be fixed before any conclusion.

**Q35. D4 retuning evaluated on 16 videos. Which videos have nonzero event F1 after retuning, and which remain at zero?**

- **File/line:** `d4_retuned/verdict.json` — only reports aggregate F1, not per-video breakdown.
- **Current answer:** The aggregate F1=0.347 could come from 5 videos with good detections (F1≈0.8) and 11 videos with F1=0.0. If so, the decoder works on some recordings and completely fails on others — which would point to recording-specific YOLOv8m detection quality variation (lighting, worker clothing, occlusion). The headline "F1=0.000→0.347" would be misleading if it's driven entirely by a subset.
- **What to verify:** Compute per-video D4 event F1 at the best global configuration. Report the distribution (min, median, max, quartiles). If any video has F1=0 despite retuning, investigate what's different (is YOLOv8m detecting anything on that video at all?).

**Q36. The D4 retuning sweep used global thresholds (same hi/lo/min for all components). Does per-component threshold retuning for D4 yield higher F1 than global, if we account for ordering constraints?**

- **File/line:** `d4_retuned/verdict.json` — the "per-component optimal" (0.261) was computed independently, which the Q33 analysis suggests is incorrect due to ordering interactions.
- **Current answer:** A joint search (optimize all 11 component thresholds simultaneously, or at least optimize the three hysteresis parameters) might find a configuration that beats global 0.347. The search space is small: hi ∈ {0.1, 0.3, 0.5}, lo ∈ {0.05, 0.1, 0.3}, min ∈ {1, 2, 3} — 27 combinations total. A full grid search on cached logits would take <1 hour.
- **What to verify:** Run the full 27-combination grid search for D4. Report the best F1 and the corresponding (hi, lo, min). If it beats 0.347, the current "best" is not actually the global optimum. If 0.347 is confirmed, the decoder improvement is saturated at YOLOv8m's detection ceiling.

**Q37. The D4 retuning changed min from 3 to 2. What happens at min=1 (fire on any single frame above threshold)?**

- **File/line:** `d4_retuned/verdict.json` best_global_config includes sustain_min=2.
- **Current answer:** YOLOv8m produces isolated single-frame detections (not sustained runs). If min=1, the decoder fires on any isolated detection, potentially catching more true transitions (higher recall) but also firing on spurious detections (lower precision). The optimal min depends on whether YOLOv8m's isolated detections correlate with PSR transitions.
- **What to verify:** Sweep min ∈ {1, 2, 3, 4} at the best hi/lo values. If min=1 gives F1 > 0.347, the best configuration hasn't been found. If min=1 gives lower F1, the decoder needs sustained evidence to avoid false positives. The shape of the min-performance curve tells us about YOLOv8m's detection reliability.

**Q38. MonotonicDecoder on ConvNeXt logits achieves F1=0.75 (SOTA_STATUS.md L22). On YOLOv8m logits after retuning, F1=0.347. What is the F1 when both use the same decoder and same thresholds?**

- **File/line:** No direct comparison exists — the ConvNeXt PSR F1 (0.7217 per global 0.10, 0.7499 per-comp) uses per-frame sigmoid at fixed thresholds, not the MonotonicDecoder's Q48 hysteresis. The D4 uses the decoder. They are different evaluation pipelines.
- **Current answer:** There is no "apples-to-apples" comparison of ConvNeXt→decoder vs YOLOv8m→decoder. The ConvNeXt PSR numbers (0.7217, 0.7499) are per-frame sigmoid threshold evaluations. The Decoder evaluation (Q48 hysteresis) has never been run on ConvNeXt logits. Without this comparison, we cannot attribute the D4 gap to backbone quality — part of it may be the decoder evaluation being stricter than per-frame thresholding.
- **What to verify:** Run the MonotonicDecoder (with the same Q48 hysteresis and same threshold sweep) on ConvNeXt-derived PSR logits. Compute decoder-based F1 for ConvNeXt. If ConvNeXt→decoder F1 ≈ 0.75, then the decoder evaluation is backbone-agnostic and YOLOv8m is genuinely worse. If ConvNeXt→decoder F1 ≈ 0.40, then the decoder itself is a bottleneck regardless of backbone, and the D4 gap shrinks.

**Q39. D4 F1=0.000 at default thresholds — does any decoder predict any transition for any frame?**

- **File/line:** `d4_retuned/verdict.json` L1: `f1_at_t_original: 0.0`
- **Current answer:** F1=0.000 means either (a) the decoder predicted no transitions at all, or (b) the decoder predicted transitions but none matched GT transitions within tolerance. The former (all-zero predictions) is consistent with YOLOv8m's extreme detection sparsity at default thresholds. But we should verify: did ANY frame in any of the 16 videos trigger a transition at default Q48?
- **What to verify:** Count the number of frames where the decoder predicted a transition (state change) at default thresholds across all 16 videos. If it's zero, the YOLOv8m→decoder pipeline was entirely silent — no detections passed even the lowest threshold. If it's >0 but F1 is still 0, the predicted transitions are all false positives. The count distinguishes "decoder never fires" from "decoder fires but always wrong."

**Q40. After the in-flight training with PSR_HEAD_REPAIR=1, will the PSR head produce outputs that can be fed back into D4 (YOLOv8m features → repaired PSR head → transitions)?**

- **File/line:** The head repair only affects the PSRTransitionPredictor (transition_heads in psr_transition.py). The D4 experiment uses a standalone MonotonicDecoder on YOLOv8m logits.
- **Current answer:** The head repair doesn't change D4 because D4 doesn't use the PSRTransitionPredictor — it uses raw YOLOv8m detection logits fed directly to a standalone MonotonicDecoder. However, a D4 variant that runs YOLOv8m logits through the repaired transition heads (the small MLPs) before the decoder might improve F1. This would test whether the head repair helps even with sparse YOLOv8m features.
- **What to verify:** After the in-flight training converges, extract the repaired transition_heads weights and run YOLOv8m logits through them before the decoder. Compare F1 to the direct YOLOv8m→decoder result. If the heads boost F1 beyond 0.347, the repair helped the feature extraction, not just the decoder interface. If F1 stays at 0.347, the bottleneck is purely YOLOv8m's detection density.

---

## §5. SOTA Comparison — STORM/B3 Paradigm Gap (10 questions)

**Q41. STORM achieves F1=0.901, B3 achieves F1=0.883. Both are event-level transition metrics. Our "transition F1" (from psr_transition_f1.py) computes event matching within ±3 frames. How comparable are these?**

- **File/line:** `SOTA_STATUS.md` L24: STORM F1=0.901, B3 F1=0.883. `psr_transition_f1.py` L23-46: `event_f1()` with tolerance ±3.
- **Current answer:** STORM and B3 use the same event-level matching protocol with tolerance ±3 frames (the IndustReal benchmark standard). Our `event_f1()` function (L23-46) implements the same algorithm. The metric IS comparable — the gap is not a protocol mismatch. However, per 133 SOTA-2: "Different quantities under the same symbol. Never in one table column." The gap is genuine but reflects different paradigms (per-frame sigmoid vs. learned transition detection) and different backbones (ConvNeXt-Tiny vs. STORM's Video Transformer).
- **What to verify:** Run `psr_transition_f1.py` with `--tolerance 3` on the current checkpoint and record the transition macro F1. If the number is significantly below 0.75 (e.g., 0.4-0.5), the paradigm gap is even larger than the per-frame numbers suggest, and the "competitive" label in SOTA_STATUS.md should be downgraded to "not comparable."

**Q42. STORM uses a large video transformer with temporal attention. Our architecture uses ConvNeXt-Tiny + per-frame pooling + causal transformer. How much of the STORM gap is explainable by backbone compute?**

- **File/line:** `133_OPUS_COMPLETE_ANSWERS.md` PW-3: claim-strength rubric — "competitive" = within 10% relative under identical protocol.
- **Current answer:** STORM uses a 200M+ parameter architecture with 3D convolutions and temporal attention over 16-frame clips. Our model uses ConvNeXt-Tiny (28M backbone, ~50M total). The compute gap is ~4x in parameters and likely ~10x in FLOPs. If our per-comp optimal F1 (0.7499) is close to the compute-parameter envelope for a 50M model, then matching STORM's 0.901 requires a larger backbone. The question is: how much of the 0.151 F1 gap is architecture (fixable with larger model) and how much is paradigm (per-frame vs. transition-based training)?
- **What to verify:** Compute the correlation between model capacity and PSR F1 in the literature. If every published ConvNeXt-Tiny model achieves F1<0.80 on IndustReal PSR, the ceiling is backbone-defined. If some achieve >0.85 with the same backbone, the paradigm (training objective, decoder design) is the binding constraint.

**Q43. B3 incorporates procedural knowledge (assembly graph, contact points). Does our MonotonicDecoder's procedure_order prior capture similar information, and what's the gap?**

- **File/line:** `psr_transition.py` L94-98: procedure_order = [(0,1), (1,2), ..., (9,10)] (sequential). This is the simplest possible ordering constraint.
- **Current answer:** B3 uses a learned procedural knowledge graph with task-specific constraints (sub-assemblies, parallel branches). Our procedure_order is a linear chain — every component must be placed after the previous one. This is a simplification that (a) doesn't capture parallel assembly operations (two components placed simultaneously) and (b) doesn't capture component grouping (comp3 and comp4 are both on a sub-assembly that must arrive before comp5). The procedural knowledge gap alone could account for 0.05-0.10 of the 0.151 F1 deficit.
- **What to verify:** Analyze the ground-truth component placement orders across all recordings. How often does the linear order (0→1→...→10) hold? If there are recordings where components are placed out of sequential order, the linear constraint is actively harmful (preventing correct predictions). The procedure_order should be data-derived, not hardcoded.

**Q44. STORM and B3 both use transition-focused training (event detection loss), not per-frame BCE. Our training uses per-frame BCE with Gaussian-smeared targets. Does our loss function approximate the transition-focused paradigm?**

- **File/line:** `psr_transition.py` L300-359: `compute_loss()` with Gaussian-smeared targets (sigma=3, radius=9) and focal-like weighting.
- **Current answer:** The Gaussian-smeared loss is an approximation to transition-focused training: instead of learning to predict the exact transition frame, it learns a "soft transition region" of ±9 frames around the true transition. For the version originally active (not the repaired head), the sigmoid outputs were nearly constant (dead heads), so the loss function didn't matter. After repair, the loss function becomes relevant: does Gaussian-smeared loss with sigma=3 produce better or worse transition detection than a hard BCE loss on exact transition frames?
- **What to verify:** After the head repair stabilizes, compare three loss variants on the repaired model: (1) Gaussian-smeared (current), (2) hard BCE at transition frame only, (3) a learned temperature. Compute transition F1 (event-level) for each. If variant (2) is better, the Gaussian smearing introduces label noise near boundaries and should be removed.

**Q45. The STORM paper reports per-component F1 breakdown. Can we compute our per-component transition F1 (event-level, not per-frame) and compare directly?**

- **File/line:** `psr_transition_f1.py` L139-170 computes per-recording transition F1 but reports macro mean (L209), not per-component mean across recordings.
- **Current answer:** The `psr_transition_f1.py` output (per_frame_vs_transition.json) has per-component per-recording F1 data but only reports the macro across components. STORM's per-component breakdown (reported in their Table 2) would allow direct comparison. We need to extract per-component transition F1 from our results.
- **What to verify:** Add per-component transition F1 to the `psr_transition_f1.py` output. Compare against STORM's per-component numbers. If our low-prevalence components (comp4, comp10) are close to STORM's while high-prevalence components show the gap, the bottleneck is different for easy vs. hard components. If all components show a uniform gap, the paradigm difference is systematic.

**Q46. B3's F1=0.883 uses a learned decoder with procedural knowledge. Our MonotonicDecoder is hand-crafted with hard-coded sequential order. What is the MonotonicDecoder's oracle upper bound?**

- **File/line:** `psr_transition.py` L79-182 — MonotonicDecoder is deterministic, rule-based, zero learned parameters. Its performance is bounded by the quality of the transition logits it receives.
- **Current answer:** The MonotonicDecoder is a rules engine, not a learned component. It can never find patterns in the data that aren't in the transition logits. A learned decoder (like B3's) could potentially extract higher-order structure (e.g., "when component 4 fires, component 10 often fires 5-10 frames later"). The oracle bound for a perfect decoder on our logits is: run the decoder with GT transition timing (infinite sigmoid at every GT transition, zero elsewhere). The resulting F1 is the upper bound on what any decoder could achieve with our backbone's features.
- **What to verify:** Feed oracle transition logits (perfectly timed spikes at GT transitions) into the decoder. The resulting F1 is the decoder ceiling. If it's >0.95, the decoder isn't the bottleneck. If it's <0.85, the decoder's hard constraints (sequential order, hysteresis) are limiting even perfect logits, and a learned decoder is necessary to beat STORM/B3.

**Q47. The null-delta analysis (comp4: +0.097, comp10: +0.093) shows signal beyond prevalence. What is STORM's comparable null-delta?**

- **File/line:** `psr_null_delta_table.md` — STORM doesn't report null-delta in its paper. This is a novel diagnostic we could contribute.
- **Current answer:** The null-delta analysis is not standard in the PSR literature. STORM and B3 report raw F1 without comparing to an always-positive baseline. If we compute null-delta from their published per-component F1 + prevalence, we can contextualize our numbers. If STORM's comp4 delta is +0.30 and ours is +0.097, the gap is 3:1 even after controlling for prevalence — which quantifies the paradigm gap in a prevalence-adjusted way.
- **What to verify:** Extract STORM and B3 per-component prevalence from their published tables (or estimate from their dataset split). Compute their null-delta using the same formula (F1 - 2p/(1+p)). Report a 3-way comparison table: (model, comp, F1, null_F1, delta). This is the fairest available comparison and doesn't require re-running their models.

**Q48. The in-flight training (PSR_HEAD_REPAIR=1 + KENDALL_FIXED_WEIGHTS=1) combines two changes. After convergence, how do we attribute improvement to each factor?**

- **File/line:** `run_psr_kendall_fixed.sh` — both env vars set to 1. `133_OPUS_COMPLETE_ANSWERS.md` PSR-3: "head repair before Kendall ablation" but the in-flight training applies both simultaneously.
- **Current answer:** The in-flight training bundles both fixes. If PSR F1 improves from 0.7499 to, say, 0.8000, we cannot attribute the +0.050 to head repair vs. fixed Kendall. A factorial ablation (head repair only, fixed Kendall only, both) would be needed. However, 133 PSR-3 states the evidence points at dead heads as the primary cause, not Kendall down-weighting (which is 4-8%). If the improvement is >+0.10, heads are the primary cause. If it's <+0.03, Kendall was the primary cause.
- **What to verify:** After the in-flight training finishes, if time permits, run two additional short trainings: (a) PSR_HEAD_REPAIR=1 only (no KENDALL_FIXED_WEIGHTS), (b) KENDALL_FIXED_WEIGHTS=1 only (no PSR_HEAD_REPAIR). Each needs only 10-15 epochs on the RTX 5060 Ti (~30-45 hours each). If no time, estimate: compare the improvement rate in the first 5 epochs of the in-flight training to the rate from epoch 18-23 of the original training. A faster early improvement suggests head repair dominance.

**Q49. STORM and B3 are evaluated on the full 38k validation set. Our 0.7499 is on 10k frames (psr_optimal_thr). Does the 10k-to-38k gap affect comparability?**

- **File/line:** `psr_optimal_thr/optimal_thresholds.json` L3: n_frames=10000. `full_eval_ep18_stream/metrics.json` L2: n_batches=38036.
- **Current answer:** The optimal thresholds were computed on 10k frames, but SOTA_STATUS.md reports the per-comp optimal F1 as 0.7499 (from the threshold optimization, which uses only 10k). The full 38k eval at global 0.10 gives macro F1=0.677 (from `full_eval_ep18_stream/metrics.json` L94). The per-comp optimal F1 on the full 38k has never been computed. If the 10k optimal thresholds perform worse on the full 38k (overfitting to the subset), the 0.7499 is an upper bound, not a representative number.
- **What to verify:** Apply the optimal thresholds (from `psr_optimal_thr/optimal_thresholds.json`) to all 38k frames and compute macro F1. If it's similar to 0.7499 (within ±0.01), the threshold is stable across frame subsets. If it's significantly lower (e.g., 0.70-0.72), the reported 0.7499 is an overestimate and the honest number is the full-eval figure.

**Q50. All PSR numbers reported use best.pth (frost epoch 18). After the in-flight training with head repair + fixed Kendall converges, what is the "combined improvement" over SOTA?**

- **File/line:** `SOTA_STATUS.md` L82-88 (current wins for epoch 18). The in-flight training (`run_psr_kendall_fixed.sh`) is expected to produce a new best checkpoint.
- **Current answer:** We have three PSR improvement mechanisms that have never been combined: (1) per-comp optimal thresholds (+0.028 over global 0.10), (2) head repair (+unknown, expected +0.03-0.08), (3) fixed Kendall weights (+unknown, expected +0.01-0.03). The combined potential is 0.7499 (current) + repair_delta + kendall_delta. If repair gives +0.05 and Kendall gives +0.02, macro F1 could reach 0.82 — still below STORM's 0.901 but "approaching competitive" by the PW-3 rubric. If combined F1 exceeds 0.85, the paradigm gap narrative changes from "fundamentally different" to "bridgeable with architectural improvements."
- **What to verify:** After the in-flight training, run the full evaluation pipeline: per-comp optimal thresholds → transition F1 → null-delta → LOO-CV. Compare each metric to both epoch 18 values and the STORM/B3 published numbers. The combined delta tells us whether our approach can scale to SOTA or requires a paradigm shift.

---

## §6. Adversarial Review (Built-in Debate)

**Debate 1 — The null-delta analysis proves there is learned signal on low-prevalence components (comp4: +0.097, comp10: +0.093). But is this signal actually useful for transition detection, or is it just improved calibration?**

- **Prosecution:** The null-delta F1 improvement on comp4 (0.249 → 0.346) comes from thresholding: at optimal threshold 0.80, the model predicts "negative" for 80% of frames and "positive" for 20%. Since comp4 has GT prevalence 0.142, the model's optimal strategy is to predict "negative" for ~86% of frames (precision: 0.11 at global 0.10 → 0.35 at optimal 0.80). This is threshold calibration, not transition detection. The model learned "comp4 is rare" and adjusted its output accordingly, but it doesn't know WHEN comp4 transitions occur.
- **Defense:** The null-delta compares against an always-positive predictor, which already "knows" prevalence (it predicts positive on every frame). The delta +0.097 means the model beats the prevalence-aware null by 39% relative. This requires per-frame discrimination: on some frames the model outputs higher sigmoid, on others lower. Even if the delta is partly calibration, the correlation with frame-level features proves the backbone sees comp4-relevant information.
- **Evidence needed:** Compute the mutual information between ConvNeXt features and comp4 transition timing. If MI > 0, frame-level features carry transition information. If MI ≈ 0, the delta is purely calibration.
- **Verdict threshold:** If the head-repair training shows that per-component F1 for comp4 jumps from 0.346 to >0.50 (+0.15), the defense wins: the old heads were masking real signal. If comp4 F1 stays at 0.35-0.40, the prosecution wins: the old heads were already doing calibration, and repair can't find transitions that aren't in the features.

**Debate 2 — The LOO-CV improvement +0.0358 ± 0.0216 proves threshold generalization across recordings. But the LOO-CV uses the same 16 recordings that the model was trained on (some in train, some in val). Does this confound the result?**

- **Prosecution:** The 16 recordings used in LOO-CV come from the dataset's training split (the model was trained on them). LOO-CV measures generalizability across recordings but NOT from train to unseen recordings. The +0.0358 improvement is within-training-distribution generalization, which is weaker than held-out dataset generalization. A proper test would compute thresholds on the 16 training recordings and evaluate on a held-out set of recordings the model has never seen (if such a set exists).
- **Defense:** LOO-CV tests whether per-component thresholds are stable across recordings within the dataset. The +0.0358 with std 60% of mean shows some recordings benefit significantly while others don't — but no recording shows negative improvement (degradation). The thresholds don't overfit to a specific recording's noise. Cross-dataset generalization is a separate concern (and not applicable since only IndustReal has PSR labels).
- **Evidence needed:** Partition the 16 recordings into train/val/test by recording. Train thresholds on train-set recordings, evaluate on val-set recordings. If the improvement on val-set recordings is similar to the LOO mean (+0.0358), the generalization is real. If lower (e.g., +0.01), the improvement is partially recording-specific.
- **Verdict threshold:** Val-set improvement > +0.02 → defense wins (thresholds generalize). Val-set improvement < +0.01 → prosecution wins (thresholds are in-distribution noise).

**Debate 3 — The head repair (LeakyReLU + bias=0 + Xavier init) might "fix" the dead heads, but if the dead heads were already near-optimal for the task (class prior fitting), the repair could make things WORSE by enabling overfitting.**

- **Prosecution:** The old heads achieved macro F1=0.7499 by fitting prevalence priors and outputting near-constant sigmoid values. This is the optimal strategy for a limited-capacity head on a noisy signal. The repaired heads with Xavier init (larger weights) and bias=0 (no prior) will learn more aggressively and may overfit to the 25% training batches that have PSR signal. After repair, train F1 may reach 0.95 while val F1 stays at 0.75 — classic overfitting without generalization benefit.
- **Defense:** The old heads were dead — ReLU gating means negative pre-activations produce exactly 0, and the bias=-1.0 means the final sigmoid saturates at 0.27 regardless of input. No learning was happening for 70% of training steps (PSR-7). The repair doesn't just "enable" learning — it enables ANY learning. Even if the first repaired checkpoint shows overfitting, dropout (0.1) and training regularization (monotonicity loss) exist to control it.
- **Evidence needed:** Compare train vs val PSR F1 for the first 10 epochs of repaired training. If train quickly exceeds val by >0.10 AUC, overfitting is happening. Add weight decay or increase dropout to 0.3.
- **Verdict threshold:** Val PSR F1 after repair > 0.75 (baseline) → defense wins (repair didn't hurt). Val PSR F1 < 0.72 → prosecution wins (repair disrupted optimal prior fitting for marginal gain).

**Debate 4 — The Gaussian-smeared transition loss (sigma=3) is fundamentally the wrong loss for transition detection because it encourages fuzzy boundaries, not sharp event predictions.**

- **Prosecution:** A transition event is a single frame index where the component changes. The Gaussian-smeared loss with sigma=3 spreads the target over ±9 frames, making the loss 0.61 * max at 3 frames from the true transition. The model is trained to output confidence proportional to proximity to the transition, which (a) dilutes the event detection signal, (b) penalizes sharp predictions (sigmoid output 0.9 at the exact frame is treated as "overconfident" compared to the Gaussian peak of 1.0), and (c) makes the decoder's job harder by providing soft transition probabilities instead of hard event indicators.
- **Defense:** The Gaussian smearing is intentional: it converts a hard binary matching problem (which is non-differentiable) into a differentiable regression problem. The decoder's hysteresis (Q48) then determines the exact transition frame. Additionally, the tolerance-based evaluation (±3 frames) means that being within 3 frames is as good as exact — the Gaussian width matches the evaluation protocol. A hard event loss (1 at transition frame, 0 elsewhere) produces near-zero gradients for 99.9% of frames and never trains the low-prevalence components.
- **Evidence needed:** After head repair stabilizes, compare models trained with (a) Gaussian sigma=3, (b) Gaussian sigma=1, (c) Hard BCE on transition frame only. Evaluate using transition F1 at tolerance=3. If sigma=3 is clearly best, defense wins. If hard BCE is comparable or better, smearing is unnecessary.
- **Verdict threshold:** Transition F1 of sigma=3 > hard BCE by >0.02 → defense wins. Hard BCE >= sigma=3 → prosecution wins (simpler loss is better).

**Debate 5 — The POS paradox (null models achieve POS≈0.999) is presented as a new finding, but it's a known property of ordered-pair metrics on sparse-event sequences. Is this really a "discovery" or an expected mathematical fact?**

- **Prosecution:** POS is defined as the fraction of frame-pairs with correct ordering sign. For a monotonic sequence with T frames and N << T transitions, any constant predictor achieves POS = 1 - N/T. With T=1000 and N=3, POS(constant) = 0.997. This is not an "artifact" — it's basic algebra. The finding that null models achieve POS≈0.999 is equivalent to "the sky is blue." Presenting it as a discovery in the paper wastes reviewer goodwill. Reduce to one sentence: "POS is upper-bounded by 1 - transitions_ratio and is therefore uninformative for sparse event data."
- **Defense:** The fact that POS has a structural upper bound that 99% of models (including perfect ones) can't exceed is NOT obvious to the average ML reader. The null-model experiment demonstrated this empirically with concrete numbers, which is worth a paragraph in §5.2.1. The POS@tolerance variant (proposed in 133) shows that the metric CAN be informative with the right evaluation window. The "discovery" isn't the algebra — it's that the metric community uses for PSR evaluation is structurally incapable of distinguishing good models from trivial ones on this task.
- **Evidence needed:** Check the STORM and B3 papers: do they report POS? If they do and claim SOTA POS, then the prosecution's "obvious" argument fails — the community doesn't know about the artifact. If they don't report POS, the metric is already being abandoned and our "discovery" is late.
- **Verdict threshold:** STORM/B3 report POS → defense wins (important to disclose the artifact). STORM/B3 don't report POS → prosecution partially wins (artifact is known, our contribution is the null-model quantification, not the discovery itself).

---

## §7. Open Decisions for Opus

The following decisions depend on evidence that will arrive during or after the in-flight training. Each is gated on a specific checkpoint.

**D1. Head repair results timeline.** The in-flight training is ~3h/epoch. At epoch 26 start + 10 epochs = ~30 hours for initial convergence signal. The first actionable checkpoint will be at ~epoch 30 (~12 hours from now). Actionable decision: if PSR macro F1 at global threshold 0.10 increases from 0.6773 (current) to >0.70 by epoch 30, continue training to epoch 50. If no improvement by epoch 30 (+2 F1 epochs), stop and diagnose transformer health (Q4).

**D2. POS@tolerance implementation priority.** POS@tolerance is the salvageable variant. Estimated implementation time: 2 hours (modify `psr_transition_f1.py` to accept `--pos-tolerance k` argument, add nested null-model loop). Decision: implement during the first 12 hours of in-flight training (while waiting for epoch 30 results), before writing the PSR section of the paper.

**D3. Full 38k per-comp optimal F1 computation.** The current 0.7499 is on 10k frames. Full 38k eval at per-comp optimal thresholds may yield a different number. Estimated compute: 30 min on RTX 3060 (eval of existing cached logits). Decision: run immediately on the RTX 3060 while the RTX 5060 Ti trains. This is zero-cost (doesn't interfere with training).

**D4. D4 per-video breakdown.** The aggregate F1=0.347 may hide per-video variation. Estimated compute: 1 hour on RTX 3060 (re-run D4 eval with per-video output). Decision: run immediately on RTX 3060 alongside D3.

**D5. ConvNeXt→decoder comparison (Q38).** This is the apples-to-apples decoder comparison. Estimated compute: 2 hours (modify eval pipeline to feed ConvNeXt logits through MonotonicDecoder with Q48 sweep). Decision: run after D3/D4 are complete, while waiting for in-flight training epoch 30.

**D6. Per-component train/val prevalence comparison (Q27).** This checks whether the null-delta is inflated by train/val prevalence shift. Estimated compute: 10 minutes (add prevalence computation to data loader). Decision: run immediately on CPU.

**D7. Transformer hidden state variance diagnostic (Q4).** One-line debug print in the forward pass of the in-flight training. Estimated effort: 5 minutes to add `print(encoded.std().item())` to the training loop. Decision: add to the training code immediately (next epoch start) by editing the checkpoint and resuming.

**D8. MonotonicDecoder oracle bound (Q46).** Perfect logit feeding. Estimated compute: 2 hours (modify eval to create oracle logits with spikes at GT transitions). Decision: run after D5, as a secondary diagnostic to bound potential improvement.

**D9. PSR-focused fine-tune at end of training (Q9).** After the combined training converges, a short fine-tune with `PSR_SEQ_EVERY_N_BATCHES=1`. Estimated compute: 5-10 epochs on RTX 5060 Ti (~15-30 hours). Decision: gate on post-repair PSR F1. If >0.75 (competitive), run the fine-tune as a "bonus" before the freeze. If <0.72, skip and investigate.

**D10. Factorial head repair ablation (Q48).** After the combined training finishes, run: PSR_HEAD_REPAIR=1 only and KENDALL_FIXED_WEIGHTS=1 only. Estimated compute: 20-30 epochs each (~60-90 hours total on RTX 5060 Ti). Decision: this is high-effort. Run only if the combined training shows >+0.05 improvement AND the paper deadline allows 3+ days of ablation. Otherwise, accept the bundled intervention and note the confound in §5.4.

---

**End of 135. Prepared for Opus consumption alongside 136 (answers). All file references are relative to `src/runs/rf_stages/checkpoints/` unless otherwise noted.**
