# 55 — Ablation Design, Load-Bearing Code Verification, and Risk Register (Grounded)

> **Generated:** 2026-06-22 (Opus). Companion to `51`–`54`. Answers docs 48 (Q6) and 50 (§7, §11, §13, §14.2). The "what we don't know" list in 50 §14.2 is resolved here against source.

---

## 1. Ablation A — single-task vs multi-task (the contribution)

**Claim under test:** sharing one backbone across tasks does not catastrophically degrade any single task vs a specialist on the *same* backbone.

**Design (minimum viable, reviewer-credible):**
- **Specialist:** `recovery_det_only` preset (`config.py:932`) — `train_det=True, train_act=False, train_psr=False, train_head_pose=True`. Same ConvNeXt-T + FPN, detection (+ cheap head-pose) only.
- **Unified:** the rf2 run (det + pose + head-pose multi-task).
- **Comparison metric:** `det_mAP50_pc` (present-class), identical eval protocol, identical backbone init.
- **Result cell:** `Δ = mAP50_pc(unified) − mAP50_pc(specialist)`.

**Why this is enough:** the single most load-bearing comparison is detection (the hardest, most interference-prone task per `52`). One clean det single-vs-multi row + the head-pose single-vs-multi row covers the claim. Five-head full Ablation A (5 specialist runs) is **not** required and **not** affordable on one GPU (`50` Q7.2) — say so in the paper.

**Interpretation guide:** small |Δ| → thesis supported. Large negative Δ → interference is real → *still a publishable finding* ("unifying egocentric assembly tasks trades fine-grained detection accuracy for efficiency"). There is no outcome here that kills the paper; both directions are results.

---

## 2. Ablation B — FiLM conditioning ladder

**Claim under test:** cross-task FiLM (PoseFiLM, HeadPoseFiLM) improves the modulated task (activity).

**Design:** four configs — no-FiLM / PoseFiLM-only / HeadPoseFiLM-only / both — same data, same schedule, measure activity Top-1.
- **Architecture is implemented:** `PoseFiLM` and `HeadPoseFiLMModule` exist (`model.py:684, 721-791`), modulating C5 at 768ch.
- **Blocking dependency:** activity must train first (Phase 3). If activity stays < ~10% Top-1, FiLM has nothing to modulate → report B inconclusive (`53` §7.4).
- **Cheap path:** if the embedding cache works (§3), run all four FiLM configs on cached features at high epochs/hour. If not, run on-the-fly (slower, fewer configs).

---

## 3. Load-bearing code verification (resolving 50 §14.2's 7 unknowns)

| # (50 §14.2) | Unknown | Verified status | Evidence |
|---|---|---|---|
| 1 | True detection ceiling with all fixes | Not all-at-once tested, but mechanism is fine-grained state separability (`52`), not a fixable bug | `config.py:180-205`, DET_PROBE |
| 2 | Can PSR train at all? | **Never has** (`psr_f1_at_t=0.0` everywhere). Settle via go/no-go (`54` Ph2) | `eval_metrics.json`, `metrics.jsonl` |
| 3 | Will activity produce signal on 35%×15ep? | Untested; expect 10–30% Top-1; at init it collapses to one class | `metrics.jsonl` ep0 confusion matrix |
| 4 | Does IKEA eval code work? | **Unverified / likely not wired** — no IKEA preset found in `config.py` preset scan | preset grep (only IndustReal presets) |
| 5 | Does `embedding_cache.py` produce correct cached features? | **Untested end-to-end.** Only an `__main__`; no cache dir; no smoke test | `embedding_cache.py:518`; no `src/runs/cache*` |
| 6 | VideoMAE V2 fits 12GB? | **Feasible.** Uses `MCG-NJU/videomae-small` (ViT-S/16, ~22M frozen, 384-D) with a graceful fallback if the checkpoint is missing | `model.py:797-875` |
| 7 | Was the ViT/FeatureBank activity head ever tested end-to-end? | Forward path exists; only ever run at epoch-0 (collapsed output) — **never validated trained** | `model.py` activity path; `metrics.jsonl` ep0 |

**Net:** the two genuine code risks are **#5 (embedding cache untested)** and **#4 (IKEA pipeline)**. Both are *avoidable*: Ablation A has a non-cache path (`recovery_det_only`), and IKEA is the first thing to cut (`53` §7). VideoMAE (#6) is fine. PSR (#2) and activity (#3, #7) are *training* unknowns, resolved by the Phase-2/3 runs, not code blockers.

---

## 4. Evaluation-protocol correctness (answering 50 §11.2)

The eval harness defines and labels each protocol — verified present in `eval_metrics.json` `_*_protocol` keys:

| Task | Protocol (as coded) | Matches baseline? |
|---|---|---|
| Detection AP | `coco` (`_det_ap_protocol`), plus present-class + all-frames variants | ✓ standard; report which frames (annotated-only vs all) explicitly |
| Activity | `clip_level_majority_vote`; **note: baseline MViTv2 is RGB+VL+stereo multi-modal, POPW is RGB-only** (`_ar_baseline_protocol`) | ⚠ modality-not-model — state this in the table caption |
| PSR F1 | `symmetric_bidirectional_greedy_per_stepid` (`_psr_f1_at_t_protocol`) | ✓ temporal-window matching; report `f1_at_t`, **not** `overall_f1` |
| PSR edit | `normalized_damerau_levenshtein_osa` | ✓ |
| PSR POS | `runs_based_adjacent_pairs_maxpos_ordering` | ✓ |
| Head pose | 9-DoF, unit-vectors → angular; watch the `non_unit_vectors` fallback that reports 0.0° | ⚠ ensure `head_pose_status == unit_vectors_ok` before trusting MAE |

**Two caption-level honesty flags:** (a) activity comparison is **modality-not-model** — POPW RGB-only vs multi-modal MViTv2; (b) **never report `psr_overall_f1`** as the PSR result — it is inflated by constant components (`comp0=1.0`); the real number is `psr_f1_at_t`.

---

## 5. Risk register (grounded mitigations)

| Risk | Likelihood | Grounded mitigation |
|---|---|---|
| **R1** rf2 finishes < 0.40 pc | Certain (~0.30–0.38) | **Non-issue.** Gate is 0.28 (passed). Paper doesn't need 0.40 (`53`). Report present-class + reframe. |
| **R2** rf3/activity crashes (LDAM/ViT/OOM) | Medium | Fix DRW schedule (`54` Ph3); GPU mem is only 1.34/12GB headroom is large; fallback = simpler CE + class weights. |
| **R3** PSR never trains | High | Go/no-go (`54` Ph2) → drop to negative result. Paper is 4-head, still complete (`53` §7.3). |
| **R4** GPU/process dies | Low-Medium | `crash_recovery.pth` mechanism exists (GUIDE 6); checkpoints on disk; Run-3 already proved restart works. |
| **R5** IKEA pipeline broken | High | **Cut IKEA** (`53` §7.1). IndustReal-only egocentric paper. Removes the dependency entirely. |
| **R6** Detection gap dominates narrative | Medium | Lead with efficiency + interference + head pose; detection second, reframed as state-discrimination (`53` §3). |
| **R7** Single-seed only | Certain | Single seed for submission; 3-seed promised for camera-ready (`50` R7 — standard). |
| **R8** Novelty challenge | Medium | Novelty = the *interference study* + FiLM ladder + commodity-hardware efficiency, not the backbone. Frame contribution as empirical, not architectural. |
| **R9** Embedding cache produces wrong features | Medium | Avoid the dependency: Ablation A via `recovery_det_only` (no cache). Only use cache for FiLM-on-activity if it passes a shape/sanity check first. |

---

## 6. The meta-answer (50 Q14.3)

**Is the plan "run in the right order, fill numbers, write honest, submit"? Yes — with one correction.** The fundamental problem the team was not seeing is **not** the detection ceiling; it is that **all compute went to the least load-bearing task (detection) and none to the ablations that *are* the thesis.** The grounded facts make the fix concrete and immediate:

1. Detection already clears its gate — **stop optimizing it** (`51` F1, `52`).
2. The thesis (no-interference, FiLM) has **zero evidence** — run Ablation A this week via an *already-working preset* (`recovery_det_only`), no untested cache required (`55` §1).
3. Activity is **unblocked today** — advance to rf3 (`54` Ph3).
4. PSR gets a 1-hour verdict (`54` Ph2); efficiency is free (`54` Ph0).

There is no hidden blocker that makes this impossible. The blockers were a **misread gate** and a **misallocated week**, both now corrected. Execute `54` in order and the paper assembles itself from numbers the harness already knows how to produce (`51` F5).

---

*End of the 51–55 set. Start with `54` Phase 0 — it costs no GPU and fills real tables today.*
