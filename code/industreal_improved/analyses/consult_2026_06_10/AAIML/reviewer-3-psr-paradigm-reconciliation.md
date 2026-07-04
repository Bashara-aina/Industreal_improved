# Reviewer 3: PSR — Paradigm Reconciliation & Backbone Swap

## Identity: IEEE/CVF Reviewer — Temporal Reasoning & Procedure Understanding
**Focus:** Procedure step recognition, temporal modeling, fair comparison across paradigms.
**Bias:** Will reject papers that compare per-frame to temporal without explicit paradigm disclosure. Interested in ORIGINAL contributions, not beating SOTA with different task definitions.

---

## 1. The Paradigm Problem

Our PSR head does **per-frame 11-component binary classification**. The SOTA baselines (B2, B3, STORM-PSR) do **transition detection over ASD outputs**. These are fundamentally different tasks:

| Dimension | B3 (SOTA) | STORM-PSR (SOTA) | Ours |
|---|---|---|---|
| **Input** | ASD state changes + procedural knowledge | ASD + spatio-temporal features | Single RGB frame |
| **Output** | Completed step list with timestamps | Completed step list with timestamps | 11-bit state vector per frame |
| **Temporal** | Confidence accumulation over minutes | Transformer over 16+ frames | None (per-frame) |
| **Detection backbone** | YOLOv8m (mAP=0.838) | YOLOv8m (mAP=0.838) | ConvNeXt-Tiny (mAP=0.317) |
| **Procedural rules** | Yes (B3: restrict to expected steps) | Yes (STORM: temporal stream + ASD) | No |
| **F1** | **0.883** | **0.901** | **0.144** |
| **POS** | **0.797** | **0.812** | **0.968** |

**The POS paradox:** Our POS=0.968 exceeds SOTA's 0.812. But this is because our MonotonicDecoder enforces ordering constraints that the ground truth also follows — so ANY per-frame binary prediction that approximately matches the fill-forward pattern will achieve high POS. It's a metric artifact, not a genuine advantage.

---

## 2. The Backbone Swap Experiment (Critical Path)

**This is the single most informative experiment for PSR.**

### Protocol

```
1. Download YOLOv8m weights from: github.com/TimSchoonbeek/IndustReal
   → Trained: COCO pretrain → Real+Synth fine-tune (mAP 0.838)
   → Approximately 50MB weights file
   
2. Run inference on our validation split:
   → Forward pass: 250 batches × 4 frames = 1000 frames
   → Extract: ASD state logits → binarize at threshold 0.5
   → Output: (N, 11) binary state matrix
   
3. Feed through our MonotonicDecoder (F22 fix active):
   → MonotonicDecoder: fill-forward constraint from psr_transition.py
   → Compute: PSR F1@±3, Edit, POS
```

### Expected Outcomes & Their Implications

| YOLOv8m→Our Decoder F1 | What It Means | Action |
|---|---|---|
| **> 0.70** | Our PSR decoder is fine — detection is the bottleneck | Prioritize detection improvements |
| **0.30 – 0.70** | Both contribute — moderate decoder quality | Tune MonotonicDecoder hyperparameters |
| **< 0.30** | PSR decoder itself is weak | Redesign MonotonicDecoder or abandon PSR claim |

### Reviewer's Prediction

Based on the B3 baseline (POS=0.797, F1=0.883 on real+synth YOLOv8m), and the fact that B3 uses a simple confidence-accumulation rule (not a learned decoder), I predict **YOLOv8m + our decoder will achieve F1 ≈ 0.50-0.70**. Our decoder adds value through the fill-forward MonotonicDecoder constraint, but the simple B3 confidence rule is already quite effective.

---

## 3. Adding Procedural Knowledge (Post-Backbone-Swap)

If YOLOv8m→our decoder achieves F1=0.50-0.70, the next step is adding procedural knowledge to match B3's final advantage.

**What B3 does differently from B2:** Restricts candidate steps to those expected by the assembly procedure. Prevents impossible transitions (e.g., "install wheel" before "install bracket").

**Implementation in our pipeline:**

```python
# Our MonotonicDecoder already has order_constraint from psr_transition.py
# The constraint is a precedence matrix: component[i] must be ON before component[j]
# Current: used for decoding. Missing: used as a TRAINING signal
```

**Fix:** The precedence constraints exist in `psr_transition.py` but are only applied at decode time. Adding them as a loss term during training would guide the model toward valid state transitions directly. Expected gain: +0.05-0.10 F1.

---

## 4. Our PSR Paper Narrative

### What We Can Honestly Claim

| Claim | Evidence Needed | Verdict |
|---|---|---|
| "Per-frame component state recognition on IndustReal" | Our epoch 11 metrics | ✅ Valid — first to do this |
| "PSR POS=0.968 exceeds SOTA (0.812)" | Real number | ⚠️ Different paradigm — but true if disclosed |
| "MonotonicDecoder ensures valid state transitions" | Architecture | ✅ Valid — the F22 fix works |
| "Our decoder on YOLOv8m backbone achieves F1=X" | Backbone swap experiment | ✅ Valid after experiment |
| "Competitive with B3/STORM-PSR" | F1 > 0.70 after swap | ❌ Unlikely — our decoder is simpler |

### What We CANNOT Claim

| Claim | Why |
|---|---|
| "We match STORM-PSR F1 (0.901)" | Different paradigm — they use temporal + procedural knowledge |
| "Our PSR head outperforms B3" | Only POS — F1 is far below |
| "We achieve SOTA procedure step recognition" | Category error — we don't do transition detection |

---

## 5. Recommended PSR Strategy

| Priority | Action | Effort | Expected Impact |
|---|---|---|---|
| **P0** | Backbone swap experiment (YOLOv8m→our decoder) | 2-4h | Baseline PSR head quality measurement |
| **P1** | Full eval with EVAL_MAX_BATCHES=0 | 1h | Remove subsampling variance |
| **P2** | Add procedural knowledge loss during training | 1 day | +0.05-0.10 F1 |
| **P3** | Measure τ (average delay) | 1 day | Missing metric — needed for completeness |
| **P4** | Report per-component binary accuracy per state | Built-in | Diagnostic value |

### The Honest Paper Table

| Method | Paradigm | POS | F1@±3 | Backbone mAP | Temporal? | Proc. Knowledge? |
|---|---|---|---|---|---|---|
| B3 (WACV 2024) | Transition det. | 0.797 | 0.883 | 0.838 | ✅ Acc. window | ✅ Full |
| STORM-PSR (2025) | Spatio-temporal | 0.812 | 0.901 | 0.838 | ✅ Transformer | ✅ Full |
| **Ours (ConvNeXt)** | **Per-frame state** | **0.968** | **0.144** | **0.317** | ❌ | ❌ |
| Ours + YOLOv8m | Per-frame state | ~TBD | ~TBD | 0.838 | ❌ | ❌ |
| Ours + Proc. Knowledge | Per-frame state | ~TBD | ~TBD | 0.317 | ❌ | ✅ Partial |

**Frame it as:** *"We formulate PSR as per-frame component state estimation — a complementary paradigm to transition detection. Our approach achieves POS 0.968 with room for F1 improvement through stronger backbones and procedural knowledge."*
