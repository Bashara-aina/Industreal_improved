# AAIML 2027 — Honest Weaknesses and Preemptive Rebuttals
## Post-Review Revision: Every weakness acknowledged, quantified, and addressed

---

## 1. Detection Performance

**Honest**: mAP50_pc = 0.34 vs YOLOv8m 0.838. FPR = 0.12 at threshold 0.5.

**Root cause**: 24 ASD classes = 11-bit binary state encodings. 70% errors are 1-bit-adjacent (coarse state correct, single component confused). No 260K synthetic images.

**Rebuttal**: At threshold 0.7, FPR = 0.04 (usable for high-precision). Bootstrap CI [0.31, 0.37] stable. Human inter-annotator agreement on 11-bit states is not 100% either. Three-seed for camera-ready.

**Done in paper**: ✅ Controlled ablation equalizes gradient updates (not confounded). ✅ SOTA comparison explains data differences. ✅ Confusion analysis quantifies 1-bit adjacency.

---

## 2. Ablation A: Unfair Comparison

**Honest**: Reviewers from R1/R2 noted multi-task model gets fewer detection-specific updates than single-task.

**Fix**: Both models receive identical detection iterations (30 epochs, same backbone, same LR, same data). Δ = −0.03 is STRUCTURAL interference, not underfitting.

**Done in paper**: ✅ Explicit training details. ✅ Naive joint training baseline (collapsed). ✅ Δ reported with CI.

---

## 3. No MTL Baselines

**Honest**: No PCGrad, CAGrad, or naive joint training as baselines (except naive joint, reported as collapsed). This is the #1 gap for an MTL paper.

**Mitigation**: Acknowledge explicitly in Limitations. Commit to adding 2 MTL baselines in extended version. The paper's primary MTL claim is "feasibility with minimal interference" not "beating specialized MTL methods."

---

## 4. Activity Recognition 18.3%

**Honest**: Wrong in >80% of frames at per-frame level.

**Context**: 74 classes, 14× chance baseline (1.35%). Top-5 = 41.2%. Per-step accuracy with temporal smoothing would be higher (future work). FiLM improves by 2.2pp (p = 0.032).

---

## 5. Single Dataset

**Honest**: IndustReal only. Commitment to IKEA ASM/IndEgo extended version.

---

## 6. Blockchain Motivation

**Honest reviewed weakness**: Blockchain is decorative. Oracle problem unsolved.

**Response**: We now frame blockchain as "feasibility demonstration," not "security guarantee." Acknowledge oracle problem explicitly. Add "verifiability without trust would require threshold signatures or decentralized verification" — honest about limitations.

---

## 7. Single Seed

Promised three-seed for camera-ready. Bootstrap CI on detection mitigates somewhat. This is standard for conference submissions.

---

## 8. Pilot (N=20, p=0.04)

NASA-TLX does not survive Bonferroni correction (α = 0.0125). Reported as pilot effect, not definitive. SUS score (72.3) and zero opt-out rate are the stronger results.
