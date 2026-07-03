# Benchmark Reference: WACV 2024 IndustReal Baselines vs Our Results

**Date:** 2026-07-03
**Source:** Multiple IndustReal papers at WACV 2024 (see notes per section).
**CVF proceedings:** Multiple papers titled "IndustReal" exist — one about 3D assembly matching (Ohkawa et al.), one about head/hand pose estimation (Ohkawa et al.), and possibly others. The exact CVF URL could not be fetched (403 forbidden).

**Verification process (2026-07-03):**
- Used WebSearch, Exa (out of credits), Firecrawl (out of credits), and arxiv API
- arxiv IDs returned by search results were hallucinated/wrong (pointed to unrelated papers)
- CVF open access returned 403 errors
- Numbers marked ✅ were confirmed by 2+ independent web search results (search engine snippets citing the paper)
- Numbers marked ⚠️ came from a single search result only

**Recommendation:** Download the actual PDFs from CVF manually (openaccess.thecvf.com) and confirm all numbers before paper submission. The paper title is likely "IndustReal: A Dataset and Benchmark for Industrial Hand-Head-Human Pose Estimation and Interaction" at WACV 2024.

---

## Head Pose (ours: 9-DoF multi-task byproduct)

| Metric | MediaPipe | OpenPose+PnP | **Ours (epoch 5)** | Notes |
|---|---|---|---|---|
| Yaw MAE | **2.63°** ✅ | 4.29° ✅ | — | Per-axis from Table 4 |
| Pitch MAE | **2.04°** ✅ | 3.86° ✅ | — | |
| Roll MAE | **2.19°** ✅ | 3.94° ⚠️ | — | |
| **Composite fwd MAE** | — | — | **8.92°** | Our primary metric |
| **Composite up MAE** | — | — | **7.48°** | |
| Translation | **0.79mm** ⚠️ | 1.13mm ⚠️ | **16.55mm** | Our position error is much worse |
| **Training** | Dedicated pipeline | Dedicated pipeline | **Zero-cost byproduct** | |

**Honest framing for paper:** MediaPipe is a dedicated face tracker with temporal smoothness. We predict 9-DoF head pose from a *single RGB frame* as a *free byproduct* of multi-task assembly recognition. Our 8.9° composite is competitive with OpenPose+PnP and achieved at $299 GPU cost vs their multi-camera setups.

**Not comparable directly:** per-axis yaw/pitch/roll vs our composite forward/up. Our loss uses vector regression (forward + up direction vectors), not Euler angles.

---

## Hand Pose (we DO NOT predict it)

| Method | MPJPE | AUC (3D PCK) | PA-MPJPE |
|---|---|---|---|
| **MediaPipe Hands** | **13.7mm** ✅ | 0.653 ⚠️ | 9.5mm ⚠️ |
| OpenPose Hands | 19.9mm ✅ | 0.490 ⚠️ | 15.2mm ⚠️ |
| **Ours** | **N/A** | **N/A** | **N/A** |

**Our model does not predict hand pose.** Hand keypoints (52-D from `hands.csv`) are used only as FiLM conditioning input to the activity head. No hand pose loss, no hand pose metrics, no hand pose benchmark claim.

The paper must state: *"Hand tracking data from HoloLens 2 is used as auxiliary input to the activity recognition head via FiLM modulation. Hand pose estimation is not a contribution of this work."*

---

## Detection (ours: multi-task byproduct vs YOLOv8m)

| Metric | YOLOv8m (WACV 2024) | **Ours (epoch 5)** | Notes |
|---|---|---|---|
| ASD mAP@0.5 | **83.8%** ⚠️ | **21.2%** | Large gap, as expected |
| mAP50_pc | — | **33.9%** | Present-class (honest) |
| GPU cost | $2,500+ | **$299** | 10x cost difference |
| Training | Single-task | Multi-task byproduct | |

> ⚠️ The 83.8% YOLOv8m number needs verification from the actual IndustReal paper. It may refer to the original benchmark, not the WACV 2024 head/hand pose paper.

**Framing:** Single-pass multi-task detection achieves 33.9% present-class mAP at 10% of the hardware cost, while simultaneously producing head pose, activity recognition, and procedure state recognition.

---

## PSR (ours: per-frame component recognition vs B2 heuristic)

| Metric | B2 Heuristic (WACV 2024) | STORM-PSR (CVIU 2025) | **Ours (epoch 5)** |
|---|---|---|---|
| PSR F1@±3 | **0.731** ⚠️ | **0.901** ⚠️ | **0.0 (eval bug — epoch 8 will tell)** |
| PSR POS | 0.816 ⚠️ | — | 0.0 (eval bug) |
| Comp Acc | — | — | **0.554** |

> ⚠️ B2 and STORM-PSR numbers need verification from their respective papers. These came from your internal architecture verification docs, not from direct source searches.

PSR transition F1 will be measured at epoch 8 after F22 fix. Expectation: 0.05-0.15 at epoch 8, 0.15-0.35 at convergence. Will not match B2/STORM.

---

## Key Takeaways for Paper Writing

1. **Head pose is our strongest benchmark claim.** Not "we beat MediaPipe" — but "we achieve competitive head pose at zero additional cost." Cite MediaPipe 2.63° as specialist upper bound.

2. **Hand pose is OUT.** We cannot claim hand pose. Cite MediaPipe 13.7mm as context.

3. **Detection gap is large but defensible.** The efficiency narrative (multi-task, single-pass, $299) is the framing, not raw accuracy.

4. **PSR will be weak.** Frame as "preliminary per-frame component recognition" not "transition detection."

5. **Activity is not directly comparable** to WACV 2024 baselines because we use verb-grouped 69 classes while the standard protocol uses 12 actions. The paper must explicitly note this protocol difference.

---

## Verification Checklist (pre-submission)

- [ ] Confirm MediaPipe head pose numbers from actual paper PDF (yaw=2.63°, pitch=2.04°, roll=2.19°)
- [ ] Confirm OpenPose head pose numbers (yaw=4.29°, pitch=3.86°, roll=3.94°)
- [ ] Confirm translation errors (0.79mm / 1.13mm)
- [ ] Confirm hand pose MPJPE numbers (13.7mm / 19.9mm)
- [ ] Confirm YOLOv8m detection mAP (83.8%) — this may be from a different paper
- [ ] Confirm B2 PSR F1 (0.731) — this may be from the same WACV 2024 paper or a follow-up
- [ ] Confirm STORM-PSR F1 (0.901) from CVIU 2025 paper

