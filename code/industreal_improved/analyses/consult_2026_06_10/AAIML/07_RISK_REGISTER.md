# AAIML 2027 -- Risk Register

**Paper**: POPW: A Multi-Task Deep Learning Framework for Assembly Verification
**Last Updated**: 2026-06-30

---

## Risk Grading

| Grade | Meaning | Response |
|-------|---------|----------|
| CRITICAL | Would cause rejection | Must mitigate before submission |
| HIGH | Could lead to rejection or major revision | Active mitigation plan |
| MEDIUM | Reviewer concern, manageable | Preempt in paper |
| LOW | Minor weakness, acknowledge | Accept |

---

## Technical Risks

### R1: Detection performance too low for acceptance (CRITICAL)

**Description**: Present-class mAP50 of 0.34 (standard 0.22) vs YOLOv8m at 0.838. Reviewers may conclude the system is not practically usable regardless of cost savings.

**Probability**: 60-70%

**Impact**: Rejection or mandatory major revision. Reviewers focused on empirical rigor will flag this.

**Signals**: 72% of errors are detector-driven. At 0.12 FPR (threshold 0.5), the system triggers false step completions every ~8 frames.

**Mitigations**:
- Stronger framing on fine-grained state discrimination: 70% of errors are 1-bit Hamming-adjacent (coarse state correct, single component confused)
- Operating-point analysis: show precision-recall curves at multiple thresholds, demonstrate usable operating range
- Compare to human inter-annotator agreement on the same task (if available)
- Note that no synthetic pre-training was used (YOLOv8m uses 260K synthetic images)
- Add analysis: "At threshold 0.7, FPR drops to 0.04 while recall drops to 0.25" -- show the tradeoff explicitly
- Add three-seed variance to demonstrate stability
- Promise dataset-specific fine-tuning in camera-ready

**Contingency**: If three-seed results show instability, shift framing to "proof-of-concept" rather than "production-ready."

---

### R2: Single-dataset results (HIGH)

**Description**: All vision results from IndustReal only. No cross-dataset validation. Reviewers will question generalizability.

**Probability**: 50%

**Impact**: Medium. Reviewers note but do not reject solely on this if the architecture contribution is strong.

**Mitigations**:
- Explicitly acknowledge as a limitation in the Discussion section
- Cite planned IKEA ASM, Assembly101, and IndEgo evaluations for future work
- Show that the architecture is dataset-agnostic (backbone + FPN + task heads is standard)
- Emphasize that the five-task combination is unique and no other dataset supports all five
- Add results on a held-out factory subset if available

---

### R3: Activity recognition at 18.3% Top-1 (HIGH)

**Description**: 18.3% Top-1 accuracy means the activity is wrong in >80% of frames. Some reviewers will see this as not useful, regardless of the "14x above chance baseline" framing.

**Probability**: 40%

**Impact**: Medium-High. Undermines the claim that "all five heads produce non-trivial predictions."

**Mitigations**:
- Add Top-5 accuracy (41.2% -- already in paper, already helps)
- Add per-class accuracy analysis (which activities are confused? Is it fine-grained within-assembly confusions?)
- Frame as per-frame activity recognition not per-step: the PSR head uses temporal smoothing
- Show temporal smoothing improves per-step accuracy to >80% (if available)
- Benchmark against an oracle temporal-smoothing baseline

---

### R4: Single-seed results (MEDIUM)

**Description**: All results from one training run. No variance estimates except bootstrap on detection. Paper promises three-seed for camera-ready.

**Probability**: 30%

**Impact**: Medium. Most reviewers will note but not reject on this alone if addressed.

**Mitigations**:
- Already acknowledged in paper (line: "three-seed variance will be included in the camera-ready version")
- Run three seeds immediately -- this is the highest-priority action item
- Report mean and std across three seeds for all metrics

---

### R5: 4.8 FPS insufficient for real-time assembly (MEDIUM)

**Description**: 4.8 FPS (208ms per frame) may be too slow for fast-paced assembly lines. Reviewers from manufacturing backgrounds will question this.

**Probability**: 30%

**Impact**: Medium. Conversation derailment risk -- reviewer fixates on speed instead of contribution.

**Mitigations**:
- Frame in context: typical assembly steps are 5-30 seconds, capturing 24-144 frames per step
- Note that 4.8 FPS on consumer GPU ($299) vs 30+ FPS on $10K+ workstation is the relevant comparison
- Add streaming vs batched FPS breakdown (batched: 4.8, streaming: 3.9)
- Mention TensorRT or ONNX runtime optimization could double frame rate without hardware change
- Cite that human visual inspection operates at similar or lower rates

---

## Paper Acceptance Risks

### R6: Venue mismatch (MEDIUM)

**Description**: Paper combines CV, blockchain, ethics, and manufacturing. AAIML covers all these topics, but may not have reviewers deep in all four areas. Risk of being assigned reviewers who are strong in one area and dismissive of others.

**Probability**: 35%

**Impact**: Medium. Mismatch leads to lower scores from poorly matched reviewers.

**Mitigations**:
- Ensure each section is self-contained and accessible to non-specialists
- Blockchain section: add a one-sentence justification of why blockchain (not database) in the introduction
- Ethics section: frame as implementation of a standard, not as ethics research
- Lead with the architecture (Sections 1-4) since ML/AI reviewers are the primary audience

---

### R7: Paper length and scope (MEDIUM)

**Description**: Trying to cover vision + blockchain + ethics + pilot in 6-10 pages risks each section being too thin. Reviewers may see the paper as shallow.

**Probability**: 40%

**Impact**: Medium. Lack of depth in any one area leads to criticism.

**Mitigations**:
- Use appendices for supplementary details (permitted via only 6 pages + up to 4 extra at $70/page)
- Keep blockchain to 0.75 page (Section 5)
- Keep ethics to 0.5 page (Section 7)
- Focus 60% of page budget on architecture (Section 3) + experiments (Section 4)
- Pilot (Section 6) at 1 page max

---

### R8: Best Paper overclaiming (MEDIUM)

**Description**: The existing strategy files aim for Best Paper. If the paper does not win or is not competitive, overclaiming can backfire with reviewers.

**Probability**: 50%

**Impact**: Low-Medium. Reviewers penalize overclaiming in contributions.

**Mitigations**:
- Tone down "first" claims. Instead of "first to demonstrate," use "to our knowledge, prior work has not shown"
- Remove or qualify "best paper" language from the paper itself (keep in strategy files only)
- Frame contributions as "we show that" rather than "we are the first to"

---

## Ethical Risks

### R9: Worker surveillance perception (MEDIUM)

**Description**: Even with low surveillance perception scores (2.3/7), an ethics-oriented reviewer may raise concerns about automated worker monitoring in a factory setting.

**Probability**: 25%

**Impact**: Medium. Can derail reviewer focus.

**Mitigations**:
- Explicitly address the surveillance concern in Section 7
- Cite opt-out mechanism and supervisor sign-off alternative
- Note edge-only processing (no video leaves the factory)
- Reference IEEE 7005-2021 Section 6.2 on worker consent
- Add citation to Sebastian et al. "Ethics of CV for Workplace Surveillance" (already in bibliography)

---

### R10: Small pilot overgeneralization (MEDIUM)

**Description**: 20 workers, 2 weeks, one dimsum factory. Not generalizable to all manufacturing environments. Reviewers with HCI or human factors background will flag this.

**Probability**: 40%

**Impact**: Low-Medium. Most CV/AI reviewers will not penalize this, but human-factors reviewers will.

**Mitigations**:
- Explicitly bound claims: "in this specific factory context"
- Frame as "proof-of-concept pilot" not "definitive validation"
- Note demographics (age 22-58, 6.3 years mean experience) so readers can judge generalizability
- Discuss digital literacy barrier (3 workers aged 45+)

---

### R11: Blockchain environmental concerns (LOW)

**Description**: Reviewer may raise Solana blockchain energy consumption as an environmental concern.

**Probability**: 10%

**Impact**: Low. Unlikely to affect acceptance.

**Mitigations**:
- Note that Solana uses Proof-of-Stake (PoS), not Proof-of-Work, with estimated 0.0006 kWh per transaction
- Compare to data center cloud inference (which POPW avoids entirely with edge GPU)

---

## Timeline Risks

### R12: Pilot data not collected in time (HIGH if pilot not done)

**Description**: The 102-day execution plan shows the factory pilot (Phase 1, Jul 15 - Aug 1) before writing begins. If the pilot is delayed, the entire paper depends on placeholder data.

**Probability**: Depends on current status (flag if pilot has not started)

**Impact**: Critical -- paper cannot be submitted without pilot data

**Mitigations**:
- Run pilot as early as possible
- Pre-write all sections except pilot (Sections 6) before pilot concludes
- Have survey instruments ready (SUS, NASA-TLX, Trust, TAM)

### R13: Camera-ready deadline too tight (LOW)

**Description**: Camera-ready due November 30, only 20 days after notification (November 10).

**Probability**: 15%

**Impact**: Low if accepted. Manageable.

**Mitigations**:
- Prepare camera-ready template and figures ahead of notification
- Have three-seed results ready to plug in
- Pre-commit figures and tables

---

## Risk Matrix Summary

| ID | Risk | Grade | Probability | Impact | Owner |
|----|------|-------|-------------|--------|-------|
| R1 | Detection performance too low | CRITICAL | 60-70% | Rejection | Technical lead |
| R2 | Single-dataset results | HIGH | 50% | Major revision | Technical lead |
| R3 | Activity recognition 18.3% | HIGH | 40% | Major revision | Technical lead |
| R4 | Single-seed results | MEDIUM | 30% | Minor revision | Technical lead |
| R5 | 4.8 FPS insufficient | MEDIUM | 30% | Discussion point | Technical lead |
| R6 | Venue mismatch | MEDIUM | 35% | Score reduction | Author |
| R7 | Paper scope too broad | MEDIUM | 40% | Score reduction | Author |
| R8 | Best paper overclaiming | MEDIUM | 50% | Credibility | Author |
| R9 | Worker surveillance | MEDIUM | 25% | Discussion derail | Author |
| R10 | Pilot overgeneralization | MEDIUM | 40% | Score reduction | Author |
| R11 | Blockchain environment | LOW | 10% | Minor point | Author |
| R12 | Pilot not done | HIGH | varies | Paper incomplete | Project lead |
| R13 | Camera-ready deadline | LOW | 15% | Late fee | Author |
