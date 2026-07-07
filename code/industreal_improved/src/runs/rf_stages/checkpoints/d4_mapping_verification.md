# D4 Mapping Verification: 24-class ASD to 11-component PSR

**Date:** 2026-07-06
**Agent:** Agent 13 (24-to-11 Mapping Verification Specialist)
**Opus Reference:** 141 Q34 — "RUN — REQUIRED before D4 text finalizes"
**Context:** The D4 experiment converts 24-class YOLOv8m detection outputs into 11-component PSR transitions via s2 feature conversion. A wrong mapping would be "D4's version of the variable-shadow bug."

## Result: MAPPING CORRECT

Both the `eval_yolov8m_psr.py` and `d4_threshold_retune.py` scripts produce identical, correct PSR_MASK matrices that faithfully encode the IndustReal 24-class ASD taxonomy into 11 PSR components.

---

## 1. Mapping Function Location

| File | Function | Lines |
|---|---|---|
| `src/config.py` | `DET_CLASS_NAMES` dict (data source) | 211-236 |
| `src/evaluation/eval_yolov8m_psr.py` | `_build_psr_mask()` | 68-89 |
| `src/evaluation/eval_yolov8m_psr.py` | `s2_from_yolo_detections()` | 143-208 |
| `src/evaluation/d4_threshold_retune.py` | `_build_psr_mask()` | 219-238 |
| `src/evaluation/d4_threshold_retune.py` | `s2_from_yolo_detections()` | 244-289 |

## 2. Mapping Mechanism

The 24 ASD classes decompose as:
- **Class 1:** `background` — no PSR mapping
- **Classes 2-23:** 22 assembly states, each encoded as an 11-character binary string
- **Class 24:** `error_state` — no PSR mapping

Each binary string position corresponds to a PSR component in procedure order (comp0=base plate ... comp10=wheels). A '1' at position i means component i is assembled in that state.

The `_build_psr_mask()` function reads the `DET_CLASS_NAMES` dict and builds a `[24, 11]` binary matrix where `PSR_MASK[c, comp] = 1` if detection class c activates component comp.

### The 22 Assembly States:

```
Class  2: 10000000000 -> comp[0]
Class  3: 10010010000 -> comp[0, 3, 6]
Class  4: 10010100000 -> comp[0, 3, 5]
Class  5: 10010110000 -> comp[0, 3, 5, 6]
Class  6: 11100000000 -> comp[0, 1, 2]
Class  7: 11110010000 -> comp[0, 1, 2, 3, 6]
Class  8: 11110100000 -> comp[0, 1, 2, 3, 5]
Class  9: 11110110000 -> comp[0, 1, 2, 3, 5, 6]
Class 10: 11110111100 -> comp[0, 1, 2, 3, 5, 6, 7, 8]
Class 11: 11110111110 -> comp[0, 1, 2, 3, 5, 6, 7, 8, 9]
Class 12: 11110110001 -> comp[0, 1, 2, 3, 5, 6, 10]
Class 13: 11110111101 -> comp[0, 1, 2, 3, 5, 6, 7, 8, 10]
Class 14: 11110111111 -> comp[0, 1, 2, 3, 5, 6, 7, 8, 9, 10]
Class 15: 11110101111 -> comp[0, 1, 2, 3, 5, 7, 8, 9, 10]
Class 16: 11110011111 -> comp[0, 1, 2, 3, 6, 7, 8, 9, 10]
Class 17: 11110011110 -> comp[0, 1, 2, 3, 6, 7, 8, 9]
Class 18: 11110101110 -> comp[0, 1, 2, 3, 5, 7, 8, 9]
Class 19: 11100001110 -> comp[0, 1, 2, 7, 8, 9]
Class 20: 11101101110 -> comp[0, 1, 2, 4, 5, 7, 8, 9]
Class 21: 11101011110 -> comp[0, 1, 2, 4, 6, 7, 8, 9]
Class 22: 11101111110 -> comp[0, 1, 2, 4, 5, 6, 7, 8, 9]
Class 23: 11101111111 -> comp[0, 1, 2, 4, 5, 6, 7, 8, 9, 10]
```

## 3. s2 Feature Conversion (s2_from_yolo_detections)

For each frame, the conversion works as follows:

1. Initialize per-component logits to -3.0 (sigmoid(-3) approx 0.047 — no-evidence default)
2. For each detection above threshold:
   a. Get component mask: `PSR_MASK[det_class_id]` (which components this class activates)
   b. Convert confidence to logit: `logit_val = log(conf / (1-conf))`
   c. For each component in the mask: `logit[comp] = max(logit[comp], logit_val)`
3. The resulting [11] logit vector is the per-frame PSR prediction

This is max-accumulation: the highest-confidence detection for a component determines its logit.

## 4. Verified Properties

| Property | Result |
|---|---|
| PSR_MASK identical between both files | YES |
| All 22 state strings unique | YES (22/22) |
| comp0 active for all assembly states | YES (22/22) |
| String format valid (11 chars, 0/1 only) | YES |
| background (class 1) excluded | YES (mask[0] sum = 0) |
| error_state (class 24) excluded | YES (mask[23] sum = 0) |
| PSR_MASK shape [24, 11] | YES |
| s2 conversion produces [B, 11] output | YES |
| Max-accumulation semantics correct | YES |
| Default -3.0 logit for no-evidence | YES |
| Non-monotonic states (parallel sub-assemblies) | Expected by IndustReal design |
| Class order matches YOLOv8m output order | YES (confirmed via D1 eval metrics) |
| Paper reference matches | YES (AAIML tex: "11-bit binary states") |

## 5. Sample Test Results

**Test 1: Single class-6 detection at conf=0.95**
- class-6 = `11100000000` -> comp[0, 1, 2]
- logit(0.95) = 2.9444
- Result: comp0,1,2 = 2.9444; comp3-10 = -3.0
- PASS

**Test 2: Two detections: class-6(conf=0.3) + class-23(conf=0.8)**
- class-6 = `11100000000` -> comp[0, 1, 2], logit(-0.8473)
- class-23 = `11101111111` -> comp[0, 1, 2, 4, 5, 6, 7, 8, 9, 10], logit(1.3863)
- Max-accumulation: comp0,1,2 = 1.3863; comp4,5,6,7,8,9,10 = 1.3863; comp3 = -3.0
- PASS

**Test 3: No detections**
- All components = -3.0 (sigmoid approx 0.047)
- PASS

## 6. Conclusion

The 24-to-11 PSR mapping is **correct** and **identical** across both code paths. No error analogous to the variable-shadow bug exists here. The mapping faithfully reproduces the IndustReal dataset's ASD taxonomy, is self-consistent, matches the YOLOv8m output channel order, and produces correct PSR logit vectors from detection inputs.

**No code changes needed.** The D4 disclosure text can finalize with confidence that the mapping is verified.

## File References
- Mapping definition: `/media/newadmin/master/POPW/working/code/industreal_improved/code/industreal_improved/src/config.py:211-236`
- Primary mapping builder + s2 conversion: `src/evaluation/eval_yolov8m_psr.py:68-89, 143-208`
- Duplicate mapping builder + s2 conversion: `src/evaluation/d4_threshold_retune.py:219-238, 244-289`
- Paper reference: `analyses/consult_2026_06_10/AAIML/popw_aaiml2027.tex:92`
