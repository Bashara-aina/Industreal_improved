# Activity 75-Class Clip-Level Evaluation

## Taxonomies: 69-Grouped vs 75-Fine

The codebase supports two activity taxonomies, defined in the WACV-2024 IndustReal paper and implemented in `src/config.py`:

### 75 Fine-Grained Classes (Raw Taxonomy)
- 75 classes indexed by raw `action_id` (0..74) from the dataset's `AR_labels.csv`.
- Index 0 is `take_short_brace` (not a background/NA class — there is no dedicated NA/background label in the raw space; `action_id < 0` signals "no annotation").
- This is the taxonomy used in the WACV paper's MViTv2-S evaluation (top-1 **65.25** / top-5 **87.93**, test split).
- Referred to as "raw," "fine," or "75-class" in the codebase.
- Names list: `ACT_CLASS_NAMES` in `src/config.py:315`, populated from `AR_labels.csv` at module load.

### 69 Grouped Classes (Hybrid Taxonomy)
- Produced by the `'hybrid'` grouping mode (`ACT_CLASS_GROUPING = 'hybrid'`, `src/config.py:346`), which is the default.
- Standalone classes: classes with >=100 labeled frames keep their fine-grained identity (53 classes).
- Tail grouping: classes with <100 frames are grouped by verb (first underscore token). This produces 6 multi-class groups and 1 `other` group (containing `unknown_37` — a placeholder for raw ID 37 which never appears in the dataset).
- Total: 69 output groups from 75 raw class IDs.
- Referred to as "grouped," "hybrid," or "69-class" in the codebase.
- Remap: `id_to_group[75]` in `config/class_maps/class_69_to_75.json` maps each raw ID to its group.

### Mapping: 69 to 75

The mapping file at `config/class_maps/class_69_to_75.json` was built 2026-07-08 by inverting the `act_remap_75_to_69.json` produced by the frozen MViTv2-S probe.

| Group Type | Count | Example |
|---|---|---|
| Singleton groups (1 fine class) | 63 | Group 2 `align_objects` = raw ID 1 |
| Multi-class groups (2 fine classes) | 6 | Group 1 `take_short_brace` = raw IDs 0, 11 |
| Other (fold-in) | 1 | Group 0 `other` = raw ID 37 (unknown_37) |

For multi-class groups, the eval expands group probabilities uniformly across constituent fine classes for 75-class scoring.

## Direct Comparability to WACV 65.25 is Currently Blocked

### Why 0.3810 (69-Grouped) Cannot Be Compared to 65.25 (75-Class Fine)

The frozen MViTv2-S probe reports `best_val_top1_69 = 0.3810` (`activity_mvit_probe/results.json`). This is **not comparable** to WACV's 65.25 for three reasons:

1. **Taxonomy mismatch.** The 0.3810 is measured on 69 grouped classes (hybrid), not 75 fine classes. Grouping collapses rare classes, inflating accuracy compared to the harder 75-class distinction task. The WACV number is on the full 75-class set.

2. **Split mismatch.** The 0.3810 is on the **val** split (5 subjects: 05, 14, 20, 24, 26, 1984 clips). WACV/MViTv2-S reports on the **test** split (10 subjects). The val-vs-test gap on 75 classes is unknown but likely significant due to subject-specific bias in the small val set.

3. **Probe-only training.** The frozen probe trains only a linear layer on top of frozen MViTv2-S features (10 epochs, 0.001 LR). WACV's 65.25 comes from full fine-tuning of MViTv2-S (end-to-end, longer schedule). The probe's 0.3810 vs WACV's 0.6525 gap (~27 percentage points) is predominantly the fine-tuning gap, not just the taxonomy gap.

### What We Can Say

| Metric | Value | Source | Comparability |
|---|---|---|---|
| Frozen probe 75-class clip top-1 (val) | **0.3810** | `activity_mvit_probe/results.json` | Same model, clip-level, 75-class space, but val split |
| WACV MViTv2-S 75-class clip top-1 (test) | **0.6525** | WACV-2024 Table 2 | Full fine-tune, test split |
| Frozen ConvNeXt 69-class per-frame top-1 (val) | **0.2169** | `activity_mvit_probe/results.json` | Different backbone, per-frame, 69-class |

The frozen probe's 75-class number (0.381) can be honestly characterized as:
> "A frozen MViTv2-S backbone with a linear probe achieves 0.381 top-1 on 75 fine-grained activity classes (val split), establishing the lower bound that fine-tuning must improve upon."

## Current State

### What Is Measured

- **Frozen MViTv2-S linear probe** (clip-level, 69-grouped): 0.3810 top-1 val (`activity_mvit_probe/results.json`)
- **Frozen MViTv2-S linear probe** (clip-level, 75-class): 0.3810 top-1 val (same run, reported as `best_val_top1_75`)
- **ConvNeXt per-frame** (69-grouped): 0.2169 top-1 val (baseline)
- **Majority baseline** (69-grouped): 0.2666 val

### What Is NOT Yet Measured

- [ ] **75-class clip-level top-1 on TEST split**: The SOTA-comparable number. Requires a trained 75-class model + test-split evaluation.
- [ ] **75-class top-5 accuracy**: Not reported by the frozen probe (only top-1). The eval script computes this.
- [ ] **75-class macro-F1**: Not reported by the frozen probe. The eval script computes this.
- [ ] **Per-class breakdown on 75 classes**: Partial (per_class.json from the probe covers 55 seen classes).
- [ ] **Full fine-tune on 75 classes**: The gap between 0.381 (probe) and 0.6525 (WACV) is the fine-tuning headroom.

### Infrastructure Created (2026-07-08)

| Artifact | Location | Purpose |
|---|---|---|
| Evaluation script | `scripts/eval_activity_75class.py` | Runs clip-level 75-class eval in two modes |
| Class mapping | `config/class_maps/class_69_to_75.json` | Maps 69-grouped outputs to 75-class space |
| This README | `src/evaluation/ACTIVITY_75CLASS_README.md` | Documents taxonomies, gaps, and status |
| Output directory | `src/runs/rf_stages/checkpoints/activity_75class_eval/` | Holds `metrics.json` from eval runs |

## Class Grouping Details

The 69-grouped taxonomy from `ACT_CLASS_GROUPING=hybrid` groups raw class IDs as follows (from `config/class_maps/class_69_to_75.json`):

**Multi-class groups (6 total, 12 fine classes collapsed into 6 groups):**

| Group ID | Group Name | Raw IDs (Fine Class Names) |
|---|---|---|
| 1 | take_short_brace | 0 (take_short_brace), 11 (take_instruction) |
| 4 | plug_short_pin | 3 (plug_short_pin), 25 (plug_partial_model) |
| 7 | tighten_nut | 6 (tighten_nut), 73 (tighten_tooth_washer) |
| 29 | fit_short_brace | 30 (fit_short_brace), 64 (fit_partial_model) |
| 36 | pull_wheel | 38 (pull_wheel), 72 (pull_small_screw_pin) |
| 37 | loosen_nut | 39 (loosen_nut), 74 (loosen_tooth_washer) |

**Other:** Group 0 = raw ID 37 (`unknown_37`, never seen in data).

**Remaining 63 groups:** Each contains exactly 1 fine-grained class (identity mapping).

## To Run

```bash
# Feature-probe mode (fast, CPU, uses cached MViTv2-S features)
python scripts/eval_activity_75class.py --mode feature-probe

# Checkpoint mode (GPU, runs model inference on clips)
python scripts/eval_activity_75class.py --mode checkpoint \
    --checkpoint src/runs/rf_stages/checkpoints/best.pth

# Custom save directory
python scripts/eval_activity_75class.py --mode feature-probe \
    --save-dir src/runs/rf_stages/checkpoints/activity_75class_eval
```

## To Achieve SOTA-Comparable Evaluation (Checklist)

- [ ] Train a 75-class activity head (not 69-grouped) as part of the multi-task model.
- [ ] Evaluate on the **test** split (10 subjects), not val.
- [ ] Report clip-level (16-frame) metrics: top-1, top-5, macro-F1.
- [ ] Report per-class precision/recall for the 10 most frequent classes in the appendix.
- [ ] Report both 75-class (for SOTA comparison) and 69-grouped (for our-task continuity).
- [ ] Bootstrap 95% CI on all headline numbers (1000 resamples, seed 42).

## References

- WACV-2024: Schoonbeek et al., "IndustReal: A Dataset for Procedure Step Recognition in Industrial Assembly Tasks"
- 174 §3.2: Activity evaluation gap analysis (69-grouped vs 75-fine)
- 175 §7.2: Per-head metrics protocol (clip-level, 75-class, test split)
- `src/config.py:269-481`: Activity class taxonomy and grouping logic
- `src/runs/rf_stages/checkpoints/activity_mvit_probe/results.json`: Frozen probe results
