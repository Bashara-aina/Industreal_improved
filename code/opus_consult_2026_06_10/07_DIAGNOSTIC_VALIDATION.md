# Post-Opus Diagnostic Validation (2026-06-11)

All 6 diagnostic scripts proposed in `06_OPUS_ANSWER.md` §Diagnostic scripts
were run against `latest.pth` (epoch 43, raw weights). Results below.

## Summary Table

| Diag | RC | Verdict | Key Metric |
|------|-----|---------|------------|
| D1  | RC-13 (EMA contamination) | **DENIED** | All 3 ckpts identical (cos=1.0). EMA disabled (P2). best.pth = raw copy. |
| D2  | RC-22 (anchor bottleneck) | **DENIED** | p50 best-IoU = 0.723, 100% GT have IoU≥0.5 anchor match |
| D3  | RC-22 (level distribution) | p5 fires 100%, p6 silent | p5: 100% scores>0.5, p6: 0% despite p50 IoU=0.68 |
| D4  | RC-19 (det_conf constant) | **CONFIRMED** | Per-dim std/|mean| = 0.349% (<1%). det_conf is near-constant O(10²). |
| D5  | RC-17 (VideoMAE zero) | **DORMANT** | Zero delta between real and zeroed clip_rgb across all activity metrics |
| D6  | RC-16 (attention saturation) | **HEALTHY** | Avg off-diag mass=0.925, entropy=1.992 nats. P5 fix is intact. |

## Model State: Fully Collapsed (epoch 43, latest.pth)

All 3 functional heads are dead as confirmed by D5's evaluate_all output:

- **Detection**: det_mAP50 = 0.0000. 0 GT boxes matched at IoU>0.5
  across 200 val frames. Median detection logit ≈ 1e-39.
- **Activity**: predicts 1/75 classes (class 8, 100% of frames).
  act_f1_macro = 0.0000.
- **PSR**: 1 unique binary pattern across 80 frames.
  psr_overall_f1 = 0.0909.

## RC Status Map

| RC | Description | Validated? |
|----|-------------|------------|
| RC-13 | EMA shadow no-op reset + collapsed restore | DENIED by D1 (EMA disabled entirely in this run) |
| RC-14 | Detection reinit misses cls_subnet/reg_subnet | UNTESTED — requires code inspection of reinit loop |
| RC-15 | Mixup/CutMix corrupt activity labels | UNTESTED — requires controlled retrain |
| RC-16 | Inverted attention scaling (scale=1/sqrt(d)) | HEALTHY per D6 — attention is differentiated |
| RC-17 | Train/eval clip_rgb mismatch | DORMANT per D5 — zeroing clip_rgb has no effect because VideoMAE features are already dead/identical |
| RC-18 | FeatureBank dead (video_ids=None always) | UNTESTED directly, but D6's healthy attention despite RC-18 is consistent (identical tokens → uniform attention, which D6 doesn't show because the ViT still sees non-identical GAP+det_conf features) |
| RC-19 | det_conf raw logits dominate activity input | CONFIRMED by D4 — det_conf per-dim variance is 0.35% of mean |
| RC-20 | Combined metric is pose-only | UNTESTED — metric formula confirmed in code |
| RC-21 | MATCH_PROBE never fires | UNTESTED — code logic confirmed |
| RC-22 | Anchor/GT scale mismatch | DENIED by D2 (anchors adequate for GT sizes) |
| RC-23 | Eval slice unrepresentative | UNTESTED — confirmed via D5's confusion matrix |
| RC-24 | 5% subset structural ceiling | UNTESTED — requires full-data run |

## Next Actions (per opus answer Tier 1)

1. **Fix the measurement chain first** (P1-P6, P8): zero GPU cost — evaluate `latest.pth` with the right collate and bigger slice
2. **Apply P1-P8 patches** before any retrain
3. **Re-run at 25% subset**, 3 epochs minimum (RC-24 means 5% is structurally capped)
