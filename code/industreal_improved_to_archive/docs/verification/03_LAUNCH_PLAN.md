# 03 — Launch Plan: Step by Step From Here

This is the operational sequence. Concrete commands, expected outputs, decision points, and stop conditions.

---

## Step 0 — Fix the two evaluate.py bugs (5 minutes, BLOCKER)

### 0.1 Open `evaluate.py` and apply two changes:

**Change 1 — class name:**

```python
# Find this near line 1813:
from model import MultiTaskIndustReal
# ...
model = MultiTaskIndustReal(pretrained=False).to(device)

# Replace with:
from model import POPWMultiTaskModel
# ...
model = POPWMultiTaskModel(
    pretrained=False,
    backbone_type=str(getattr(C, 'BACKBONE', 'resnet50')),
    use_headpose_film=bool(getattr(C, 'USE_HEADPOSE_FILM', False)),
    use_videomae=bool(getattr(C, 'USE_VIDEOMAE', False)),
).to(device)
```

**Change 2 — indentation:**

Lines 1808–1810 currently have 8 spaces of indentation (inside `_make_loader`):
```python
        args = parser.parse_args()

        logging.basicConfig(level=logging.INFO)
```

Dedent them to 4 spaces (back at outer scope):
```python
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
```

### 0.2 Verify:

```bash
python evaluate.py --help
```

Should print the help text without errors. If you get `NameError` or `ImportError`, the fix is incomplete.

### 0.3 Optionally fix the angular MAE normalization (Doc 01 §A.2)

This isn't strictly needed for training to work, but you want it before reporting head-pose numbers. 4-line fix in `compute_head_pose_metrics` around line 729.

---

## Step 1 — Install missing Python packages (5 minutes)

```bash
pip install lion-pytorch transformers fvcore onnxruntime psutil scikit-learn tqdm torchmetrics
```

Why each:
- `lion-pytorch` — `USE_LION=True` is the default; without it the code falls back to AdamW with a warning, costing ~0.5%
- `transformers` — only needed if you flip `USE_VIDEOMAE=True` later
- `fvcore` — needed for `efficiency_report.py` GFLOPs measurement
- `onnxruntime` — needed for `efficiency_report.py --onnx_export` ONNX FPS
- `psutil` — used by `pretrain_synthetic.py` for memory checks
- `scikit-learn` — `calibrate_anchors.py` uses k-means; `evaluate.py` uses metric helpers
- `torchmetrics` — `pretrain_synthetic.py` and parts of `evaluate.py` use mAP metrics

If any of these are already installed, the command is a no-op.

---

## Step 2 — Build the synthetic-pretraining checkpoint (overnight)

`PRETRAIN_DET_ON_SYNTH = True` is in config but the main `train.py` doesn't run pretraining itself. You run `pretrain_synthetic.py` as a separate first step, then use its output as the initialization for the main run.

### 2.1 Sanity-check the pretraining script

```bash
python pretrain_synthetic.py --help 2>&1 | head -20
```

If this fails, fix imports before launching the long run.

### 2.2 Launch pretraining

```bash
cd /home/newadmin/swarm-bot/project/popw/working/code/industreal_improved
python pretrain_synthetic.py 2>&1 | tee runs/pretrain_synthetic_log.txt
```

**Expected timeline:** 20 epochs × ~25 min/epoch on RTX 3060 = ~8 hours. Run overnight.

**Expected outcome:** mAP@0.5 around 75–82% on val by epoch 20. Final checkpoint saved to `runs/pretrain_synthetic/checkpoints/best.pth`.

### 2.3 Decision point

If pretraining mAP@0.5 < 70%, something is wrong (data path, label format, etc.) — investigate before continuing. If 75–82%, proceed.

---

## Step 3 — Smoke test the main training loop (30 minutes)

Before you commit to a multi-day run, verify the full pipeline works in debug mode.

```bash
python train.py --debug --max-epochs 1 --resume runs/pretrain_synthetic/checkpoints/best.pth 2>&1 | tee runs/smoke_test.log
```

The `--debug` flag (per `config.py` `DEBUG_MODE` and `DEBUG_MAX_VIDEOS=20`) limits the dataset to 20 recordings and uses larger frame strides — runs in 5–10 minutes.

### 3.1 Things to watch for in the log:

| What you should see | What it confirms |
|---|---|
| `Optimizer: Lion (LR ~3× smaller, WD ~3× larger than AdamW)` | Lion installed and active |
| `Epoch 0 [stage=1]` in progress bar | Staged training is on |
| `Stage 1: frozen=20.4M, trainable=39.0M` (or similar) | Stage param freezing works |
| `VideoMAE stream   : False` | VideoMAE off (correct default) |
| `Resuming from runs/pretrain_synthetic/checkpoints/best.pth` | Synthetic-pretrained backbone loads |
| Detection loss decreasing in stage 1 | Backbone fine-tuning works |
| `eval_results: act_accuracy: ...` after 1 epoch | Eval pipeline runs |

### 3.2 Things that should NOT happen:

| Warning sign | What it means |
|---|---|
| `Lion not installed, falling back to AdamW` | Run `pip install lion-pytorch` |
| `NaN/Inf loss at epoch 0 step ...` (more than ~5 times) | Loss scaling problem; investigate |
| `CUDA out of memory` | Reduce `BATCH_SIZE` in config (try 1) and bump `GRAD_ACCUM_STEPS` (to 32) |
| `cannot import name 'MultiTaskIndustReal'` | You forgot to apply Step 0.1 fix |

### 3.3 If everything looks good, proceed to Step 4.

---

## Step 4 — Launch the real training run (3–5 days)

### 4.1 Pick a seed and launch

```bash
python train.py \
    --resume runs/pretrain_synthetic/checkpoints/best.pth \
    --max-epochs 60 \
    --seed 42 \
    2>&1 | tee runs/main_run_seed42.log
```

**Expected timeline on RTX 3060 with default config (ConvNeXt-Tiny, batch=2, grad_accum=16):**
- ~1.0–1.3 hours per epoch (faster than ResNet-50)
- 60 epochs = 60–80 hours = 2.5–3.5 days
- With early stopping (patience=10), realistic 35–45 epochs = 1.5–2 days

### 4.2 Monitor every ~5 epochs

The log emits validation metrics each `VAL_EVERY` epochs (default every epoch when `BENCHMARK_MODE=True`). Track:
- `det_mAP50` — should climb from ~75% (loaded checkpoint) toward 84–87% by epoch 30
- `act_accuracy` — should climb from ~5% (random) toward 65–71% by epoch 30 (after activity ramp-in at epoch 5)
- `psr_macro_f1` — should climb from ~0.5 toward 0.86–0.89 by epoch 30
- `head_pose_MAE` — should decrease

### 4.3 Decision points during training

- **By epoch 15:** if `det_mAP50` is below 78%, something is off with the synthetic-pretrained checkpoint loading. Check the resume log.
- **By epoch 25:** if `act_accuracy` is below 62%, the activity head is struggling. Possible causes: LDAM-DRW DRW epoch (60) hasn't activated yet (expected — DRW only kicks in at epoch 60); or class imbalance is more severe than LDAM can handle alone.
- **By epoch 30:** if all three (`det_mAP50` >82%, `act_accuracy` >65%, `psr_macro_f1` >0.85), you're on track for a publishable result.

### 4.4 Early-stopping considerations

`PATIENCE = 10` in config. If val metrics plateau for 10 epochs, training stops. This is fine — don't override unless you have a specific reason.

---

## Step 5 — Evaluate on test split (30 minutes)

```bash
python evaluate.py \
    --checkpoint runs/.../checkpoints/best.pth \
    --split test \
    --save-dir runs/eval_seed42_no_tta \
    2>&1 | tee runs/eval_seed42_no_tta.log
```

This produces the headline numbers without TTA. Save them; they're your primary result.

### 5.1 Then evaluate with TTA for the "best" numbers:

```bash
python evaluate.py \
    --checkpoint runs/.../checkpoints/best.pth \
    --split test \
    --flip-tta \
    --crop-tta \
    --save-dir runs/eval_seed42_with_tta \
    2>&1 | tee runs/eval_seed42_with_tta.log
```

Note: 5-crop TTA is **5× slower** than no-TTA. Budget ~2 hours for full test eval with both TTAs.

Report both. The honest table looks like:

| Metric | No TTA | + flip TTA | + flip + 5-crop TTA |
|---|---|---|---|
| Activity Top-1 | 67.5% | 68.2% | 69.1% |
| ASD mAP@0.5 | 84.7% | 85.4% | 86.1% |
| ... | ... | ... | ... |

This shows reviewers your "real" improvement and the "best possible" number.

### 5.2 Decision point — do you want more?

If your headline numbers in 5.1 are above benchmark targets:

- ASD mAP@0.5 > 83.8% ✓
- Activity Top-1 > 66.45% ✓
- Activity Top-5 > 88.43% ✓
- PSR F1 > 0.883 (B3) ✓ — STORM-PSR's 0.901 is the harder bar

**You can stop here.** The result is publishable. Spend remaining time on Step 7 (multi-seed) and writing.

If any major target is missed, go to Step 6 (Phase 2 work).

---

## Step 6 — Phase 2: critical fixes (only if you missed targets)

### 6.1 If PSR is below STORM-PSR target

Implement sequence-mode PSR training. Outline:
1. In `industreal_dataset.py`, add a `sequence_mode` flag that returns contiguous frame sequences instead of single frames.
2. In `train.py`, alternate between random-frame batches (for det/pose/activity) and sequence batches (for PSR).
3. In `PSRHead.forward`, when in training mode AND given a sequence batch, run the causal Transformer on the whole sequence (causal mask works correctly).

Estimated effort: 1.5–2 days. Estimated gain: PSR F1 +0.04–0.06.

### 6.2 If Activity Top-1 is below 66.45%

Enable VideoMAE V2 stream. The plumbing is already there — flip the flag and install transformers:

```bash
pip install transformers
```

In `config.py`:
```python
USE_VIDEOMAE = True
```

Then re-train (you can warm-start from your best checkpoint):
```bash
python train.py \
    --resume runs/.../checkpoints/best.pth \
    --max-epochs 30 \
    --seed 42
```

Estimated gain: +5–7% Activity Top-1. Memory cost: VideoMAE-Small adds ~22M params (~600MB GPU mem). Should still fit on RTX 3060 with batch_size=1.

### 6.3 Run SWA after main training

In `config.py`:
```python
USE_SWA = True
SWA_EPOCHS = 8
SWA_LR = 1e-5
```

The training loop will run 8 extra epochs of SWA at the end, producing `runs/.../checkpoints/swa.pth`. Evaluate this checkpoint same as the main one — it usually adds +0.3–0.5% across most metrics.

---

## Step 7 — Multi-seed and ablation (Phase 3, ~1 week)

Once you have a final config you're happy with, run multi-seed for statistical significance.

### 7.1 Multi-seed runs

```bash
python run_multi_seed.py --seeds 42,123,7 --epochs 60
```

This trains 3 separate runs with different seeds, then evaluates each, then prints a mean ± std table. Total time: 3× your single-seed time, so ~1 week on RTX 3060.

If you don't have time for 3 full runs, run 2 (e.g. seeds 42 and 123). Reviewers will accept "2 seeds" with a note about compute constraints.

### 7.2 Ablation table

After the multi-seed runs, run `generate_ablation_table.py` to compile results across configs into a single markdown table.

For ablations, you want shorter runs (~30 epochs) with a single seed each. Keep the table to ~6 rows:
1. Baseline (XML diagram only)
2. + Architectural changes (TCN, T=16, 2× ViT, causal PSR)
3. + Loss & training (LDAM, GIoU, RandAugment, CutMix, layer4 BN)
4. + ConvNeXt + HeadPoseFiLM
5. + Synthetic pretraining
6. POPW-Full (everything)

Each row is ~1 day of training, so 6 days total. **Plan this in advance** because it's the longest single time-sink.

### 7.3 Cross-validation

```bash
python cross_validate.py --folds 5 --epochs 20
```

5-fold CV is **expensive** — 5× shorter training runs (20 epochs each). About 5–7 days total on RTX 3060. Optional but reviewers love it. If pressed for time, skip CV and rely on multi-seed.

---

## Step 8 — Generate the figures and efficiency report

### 8.1 Run efficiency benchmark

```bash
python efficiency_report.py --baseline_compare --onnx_export
```

This produces the comparison table you'll put in the paper. Includes streaming FPS, ONNX FPS, params, GFLOPs.

### 8.2 Qualitative figures

```bash
python visualize_psr_transitions.py    # PSR component state timeline
python visualize_head_pose.py          # Head pose forward-vector overlay
```

These produce the figures you'll include for qualitative analysis.

### 8.3 Per-class breakdown

The eval pipeline already saves per-class F1 to JSON. Plot the bottom-5 / top-5 hardest activity classes as a bar chart for the paper.

---

## Step 9 — Stop conditions

Don't keep iterating past these:

| Stop when | Reason |
|---|---|
| Phase 1 (default config + synthetic pretraining) clears 5+ targets | You have a publishable result; further work is diminishing returns |
| You're 2 weeks in and Phase 1 results are below my probability estimates | Either real bugs remain or hardware is throttling — investigate before scaling up |
| Phase 2 doesn't move the needle by epoch 15 | The model has hit its ceiling; stop and write |
| You're 6+ months in and haven't run multi-seed | Lock in the config, run seeds, write |

---

## Total realistic timeline

| Phase | Time | Cumulative |
|---|---|---|
| Step 0 — fix evaluate.py | 5 min | 5 min |
| Step 1 — install deps | 5 min | 10 min |
| Step 2 — synthetic pretraining | overnight | 1 day |
| Step 3 — smoke test | 30 min | 1.5 days |
| Step 4 — main training | 2–3 days | 4 days |
| Step 5 — evaluation | 30 min – 2 hours | 4 days |
| Decision point: stop or continue |
| Step 6 — Phase 2 (if needed) | 2–4 days | 8 days |
| Step 7 — multi-seed | 5–7 days | 14 days |
| Step 8 — figures + report | 1 day | 15 days |
| Writing the thesis | ??? | weeks |

**Realistic to a publishable result: 5–7 days from now if Step 5 hits targets, 2 weeks if you need Phase 2.**

---

## What I'd actually do if I were you, in order

1. **Right now** (10 min): Apply the evaluate.py fixes from Step 0, install deps from Step 1.
2. **Tonight** (overnight): Launch `pretrain_synthetic.py`. Sleep.
3. **Tomorrow morning** (30 min): Smoke test from Step 3. If it passes, immediately launch the main training run from Step 4.
4. **Days 2–4**: Main training runs. Check the log every few hours just to confirm losses look healthy. Use this time to write the methods section of your thesis based on what's actually in the code.
5. **Day 4–5**: Evaluate from Step 5. Look at the numbers and decide: stop or Phase 2.
6. **Days 5–7 (if stopping)**: Multi-seed seeds 123 and 7, then figures + writing.
7. **Days 5–14 (if Phase 2)**: Sequence-mode PSR or VideoMAE, retrain, re-evaluate, then multi-seed.

That's the path. The model is ready. Fix two lines in evaluate.py and go.
