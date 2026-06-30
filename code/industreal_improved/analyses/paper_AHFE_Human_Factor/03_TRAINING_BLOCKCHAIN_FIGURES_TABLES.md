# Plan 3: Training Execution, Blockchain Implementation, Figures, and Tables

> **Single paper. Hardware:** RTX 5060 Ti (16 GB, GPU 1) + RTX 3060 (12 GB, GPU 0)
> **Deadline:** July 24, 2026 (27 days). All figures and tables must be publication-ready.

---

## 1. Dual-GPU Training Execution

### Current State (June 27)

```
GPU 0 (RTX 3060 12 GB):  IDLE — 70 MiB / 12 GB
GPU 1 (RTX 5060 Ti 16 GB): IDLE — 455 MiB / 16 GB
Checkpoint: crash_recovery.pth (epoch 0, step 500 — SIGTERM, not code bug)
Prior best: det_mAP50_pc = 0.304, head pose MAE = 9.13 deg
```

### Parallel Execution Map

| Date | 5060 Ti (GPU 1) | 3060 (GPU 0) |
|---|---|---|
| Jun 27 | LAUNCH: RF2 from crash_recovery.pth | EFFICIENCY: params, GFLOPs, FPS |
| Jun 28 | RF2 continues | PSR go/no-go (1h test) |
| Jun 29 | RF2 complete → RF3 advance | ABLATION A: recovery_det_only |
| Jun 30-Jul 1 | RF3 activity (epochs 1-10) | Ablation A continues |
| Jul 2-3 | RF3 activity (epochs 11-15) | Ablation A eval + confusion matrix |
| Jul 4 | RF3 complete → activity numbers | Diagnostics complete |
| Jul 5-12 | Contingency / FiLM ladder | Contingency / extra evals |
| Jul 13-18 | FINAL FULL EVAL | FINAL FULL EVAL |
| Jul 19-24 | WRITING + FORMAT + SUBMIT | WRITING + FORMAT + SUBMIT |

### Exact Commands

```bash
# STEP 1: EFFICIENCY (3060, 5 min, instant)
CUDA_VISIBLE_DEVICES=0 python3 src/evaluation/evaluate.py \
  --ckpt src/runs/full_multi_task_tma_tbank/checkpoints/crash_recovery.pth \
  --profile-efficiency-only

# STEP 2: RF2 TRAINING (5060 Ti, ~2 days)
CUDA_VISIBLE_DEVICES=1 nohup python3 -u src/training/train.py \
  --preset stage_rf2 \
  --resume src/runs/full_multi_task_tma_tbank/checkpoints/crash_recovery.pth \
  --seed 42 --num-workers 4 \
  > src/runs/phase_A_5060ti/logs/train.log 2>&1 &

# STEP 3: PSR GO/NO-GO (3060, 1h)
CUDA_VISIBLE_DEVICES=0 python3 src/training/train.py \
  --preset stage_rf2 --train_psr True \
  --resume src/runs/full_multi_task_tma_tbank/checkpoints/crash_recovery.pth \
  --epochs 2 --subset 0.35

# STEP 4: ABLATION A (3060, parallel with RF3)
CUDA_VISIBLE_DEVICES=0 nohup python3 -u src/training/train.py \
  --preset recovery_det_only --seed 123 --num-workers 4 \
  > src/runs/ablation_A_3060/logs/train.log 2>&1 &

# STEP 5: RF3 ACTIVITY (5060 Ti, ~5 days)
CUDA_VISIBLE_DEVICES=1 nohup python3 -u src/training/train.py \
  --preset stage_rf3 \
  --resume src/runs/phase_A_5060ti/checkpoints/best.pth \
  --seed 42 --num-workers 4 \
  > src/runs/phase_B_5060ti/logs/train.log 2>&1 &

# STEP 6: FINAL EVALUATION (both GPUs, Jul 13-18)
CUDA_VISIBLE_DEVICES=1 python3 src/evaluation/evaluate.py \
  --ckpt src/runs/phase_B_5060ti/checkpoints/best.pth --split test
CUDA_VISIBLE_DEVICES=0 python3 src/evaluation/evaluate.py \
  --ckpt src/runs/ablation_A_3060/checkpoints/best.pth --split test
CUDA_VISIBLE_DEVICES=0 python3 src/diag_per_class_truth.py \
  --run src/runs --output src/runs/per_class_report.json
```

---

## 2. Blockchain Implementation (3 Days, Parallel)

### Day 1: Deploy x402 Template

```bash
# Install Solana CLI
sh -c "$(curl -sSfL https://release.anza.xyz/v2.1.0/install)"

# Use official Solana x402 Rust template
npx create-solana-dapp x402-popw --template x402-solana-rust
cd x402-popw

# Devnet config
export SOLANA_NETWORK=solana-devnet
export SOLANA_RPC_URL=https://api.devnet.solana.com
export USDC_MINT_ADDRESS=4zMMC9srt5Ri5X14GAgXhaHii3GnPAEERYPJgZJDncDU
export DEFAULT_PRICE=10000  # $0.01 USDC

cargo build && cargo run --bin facilitator
```

### Day 2: Python Bridge

```python
import aiohttp, asyncio, time

async def submit_and_measure(facilitator_url, worker_id, event_type, amount):
    nonce = f"{worker_id}:{event_type}:{time.time():.3f}"
    payload = {
        "x402Version": 1, "scheme": "exact",
        "network": "solana-devnet",
        "payload": {"worker": worker_id, "event": event_type,
                    "amount": amount, "nonce": nonce}
    }
    t0 = time.time()
    async with aiohttp.ClientSession() as s:
        async with s.post(f"{facilitator_url}/verify", json=payload) as r:
            v = await r.json()
        t1 = time.time()
        async with s.post(f"{facilitator_url}/settle", json=v) as r:
            st = await r.json()
        t2 = time.time()
    return {"verify_ms": (t1-t0)*1000, "settle_ms": (t2-t1)*1000,
            "total_ms": (t2-t0)*1000}
```

### Day 3: 100-Cycle Measurement

Run 100 iterations. Compute mean, p50, p95 for each stage. Fill Table 5.

---

## 3. The 6 Publication-Ready Figures

| Figure | Paper Section | Type | Dependency | Time |
|---|---|---|---|---|
| **Fig 1**: Application scenario — worker at workstation with egocentric camera, GPU box, callouts | 3.1 (System Design) | Illustration | None | 4-6h |
| **Fig 2**: Payment pipeline — CCTV → POPW → Verifier → x402 → Solana → Wallet | 5.1 (Blockchain) | Flow diagram | None | 1h |
| **Fig 3**: Detection confusion matrix — 24x24 heatmap | 4.3 (Results) | Matplotlib heatmap | Training results | 1h |
| **Fig 4**: Hardware cost comparison bar chart | 4.5 (Cost) | Matplotlib bar | None | 1h |
| **Fig 5**: Head pose overlay — 4 frames with gaze arrows | 4.2 (Results) | Python + ffmpeg | Training results | 2-3h |
| **Fig 6**: Ethical design principles — 4 pillars diagram | 6.2 (Ethics) | Illustration | None | 1h |

### Figure Descriptions

**FIGURE 1 (Most Important):** Worker at beverage bottle assembly station wearing egocentric camera. Small PC labeled "RTX 3060 — $299 — Local Processing." Callout bubbles: "Step 5/12: Attaching Cap — VERIFIED," "Gaze: Assembly Area," bounding boxes on bottle and components. Dashboard inset showing "Today: 157 verified — $15.70 earned."

**FIGURE 2:** CCTV Frame → POPW Inference (31ms) → Verification Engine (1ms) → x402 Facilitator (/verify + /settle) → Solana Devnet (~400ms) → Worker Wallet. Each box shows measured latency.

**FIGURE 3:** 24x24 confusion matrix heatmap with annotation: "Error mass concentrated on 1-bit-Hamming-adjacent assembly states. Task is fine-grained state discrimination, not object detection."

**FIGURE 4:** 4 groups (Hardware Cost, Parameters, Power, Inference Passes) with 2 bars each (Traditional = blue, POPW = green). Values on bars.

**FIGURE 5:** 2x2 grid of egocentric frames with colored gaze direction arrows. Annotations show "Assembly Area," "Instructions," "Away."

**FIGURE 6:** 4 columns (Consent, Privacy, Fairness, Transparency) with 3 checkmark items each. IEEE 7005-2021 sections referenced.

---

## 4. The 6 Publication-Ready Tables

| Table | Section | Content | Dependency |
|---|---|---|---|
| **Table 1**: Competitor Analysis | 2.2 (Related Work) | 6 approaches vs POPW | None |
| **Table 2**: Primary Results | 4.2 (Results) | 6 metrics + HF meaning | Training results |
| **Table 3**: Ablation A | 4.4 (Ablation) | Single vs multi-task | Ablation A complete |
| **Table 4**: 3-Year TCO | 4.5 (Cost) | Traditional vs POPW | None |
| **Table 5**: Payment Latency | 5.2 (Blockchain) | 5 pipeline stages | Devnet test |
| **Table 6**: Ethical Principles | 6.2 (Ethics) | 6 principles + IEEE refs | None |

---

## 5. Daily Check-In Signals

| Date | Signal | Alarm |
|---|---|---|
| Jun 27 | RF2 loss < 10.0? | Increasing → check LR |
| Jun 28 | PSR f1 > 0.3? | 0.0 → drop PSR, 1 paragraph |
| Jun 29 | RF2 val mAP > 0.22? | < 0.18 → checkpoint issue |
| Jul 1 | Activity Top-1 > 5%? | < 3% → collapsed, adjust LR |
| Jul 4 | Activity Top-1 > 10%? | < 5% → report as preliminary |
| Jul 8 | Ablation delta < 0.05? | > 0.10 → real interference |

---

## 6. AHFE Format Compliance

**Submission System:** edition.ahfe-cms.org
**Files:** Aina_Bashara_PaperID.doc + Aina_Bashara_PaperID.pdf + Consent_Aina_Bashara_PaperID.pdf
**Length:** 10 pages maximum (including references)
**Figures:** 300+ DPI, embedded in Word document, RGB color
**Font:** 10pt body, 8pt min for tables/references
**Template:** Download from AHFE submission system
**Volume Editor field:** BLANK in consent form
