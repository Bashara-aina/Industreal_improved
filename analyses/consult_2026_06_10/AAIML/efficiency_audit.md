# Efficiency Audit — Replacement of Fabricated Claims in 167/170

## Fabricated Claims in 167 and 170

**File 167 (`167_MULTITASK_ARCHITECTURE_STRATEGY.md`), section 4 "Efficiency gain from multi-task":**

> Table: | Metric | Single-task | V5 multi-task | V8 multi-task |
> |---|---|---|---|
> | Parameters | 4x (~150M total) | 1x (~30M) | 1x (~90M) |

And asserts: "V8 gives 4x efficiency gain over single-task."

**File 170 (`170_DISCUSSION_CONCLUSION.md`), section 2 "Multi-Task Efficiency Analysis":**

> Table: | Metric | Single-task | V8 multi-task | Gain |
> |---|---|---|---|
> | Parameters | 600M | 90M | **6.7x** |

And asserts: "V8 multi-task is 4x more efficient than single-task on all dimensions."

Both documents fabricated the following specific numbers:
- **600M** parameters for four separate single-task models (line 50: "4 x ~150M = 600M total")
- **90M** parameters for V8 multi-task (line 56: "1 x ~90M (shared backbones)")
- **4x** compute efficiency claim across training, storage, inference
- **6.7x** parameter savings claim
- The parameter table in 167 further claims V5 at **~30M** parameters

None of these numbers reflect actual measurements. The "600M" figure assumes each single-task model is ~150M parameters, which is without basis -- no model used in this project (ConvNeXt-Tiny, ResNet-50, MViTv2-S, YOLOv8m) approaches 150M parameters individually. The V8 "90M" is also fabricated.

---

## Measured Numbers (from fvcore, identical hardware)

**Hardware:** NVIDIA GeForce RTX 5060 Ti, CUDA 12.x
**Measurement tool:** `fvcore.nn.FlopCountAnalysis` (params + FLOPs), CUDA events (latency), `torch.cuda.max_memory_allocated` (VRAM)
**Protocol:** 50 timed forward passes after 10 warmup passes, batch=1 and batch=2

### Raw Measurements

| Model | Params | Trainable | FLOPs | FPS | Latency | VRAM | Storage |
|---|---|---|---|---|---|---|---|
| V5 b1 | 46.47M | 46.47M | 245.73G | 13.5 | 74.3ms | 0.48GB | 177.3MB |
| V5 b2 | 46.47M | 46.47M | 491.45G | 7.6 | 132.2ms | 0.75GB | 177.3MB |
| V8 b1 | 53.80M | 19.26M | 67.11G | 17.7 | 56.6ms | 0.51GB | 205.2MB |
| V8 b2 | 53.80M | 19.26M | 134.21G | 10.5 | 95.3ms | 0.77GB | 205.2MB |
| V8-Simple b1 | 34.57M | 0.03M | 64.46G | 18.0 | 55.7ms | 0.42GB | 131.9MB |

**Notes:**
- V5 = `POPWMultiTaskModel` with ConvNeXt-Tiny backbone (frame-level, 720x1280)
- V8 = `VideoMultiTaskModel` with MViTv2-S backbone + FPN + 4 heads (clip-level, 16x224x224)
- V8-Simple = `V8Model` from `train_v8_multitask.py` (frozen MViTv2-S backbone, linear heads only)
- FLOPs are lower-bound due to unsupported fvcore operators (gelu, attention, meshgrid, etc. counted as 0)

### Efficiency Comparison (MTL vs 4x Single-Task)

A fair single-task baseline for comparable architectures on IndustReal:

| Component | Model | Params |
|---|---|---|
| Detection alone | YOLOv8m | ~25.9M |
| Activity alone | MViTv2-S | ~34.5M |
| PSR alone | Small classifier | ~3-5M |
| Pose alone | MViTv2-S | ~34.5M |
| **Sum 4x single-task** | | **~98-100M** |

**MTL efficiency savings vs 4x single-task:**

| Metric | 4x Single-Task | V5 (MTL) | V8 (MTL) |
|---|---|---|---|
| Total params | ~100M | 46.47M | 53.80M |
| FLOPs (per forward) | architecture-dependent | 245.73G | 67.11G |
| Storage (FP32) | ~400MB | 177.3MB | 205.2MB |
| Inference passes | 4 sequential | 1 | 1 |

**Real vs fabricated comparison:**

| Claim | Fabricated | Measured (real) |
|---|---|---|
| Single-task total params | 600M | ~100M |
| V8 total params | 90M | 53.80M |
| V5 total params | 30M | 46.47M |
| Parameter savings (V8 vs 4xST) | 6.7x | ~1.86x |
| Compute efficiency gain | 4x | ~1.86x (params) / faster per pass (FLOPs) |
| V8 vs V5 params savings | "V8 is more efficient" | V8 is larger (53.80M vs 46.47M) but different architectures |

---

## Honest Assessment

**Where the numbers support the efficiency claim:**
- V8 multi-task uses 53.80M params for all 4 tasks versus approximately 100M for 4 separate single-task models. The parameter saving is real: approximately **1.86x** (not 6.7x).
- V5 multi-task at 46.47M provides an approximately **2.15x** parameter saving versus 4 separate single-task models.
- Both MTL models perform all 4 tasks in one forward pass, saving inference latency versus sequential single-task passes.
- V8 has significantly lower FLOPs (67G vs 246G for V5) due to operating on lower-resolution inputs (224x224 clips vs 720x1280 frames), and is faster in FPS (17.7 vs 13.5).

**Where the numbers undercut the claims:**
- The fabricated 600M single-task total (and 6.7x/4x savings) is false. No model in this project has 150M parameters. Real single-task sum is approximately 100M.
- The fabricated 90M for V8 is also wrong -- the measured V8 is 53.80M. The fabricated number was higher than reality, but the savings ratio was inflated by using a wildly inflated single-task baseline.
- V8 is not smaller than V5 in params -- V8 (53.80M) has more parameters than V5 (46.47M), because MViTv2-S (34.5M) is larger than ConvNeXt-Tiny (28M), plus the FPN and 4 heads add approximately 19M trainable parameters.
- The FLOPs savings of V8 over V5 (67G vs 246G) are not due to multi-task efficiency but due to lower input resolution (224x224 clips vs 720x1280 frames). This is an apples-to-oranges comparison.

**Parity with single-task:**
The honest efficiency claim is approximately 2x parameter savings, not 4x or 6.7x. The real mechanism is: one backbone shared across 4 heads versus 4 separate backbones. This saving is genuine and publishable. The fabricated 4x/6.7x numbers invite desk rejection -- the real 2x is defensible.

**FLOPs caveat (per 175 section 7.4):**
The two frozen-weight modes (temporal mode at 224x224, detection mode at potentially higher resolution) will exceed a single ConvNeXt pass in FLOPs if both modes are executed. The numbers above measure one forward pass per model. If dual-mode is deployed (temporal mode + detection mode on the same backbone), total FLOPs could approach 67G (temporal) + detection-mode FLOPs (which depends on detection resolution). This should be measured when Tier F implements the weight-shared dual-mode forward.

---

## Data Provenance

- **Script:** `scripts/measure_efficiency.py`
- **Models:** V5 (`POPWMultiTaskModel`, ConvNeXt-Tiny), V8 (`VideoMultiTaskModel`, MViTv2-S + 4 heads), V8-Simple (`V8Model`, frozen MViTv2-S + linear heads)
- **fvcore version:** checked at measurement time
- **GPU:** NVIDIA GeForce RTX 5060 Ti (2 available, 1 used)
- **Pretrained weights:** None loaded -- architectures measured in random-init state (weights affect FLOPs identically; params are architecture-only)
- **Tier F (Hiera-B):** Not yet available -- Agent 2 has not completed the architecture implementation

---

## References

- **175 section 7.4:** Efficiency measurement protocol specifying fvcore, FPS, VRAM, honest FLOPs reporting
- **175 preflight P8:** "Delete fabricated efficiency table (600M/4x); measure with fvcore (section 7.4)"
- **167 section 4:** Fabricated efficiency table (4x ~150M / 1x ~30M / 1x ~90M)
- **170 section 2:** Fabricated efficiency table (600M / 90M / 6.7x / 4x)
- **172 (Opus audit):** Identified the efficiency table as fabricated; documented in P8
