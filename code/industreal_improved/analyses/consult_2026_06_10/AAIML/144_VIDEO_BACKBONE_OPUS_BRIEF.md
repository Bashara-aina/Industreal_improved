# 144 — Video Backbone Activity Recovery Brief for Opus

**Date:** 2026-07-07
**Context:** User approved full SOTA push: video backbone + fine-tuning + multi-task
**Current bottleneck:** ConvNeXt-Tiny frozen features give 0.2169 ≈ 0.2217 baseline
**Goal:** Close gap to MViTv2-S 0.622

## 0. Current Status

- Linear probe (frozen ConvNeXt): 0.2169 ≈ 0.2217 (zero signal). The frozen backbone cannot discriminate assembly activities above chance.
- TCN+ViT architectures built (commits `a3bad7356`, `693b119b`) but not trained. These remain candidates for a post-backbone temporal head if the video backbone succeeds.
- 41/69 classes have zero accuracy (class collapse confirmed). The per-class breakdown shows systematic collapse in low-frequency activities, suggesting the frozen features lack temporal structure entirely.
- PSR head repair (LeakyReLU) applied; full D4+D1R F1=0.6364 (3-video subset) decisive test. The repair stabilised PSR but does not address the backbone bottleneck.

## 1. Three Improvements Requested

### 1. Video Backbone (Kinetics-400 pretrained)

Three candidates for replacing the frozen ConvNeXt-Tiny with a temporally-aware video backbone:

- **MViTv2-S** (~25M params). Multiscale Vision Transformer with pooling attention. Strong on Something-Something-V2 (action recognition). Memory-efficient relative to ViT-L.
- **VideoMAE** (masked autoencoder). Self-supervised pretraining on K400. Reports strong fine-tuning performance on downstream video tasks.
- **TimeSformer** (pure attention). Division of space-time attention into separate spatial and temporal heads. Lower FLOPs than factored-joint attention but potentially less expressive.

Key question is which pretraining dataset best transfers to industrial assembly actions.

### 2. Backbone Fine-tuning

Currently the ConvNeXt is frozen for the linear probe. The proposal is to replace it with a video backbone and fine-tune it jointly with the task heads:

- Multi-task loss gradient flow through the backbone, rather than frozen features.
- Discriminative learning rates: backbone ~1e-5, heads ~1e-4.
- Warm-start from current `best.pth` or fresh Kinetics-pretrained weights.
- The fine-tuning strategy must work within the constraints of the multi-task setup (4 heads, different loss scales).

### 3. Multi-task Training (4 heads)

- **Detection** (existing, proven at ~0.995 mAP@50 on D1R). Strong gradient signal; risk of dominating the backbone features.
- **Head pose** (existing, 7.78 degrees MAE). Moderate gradient magnitude.
- **PSR** (existing, head repaired with LeakyReLU, F1=0.6364 (3-video subset)). Handles procedure state recognition.
- **Activity** (currently 0.0236). Severely undertrained; likely needs the largest gradient contribution per step.

The multi-task dynamics are unknown because only linear-probe and single-task runs exist.

## 2. 30 Questions for Opus

### Video Backbone Choice (10)

1. MViTv2-S vs VideoMAE vs TimeSformer — which is best for short assembly action clips with 41/69 zero classes?
2. Kinetics-400 vs Kinetics-700 vs pre-fine-tune on IndustReal first before Kinetics pretraining is used?
3. Should we use the full MViTv2-S or a smaller variant (MViTv2-T) given the 12GB memory budget on RTX 3060?
4. Input resolution: 224x224 (default), 256x256, or 384x384 for assembly actions where fine-grained manipulation needs spatial detail?
5. Frame sampling: dense 16 frames vs strided 8+1 vs uniform sampling for action segments of variable duration?
6. Should BatchNorm statistics be frozen or updated during fine-tuning given small batch sizes from memory constraints?
7. What learning rate range for video backbone fine-tuning (1e-5, 1e-4, or layer-wise)?
8. Should a CLIP-style joint text-image model be pretrained first on assembly transcripts, or is video-only pretraining sufficient?
9. Memory budget: 16-frame 224x224 MViTv2-S needs approximately 8GB. Should gradient checkpointing or activation offloading be used to make room for 4 heads?
10. Should weights be initialised with `@trunc_normal_` (ViT default) or does Kinetics-pretrained already have appropriate weight distributions?

### Fine-tuning Strategy (10)

11. Warm-start from current `best.pth` (ConvNeXt-Tiny weights) or start fresh from Kinetics-pretrained video backbone?
12. Learning rate ratio backbone:heads — 0.01 (two orders apart) or 0.1 (one order) given the multi-task gradient magnitudes?
13. Discriminative learning rates per backbone layer (last block 10x, first block 0.1x) or uniform?
14. Freeze BN statistics or use SyncBN for multi-GPU (if available)? Single GPU constraint defaults to freezen stats.
15. Should EMA (exponential moving average) be applied to backbone weights for inference, or to head weights only?
16. Use mixed-precision (bf16) for video backbone forward/backward?
17. Gradient accumulation steps: how many to reach effective batch size of 16-32 with single GPU?
18. Cosine or linear LR schedule for 2-week fine-tuning horizon?
19. Warmup steps: 500, 1000, or a percentage (e.g. 10%) of total training budget?
20. Early stopping criteria based on activity F1 plateau, or do all heads need to plateau?

### Multi-task Integration (10)

21. How to share backbone features between 4 heads without excessive compute — single shared temporal head or per-head adapters?
22. Spatial heads (detection, pose) should extract features from which backbone layer (early, mid, or late)? Different from temporal heads?
23. Temporal heads (PSR, activity) at final layer only, or with additional temporal pooling?
24. Per-head loss weights: should activity get a larger weight given its 0.0236 baseline, or would that destabilise detection and pose?
25. Should per-head uncertainty weighting (Kendall et al., 2018) be used to learn loss weights, or fixed weights based on baseline performance?
26. Curriculum: train detection first (already strong), then add pose, then PSR, then activity — or all from the start?
27. Should there be a shared temporal aggregation layer (e.g. a small TCN or attention pooling) on top of the video backbone, or does MViTv2-S's own temporal attention suffice?
28. How to handle different input frame requirements per head: detection needs dense frames (all 16), activity needs strided (8 of 16), pose needs centre frame?
29. Task-specific data augmentation differences: detection needs heavy spatial augmentation, activity needs temporal jitter — can they share a single dataloader?
30. Total memory budget estimate for 4 heads + MViTv2-S on RTX 3060 (12GB): what fits, what needs gradient checkpointing, what needs offloading?

## 3. Specific Recommendations Wanted

- **Time-box:** 1 week, 2 weeks, or longer given the gap from 0.2169 to 0.622?
- **Cut criteria:** if activity F1 < 0.40 by the deadline, should the activity head be abandoned or the video backbone reverted to frozen?
- **Order of operations:** backbone choice -> fine-tuning -> multi-task training -> TCN+ViT comparison. Or should TCN+ViT be evaluated first since the code already exists?
- **Go/no-go:** whether to attempt video backbone at all given the 2-week budget and 0.622 target. The current activity F1 (0.0236) is so far from 0.622 that even with a strong backbone the multi-task dynamics may not converge.

## 4. Fail-Safe Plan

If video backbone and fine-tuning fail to beat 0.40 activity F1:

- **Paper fallback:** frame the activity result as a diagnostic contribution — backbone quality dominates head architecture, and the current 0.0236 establishes a worst-case baseline. Compare with fine-tuned result to quantify backbone impact.
- **Cut activity entirely:** if video backbone does not move activity above chance, remove the activity head from the paper and focus on detection+PSR+pose three-task contribution.
- **Frame as diagnostic:** title a section "What frozen features hide: the backbone bottleneck in multi-task activity recognition" and use the negative result to argue that video pretraining is necessary for assembly action recognition.
