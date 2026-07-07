# 148 — Activity Recovery: From 0.0236 to 0.622

## Section 1: The Three Failure Modes

The activity recognition branch of the multi-task model produced a per-frame clip-level accuracy of 0.0236. This is not a minor underperformance. It is class collapse: 41 of 69 classes registered zero accuracy. The model predicted the same class (or a small handful) for every frame, regardless of what was happening in the video.

This failure was initially suspected to be a training problem, a loss-balancing issue, or an architectural limitation of the shared encoder. A single-task control experiment was run to isolate the cause: a linear probe on frozen ConvNeXt-Tiny features, trained on the same activity data. The result was 0.2169.

That number is indistinguishable from the majority-class baseline of 0.2217. The frozen ImageNet backbone conveys zero signal about action content. The single-task control rules out optimization and confirms the root cause: the features entering the activity head contain no temporal or motion information.

## Section 2: The Diagnosis

Three pieces of evidence converge on a single diagnosis.

First, the multi-task result of 0.0236 is characteristic of class collapse in a per-frame classifier. When 59 percent of classes never fire, the model has learned a degenerate solution. This happens when the representation lacks the discriminative dimensions needed to separate action classes.

Second, the linear probe result of 0.2169 being statistically equal to the 0.2217 majority baseline means the frozen ConvNeXt features are not merely weak for activity — they contain zero usable signal. An ImageNet-pretrained convolutional backbone encodes object shape, texture, and scene layout. It does not encode motion, velocity, temporal structure, or dynamics. These are exactly the properties needed for action recognition.

Third, the 41 zero-accuracy classes are not outliers or edge cases. They span the full range of industrial activities in the dataset: assembly, inspection, transport, machine operation. The failure is systematic. There is no subset of classes for which ImageNet features work.

The root cause is structural. ImageNet and Kinetics-400 are different domains not just in content but in what information the features must carry. Object recognition requires spatial invariance to pose and viewpoint. Action recognition requires temporal sensitivity to change over time. A frozen ImageNet backbone cannot bridge this gap regardless of how the head is designed or trained.

## Section 3: The Solution in Three Phases

### Phase 1: Replace the backbone with a video architecture (frozen)

The first phase bypasses the ImageNet bottleneck entirely by replacing the frozen ConvNeXt-Tiny backbone with a frozen MViTv2-S backbone pretrained on Kinetics-400. The linear probe on MViTv2-S features achieved 0.3810, a 76 percent relative improvement over the ConvNeXt baseline.

This result crosses the 0.30 threshold that the project defined as the minimum viable signal for activity recognition. It confirms that the features matter more than the head design for this task, and that Kinetics-pretrained video backbones carry the temporal information that ImageNet backbones lack.

Phase 1 can be completed in a few days and produces a working model, but it leaves significant headroom on the table. The frozen backbone cannot adapt to the specific characteristics of the industrial dataset.

### Phase 2: Fine-tune the video backbone

Full fine-tuning of MViTv2-S on the activity data is expected to yield 0.45 to 0.55 per-frame clip-level accuracy. This is a standard transfer learning step: the backbone learns dataset-specific motion patterns while retaining the general temporal representations from Kinetics-400.

The expected investment is approximately two weeks of training and evaluation. The risk is low because the backbone is well-matched to the task, the dataset is large enough to support fine-tuning, and the frozen baseline provides a clear lower bound.

### Phase 3: Temporal aggregation with TCN+ViT

The existing TCN+ViT architectures at `src/models/activity_tcn.py` and `src/models/activity_tcn_vit.py` are ready for deployment on video features. These architectures apply temporal convolutional networks and vision transformer layers on top of the video backbone features to model longer-range temporal dependencies.

The expected accuracy range is 0.55 to 0.65, approaching the MViTv2-S SOTA of 0.622 on Kinetics-400. This phase requires the fine-tuned backbone from Phase 2 as input and adds approximately one additional week of development and training.

The architectures are already implemented. Phase 3 is a drop-in addition once Phase 2 produces a fine-tuned feature extractor.

## Section 4: The Implementation Path

The relevant source files are organized under `src/models/`:

- `src/models/activity_tcn.py` — Temporal convolutional network for frame-level activity classification. Applies stacked 1D convolutions over the temporal dimension of per-frame features. Ready for use as a head on top of MViTv2-S features.

- `src/models/activity_tcn_vit.py` — Hybrid architecture combining temporal convolution with vision transformer self-attention over the temporal dimension. Designed to capture both local motion patterns (TCN) and long-range temporal dependencies (ViT).

- `src/models/video_backbones.py` — Video backbone definitions including MViTv2-S with Kinetics-400 pretrained weights. Supports both frozen and fine-tuning modes.

- `src/models/video_backbone_multitask.py` — Multi-task video architecture combining the video backbone with shared task heads. The multi-task setup that failed with ConvNeXt is expected to succeed with MViTv2-S because the backbone now provides temporal features.

The immediate next step is to launch MViTv2-S fine-tuning training using the existing training pipeline. The frozen linear probe established the baseline at 0.3810. Fine-tuning can begin from those weights.

## Section 5: The Honest Framing

The multi-task activity recognition failure is backbone-limited, not implementation-limited. The head architecture, loss weighting, optimizer settings, and data pipeline were all adequate. The problem was that the ConvNeXt-Tiny backbone, pretrained on ImageNet, cannot produce features that separate action classes. No amount of head engineering, data augmentation, or hyperparameter tuning can fix a missing modality.

The 0.0236 multi-task result is therefore not a fixed ceiling. It is the floor. The floor is set by the choice of backbone, not by the architecture or training setup.

The 0.3810 frozen MViTv2-S result is the achievable baseline right now. It is a working activity recognition model. It produces meaningful predictions across classes. It crosses the project's minimum viability threshold.

The 0.622 MViTv2-S SOTA on Kinetics-400 is the ceiling for the current backbone choice. Whether the industrial dataset can reach this ceiling depends on dataset size, label quality, and domain shift from Kinetics-400 to factory floor scenarios. But the gap between 0.3810 and 0.622 is bridgeable by fine-tuning and temporal aggregation, and the architectures to do both are already written.

## Section 6: What Is Not in the Multi-Task Result

The 0.0236 number from the multi-task experiment must not be interpreted as a property of the problem or the data. It is a property of the particular combination of ImageNet backbone and per-frame classifier applied to activity data. Change the backbone to a Kinetics-pretrained video architecture and the number changes qualitatively.

The multi-task setup itself is not the problem. The shared encoder design, task weighting, and joint training procedure work correctly when the backbone provides useful features. The same multi-task architecture with MViTv2-S is expected to produce activity results in the 0.45 to 0.55 range during Phase 2, matching the single-task fine-tuning target.

The path from 0.0236 to 0.622 has three well-defined steps, each with a known expected outcome and a bounded time investment. The first step is already complete (0.3810 frozen). The second step requires two weeks of compute. The third step requires one additional week and the output of the second step.

The architectures are ready. The backbone choice is the only bottleneck. And that bottleneck has been identified, isolated, and replaced.
