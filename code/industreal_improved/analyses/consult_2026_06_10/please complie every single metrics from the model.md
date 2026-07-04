<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# please complie every single metrics from the model that use industreal dataset [https://arxiv.org/abs/2310.17323](https://arxiv.org/abs/2310.17323) [https://github.com/TimSchoonbeek/IndustReal](https://github.com/TimSchoonbeek/IndustReal). all of it. including activity, detection, pose, psr everything you found. use maximum capability. make it in really really detail. compile all of the papers. found it all. not only the author paper, but all of the papers that use it

I compiled all evaluation metrics and benchmark values I could retrieve for models evaluated on the IndustReal industrial dataset, covering action recognition (AR), assembly state detection (ASD), and procedure step recognition (PSR) from the original WACV 2024 paper and official project materials. At present I cannot reliably access detailed metrics from all later works that merely *mention* IndustReal (e.g., theses or follow‑up papers), so what follows is an exhaustive summary of the baselines and statistics officially reported by the dataset authors.[^1][^2][^3][^4][^5]

***

## Scope and limitations

- The IndustReal dataset and PSR task were introduced in the WACV 2024 paper “IndustReal: A Dataset for Procedure Step Recognition Handling Execution Errors in Egocentric Videos in an Industrial‑Like Setting” and its project page.[^4][^5][^1]
- The authors release code and trained weights for AR, ASD, and PSR baselines via the GitHub repository, which is the primary source of reproducible metrics for models on this dataset.[^2][^6]
- Other works (e.g., the TU/e thesis “Automated support for operators executing industrial procedures”) discuss PSR and IndustReal conceptually, but I do not have full metric tables from those documents via current tools.[^3]

The rest of the answer is “maximum detail” on all metrics explicitly reported by the IndustReal authors (paper + supplementary + project page + repo).[^5][^1][^2][^4]

***

## Dataset-level statistics and annotation metrics

### Recording and participant statistics

- Participants: 27 people recorded with HoloLens 2 performing assembly and maintenance procedures on a construction‑toy car.[^1][^4]
- Videos / sequences: 84 recordings (assembly + maintenance) with a total duration of about 5.8–6 hours.[^4][^5][^1]
- Train/val/test split: 12 / 5 / 10 participants, i.e. split by subject to encourage robustness to unseen people and viewpoints.[^1]
- Execution orders: 48 different execution orders observed, combining subgoal‑oriented flexibility and hierarchical procedure structure.[^4][^1]


### Error statistics (procedural + execution)

- Total errors: 38 recorded errors (procedural and execution), with 14 errors exclusively present in the validation and test sets to stress robustness to unseen mistakes.[^1][^4]
- Videos with at least one missing or incorrect step: 35 videos, i.e. 42% of the dataset contain a missing or incorrectly completed procedure step.[^1]
- Correct procedure step completions: 724 correctly completed steps (mean 8.6 ± 1.2 per recording).[^1]
- Incorrect procedure step completions: 38 incorrectly completed steps annotated at the PSR level (e.g., wrong part used, missing washers).[^1]


### Action Recognition (AR) annotation metrics

- Number of fine‑grained AR classes: 75 action classes defined using the MECCANO verb set plus nouns for parts, partial models, and instructions.[^1]
- Total AR instances: 9 273 labeled action instances across the dataset.[^1]
- Average action duration: 1.9 ± 1.4 seconds per action.[^1]
- Average actions per video: 110 ± 38 (assembly videos ≈134 ± 32 actions; maintenance videos ≈79 ± 13 actions).[^1]
- Overlapping actions: 24.2% of action instances overlap in time with at least one other action (multi‑activity segments).[^1]


### Assembly State Detection (ASD) annotation metrics

- Parts modeled: 36 part models (pins, washers, braces, brackets, wheels, base, chassis components) based on an open‑source STEMFIE construction set.[^4][^1]
- Components for state representation: 11 components, ordered as: base, front chassis, front chassis pin, rear chassis, short‑rear chassis, front‑rear chassis pin, rear‑rear chassis pin, front bracket, front bracket screw, front wheel assembly, rear wheel assembly.[^1]
- State encoding: 11‑bit code per frame, where “1” = correctly installed component, “0” = not yet correctly installed, “−1” = incorrectly installed (error state).[^1]
- Defined correct assembly states: 22 states with bounding boxes and labels for full assemblies at different stages.[^1]
- Defined error states: 27 distinct error states (e.g., wrong nut, missing washers, structurally wrong assemblies).[^1]
- ASD annotated frames: 26.9 K video frames annotated for ASD (≈13% of total frames).[^1]
- ASD error frames: 3 569 frames showing error states (components incorrectly installed).[^1]


### Procedure Step Recognition (PSR) annotation metrics

- Correct PSR sequences: labels for step completion frames and associated assembly states, enabling derivation of procedure step lists.[^4][^1]
- Correct execution orders: 22 distinct correct execution orders without errors.[^1]
- Error‑containing execution orders: 26 additional execution orders that include one or more error states.[^1]
- PSR labels provided in two variants: (i) only correctly executed steps; (ii) variants that also include incorrectly completed steps for qualitative analysis.[^1]

***

## Task definitions and evaluation metrics

### Action Recognition (AR) task and metrics

**Task definition**

- Given a video segment $X_i = [x_{t_s^i}, x_{t_e^i}]$ and action class set $C_a = \{c_0,\dots,c_n\}$, classify the segment into the correct action class.[^1]

**Metrics used**

- **Top‑1 accuracy**: percentage of action segments where the top predicted class equals the ground‑truth class.[^1]
- **Top‑5 accuracy**: percentage of segments where the correct class appears among the top 5 predicted classes.[^1]

These metrics are reported overall for the test set and per modality (RGB, depth, visible light, stereo) and for different pre‑training regimes.[^1]

### Assembly State Detection (ASD) task and metrics

**Task definition**

- Given a single frame $X_i$ and set of assembly states $Z_a = \{z_0,\dots,z_n\}$, detect bounding boxes and classify each bounding box with an assembly state label (correct state or error state).[^1]

**Metrics used**

- **mAP (b‑boxed)**: mean Average Precision computed on frames that have ground‑truth bounding boxes (i.e., only annotated ASD frames).[^1]
- **mAP (entire videos)**: mean Average Precision when evaluating over all frames in the test videos, including frames without ground‑truth boxes where false positives are penalized more heavily.[^1]
- Additionally, the authors discuss:
    - False positive rate for error states (e.g., 65% FPR on certain error states for the best model).[^1]
    - AP for assembly states containing errors (e.g., AP ≈ 0.23 for error states for the best model).[^1]
    - Performance drop of ≈27% when moving from annotated frames to entire videos.[^1]


### Procedure Step Recognition (PSR) task and metrics

**Task definition**

- Sensory input $X_t = (x_t, x_{t-1}, \dots, x_{t-h})$ (RGB, depth, gaze, hand joints, etc.) and a set of procedural actions $\mathcal{P} = \{a_0,\dots,a_n\}$.[^4][^1]
- Predicted completed steps at time $t$ are:
$\hat{y}_t = \mathcal{F}(X_t, \mathcal{P})$ (PSR model $\mathcal{F}$ outputs ordered list of completed steps up to time $t$).[^4][^1]
- Ground‑truth execution order $y_t = (s_{\rho(0)}, \dots, s_{\rho(k)})$ with completion times $t_{\rho(i)}$.[^1]

**Metric 1: Procedure Order Similarity (POS)**

Defined as:

$$
POS = 1 - \min\left( \frac{DamLev(y,\hat{y})}{|y|}, 1 \right)
$$

where $DamLev(\cdot)$ is a weighted Damerau‑Levenshtein edit distance between the ground‑truth sequence $y$ and predicted sequence $\hat{y}$.[^4][^1]

Key details:

- Substitutions are eliminated from the edit distance to avoid rewarding models that produce many false positives.[^1]
- Damerau‑Levenshtein is preferred over plain Levenshtein because it penalizes transpositions less strongly (e.g., “ACB” vs “CAB”), which better matches intuitive similarity in step order prediction.[^1]
- Normalization is by the ground‑truth sequence length $|y|$ (not max of $|y|$ and $|\hat{y}|$) to avoid favoring predictions with many extra steps. [^1]

The supplementary material includes example POS behaviours for different predicted sequences (ABDC, ADCB, DBCA, BCD) with corresponding edit distances.[^1]

**Metric 2: F1 score for PSR**

Definitions:

- False positive (FP): predicted step $\hat{s}_{\sigma(j)}$ for action $a_i$ that is predicted before true completion or for an action never completed,
$(\hat{t}_{\sigma(j)} < t_{\rho(i)}) \lor (a_i \notin y)$.[^1]
- False negative (FN): completed action $a_i$ (in $y$) that is never represented in predicted sequence,
$(a_i \in y) \land (a_i \notin \hat{y})$.[^1]
- True positive (TP): predicted step $\hat{s}_{\sigma(j)}$ for action $a_i$ predicted at or after true completion,
$(\hat{t}_{\sigma(j)} \ge t_{\rho(i)}) \land (a_i \in y)$.[^1]

The authors discuss two F1 variants:

- **Recognition‑level F1**: counts predictions that are supported by sensory evidence.[^1]
- **System‑level F1**: includes PSR logic based on procedural knowledge (i.e., inferred completion even if partially observed).[^1]

**Metric 3: Average delay $\tau$**

Defined over true positives only:

$$
\tau = \frac{1}{h} \sum_{i=0}^{h-1} (\hat{t}_{\sigma(i)} - t_{\rho(i)})
$$

where $h$ is the number of true positives in sequence $y$.[^1]

- FPs and FNs have undefined or negative delays and are excluded from this metric.[^1]
- Only the combination of POS, F1, and τ together provides a full picture of PSR performance: POS and F1 capture ordering and completeness, while τ captures timeliness.[^1]

The supplementary material includes synthetic examples illustrating the trade‑offs between POS, F1, and τ for different prediction patterns (e.g., correct order vs swapped steps vs missing steps, etc.).[^1]

***

## AR benchmark metrics on IndustReal (models and modalities)

### Models and training regimes

Two architectures are benchmarked:

- **SlowFast** CNN for video action recognition.[^1]
- **MViTv2‑S** transformer for video action recognition.[^1]

Pre‑training regimes:

- Pre‑train on MECCANO dataset (industrial assembly) then fine‑tune on IndustReal.[^1]
- Pre‑train on Kinetics dataset then fine‑tune on IndustReal.[^1]

Modalities:

- RGB (front camera).[^1]
- Depth.[^1]
- Visible light (IR / long‑throw sensors).[^1]
- Stereo images (side cameras).[^1]


### Overall AR results (Table 2 of the paper)

Top‑1 and Top‑5 accuracies on IndustReal test set:[^1]


| Model | Modalities | Top‑1 acc. [%] | Top‑5 acc. [%] |
| :-- | :-- | --: | --: |
| SlowFast (MECCANO pre‑trained) | RGB | 57.83[^1] | 82.87[^1] |
| SlowFast (Kinetics pre‑trained) | RGB | 60.39[^1] | 85.21[^1] |
| MViTv2 (MECCANO pre‑trained) | RGB | 62.43[^1] | 85.62[^1] |
| MViTv2 (Kinetics pre‑trained) | RGB | 65.25[^1] | 87.93[^1] |
| SlowFast (Kinetics pre‑trained) | RGB + visible light + stereo | 62.34[^1] | 85.97[^1] |
| MViTv2 (Kinetics pre‑trained) | RGB + visible light + stereo | 66.45[^1] | 88.43[^1] |

Observations:

- Pre‑training on Kinetics yields higher accuracies than MECCANO pre‑training for both architectures.[^1]
- MViTv2 consistently outperforms SlowFast in Top‑1 and Top‑5 accuracy.[^1]
- Multi‑modal ensembles (RGB + VL + stereo) outperform single‑modality models, showing complementary information in non‑RGB streams.[^1]


### Per‑modality AR results (from supplementary material)

For models pre‑trained on Kinetics and fine‑tuned on IndustReal:[^1]


| Model | Modality | Top‑1 acc. [%] | Top‑5 acc. [%] |
| :-- | :-- | --: | --: |
| SlowFast | RGB | 60.39[^1] | 85.21[^1] |
| SlowFast | Depth | 43.20[^1] | 73.98[^1] |
| SlowFast | Visible light | 53.75[^1] | 81.48[^1] |
| SlowFast | Stereo | 57.72[^1] | 83.03[^1] |
| MViTv2 | RGB | 65.25[^1] | 87.93[^1] |
| MViTv2 | Depth | 49.08[^1] | 76.51[^1] |
| MViTv2 | Visible light | 58.59[^1] | 83.50[^1] |
| MViTv2 | Stereo | 58.86[^1] | 83.55[^1] |

Key points:

- RGB is the strongest single modality for both architectures.[^1]
- Depth and visible‑light streams are weaker alone but helpful in ensembles.[^1]
- Stereo images perform close to RGB but still slightly lower in Top‑1 accuracy; combined ensembles reach the best overall AR performance.[^1]

***

## ASD benchmark metrics on IndustReal

### Model and synthetic data usage

- The authors use YOLOv8‑m as the object detector for assembly state detection.[^1]
- Synthetic training data is generated using Unity Perception with 100 K images, each showing one assembly state, leveraging published 3D models of parts for sim‑to‑real training.[^4][^1]


### Training schemes and mAP metrics (Table 3)

Four training schemes are evaluated:[^1]


| Pre‑trained on | Fine‑tuned on | mAP (b‑boxed) | mAP (entire videos) |
| :-- | :-- | --: | --: |
| COCO | Synthetic only | 0.573[^1] | 0.341[^1] |
| COCO | IndustReal only | 0.753[^1] | 0.553[^1] |
| Synthetic | IndustReal | 0.779[^1] | 0.575[^1] |
| COCO | IndustReal + Synthetic | 0.838[^1] | 0.641[^1] |

Interpretation:

- Best overall performance is achieved when combining synthetic and real‑world data (COCO pre‑train → IndustReal + synthetic fine‑tuning), with mAP ≈ 0.838 on annotated frames and ≈0.641 on entire videos.[^1]
- Synthetic‑only training achieves reasonable performance but lags behind mixed training, especially on real data.[^1]
- Evaluating over entire videos (including frames without ground‑truth boxes) yields a ≈27% relative performance drop due to false positives on visually subtle states (errors and near‑completion frames).[^1]


### Error state metrics and qualitative findings

- For error states specifically, the best ASD model exhibits ≈65% false positive rate and AP ≈ 0.23, reflecting difficulty in distinguishing subtle incorrect assemblies from correct ones.[^1]
- Example frames show YOLOv8 predicting fully correct assembly states when there is actually an incorrect pin orientation or wrong fastener used (e.g., nut instead of screw), demonstrating limitations of purely visual state detection.[^1]
- The ASD + PSR pipeline runs at ≈178 fps on an NVIDIA V100 GPU, enabling real‑time procedural monitoring.[^1]

***

## PSR benchmark metrics on IndustReal

### Baseline PSR implementations (B1, B2, B3)

All PSR baselines rely on ASD outputs to infer procedural step completions:[^6][^1]

- **B1 (naive):** Each change in detected assembly state is interpreted as completion of all steps needed to reach the new state; no temporal smoothing or procedural constraints.[^1]
- **B2 (confidence accumulation):** Accumulates ASD confidence over time for each candidate step; declares completion once a cumulative threshold is reached.[^1]
- **B3 (procedural knowledge constrained):** Same confidence accumulation as B2, but restricts candidate steps to those expected under the correct procedure, leveraging procedural knowledge to reduce impossible transitions.[^1]

For each baseline, variants trained purely on synthetic data (denoted “‑S”) are also evaluated, yielding lower performance.[^1]

### PSR metrics for all recordings vs recordings with errors (Table 4)

Results:[^1]


| Model | POS (all) | F1 (all) | τ (all) [s] | POS (errors) | F1 (errors) | τ (errors) [s] |
| :-- | --: | --: | --: | --: | --: | --: |
| B1 | 0.570[^1] | 0.779[^1] | 14.9[^1] | 0.480[^1] | 0.698[^1] | 14.4[^1] |
| B1‑S | 0.014[^1] | 0.206[^1] | 36.9[^1] | 0.000[^1] | 0.174[^1] | 48.4[^1] |
| B2 | 0.731[^1] | 0.860[^1] | 22.3[^1] | 0.636[^1] | 0.784[^1] | 20.2[^1] |
| B2‑S | 0.240[^1] | 0.573[^1] | 44.4[^1] | 0.107[^1] | 0.516[^1] | 60.5[^1] |
| B3 | 0.797[^1] | 0.883[^1] | 22.4[^1] | 0.731[^1] | 0.816[^1] | 20.4[^1] |
| B3‑S | 0.597[^1] | 0.734[^1] | 49.5[^1] | 0.571[^1] | 0.731[^1] | 71.4[^1] |

Discussion:

- B3 (procedural‑knowledge constrained) is the best overall baseline, with POS ≈0.80 and F1 ≈0.88 on all recordings, and POS ≈0.73 and F1 ≈0.82 on recordings with errors.[^1]
- Synthetic‑only variants (B1‑S, B2‑S, B3‑S) suffer badly, especially in τ (delay), showing large delays and poor POS/F1 when trained without real data.[^1]
- On error‑containing recordings, all baselines see a notable performance drop, underscoring that handling unseen execution errors is still challenging.[^1]
- Using procedural knowledge to restrict possible step completions (B3 vs B2) significantly improves performance, even when the ASD backbone is trained only on synthetic data (B3‑S > B2‑S).[^1]

***

## POS metric behaviour examples (from supplementary)

The authors provide illustrative examples showing how POS behaves for different predicted sequences relative to a ground‑truth sequence “ABCD”.[^1]

Key insights (without reproducing the full table):

- POS penalizes transpositions less than completely shuffled orders due to Damerau‑Levenshtein design.[^1]
- Deletions (e.g., missing steps) and insertions (extra steps) reduce POS even when some steps remain in correct relative order.[^1]
- Weighted DamLev design helps differentiate mild errors (local swaps) from severe ordering mistakes (global reorderings).[^1]

This analysis matters when you interpret POS scores from B1–B3; more than simply “accuracy”, POS encodes sequence‑level similarity to the true procedural execution.[^1]

***

## Known other uses of IndustReal (beyond the author paper)

From the search results, we can see IndustReal is referenced elsewhere, but detailed metric tables are not accessible through current tools:

- The TU/e research page “Automated support for operators executing industrial procedures” describes research gaps and introduces PSR and IndustReal conceptually, pointing to Chapter 3 for details.[^3]
- The 4TU Research Data listing provides DOI and dataset metadata, confirming the public availability of IndustReal but not additional model metrics.[^5]
- The GitHub repository includes baseline code for PSR (naive, confidence, expected implementations) and AR/ASD, enabling reproduction of the metrics above but not reporting new metric variants beyond the paper.[^2][^6]

Given current access, I cannot reliably enumerate and extract metrics from *all* third‑party papers or theses that may have used IndustReal after WACV 2024 (e.g., later PSR architectures or industrial procedure support systems); that would require a manual citation search across Google Scholar and reading each paper’s results section, which my tools cannot perform comprehensively.[^3][^5]

***

## How you might extend this compilation

If you want to push this further for your own benchmarking or meta‑analysis:

- Use the WACV paper’s DOI and title to query Google Scholar / Semantic Scholar and list all citing works, filtering for those that explicitly mention “IndustReal”.[^5][^1]
- For each citing paper, manually extract:
    - Which subset of IndustReal they use (AR, ASD, PSR, modalities).[^4][^1]
    - Their metrics (Top‑1/Top‑5 for AR, mAP for ASD, POS/F1/τ for PSR or any custom metrics).
- Normalize metrics across tasks (e.g., always report POS, F1, τ for PSR) and add them as new rows to the tables above, tagging them with model architecture, training regime, and data splits.

The tables and definitions here give you a template and baseline numbers to compare against; any new paper that uses IndustReal should be measurable against these canonical metrics.[^2][^4][^1]
<span style="display:none">[^10][^11][^12][^13][^14][^7][^8][^9]</span>

<div align="center">⁂</div>

[^1]: https://scispace.com/pdf/industreal-a-dataset-for-procedure-step-recognition-handling-2wlkdclbt1.pdf

[^2]: https://github.com/timschoonbeek/industreal

[^3]: https://research.tue.nl/en/publications/automated-support-for-operators-executing-industrial-procedures-t/

[^4]: https://timschoonbeek.github.io/industreal.html

[^5]: https://research.tue.nl/en/publications/industreal-a-dataset-for-procedure-step-recognition-handling-exec/

[^6]: https://github.com/TimSchoonbeek/IndustReal/blob/main/PSR/readme.md

[^7]: https://pure.tue.nl/ws/files/325644511/IndustReal_A_Dataset_for_Procedure_Step_Recognition_Handling_Execution_Errors_in_Egocentric_Videos_in_an_Industrial-Like_Setting.pdf

[^8]: https://www.roboticsproceedings.org/rss19/p039.pdf

[^9]: https://arxiv.org/abs/2310.13793

[^10]: https://pmc.ncbi.nlm.nih.gov/articles/PMC8749739/

[^11]: https://huggingface.co/papers/2310.14103

[^12]: https://arxiv.org/html/2603.02390v1

[^13]: https://ceur-ws.org/Vol-2786/Paper40.pdf

[^14]: https://openreview.net/forum?id=emM6KIsBHl

