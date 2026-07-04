# Metrics Used to Evaluate Industrial Vision and Activity Models

## Overview

Industrial AI models for inspection, monitoring, and robotics are evaluated with a wide set of metrics covering detection, segmentation, anomaly scoring, pose accuracy, time‑series faults, and human activity recognition.  This report compiles the main metric families used on widely‑adopted industrial datasets and tasks: anomaly detection (MVTec, VisA, NEU‑DET), defect/object detection, pixel segmentation, 6D pose estimation, radar/correlation tracking (PSR), time‑series anomaly detection, and activity/action recognition.[^1][^2][^3][^4]


## Industrial Visual Anomaly Detection (MVTec, VisA, etc.)

Industrial anomaly detection benchmarks such as MVTec AD, MVTec 3D‑AD, VisA, KolektorSDD2, NEU‑DET and related datasets focus on detecting defects in manufactured parts and surfaces.  Evaluation typically uses detection‑level and pixel‑level metrics.[^5][^2][^1]

### Image‑level metrics

- **Image AUROC (I‑AUROC)** – Area under the ROC curve using per‑image anomaly scores, the primary scoreboard on MVTec AD and related benchmarks.[^2][^6][^1]
- **Overall AUROC / Mean AUROC** – AUROC averaged over all categories or subsets.[^6][^2]
- **AP‑I / Image‑level AP** – Average precision computed from image‑level predictions in some works.[^6]

These metrics quantify the trade‑off between true‑positive rate and false‑positive rate across thresholds, with top systems reaching ≈99–100% image‑level AUROC on MVTec AD.[^1][^6]

### Pixel‑ and region‑level metrics

- **Pixel AUROC (P‑AUROC)** – AUROC evaluated at pixel resolution, using dense anomaly scores.[^2][^1][^6]
- **PRO (Per‑Region Overlap)** – Averages the IoU of predicted anomaly regions with ground truth regions, designed to better reflect performance on small defects.[^1][^2][^6]
- **PRO‑AUC** – Area under the PRO curve over thresholds, proposed as a more balanced segmentation metric.[^2]

Pixel AUROC and PRO are standard segmentation metrics on MVTec, sometimes used together to separate models that only flag defective images from those that localize defects accurately.[^1][^2]


## Industrial Defect and Object Detection (Steel, Welds, NEU‑DET)

Industrial defect detection models (YOLO variants, Faster R‑CNN, SSD, etc.) evaluated on steel surface datasets such as NEU‑DET and other defect corpora use classic object detection metrics plus task‑specific ones.[^7][^8][^9]

### Core detection metrics

- **Precision (P)** – Fraction of predicted defect bounding boxes that are correct; measures false alarm rate.[^8][^9][^7]
- **Recall (R)** – Fraction of ground‑truth defects that are detected; measures missed defects.[^9][^7][^8]
- **Average Precision (AP)** – Area under the precision‑recall curve for each defect class.[^8]
- **Mean Average Precision (mAP)** – Mean of AP across all defect classes; can be reported at a single IoU threshold (mAP@0.5) or averaged over thresholds (mAP@0.5:0.95).[^7][^9][^8]

For example, improved Faster R‑CNN and YOLO‑based models report per‑class AP for NEU‑DET classes (Cr, In, Pa, Ps, Rs, Sc) and an overall mAP that indicates overall detection quality.[^9][^8]

### Extended industrial accuracy metrics

Industrial defect detection research on lithium batteries and steel surfaces further introduces specialized metrics:[^10][^8]

- **Micro‑defect Recall Rate (MRR)** – Recall computed only on defects whose pixel area is within a small (e.g., 1–5%) ratio of the image, emphasizing tiny defects.[^10]
- **False Negative Rate (FNR)** – FN/(TP+FN), share of real defects missed; directly tied to safety risk.[^10]
- **False Positive Rate (FPR)** – FP/(TN+FP), normal samples misclassified as defective; relates to inspection cost.[^10]
- **Per‑class AP** – AP per defect category to highlight which defect types remain hard to detect.[^8][^9][^10]

### Robustness and deployment metrics

Industrial deployments report robustness and system‑level metrics.[^8][^10]

- **Noise Sensitivity (NS)** – Drop in mAP@0.5 after adding noise (Gaussian, Poisson) to images; "Excellent" if drop ≤10%.[^10]
- **Illumination Adaptability (LA)** – Drop in MRR under ±30% visible‑light illumination change.[^10]
- **Modality Missing Adaptability (MMA)** – mAP@0.5 when only a single modality (e.g., visible or X‑ray) is available.[^10]
- **Frames Per Second (FPS)** – Inference speed including preprocessing and NMS, measured on server GPUs and edge devices.[^8][^10]
- **End‑to‑End Latency** – Total time from acquisition to output; for some industrial lines the requirement is ≤50 ms.[^10]
- **Model Parameters / FLOPs** – Parameter count used to assess whether models fit edge constraints (e.g., ≤15M parameters).[^10]

These metrics capture accuracy, robustness to environmental variation, throughput, and resource usage, all critical for industrial quality inspection.[^8][^10]


## Industrial Anomaly Detection Dataset Register and Metrics

Industrial anomaly detection datasets explicitly document metrics to standardize benchmarking.[^2][^1]

### MVTec AD and 3D‑AD

- Focus on unsupervised anomaly detection for inspection across 15 categories in 2D, plus 10 categories in 3D.[^5][^1]
- Default metrics: **Detection AUROC** (image‑level) and **Segmentation AUROC** (pixelwise), plus PRO/PRO‑AUC for improved segmentation evaluation.[^6][^1][^2]

### VisA, KolektorSDD2, Weld Defect X‑ray, Severstal, NEU‑DET

- **VisA**: complex industrial scenes; metrics mainly image‑level AUROC and sometimes pixel‑level AUROC.[^1]
- **KolektorSDD2**: commutator defects; uses AUROC as primary measure.[^1]
- **Weld Defect X‑Ray**: weld inspection; uses AUROC and detection metrics per defect type.[^1]
- **Severstal Steel Defect (Kaggle)**: defect segmentation and detection; uses mAP and IoU‑based measures.[^1]
- **NEU‑DET**: 6 steel defect categories; uses precision, recall, AP per class, mAP@0.5 and mAP@0.5:0.95.[^7][^9]


## Industrial Human Activity / Action Recognition

Industrial human activity recognition datasets like OpenMarcie and InHARD evaluate models for monitoring assembly tasks and worker actions.[^11][^4][^12]

### Task‑level metrics

OpenMarcie benchmarks three tasks:[^4]

- **Activity classification** – Accuracy, top‑k accuracy, precision, recall, F1 score per action class.
- **Open‑vocabulary captioning** – Caption quality via BLEU, METEOR, CIDEr, ROUGE‑L and similar NLP metrics.
- **Cross‑modal alignment** – Retrieval and alignment metrics such as Recall@K and mean Average Precision for matching video to text or sensor streams.

Industrial HAR experiments additionally report confusion matrices, per‑class F1 scores, and sometimes temporal segmentation accuracy for start/end of actions.[^12]

### InHARD and related datasets

The Industrial Human Action Recognition Dataset (InHARD) proposes usage metrics and split strategies:[^11]

- Actions split into **expert vs beginner** operators based on total manipulation duration, allowing separate evaluation on expert and novice subsets.[^11]
- Standard performance metrics: accuracy, precision, recall, F1 over the action classes, sometimes stratified by expertise level.[^12][^11]

These metrics support safety monitoring, productivity analysis, and ergonomic assessments in industrial environments.[^4][^12]


## Industrial Pose Estimation (6D Object Pose, Robotics)

Industrial robotics and manipulation require accurate 6D pose estimation of parts.[^13][^14][^15]

### Geometric pose metrics

Common metrics across 6D pose benchmarks (e.g., YCB‑Video, industrial parts datasets) include:[^14][^15][^13]

- **ADD (Average Distance of Model Points)** – Mean distance between transformed model points under predicted and ground‑truth pose; used for non‑symmetric objects.[^15][^14]
- **ADD‑S** – Variant handling symmetrical objects by measuring minimal distance between point sets.[^14]
- **2D Projection Error** – Average pixel distance between projected 3D keypoints under predicted vs ground‑truth poses.[^14]
- **Pose Success Rate** – Percentage of poses whose ADD/ADD‑S error is below a threshold (e.g., 10 cm) or whose 2D projection error is below a threshold (e.g., 40 px).[^14]

Some industrial works introduce probabilistic task‑success metrics that estimate the probability that a robot successfully completes a grasp or manipulation using the predicted pose, based on empirical distributions from physical trials.[^13]


## PSR and Related Confidence Metrics in Industrial Tracking

Peak‑to‑Sidelobe Ratio (PSR) originates in correlation‑filter based tracking and radar, and is often used in industrial visual tracking to monitor tracker reliability.[^16][^17][^18]

### PSR definition and use

- In adaptive correlation filter tracking, **PSR** measures the strength of the main correlation peak relative to the average and variance of the sidelobe region in the response map.[^17][^16]
- It is used to detect occlusions or tracking failure, stop online updates, and reacquire targets when appearance changes.[^16][^17]

Industrial tracking systems can:[^18]

- Set a PSR threshold, above which tracking is considered reliable and model updates are allowed, and below which updates are reduced or suspended.
- Use PSR ranges (e.g., 20–60 as "strong peak", ≈7 indicating occlusion or loss) to adapt update rates and learning rates.[^18]

Related metrics include **average peak‑to‑correlation energy (APCE)**, maximum response value, and channel reliability measures, all used as confidence indicators for object trackers in industrial surveillance or inspection robots.[^19][^20][^18]


## Industrial Time‑Series Fault and Anomaly Detection

Industrial processes produce rich time‑series from sensors and PLCs; datasets like FactoryNet and others evaluate fault detection and early warning models.[^21][^22][^23][^3]

### Standard time‑series anomaly metrics

- **AUROC / AUPRC** – Binary anomaly vs normal discrimination using ROC and precision‑recall curves.[^23][^21]
- **Accuracy, Precision, Recall, F1** – Per‑time‑step or per‑window classification metrics.[^22]
- **Detection delay / earliness** – How early an anomaly is detected relative to its onset.[^21]
- **Alarm frequency / cardinality** – Number of alarms raised per unit time or per anomaly, addressing over‑ and under‑alarming.[^21]

A recent industrial time‑series anomaly metric framework proposes explicit earliness and alarm cardinality metrics to capture timeliness and appropriateness of alarms, beyond pure classification accuracy.[^21]

### FactoryNet and large‑scale benchmarks

FactoryNet introduces a large‑scale industrial time‑series dataset targeting foundation models:[^3][^23]

- Benchmarks tasks such as forecasting, fault detection, and representation learning with metrics like mean squared error, mean absolute error, AUROC, F1, and task‑specific scores.
- Evaluations often include domain‑specific metrics such as downtime reduction, number of correctly predicted faults, and adherence to maintenance schedules, though these are more application‑level and may be reported qualitatively.[^23][^3]


## Segmentation, Localization, and Structural Metrics

Beyond anomaly‑specific measures, industrial vision systems use general segmentation and localization metrics.[^24][^25][^2]

- **IoU (Intersection‑over‑Union)** – Overlap between predicted and ground‑truth regions; often per class and averaged.[^24][^2]
- **Dice Coefficient / F1 for segmentation** – Alternative overlap measure, especially in highly imbalanced defect vs background settings.[^24]
- **Boundary metrics** – Distances or overlap measures focused on defect boundaries, used in high‑precision inspection contexts.[^25][^24]

These metrics complement AUROC/PRO by explicitly quantifying spatial accuracy and boundary quality of defect localization.[^24][^2]


## System‑Level and Industrial Compliance Metrics

Industrial AI deployments often report system‑level and standards compliance metrics, beyond pure model accuracy.[^1][^10]

- **Industrial standard compatibility** – Whether the inspection pipeline complies with domain standards (e.g., IEC 62133‑2:2017, GB/T 30038‑2013 for battery safety).[^10]
- **Throughput and utilization** – Frames or items inspected per second, and percentage of production covered.[^1][^10]
- **Edge deployment feasibility** – Parameter count, memory footprint, and inference latency relative to constraints of embedded devices (Jetson, industrial PCs).[^10]

These metrics ensure that high‑performing models can actually be deployed on factory hardware under regulatory requirements.[^1][^10]


## Summary of Metric Families

Across industrial datasets and tasks, the key metric families include:

- **Detection & classification**: accuracy, precision, recall, per‑class AP, mAP, AUROC.[^9][^7][^8][^10]
- **Segmentation & localization**: pixel AUROC, PRO/PRO‑AUC, IoU, Dice, boundary metrics.[^24][^2][^1]
- **Anomaly scoring**: image‑level AUROC, pixel‑level AUROC, PRO, KID/IS for anomaly generation quality.[^6][^2][^1]
- **Pose estimation**: ADD, ADD‑S, 2D reprojection error, success rate under thresholds, probabilistic task‑success.[^15][^13][^14]
- **Time‑series faults**: AUROC/AUPRC, F1, detection delay, earliness, alarm cardinality.[^3][^23][^21]
- **Activity recognition**: accuracy, F1, confusion matrices, captioning metrics (BLEU, METEOR, CIDEr), alignment metrics (Recall@K, mAP).[^4][^12][^11]
- **Tracking confidence**: PSR, APCE, max response value, channel reliability scores.[^20][^17][^16][^18]
- **System metrics**: FPS, latency, parameter count, energy usage, industrial standard compatibility.[^10][^1]

Together, these metrics provide a comprehensive view of performance, robustness, and deployability for industrial AI models across vision, time‑series, robotics, and human activity tasks.[^3][^2][^1][^10]

---

## References

1. [Industrial Anomaly Detection Benchmarks - Codesota](https://www.codesota.com/industrial) - Find current state-of-the-art AI models by task, benchmark, metric, source, and snapshot date. Human...

2. [OpenBayes Trends - MVTecAD Dataset](https://trends.openbayes.com/dataset/mvtecad) - MVTec AD is a dataset for benchmarking anomaly detection methods with a focus on industrial inspecti...

3. [FactoryNet: A Large-Scale Dataset toward Industrial Time-Series Foundation Models](https://arxiv.org/pdf/2605.09081v4.pdf)

4. [OpenMarcie: Dataset for Multimodal Action Recognition in ...](https://openreview.net/forum?id=emM6KIsBHl) - This paper introduces a multimodal dataset named OpenMarcie for industrial human activity understand...

5. [Industrial anomaly detection benchmark dataset](https://www.mvtec.com/research-teaching/datasets/mvtec-ad) - MVTec AD is a benchmark dataset for industrial anomaly detection and localization. It contains more ...

6. [SOTA benchmarks on MVTec AD and PapersWithCode | Wizwand](https://www.wizwand.com/dataset/mvtec-ad) - Explore 184 SOTA benchmarks and 197 papers that use the MVTec AD dataset family. Wizwand is the best...

7. [Attention-guided YOLOv5s-SDF model for accurate detection of strip steel surface defects](https://www.nature.com/articles/s41598-025-26313-5) - Surface defects on strip steel are often inevitable due to limitations in raw materials and manufact...

8. [Improved faster R-CNN for steel surface defect detection in ...](https://pmc.ncbi.nlm.nih.gov/articles/PMC12358517/) - Steel surface defect detection constitutes a critical inspection task in industrial production. To a...

9. [Table 2. The detection performance of different categories on NEU-DET dataset.](https://pmc.ncbi.nlm.nih.gov/articles/PMC12604762/table/pone.0334048.t002/) - The detection of defects on steel surfaces constitutes a vital area of research in computer vision, ...

10. [Table 2.](https://pmc.ncbi.nlm.nih.gov/articles/PMC12845972/table/sensors-26-00635-t002/) - To address the limitations of traditional lithium battery defect detection—low efficiency, high miss...

11. [Industrial Human Action Recognition Dataset InHARD - GitHub](https://github.com/vhavard/InHARD) - We propose a set of usage metrics of the InHARD dataset for future utilization. Firstly, we suggest ...

12. [[PDF] Human Activity Recognition in the Context of Industrial ... - mediaTUM](https://mediatum.ub.tum.de/doc/1281524/1281524.pdf)

13. [Object Pose Estimation in Robotics Revisited](https://arxiv.org/abs/1906.02783) - Vision based object grasping and manipulation in robotics require accurate estimation of object's 6D...

14. [[PDF] 6D Object Pose Estimation using Keypoints and Part Affinity Fields](https://www.ais.uni-bonn.de/papers/RCS_2021_Zappel.pdf) - The maximum thresholds are set to 10 cm for ADD(-S) and 40 px for the 2D projection metric. 4.2 Resu...

15. [YCB-Video Dataset for 6D Pose Estimation - Emergent Mind](https://www.emergentmind.com/topics/ycb-video-dataset) - The YCB-Video Dataset offers richly annotated RGB-D video sequences that enable robust evaluation of...

16. [Visual Object Tracking using Adaptive Correlation Filters](https://www.cs.colostate.edu/~draper/papers/bolme_cvpr10.pdf) - by DS Bolme · Cited by 4748 — The Peak-to-Sidelobe Ratio (PSR), which measures the strength of a cor...

17. [computer vision笔记：Peak-to-Sidelobe Ratio应用于目标跟踪](https://gsy00517.github.io/computer-vision20200118213942/) - 在Visual Object Tracking using Adaptive Correlation Filters一文中，我看到这样一句话：“The Peak-to-Sidelobe Ratio(P...

18. [单目标跟踪 （三） 小结](https://blog.csdn.net/weixin_41386168/article/details/110187380) - 文章浏览阅读5.9k次，点赞2次，收藏36次。特征：如果目标快速变形，基于HOG的梯度模板就跟不上了，如果快速变色，基于CN的颜色模板就跟不上了。置信度指标:高置信度更新：只有在跟踪置信度比较高的时候...

19. [MOSSE Tracking Algorithm | artivis/MOSSE_tracker | DeepWiki](https://deepwiki.com/artivis/MOSSE_tracker/5-mosse-tracking-algorithm) - This document provides a detailed explanation of the MOSSE (Minimum Output Sum of Squared Error) tra...

20. [Robust Scale Adaptive Tracking by Combining Correlation Filters with Sequential Monte Carlo](https://pmc.ncbi.nlm.nih.gov/articles/PMC5375798/) - A robust and efficient object tracking algorithm is required in a variety of computer vision applica...

21. [Anticipation, earliness, alarm cardinality: A new metric for industrial time-series anomaly detection](https://hal.science/hal-04577634v1/document)

22. [[PDF] A data-driven approach to fault diagnostics for industrial process ...](https://re.public.polimi.it/retrieve/f56c2be2-aa84-409d-9b01-6b58c1c22c52/ICCAD23.pdf)

23. [FactoryNet: A Large-Scale Dataset toward Industrial Time-Series ...](https://arxiv.org/html/2605.09081v4)

24. [[PDF] Deep Learning for Automated Quality Inspection of Mechanical Parts](https://lnu.diva-portal.org/smash/get/diva2:2072478/FULLTEXT01.pdf) - The performance of automated visual inspection systems is commonly evaluated using metrics such as a...

25. [FDD: a deep learning–based steel defect detectors](https://www.cse.scu.edu/~yliu1/papers/FDD-a_deep_learning-based_steel_defect_detector.pdf)

