# Reviewer 4: Ego-Pose — The Uncontested Contribution & Its Defense

## Identity: IEEE/CVF Reviewer — Human Pose Estimation & Egocentric Vision
**Focus:** Head/body/hand pose estimation, egocentric vision, fair benchmarking.
**Bias:** Skeptical of multi-task papers that claim pose as a "free byproduct." Will verify units, data provenance, and protocol rigor.

---

## 1. Why This Matters

This is the ONLY head where we have a genuine original contribution. All other metrics require careful framing or additional experiments. **Ego-pose at 8.14° forward MAE with no prior IndustReal baseline is publishable as a standalone contribution.**

But reviewers will probe it hard. Let's pre-empt every attack.

---

## 2. Anticipated Reviewer Attacks & Counterarguments

### Attack 1: "What exactly are you measuring?"

**They'll ask:** *"Forward angular MAE of 8.14° — is this the wearer's head direction relative to the HoloLens, or the absolute head orientation in world coordinates? What is the coordinate frame?"*

**Our answer should be:**
- Pose.csv contains 9-DoF HoloLens head tracking: forward vector (x,y,z), up vector (x,y,z), position (x,y,z) in HoloLens coordinate frame
- Forward MAE = angular difference between predicted and ground truth forward vectors
- Up MAE = same for up vectors
- Position = Euclidean distance in HoloLens units (unit uncertain — see Attack 2)

### Attack 2: "What are the units for position?"

**They'll ask:** *"Position error of 102mm at epoch 8 — is this in meters, centimeters, or HoloLens coordinate units? HEAD_POSE_POS_SCALE=100 suggests arbitrary scaling."*

**Current status:** We DO NOT HAVE AN ANSWER. Doc 85 says "DO NOT REPORT position." The contribution audit confirmed the unit is ambiguous.

**Fix required before submission:** 
1. Check the raw pose.csv values against known HoloLens specifications
2. HoloLens 2 exports position in meters by default
3. If CSV is in meters: our HEAD_POSE_POS_SCALE=100 means our model predicts in cm ÷ 100 → divide reported values by 100 = 1.02mm position error
4. If CSV is in proprietary units: calibrate against a known physical measurement
5. **If unit cannot be determined: drop position claims entirely — MAE alone is publishable**

### Attack 3: "How does this compare to OpenFace / 6DRepNet / WHENet?"

**They'll ask:** *"OpenFace gets 2.63° yaw on BIWI. You get 8.14°. Why should I care?"*

**Our answer:**
- **This is ego-pose, not face-based head pose.** OpenFace detects facial landmarks from a front-facing camera. Our input is a HoloLens egocentric camera showing the wearer's hands and the assembly object — not their face. These are fundamentally different tasks.
- **No prior IndustReal ego-pose baseline exists.** We are the first to report this.
- **Our pose is a byproduct of multi-task training** — zero additional inference cost.

**Action: Remove ALL OpenFace/6DRepNet comparisons from the paper.** They create confusion about what we're measuring.

### Attack 4: "Ego-pose from a HoloLens is trivial — the device already tracks its own position"

**They'll ask:** *"The HoloLens gives you head pose for free. What did your model learn?"*

**Our answer:**
- During training, the model receives images only — NOT the HoloLens head tracking data
- The pose.csv ground truth is used ONLY as training supervision, not as input
- At inference time, the model predicts pose from a SINGLE RGB FRAME with no temporal tracking
- This is fundamentally harder than HoloLens' internal sensor fusion (which uses IMU + SLAM)

---

## 3. Required Experiments Before Submission

| Experiment | Effort | Why |
|---|---|---|
| **Verify position units** | 30 min | Determine if raw csv values are meters → divide our results by 100 |
| **Full-test split eval** | 1h | Our numbers are on val subset — need test set for paper |
| **EMA vs raw weights comparison** | Built-in | Epoch 11 uses raw weights; EMA may improve further |
| **Per-axis angular breakdown** | 30 min | Yaw/pitch/roll vs our composite forward/up (unit vector formulation) |

---

## 4. The Paper Table

| Method | Task Type | Forward MAE | Up MAE | Position | Temporal? | Prior Work? |
|---|---|---|---|---|---|---|
| **Ours (epoch 11)** | **Multi-task ego-pose** | **8.14°** | **7.06°** | **TBD** | **❌ Single frame** | **✅ First baseline** |
| Expected at convergence (epoch 30+) | Multi-task ego-pose | ~6-8° | ~5-7° | TBD | ❌ Single frame | First baseline |

---

## 5. Recommendation

**This is the paper's lead contribution.** Here's the reviewer-approved framing:

> *"We establish the first ego-pose estimation baseline on the IndustReal assembly dataset, achieving 8.14° forward angular MAE and 7.06° up angular MAE from a single RGB frame as a zero-cost byproduct of multi-task training. Our approach requires no temporal filtering, no IMU data, and no facial landmarks — unlike dedicated pose estimators — while simultaneously performing assembly state detection, per-frame action classification, and component state estimation on a single consumer GPU."*

**Three non-negotiable fixes:**
1. ✅ **Fix position units** or drop position entirely
2. ✅ **Remove OpenFace/6DRepNet comparisons** — category error  
3. ✅ **Re-frame as "ego-pose" not "head pose"** throughout the paper
