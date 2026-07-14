#!/usr/bin/env python3
"""
mediapipe_pose_baseline.py — Head pose baseline from MediaPipe Face Mesh.

V2 T1.7: Run MediaPipe Face Mesh on the IndustReal test split and compute
angular MAE between MediaPipe-derived forward/up vectors and the HoloLens 2
sensor ground truth.

Publishes an uncontested vision-only head pose baseline so our ConvNeXt-Tiny
model (currently ~9.2 deg forward angular MAE) has a meaningful reference point.

Usage:
    python3 scripts/mediapipe_pose_baseline.py [--frame-step N] [--max-frames N]

Requirements (not installed in current env — see docstring below):
    mediapipe, opencv-python, numpy

Output:
    src/runs/rf_stages/checkpoints/efficiency_measured/mediapipe_baseline.json
    {
      "per_recording": { "<rec_id>": { "forward_mae": ..., "up_mae": ..., "n_frames": ... }, ... },
      "overall": { "forward_angular_mae_deg": ..., "up_angular_mae_deg": ..., "n_frames_total": ... },
      "comparison_to_model": {
        "model_forward_mae_deg": 9.21,
        "mediapipe_forward_mae_deg": ...,
        "delta_deg": ...,
        "mediapipe_wins": ...
      },
      "config": { "frame_step": ..., "max_frames": ..., "n_recordings": ... }
    }
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger("mediapipe_baseline")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
HEAD_POSE_POS_SCALE = 100.0  # matches training code normalization (unscale for mm)

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
CODE_DIR = REPO_ROOT / "code" / "industreal_improved"
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR / "src"))

from src.config import RECORDINGS_ROOT, TEST_CSV

# ---------------------------------------------------------------------------
# MediaPipe face landmark indices for Pose-from-Landmarks (P4L)
# ---------------------------------------------------------------------------
# Standard set of 6 landmarks used in cv2.solvePnP head pose estimation
# paired with corresponding 3D coordinates from the MediaPipe canonical
# face model (units: mm, origin at nose tip).
#
# Indices from MediaPipe Face Mesh topology:
FACE_LANDMARK_INDICES = {
    "nose_tip": 1,
    "chin": 199,
    "left_eye_corner": 33,
    "right_eye_corner": 263,
    "left_mouth_corner": 61,
    "right_mouth_corner": 291,
}

# Canonical 3D coordinates (mm) corresponding to the above landmarks.
# Derived from the MediaPipe canonical face model used in the Face Mesh
# solution. These are approximately centered at the nose tip and aligned
# to the camera coordinate system (x=right, y=down, z=forward).
_MODEL_POINTS = np.array(
    [
        [0.0, 0.0, 0.0],  # nose tip
        [0.0, -330.0, -65.0],  # chin (negative y = down in image coords)
        [-225.0, 170.0, -135.0],  # left eye corner (negative x = left)
        [225.0, 170.0, -135.0],  # right eye corner
        [-150.0, -150.0, -125.0],  # left mouth corner
        [150.0, -150.0, -125.0],  # right mouth corner
    ],
    dtype=np.float64,
)


def _rotation_matrix_to_vectors(R: np.ndarray):
    """Extract forward and up vectors from a 3x3 rotation matrix.

    In OpenCV camera convention (z-forward, y-down), the columns of the
    rotation matrix are:
        col 0 = x-axis (right)
        col 1 = y-axis (down, negated for our up)
        col 2 = z-axis (forward)

    Our convention (matching HoloLens 2 ground truth):
        forward = +z  (column 2)
        up      = -y  (negate column 1 since y points down in image coords)

    Returns:
        forward: np.ndarray [3] — unit vector pointing forward
        up:      np.ndarray [3] — unit vector pointing up
    """
    forward = R[:, 2].copy()  # z-axis
    # OpenCV camera: y points down. Our coordinate system: y points up.
    up = -R[:, 1].copy()  # negate y-axis
    return forward / (np.linalg.norm(forward) + 1e-12), up / (np.linalg.norm(up) + 1e-12)


def estimate_head_pose_from_landmarks(
    landmarks: np.ndarray, img_w: int, img_h: int
):
    """Estimate head rotation from 2D-3D landmark correspondences via solvePnP.

    Args:
        landmarks: np.ndarray [468, 2] — (x, y) pixel coordinates of face mesh.
        img_w: image width in pixels.
        img_h: image height in pixels.

    Returns:
        forward: np.ndarray [3] — unit forward vector (camera frame).
        up:      np.ndarray [3] — unit up vector (camera frame).
        rvec:    np.ndarray [3] — rotation vector from solvePnP.
        success: bool — True if solvePnP converged.
    """
    # Collect 2D-3D correspondences
    img_pts = []
    model_pts = []
    for name, model_pt in zip(
        FACE_LANDMARK_INDICES.keys(), _MODEL_POINTS, strict=True
    ):
        idx = FACE_LANDMARK_INDICES[name]
        # Landmarks may be normalized [0, 1] — denormalize if needed
        lm = landmarks[idx]
        if lm[0] <= 1.0 and lm[1] <= 1.0:
            lm_denorm = np.array([lm[0] * img_w, lm[1] * img_h], dtype=np.float64)
        else:
            lm_denorm = lm.astype(np.float64)
        img_pts.append(lm_denorm)
        model_pts.append(model_pt)

    img_pts = np.array(img_pts, dtype=np.float64).reshape(-1, 2)
    model_pts = np.array(model_pts, dtype=np.float64).reshape(-1, 3)

    # Camera intrinsics (approximate — assume principal point at image center,
    # focal length ~1.2x image width as a reasonable guess for webcam/FOV).
    focal_length = img_w * 1.2
    camera_matrix = np.array(
        [
            [focal_length, 0, img_w / 2.0],
            [0, focal_length, img_h / 2.0],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    # solvePnP with iterative refinement
    success, rvec, tvec = cv2.solvePnP(
        model_pts, img_pts, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )

    if not success:
        # Fallback: return identity vectors
        return (
            np.array([0.0, 0.0, 1.0], dtype=np.float32),
            np.array([0.0, 1.0, 0.0], dtype=np.float32),
            np.zeros(3),
            False,
        )

    # Rodrigues rotation vector -> rotation matrix
    R, _ = cv2.Rodrigues(rvec)
    forward, up = _rotation_matrix_to_vectors(R)
    return forward.astype(np.float32), up.astype(np.float32), rvec.flatten(), True


def angular_mae_deg(pred: np.ndarray, target: np.ndarray) -> float:
    """Angular MAE between two unit vectors in degrees.

    Uses arccos of cosine similarity, clamped to [-1, 1] for numerical safety.
    """
    pred_u = pred / (np.linalg.norm(pred) + 1e-12)
    target_u = target / (np.linalg.norm(target) + 1e-12)
    cos_sim = float(np.clip(np.dot(pred_u, target_u), -1.0, 1.0))
    return float(np.rad2deg(np.arccos(cos_sim)))


def load_pose_csv(rec_dir: Path) -> np.ndarray:
    """Load ground truth pose from a recording's pose.csv.

    Returns:
        np.ndarray [num_frames, 9] with columns:
            [forward_x, forward_y, forward_z,
             position_x, position_y, position_z,
             up_x, up_y, up_z]
    """
    pose_path = rec_dir / "pose.csv"
    if not pose_path.exists():
        logger.warning(f"pose.csv not found: {pose_path}")
        return None

    # pose.csv format (no header):
    #   frame.jpg,forward_x,forward_y,forward_z,position_x,position_y,position_z,up_x,up_y,up_z
    # Need to determine num_frames first so we can allocate.
    max_frame = 0
    rows = []
    with open(pose_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 10:
                continue
            try:
                frame_num = int(Path(parts[0]).stem)
                values = [float(v) for v in parts[1:10]]
                rows.append((frame_num, values))
                max_frame = max(max_frame, frame_num)
            except (ValueError, IndexError):
                continue

    if not rows:
        logger.warning(f"pose.csv empty or unparseable: {pose_path}")
        return None

    pose = np.zeros((max_frame + 1, 9), dtype=np.float32)
    for frame_num, vals in rows:
        pose[frame_num] = vals

    # Apply position scale (same as dataset loader)
    pose[:, 3:6] /= HEAD_POSE_POS_SCALE

    # Normalize forward and up vectors
    for i in range(pose.shape[0]):
        fwd = pose[i, 0:3]
        fnorm = np.linalg.norm(fwd)
        if fnorm > 1e-8:
            pose[i, 0:3] /= fnorm
        upv = pose[i, 6:9]
        unorm = np.linalg.norm(upv)
        if unorm > 1e-8:
            pose[i, 6:9] /= unorm

    return pose


def main():
    parser = argparse.ArgumentParser(
        description="MediaPipe Face Mesh head pose baseline for IndustReal."
    )
    parser.add_argument(
        "--frame-step",
        type=int,
        default=1,
        help="Process every Nth frame (default: 1 = all frames). "
        "Use --frame-step 3 to process 1/3rd of frames.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Max frames per recording (default: 0 = all). Use for quick smoke tests.",
    )
    parser.add_argument(
        "--recording",
        type=str,
        default=None,
        help="Process a single recording ID only (for debugging).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    logger.info("=" * 60)
    logger.info("MediaPipe Head Pose Baseline — IndustReal Test Split")
    logger.info("=" * 60)

    # Verify mediapipe is available
    try:
        import mediapipe as mp
    except ImportError:
        logger.error(
            "mediapipe is not installed. Install with:\n"
            "  pip install mediapipe opencv-python numpy\n\n"
            "Note: This script was designed per V2 T1.7 but requires\n"
            "mediapipe to actually run. On the target hardware, install\n"
            "it and re-run. The source is self-contained."
        )
        sys.exit(1)

    # Locate test split recordings
    recordings_root = Path(RECORDINGS_ROOT)
    test_csv = Path(TEST_CSV)

    if not test_csv.exists():
        logger.error(f"Test split CSV not found: {test_csv}")
        sys.exit(1)

    # Read recording IDs from the split CSV
    # Format: recording_id,split (or just recording_id per line)
    rec_ids = []
    with open(test_csv) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            rec_ids.append(parts[0])

    if not rec_ids:
        logger.error("No recordings found in test split CSV.")
        sys.exit(1)

    if args.recording:
        if args.recording not in rec_ids:
            logger.error(f"Recording {args.recording} not in test split.")
            sys.exit(1)
        rec_ids = [args.recording]

    logger.info(f"Found {len(rec_ids)} test recordings.")

    # ------------------------------------------------------------------
    # Initialize MediaPipe Face Mesh
    # ------------------------------------------------------------------
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5,
    )

    # ------------------------------------------------------------------
    # Per-recording evaluation
    # ------------------------------------------------------------------
    per_recording = {}
    total_forward_mae = 0.0
    total_up_mae = 0.0
    total_processed = 0
    total_failures = 0

    t_start = time.perf_counter()

    for rec_id in rec_ids:
        rec_dir = recordings_root / "test" / rec_id
        rgb_dir = rec_dir / "rgb"

        if not rgb_dir.exists():
            logger.warning(f"  rgb/ directory not found: {rgb_dir}, skipping.")
            continue

        # Load ground truth pose
        pose = load_pose_csv(rec_dir)
        if pose is None:
            logger.warning(f"  pose.csv not found for {rec_id}, skipping.")
            continue

        # List available frame files, sorted by frame number
        frame_paths = sorted(rgb_dir.glob("*.jpg"), key=lambda p: int(p.stem))
        if not frame_paths:
            logger.warning(f"  No frames found for {rec_id}, skipping.")
            continue

        # Apply frame step
        if args.frame_step > 1:
            frame_paths = frame_paths[:: args.frame_step]

        # Apply max frames limit
        if args.max_frames > 0:
            frame_paths = frame_paths[: args.max_frames]

        n_rec_frames = len(frame_paths)
        logger.info(
            f"  [{rec_id}] Processing {n_rec_frames} frames "
            f"(pose has {pose.shape[0]} rows)..."
        )

        rec_forward_mae = 0.0
        rec_up_mae = 0.0
        rec_processed = 0
        rec_failures = 0

        for fp in frame_paths:
            frame_num = int(fp.stem)

            # Ensure we have ground truth for this frame
            if frame_num >= pose.shape[0]:
                total_failures += 1
                rec_failures += 1
                continue

            # Load raw image (BGR from cv2, RGB for MediaPipe)
            img_bgr = cv2.imread(str(fp))
            if img_bgr is None:
                total_failures += 1
                rec_failures += 1
                continue

            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_h, img_w = img_rgb.shape[:2]

            # MediaPipe expects [H, W, 3] uint8
            results = face_mesh.process(img_rgb)

            if not results.multi_face_landmarks:
                total_failures += 1
                rec_failures += 1
                continue

            # Extract landmark coordinates (first face)
            lmks = results.multi_face_landmarks[0]
            landmarks = np.array(
                [(lm.x, lm.y) for lm in lmks.landmark], dtype=np.float32
            )

            # Estimate head pose from landmarks
            forward_pred, up_pred, rvec, success = estimate_head_pose_from_landmarks(
                landmarks, img_w, img_h
            )
            if not success:
                total_failures += 1
                rec_failures += 1
                continue

            # Ground truth vectors
            forward_gt = pose[frame_num, 0:3]
            up_gt = pose[frame_num, 6:9]

            # Angular MAE
            fwd_err = angular_mae_deg(forward_pred, forward_gt)
            up_err = angular_mae_deg(up_pred, up_gt)

            rec_forward_mae += fwd_err
            rec_up_mae += up_err
            rec_processed += 1

        # Average for this recording
        if rec_processed > 0:
            rec_fwd_mae = rec_forward_mae / rec_processed
            rec_up_mae_val = rec_up_mae / rec_processed
        else:
            rec_fwd_mae = None
            rec_up_mae_val = None

        per_recording[rec_id] = {
            "forward_angular_mae_deg": round(rec_fwd_mae, 2) if rec_fwd_mae is not None else None,
            "up_angular_mae_deg": round(rec_up_mae_val, 2) if rec_up_mae_val is not None else None,
            "n_frames_processed": rec_processed,
            "n_frames_failed": rec_failures,
        }

        if rec_processed > 0:
            total_forward_mae += rec_forward_mae
            total_up_mae += rec_up_mae
            total_processed += rec_processed
            total_failures += rec_failures

        logger.info(
            f"    forward MAE: {rec_fwd_mae:.2f} deg  |  "
            f"up MAE: {rec_up_mae_val:.2f} deg  "
            f"({rec_processed} frames, {rec_failures} failures)"
        )

    t_elapsed = time.perf_counter() - t_start

    # ------------------------------------------------------------------
    # Aggregate results
    # ------------------------------------------------------------------
    if total_processed > 0:
        overall_forward_mae = total_forward_mae / total_processed
        overall_up_mae = total_up_mae / total_processed
    else:
        overall_forward_mae = None
        overall_up_mae = None
        logger.error("No frames processed across any recording!")

    # Comparison to our ConvNeXt-Tiny model
    MODEL_FORWARD_MAE = 9.21  # from rf_stage_state.json
    comparison = None
    if overall_forward_mae is not None:
        delta = overall_forward_mae - MODEL_FORWARD_MAE
        comparison = {
            "model_forward_mae_deg": MODEL_FORWARD_MAE,
            "model_name": "ConvNeXt-Tiny (POPWMultiTaskModel)",
            "mediapipe_forward_mae_deg": round(overall_forward_mae, 2),
            "model_vs_mediapipe_delta_deg": round(delta, 2),
            "mediapipe_wins": delta > 0,
            "interpretation": (
                "If mediapipe_wins = True, MediaPipe baseline is WORSE than our model, "
                "meaning our learned approach outperforms the classical geometry-based baseline. "
                "If False, MediaPipe is competitive and worth investigating as a lightweight alternative."
            ),
        }

    results = {
        "per_recording": per_recording,
        "overall": {
            "forward_angular_mae_deg": round(overall_forward_mae, 2)
            if overall_forward_mae is not None
            else None,
            "up_angular_mae_deg": round(overall_up_mae, 2)
            if overall_up_mae is not None
            else None,
            "n_frames_processed": total_processed,
            "n_frames_failed": total_failures,
            "n_recordings": len(rec_ids),
            "elapsed_seconds": round(t_elapsed, 1),
        },
        "comparison_to_model": comparison,
        "config": {
            "frame_step": args.frame_step,
            "max_frames_per_recording": args.max_frames if args.max_frames > 0 else "all",
            "method": "MediaPipe Face Mesh + solvePnP (6-point)",
            "landmark_indices": FACE_LANDMARK_INDICES,
        },
    }

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    output_dir = (
        REPO_ROOT
        / "code"
        / "industreal_improved"
        / "src"
        / "runs"
        / "rf_stages"
        / "checkpoints"
        / "efficiency_measured"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "mediapipe_baseline.json"

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    if overall_forward_mae is not None:
        logger.info(f"  Forward angular MAE (MediaPipe):  {overall_forward_mae:.2f} deg")
        logger.info(f"  Up angular MAE (MediaPipe):       {overall_up_mae:.2f} deg")
        logger.info(f"  Our ConvNeXt-Tiny model:          {MODEL_FORWARD_MAE:.2f} deg")
        logger.info(f"  Delta (MediaPipe vs Model):       {delta:+.2f} deg")
        if comparison["mediapipe_wins"]:
            logger.info("  => Our model BEATS the MediaPipe baseline.")
        else:
            logger.info("  => MediaPipe is competitive with our model.")
    logger.info(f"  Processed {total_processed} frames across {len(rec_ids)} recordings.")
    logger.info(f"  Frame failures (no face detected / corrupt): {total_failures}")
    logger.info(f"  Elapsed time: {t_elapsed:.1f}s")
    logger.info(f"\nResults saved to: {output_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
