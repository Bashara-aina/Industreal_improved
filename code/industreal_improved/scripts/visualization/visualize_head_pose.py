"""
Head Pose Visualization for IndustReal.
Overlays forward-direction and up-direction arrows on RGB frames from pose.csv.
The 9-DoF pose format is:
    frame.jpg, forward_x, forward_y, forward_z,
               pos_x, pos_y, pos_z,
               up_x, up_y, up_z

Arrows are drawn from the projected head position in world coordinates.
Green = forward gaze direction, Blue = up direction.
"""

import csv
from pathlib import Path

import cv2
import numpy as np

REC_DIR = Path(
    "/home/newadmin/swarm-bot/project/popw/working/data/dataset/"
    "industreal/recordings/train/01_assy_0_1"
)
IMG_DIR = REC_DIR / "rgb"
POSE_CSV = REC_DIR / "pose.csv"
OUTPUT_DIR = Path(
    "/home/newadmin/swarm-bot/project/popw/working/code/industreal_improved/headpose_visualizations"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ARROW_LEN_PX = 80
CIRCLE_RADIUS = 6
FONT = cv2.FONT_HERSHEY_SIMPLEX


def load_pose_csv(path: Path) -> dict[int, np.ndarray]:
    poses = {}
    if not path.exists():
        print(f"[head_pose_viz] pose.csv not found at {path}")
        return poses
    with open(path, encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) < 10:
                continue
            try:
                frame_num = int(Path(row[0]).stem)
                values = [float(v) for v in row[1:10]]
                poses[frame_num] = np.array(values, dtype=np.float32)
            except (ValueError, IndexError):
                continue
    return poses


def project_to_image(
    pos: np.ndarray,
    f_x: float = 900.0,
    c_x: float = 640.0,
    f_y: float = 900.0,
    c_y: float = 360.0,
) -> tuple[float, float]:
    if abs(pos[2]) < 1e-6:
        return float("nan"), float("nan")
    u = f_x * pos[0] / pos[2] + c_x
    v = f_y * pos[1] / pos[2] + c_y
    return float(u), float(v)


def draw_axis(
    img: np.ndarray,
    pos: np.ndarray,
    forward: np.ndarray,
    up: np.ndarray,
    fx: float = 900.0,
    fy: float = 900.0,
    cx: float = 640.0,
    cy: float = 360.0,
    scale: float = 0.12,
) -> None:
    px, py = project_to_image(pos, fx, fy, cx, cy)
    if np.isnan(px) or np.isnan(py):
        return

    px_i, py_i = int(px), int(py)

    fwd_scaled = forward * scale
    up_scaled = up * scale

    fwd_end_x = px + fwd_scaled[0]
    fwd_end_y = py - fwd_scaled[1]
    up_end_x = px + up_scaled[0]
    up_end_y = py - up_scaled[1]

    cv2.circle(img, (px_i, py_i), CIRCLE_RADIUS, (0, 255, 255), -1)
    cv2.putText(img, "H", (px_i + 8, py_i - 8), FONT, 0.6, (0, 255, 255), 2)

    cv2.arrowedLine(
        img, (px_i, py_i), (int(fwd_end_x), int(fwd_end_y)), (0, 255, 0), 3, tipLength=0.3
    )
    cv2.putText(img, "FWD", (int(fwd_end_x) + 5, int(fwd_end_y)), FONT, 0.5, (0, 255, 0), 1)

    cv2.arrowedLine(
        img, (px_i, py_i), (int(up_end_x), int(up_end_y)), (255, 0, 0), 3, tipLength=0.3
    )
    cv2.putText(img, "UP", (int(up_end_x) + 5, int(up_end_y)), FONT, 0.5, (255, 0, 0), 1)


def main() -> None:
    print(f"[head_pose_viz] Loading poses from {POSE_CSV}")
    poses = load_pose_csv(POSE_CSV)
    print(f"[head_pose_viz] Loaded {len(poses)} frame poses")

    if not poses:
        print("[head_pose_viz] No pose data found — exiting.")
        return

    frame_indices = sorted(poses.keys())
    test_frames = [
        frame_indices[len(frame_indices) // 4],
        frame_indices[len(frame_indices) // 2],
        frame_indices[3 * len(frame_indices) // 4],
    ]

    for frame_num in test_frames:
        pose = poses[frame_num]
        frame_name = f"{frame_num:06d}.jpg"
        img_path = IMG_DIR / frame_name

        if not img_path.exists():
            print(f"[head_pose_viz] Frame not found: {img_path}")
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        forward = pose[0:3]
        pos = pose[3:6]
        up = pose[6:9]

        img_color = img.copy()
        draw_axis(img_color, pos, forward, up)

        h, w = img_color.shape[:2]
        cv2.putText(
            img_color,
            f"Frame {frame_num} | pos=({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})",
            (10, 30),
            FONT,
            0.6,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            img_color,
            f"fwd=({forward[0]:.2f}, {forward[1]:.2f}, {forward[2]:.2f})  "
            f"up=({up[0]:.2f}, {up[1]:.2f}, {up[2]:.2f})",
            (10, 60),
            FONT,
            0.6,
            (255, 255, 255),
            2,
        )

        out_path = OUTPUT_DIR / f"headpose_{frame_name}"
        cv2.imwrite(str(out_path), img_color)
        print(f"[head_pose_viz] Saved: {out_path}")

    all_frames_output = OUTPUT_DIR / "all_frames"
    all_frames_output.mkdir(exist_ok=True)

    for frame_num in frame_indices:
        pose = poses[frame_num]
        frame_name = f"{frame_num:06d}.jpg"
        img_path = IMG_DIR / frame_name

        if not img_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        forward = pose[0:3]
        pos = pose[3:6]
        up = pose[6:9]

        draw_axis(img, pos, forward, up)

        out_path = all_frames_output / f"hp_{frame_name}"
        cv2.imwrite(str(out_path), img)

    print(f"[head_pose_viz] All frames: {all_frames_output}/  ({len(frame_indices)} images)")


if __name__ == "__main__":
    main()
