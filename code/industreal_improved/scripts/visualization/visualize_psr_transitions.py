"""
PSR Component State Transition Diagram for IndustReal.

Generates a temporal diagram showing the assembly state of each of the 11
components over time. Each row is a component (0-10); the x-axis is the
frame number. Filled cells = assembled (state=1), empty = not assembled.

Optionally overlays ground-truth vs predicted states from a model run.

Usage:
    python visualize_psr_transitions.py

Output:
    headpose_visualizations/psr_transitions_01_assy_0_1.png
"""

from pathlib import Path

import cv2
import numpy as np

REC_DIR = Path(
    "/home/newadmin/swarm-bot/project/popw/working/data/dataset/"
    "industreal/recordings/train/01_assy_0_1"
)
PSR_RAW_CSV = REC_DIR / "PSR_labels_raw.csv"
NUM_COMPONENTS = 11
OUTPUT_DIR = Path(
    "/home/newadmin/swarm-bot/project/popw/working/code/industreal_improved/psr_visualizations"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COMPONENT_LABELS = [f"Comp{i}" for i in range(NUM_COMPONENTS)]
FPS = 30

CELL_H = 40
CELL_W = 3
MARGIN_LEFT = 90
MARGIN_TOP = 60
MARGIN_BOTTOM = 20
MARGIN_RIGHT = 20


def load_dense_psr(csv_path: Path) -> np.ndarray:
    if not csv_path.exists():
        return None
    sparse_changes = {}
    max_frame = 0
    with open(csv_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 12:
                continue
            try:
                frame_num = int(Path(parts[0]).stem)
                states = [int(parts[1 + i]) for i in range(NUM_COMPONENTS)]
                sparse_changes[frame_num] = states
                if frame_num > max_frame:
                    max_frame = frame_num
            except (ValueError, IndexError):
                continue

    if not sparse_changes:
        return None

    dense = np.zeros((max_frame + 1, NUM_COMPONENTS), dtype=np.uint8)
    current = np.zeros(NUM_COMPONENTS, dtype=np.uint8)
    last_frame = 0
    for frame in sorted(sparse_changes):
        dense[last_frame : frame + 1] = current
        current = np.array(sparse_changes[frame], dtype=np.uint8)
        last_frame = frame
        dense[frame] = current
    dense[last_frame:] = current
    return dense


def draw_diagram(
    states: np.ndarray,
    title: str,
    stride: int = 1,
) -> np.ndarray:
    num_frames = states.shape[0]
    num_comp = states.shape[1]

    img_w = MARGIN_LEFT + num_frames * CELL_W + MARGIN_RIGHT
    img_h = MARGIN_TOP + num_comp * CELL_H + MARGIN_BOTTOM
    img = np.ones((img_h, img_w, 3), dtype=np.uint8) * 30

    for t in range(0, num_frames, stride):
        x = MARGIN_LEFT + t * CELL_W
        cv2.line(img, (x, MARGIN_TOP), (x, img_h - MARGIN_BOTTOM), (60, 60, 60), 1)

    for c in range(num_comp):
        y = MARGIN_TOP + c * CELL_H
        cv2.line(img, (MARGIN_LEFT, y), (img_w - MARGIN_RIGHT, y), (80, 80, 80), 1)

    for t in range(num_frames):
        x = MARGIN_LEFT + t * CELL_W
        for c in range(num_comp):
            y = MARGIN_TOP + c * CELL_H
            if states[t, c] == 1:
                cv2.rectangle(
                    img,
                    (x + 1, y + 1),
                    (x + CELL_W - 1, y + CELL_H - 1),
                    (0, 200, 130),
                    -1,
                )
            else:
                cv2.rectangle(
                    img,
                    (x + 1, y + 1),
                    (x + CELL_W - 1, y + CELL_H - 1),
                    (18, 18, 18),
                    -1,
                )

    for c, label in enumerate(COMPONENT_LABELS):
        y = MARGIN_TOP + c * CELL_H + CELL_H // 2 + 5
        cv2.putText(
            img,
            label,
            (5, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (200, 200, 200),
            1,
        )

    for t in range(0, num_frames, 100 * stride):
        x = MARGIN_LEFT + t * CELL_W
        cv2.putText(
            img,
            f"{t}",
            (x - 5, img_h - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (150, 150, 150),
            1,
        )

    cv2.putText(
        img,
        title,
        (MARGIN_LEFT, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        img,
        f"{num_frames} frames @ {FPS}fps = {num_frames / FPS:.1f}s",
        (MARGIN_LEFT, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (180, 180, 180),
        1,
    )
    return img


def main() -> None:
    print(f"[psr_viz] Loading PSR raw from {PSR_RAW_CSV}")
    dense = load_dense_psr(PSR_RAW_CSV)

    if dense is None:
        print("[psr_viz] No PSR data found — exiting.")
        return

    print(f"[psr_viz] Dense PSR shape: {dense.shape}")

    img = draw_diagram(
        dense,
        title=f"PSR Component States — 01_assy_0_1",
        stride=3,
    )

    out_path = OUTPUT_DIR / "psr_transitions_01_assy_0_1.png"
    cv2.imwrite(str(out_path), img)
    print(f"[psr_viz] Saved: {out_path}")

    transitions = []
    for c in range(NUM_COMPONENTS):
        changes = np.where(np.diff(dense[:, c]) != 0)[0]
        transitions.append(len(changes))

    print(f"[psr_viz] Per-component transition counts:")
    for c in range(NUM_COMPONENTS):
        total_assembled = int(dense[:, c].sum())
        pct = 100 * total_assembled / dense.shape[0]
        print(
            f"  Comp{c}: {transitions[c]} transitions, "
            f"{total_assembled}/{dense.shape[0]} frames assembled ({pct:.0f}%)"
        )


if __name__ == "__main__":
    main()
