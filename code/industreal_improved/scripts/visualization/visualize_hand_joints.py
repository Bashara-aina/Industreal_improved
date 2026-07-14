"""
Visualize real IndustReal hand joint annotations on RGB images.
Uses actual data from recordings/train/01_assy_0_1/

Bug fix: coordinates must be drawn at full scale on full-size image (1280x720).
"""

import sys

sys.path.insert(0, "/home/newadmin/swarm-bot/project/popw/working/code/industreal_improved")

from PIL import Image, ImageDraw
import csv
import os

img_dir = "/home/newadmin/swarm-bot/project/popw/working/data/dataset/industreal/recordings/train/01_assy_0_1/rgb"
hands_csv = "/home/newadmin/swarm-bot/project/popw/working/data/dataset/industreal/recordings/train/01_assy_0_1/hands.csv"
output_dir = "/home/newadmin/swarm-bot/project/popw/working/code/industreal_improved/hand_joint_visualizations"
os.makedirs(output_dir, exist_ok=True)

# Read hands.csv
frames = []
with open(hands_csv, "r") as f:
    reader = csv.reader(f)
    for row in reader:
        frames.append(row)

# MediaPipe hand skeleton (26 joints per hand)
# Index: 0=wrist, 1=thumb_cmc, 2=thumb_mcp, 3=thumb_ip, 4=thumb_tip,
# 5=index_mcp, 6=index_pip, 7=index_dip, 8=index_tip,
# 9=middle_mcp, 10=middle_pip, 11=middle_dip, 12=middle_tip,
# 13=ring_mcp, 14=ring_pip, 15=ring_dip, 16=ring_tip,
# 17=pinky_mcp, 18=pinky_pip, 19=pinky_dip, 20=pinky_tip
HAND_EDGES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),  # thumb
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),  # index
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),  # middle
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),  # ring
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),  # pinky
    (5, 9),
    (9, 13),
    (13, 17),  # palm
]

# Frame indices to visualize
FRAME_INDICES = [10, 50, 99]

for frame_idx in FRAME_INDICES:
    row = frames[frame_idx]
    frame_name = row[0]
    all_coords = [float(x) for x in row[1:]]  # 104 coords: left(52) + right(52)

    # Load image at full resolution
    img = Image.open(f"{img_dir}/{frame_name}").convert("RGB")
    draw = ImageDraw.Draw(img)

    for hand_start, color, label in [(0, (255, 0, 0), "LEFT"), (52, (0, 255, 0), "RIGHT")]:
        # Extract 26 joint (x,y) pairs from flat coords
        joints = []
        for i in range(26):
            x = all_coords[hand_start + i * 2]
            y = all_coords[hand_start + i * 2 + 1]
            joints.append((x, y))

        # Draw edges
        for edge in HAND_EDGES:
            i, j = edge
            x1, y1 = joints[i]
            x2, y2 = joints[j]
            if x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0:
                draw.line([(x1, y1), (x2, y2)], fill=color, width=3)

        # Draw joint points
        for i, (x, y) in enumerate(joints):
            if x > 0 and y > 0:
                r = 5 if i == 0 else 4  # wrist slightly larger
                draw.ellipse([x - r, y - r, x + r, y + r], fill=color, outline=(255, 255, 255))
                if i == 0:
                    draw.text((x + 6, y - 12), f"{label} wrist ({x:.0f},{y:.0f})", fill=color)

    # Save
    output_path = f"{output_dir}/{frame_name}"
    img.save(output_path)
    print(f"Saved: {output_path}")

print(f"\nAll visualizations saved to: {output_dir}/")
