"""
Visualization of IndustReal hand joints with TRACKING ERROR markers.
All joints drawn, but suspicious ones (off-hand pixels) are marked with X.
"""

from PIL import Image, ImageDraw, ImageFont
import csv
import os
import numpy as np

img_dir = "/home/newadmin/swarm-bot/project/popw/working/data/dataset/industreal/recordings/train/01_assy_0_1/rgb"
hands_csv = "/home/newadmin/swarm-bot/project/popw/working/data/dataset/industreal/recordings/train/01_assy_0_1/hands.csv"
output_dir = "/home/newadmin/swarm-bot/project/popw/working/code/industreal_improved/hand_joint_visualizations_final"
os.makedirs(output_dir, exist_ok=True)

frames = list(csv.reader(open(hands_csv)))

HAND_EDGES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (5, 9),
    (9, 13),
    (13, 17),
]


def is_valid_joint(x, y, img):
    x, y = int(x), int(y)
    if x < 0 or x >= 1280 or y < 0 or y >= 720:
        return False
    brightness = img[y, x].mean()
    return 15 < brightness < 100


try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
except:
    font = ImageFont.load_default()

for frame_idx in [10, 50, 99]:
    row = frames[frame_idx]
    frame_name = row[0]
    coords = [float(x) for x in row[1:]]

    img = Image.open(f"{img_dir}/{frame_name}").convert("RGB")
    img_arr = np.array(img)
    draw = ImageDraw.Draw(img)

    for hand_start, hand_label, color in [(0, "L", (255, 50, 50)), (52, "R", (50, 255, 50))]:
        joints = [(coords[hand_start + i * 2], coords[hand_start + i * 2 + 1]) for i in range(26)]

        # Draw edges (lines between joints)
        for i, j in HAND_EDGES:
            x1, y1 = joints[i]
            x2, y2 = joints[j]
            valid1 = is_valid_joint(x1, y1, img_arr)
            valid2 = is_valid_joint(x2, y2, img_arr)
            if valid1 and valid2:
                draw.line([(x1, y1), (x2, y2)], fill=color, width=5)
            elif valid1 or valid2:
                # Dashed/dotted line for partially valid connections
                draw.line([(x1, y1), (x2, y2)], fill=(255, 200, 0), width=3)

        # Draw joints
        for i, (x, y) in enumerate(joints):
            valid = is_valid_joint(x, y, img_arr)
            r = 8 if i == 0 else 5
            if valid:
                draw.ellipse([x - r - 2, y - r - 2, x + r + 2, y + r + 2], fill=(255, 255, 255))
                draw.ellipse([x - r, y - r, x + r, y + r], fill=color)
            else:
                # Draw X for tracking errors
                draw.text((int(x) + 8, int(y) - 8), f"!{i}", fill=(255, 0, 255), font=font)

    out_path = f"{output_dir}/{frame_name}"
    img.save(out_path, quality=95)
    print(f"Saved: {out_path}")

print(f"\nFinal visualizations with error markers: {output_dir}/")
