"""
High-contrast visualization of IndustReal hand joints.
Uses bright colors on top of original image for maximum visibility.
"""

from PIL import Image, ImageDraw, ImageFont
import csv
import os

img_dir = "/home/newadmin/swarm-bot/project/popw/working/data/dataset/industreal/recordings/train/01_assy_0_1/rgb"
hands_csv = "/home/newadmin/swarm-bot/project/popw/working/data/dataset/industreal/recordings/train/01_assy_0_1/hands.csv"
output_dir = "/home/newadmin/swarm-bot/project/popw/working/code/industreal_improved/hand_joint_visualizations"
os.makedirs(output_dir, exist_ok=True)

frames = []
with open(hands_csv, "r") as f:
    frames = list(csv.reader(f))

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

FRAME_INDICES = [10, 50, 99]

for frame_idx in FRAME_INDICES:
    row = frames[frame_idx]
    frame_name = row[0]
    all_coords = [float(x) for x in row[1:]]

    img = Image.open(f"{img_dir}/{frame_name}").convert("RGB")
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except:
        font = ImageFont.load_default()

    for hand_start, color, label in [(0, (255, 20, 20), "LEFT"), (52, (20, 255, 20), "RIGHT")]:
        joints = []
        for i in range(26):
            x = all_coords[hand_start + i * 2]
            y = all_coords[hand_start + i * 2 + 1]
            joints.append((x, y))

        # Draw thick lines first (underneath)
        for edge in HAND_EDGES:
            i, j = edge
            x1, y1 = joints[i]
            x2, y2 = joints[j]
            if 0 <= x1 <= 1280 and 0 <= y1 <= 720 and 0 <= x2 <= 1280 and 0 <= y2 <= 720:
                draw.line([(x1, y1), (x2, y2)], fill=color, width=8)

        # Draw large joint circles with white border
        for i, (x, y) in enumerate(joints):
            if 0 <= x <= 1280 and 0 <= y <= 720:
                r = 12 if i == 0 else 8
                # Bright white border
                draw.ellipse([x - r - 3, y - r - 3, x + r + 3, y + r + 3], fill=(255, 255, 255))
                # Filled circle with color
                draw.ellipse([x - r, y - r, x + r, y + r], fill=color)
                # Label
                if i == 0:
                    draw.text(
                        (x + 14, y - 20), f"{label} #{frame_idx}", fill=(255, 255, 0), font=font
                    )

    output_path = f"{output_dir}/{frame_name}"
    img.save(output_path, quality=95)
    print(f"Saved: {output_path}")

print(f"\nAll visualizations saved to: {output_dir}/")
