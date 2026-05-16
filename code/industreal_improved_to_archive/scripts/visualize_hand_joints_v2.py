"""
Improved visualization of IndustReal hand joint annotations.
Thicker lines, larger dots, more visible.
"""
import sys
sys.path.insert(0, '/home/newadmin/swarm-bot/project/popw/working/code/industreal_improved')

from PIL import Image, ImageDraw, ImageFont
import csv
import os

img_dir = '/home/newadmin/swarm-bot/project/popw/working/data/dataset/industreal/recordings/train/01_assy_0_1/rgb'
hands_csv = '/home/newadmin/swarm-bot/project/popw/working/data/dataset/industreal/recordings/train/01_assy_0_1/hands.csv'
output_dir = '/home/newadmin/swarm-bot/project/popw/working/code/industreal_improved/hand_joint_visualizations'
os.makedirs(output_dir, exist_ok=True)

# Read hands.csv
frames = []
with open(hands_csv, 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        frames.append(row)

# MediaPipe hand skeleton (26 joints per hand)
HAND_EDGES = [
    (0,1), (1,2), (2,3), (3,4),   # thumb
    (0,5), (5,6), (6,7), (7,8),   # index
    (0,9), (9,10), (10,11), (11,12),  # middle
    (0,13), (13,14), (14,15), (15,16),  # ring
    (0,17), (17,18), (18,19), (19,20),  # pinky
    (5,9), (9,13), (13,17),  # palm
]

# Frame indices to visualize
FRAME_INDICES = [10, 50, 99]

for frame_idx in FRAME_INDICES:
    row = frames[frame_idx]
    frame_name = row[0]
    all_coords = [float(x) for x in row[1:]]

    img = Image.open(f'{img_dir}/{frame_name}').convert('RGB')
    draw = ImageDraw.Draw(img)

    # Try to use a font, fallback to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except:
        font = ImageFont.load_default()

    for hand_start, color, label in [(0, (255, 0, 0), 'LEFT'), (52, (0, 255, 0), 'RIGHT')]:
        joints = []
        for i in range(26):
            x = all_coords[hand_start + i * 2]
            y = all_coords[hand_start + i * 2 + 1]
            joints.append((x, y))

        # Draw edges with THICK lines
        for edge in HAND_EDGES:
            i, j = edge
            x1, y1 = joints[i]
            x2, y2 = joints[j]
            if 0 <= x1 <= 1280 and 0 <= y1 <= 720 and 0 <= x2 <= 1280 and 0 <= y2 <= 720:
                draw.line([(x1, y1), (x2, y2)], fill=color, width=6)

        # Draw LARGE joint points with white outline
        for i, (x, y) in enumerate(joints):
            if 0 <= x <= 1280 and 0 <= y <= 720:
                r = 10 if i == 0 else 7  # wrist much larger
                # White outline
                draw.ellipse([x-r-2, y-r-2, x+r+2, y+r+2], fill=(255, 255, 255))
                # Colored fill
                draw.ellipse([x-r, y-r, x+r, y+r], fill=color)
                # Label wrist
                if i == 0:
                    draw.text((x + 12, y - 20), f'{label}', fill=color, font=font)

    output_path = f'{output_dir}/{frame_name}'
    img.save(output_path, quality=95)
    print(f'Saved: {output_path}')

print(f'\nAll visualizations saved to: {output_dir}/')