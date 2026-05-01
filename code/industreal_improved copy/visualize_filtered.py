"""
Improved visualization of IndustReal hand joints with filtering of HL2 tracking errors.
Filters out joints that fall on non-hand pixels (shadows, occlusions).
"""
from PIL import Image, ImageDraw, ImageFont
import csv
import os
import numpy as np

img_dir = '/home/newadmin/swarm-bot/project/popw/working/data/dataset/industreal/recordings/train/01_assy_0_1/rgb'
hands_csv = '/home/newadmin/swarm-bot/project/popw/working/data/dataset/industreal/recordings/train/01_assy_0_1/hands.csv'
output_dir = '/home/newadmin/swarm-bot/project/popw/working/code/industreal_improved/hand_joint_visualizations_v5'
os.makedirs(output_dir, exist_ok=True)

frames = list(csv.reader(open(hands_csv)))

HAND_EDGES = [
    (0,1), (1,2), (2,3), (3,4),
    (0,5), (5,6), (6,7), (7,8),
    (0,9), (9,10), (10,11), (11,12),
    (0,13), (13,14), (14,15), (15,16),
    (0,17), (17,18), (18,19), (19,20),
    (5,9), (9,13), (13,17),
]

def is_hand_pixel(r, g, b):
    """Check if pixel looks like hand skin in this scene (dark, 20-80 brightness, reddish)"""
    brightness = (r + g + b) / 3
    return 20 < brightness < 80 and r >= g >= b

def is_valid_joint(x, y, img):
    """Check if joint position is on hand-like pixels"""
    x, y = int(x), int(y)
    if x < 0 or x >= 1280 or y < 0 or y >= 720:
        return False
    r, g, b = img[y, x]
    brightness = (r + g + b) / 3
    if brightness < 15 or brightness > 100:
        return False
    return True

try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
except:
    font = ImageFont.load_default()

for frame_idx in [10, 50, 99]:
    row = frames[frame_idx]
    frame_name = row[0]
    coords = [float(x) for x in row[1:]]

    img = Image.open(f'{img_dir}/{frame_name}').convert('RGB')
    img_arr = np.array(img)
    draw = ImageDraw.Draw(img)

    stats = {'L': {'total': 0, 'filtered': 0}, 'R': {'total': 0, 'filtered': 0}}

    for hand_start, hand_label, color in [(0, 'L', (255, 0, 0)), (52, 'R', (0, 255, 0))]:
        joints = [(coords[hand_start+i*2], coords[hand_start+i*2+1]) for i in range(26)]
        valid_joints = []
        
        for i, (x, y) in enumerate(joints):
            stats[hand_label]['total'] += 1
            if is_valid_joint(x, y, img_arr):
                valid_joints.append((i, x, y))

        # Draw edges only for VALID connections
        for i, j in HAND_EDGES:
            x1, y1 = joints[i]
            x2, y2 = joints[j]
            if (is_valid_joint(x1, y1, img_arr) and 
                is_valid_joint(x2, y2, img_arr)):
                draw.line([(x1,y1),(x2,y2)], fill=color, width=6)
        
        # Draw valid joints
        for i, x, y in valid_joints:
            stats[hand_label]['filtered'] += 1
            r = 10 if i == 0 else 7
            draw.ellipse([x-r-3, y-r-3, x+r+3, y+r+3], fill=(255, 255, 255))
            draw.ellipse([x-r, y-r, x+r, y+r], fill=color)
            if i == 0:
                draw.text((x + 12, y - 18), f'{hand_label} wrist', fill=(255, 255, 0), font=font)

    print(f"Frame {frame_idx}: {frame_name}")
    print(f"  Left: {stats['L']['filtered']}/{stats['L']['total']} joints drawn")
    print(f"  Right: {stats['R']['filtered']}/{stats['R']['total']} joints drawn")

    out_path = f'{output_dir}/{frame_name}'
    img.save(out_path, quality=95)
    print(f"  Saved: {out_path}")

print(f'\nFiltered visualizations saved to: {output_dir}/')