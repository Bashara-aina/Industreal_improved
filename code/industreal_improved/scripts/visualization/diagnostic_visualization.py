"""
DIAGNOSTIC visualization: Map each HL2 joint index to its pixel position.
Show HL2 joint indices 0-25 for both hands with labels.
Use correct HL2 skeleton edges (NOT MediaPipe) to draw connections.
"""

from PIL import Image, ImageDraw, ImageFont
import csv
import os
import numpy as np

img_dir = "/home/newadmin/swarm-bot/project/popw/working/data/dataset/industreal/recordings/train/01_assy_0_1/rgb"
hands_csv = "/home/newadmin/swarm-bot/project/popw/working/data/dataset/industreal/recordings/train/01_assy_0_1/hands.csv"
output_dir = (
    "/home/newadmin/swarm-bot/project/popw/working/code/industreal_improved/diagnostic_output"
)
os.makedirs(output_dir, exist_ok=True)

frames = list(csv.reader(open(hands_csv)))

# HL2 hand joint indices/names
HL2_JOINT_NAMES = [
    "0_Palm",
    "1_Wrist",
    "2_ThumbCMC",
    "3_ThumbMCP",
    "4_ThumbIP",
    "5_ThumbTip",
    "6_IndexMCP",
    "7_IndexPIP",
    "8_IndexDIP",
    "9_IndexTip",
    "10_MiddleMCP",
    "11_MiddlePIP",
    "12_MiddleDIP",
    "13_MiddleTip",
    "14_RingMCP",
    "15_RingPIP",
    "16_RingDIP",
    "17_RingTip",
    "18_PinkyMCP",
    "19_PinkyPIP",
    "20_PinkyDIP",
    "21_PinkyTip",
    "22_Thumb_CMC_rad",
    "23_Thumb_IP_rad",
    "24_Palm-ulnar",
    "25_PinkyMCP-ulnar",  # Not sure about these
]

# CORRECTED HL2 skeleton edges based on anatomical hand structure
# Joint 0 = Palm (HL2), joint 1 = Wrist, joint 2 = Thumb CMC, etc.
HL2_EDGES_CORRECT = [
    # Thumb: Wrist -> CMC -> MCP -> IP -> Tip
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 5),
    # Index: Wrist -> MCP -> PIP -> DIP -> Tip
    (1, 6),
    (6, 7),
    (7, 8),
    (8, 9),
    # Middle: Wrist -> MCP -> PIP -> DIP -> Tip
    (1, 10),
    (10, 11),
    (11, 12),
    (12, 13),
    # Ring: Wrist -> MCP -> PIP -> DIP -> Tip
    (1, 14),
    (14, 15),
    (15, 16),
    (16, 17),
    # Pinky: Wrist -> MCP -> PIP -> DIP -> Tip
    (1, 18),
    (18, 19),
    (19, 20),
    (20, 21),
    # Palm connections
    (0, 1),  # Palm to Wrist
    (0, 6),  # Palm to Index MCP
    (0, 10),  # Palm to Middle MCP
    (0, 14),  # Palm to Ring MCP
    (0, 18),  # Palm to Pinky MCP
]

# ALSO try the MediaPipe-style edges (using HL2 index directly)
MEDIAPIPE_EDGES_HL2 = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),  # Thumb (but 0 is Palm not Wrist!)
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),  # Index (0=Palm -> wrong!)
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),  # Middle
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),  # Ring
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),  # Pinky
    (5, 9),
    (9, 13),
    (13, 17),  # Palm connections
]

try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 8)
    font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 10)
except:
    font = ImageFont.load_default()
    font_large = font


def is_valid_joint(x, y, img):
    x, y = int(x), int(y)
    if x < 0 or x >= 1280 or y < 0 or y >= 720:
        return False
    brightness = img[y, x].mean()
    return 15 < brightness < 100


for frame_idx in [10, 50, 99]:
    row = frames[frame_idx]
    frame_name = row[0]
    coords = [float(x) for x in row[1:]]

    img = Image.open(f"{img_dir}/{frame_name}").convert("RGB")
    img_arr = np.array(img)

    # Version 1: HL2 correct skeleton
    img1 = Image.open(f"{img_dir}/{frame_name}").convert("RGB")
    draw1 = ImageDraw.Draw(img1)

    for hand_start, hand_label, color in [(0, "L", (255, 80, 80)), (52, "R", (80, 255, 80))]:
        joints = [(coords[hand_start + i * 2], coords[hand_start + i * 2 + 1]) for i in range(26)]

        # Draw edges with correct HL2 skeleton
        for i, j in HL2_EDGES_CORRECT:
            x1, y1 = joints[i]
            x2, y2 = joints[j]
            valid1 = is_valid_joint(x1, y1, img_arr)
            valid2 = is_valid_joint(x2, y2, img_arr)
            if valid1 and valid2:
                draw1.line([(x1, y1), (x2, y2)], fill=color, width=5)
            elif valid1 or valid2:
                draw1.line([(x1, y1), (x2, y2)], fill=(255, 200, 0), width=3)

        # Draw ALL joints with index labels
        for i, (x, y) in enumerate(joints):
            valid = is_valid_joint(x, y, img_arr)
            r = 6 if i <= 1 else 4
            col = (0, 255, 0) if valid else (255, 0, 255)
            draw1.ellipse([x - r - 2, y - r - 2, x + r + 2, y + r + 2], fill=(255, 255, 255))
            draw1.ellipse([x - r, y - r, x + r, y + r], fill=col)
            # Label with joint index
            draw1.text((int(x) + 4, int(y) - 2), str(i), fill=(255, 255, 0), font=font)

        # Label hand
        wrist_x, wrist_y = joints[1]
        draw1.text(
            (int(wrist_x) + 15, int(wrist_y) - 15),
            f"{hand_label} (HL2 corrected)",
            fill=(255, 255, 0),
            font=font_large,
        )

    draw1.text(
        (10, 10),
        f"Frame {frame_idx} - HL2 CORRECT skeleton (edges: Wrist=1 as base)",
        fill=(255, 255, 255),
        font=font_large,
    )
    img1.save(f"{output_dir}/HL2_correct_{frame_name}", quality=95)
    print(f"Saved HL2_correct: {output_dir}/HL2_correct_{frame_name}")

    # Version 2: MediaPipe skeleton with HL2 data (ORIGINAL - potentially wrong)
    img2 = Image.open(f"{img_dir}/{frame_name}").convert("RGB")
    draw2 = ImageDraw.Draw(img2)

    for hand_start, hand_label, color in [(0, "L", (255, 80, 80)), (52, "R", (80, 255, 80))]:
        joints = [(coords[hand_start + i * 2], coords[hand_start + i * 2 + 1]) for i in range(26)]

        # Draw with MediaPipe skeleton edges
        for i, j in MEDIAPIPE_EDGES_HL2:
            x1, y1 = joints[i]
            x2, y2 = joints[j]
            valid1 = is_valid_joint(x1, y1, img_arr)
            valid2 = is_valid_joint(x2, y2, img_arr)
            if valid1 and valid2:
                draw2.line([(x1, y1), (x2, y2)], fill=color, width=5)
            elif valid1 or valid2:
                draw2.line([(x1, y1), (x2, y2)], fill=(255, 200, 0), width=3)

        for i, (x, y) in enumerate(joints):
            valid = is_valid_joint(x, y, img_arr)
            r = 6 if i <= 1 else 4
            col = (0, 255, 0) if valid else (255, 0, 255)
            draw2.ellipse([x - r - 2, y - r - 2, x + r + 2, y + r + 2], fill=(255, 255, 255))
            draw2.ellipse([x - r, y - r, x + r, y + r], fill=col)
            draw2.text((int(x) + 4, int(y) - 2), str(i), fill=(255, 255, 0), font=font)

        wrist_x, wrist_y = joints[1]
        draw2.text(
            (int(wrist_x) + 15, int(wrist_y) - 15),
            f"{hand_label} (MediaPipe skeleton)",
            fill=(255, 255, 0),
            font=font_large,
        )

    draw2.text(
        (10, 10),
        f"Frame {frame_idx} - MediaPipe skeleton on HL2 coords (WRONG if Palm!=Wrist)",
        fill=(255, 255, 255),
        font=font_large,
    )
    img2.save(f"{output_dir}/MediaPipe_{frame_name}", quality=95)
    print(f"Saved MediaPipe: {output_dir}/MediaPipe_{frame_name}")

    # Version 3: Raw dots only (no edges) - to see actual joint positions
    img3 = Image.open(f"{img_dir}/{frame_name}").convert("RGB")
    draw3 = ImageDraw.Draw(img3)

    for hand_start, hand_label, color in [(0, "L", (255, 80, 80)), (52, "R", (80, 255, 80))]:
        joints = [(coords[hand_start + i * 2], coords[hand_start + i * 2 + 1]) for i in range(26)]
        for i, (x, y) in enumerate(joints):
            valid = is_valid_joint(x, y, img_arr)
            r = 5
            col = (0, 200, 0) if valid else (200, 0, 200)
            draw3.ellipse([int(x) - r, int(y) - r, int(x) + r, int(y) + r], fill=col)
            draw3.text((int(x) + 6, int(y) - 3), str(i), fill=(255, 255, 0), font=font)

    draw3.text(
        (10, 10),
        f"Frame {frame_idx} - Raw HL2 joint positions (no edges)",
        fill=(255, 255, 255),
        font=font_large,
    )
    img3.save(f"{output_dir}/RawDots_{frame_name}", quality=95)
    print(f"Saved RawDots: {output_dir}/RawDots_{frame_name}")

print(f"\nDiagnostic output: {output_dir}/")
