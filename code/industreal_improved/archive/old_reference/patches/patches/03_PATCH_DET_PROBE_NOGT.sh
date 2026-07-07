#!/bin/bash
# V3 Patch 2: DET_PROBE no-GT misdiagnosis fix
# Reason: V2 logs showed "TOTAL COLLAPSE" verdict for batches with 0 GT (n_gt=0),
# which is misleading — it's not a collapse, the batch simply has no objects.
# Patch: check n_gt and emit a distinct "NO-GT (n_gt=0)" verdict instead.
set -e

PROJECT_ROOT="/home/newadmin/swarm-bot/project/popw/working/code/industreal_improved"
EVAL="${PROJECT_ROOT}/src/evaluation/evaluate.py"

# Idempotency check
if grep -qE "V3_PATCH_NO_GT_VERDICT" "$EVAL"; then
  echo "[PATCH-2] No-GT verdict patch already applied — no change"
  exit 0
fi

# Find the verdict logic block
# Original:
#   if n_matched == 0 and summary["bestIoU_max"] < iou_match:
#       verdict = f"TOTAL COLLAPSE (0 preds at IoU>{iou_match}, max={summary['bestIoU_max']:.2f})"
#   elif n_matched == 0:
#       verdict = f"NEAR-COLLAPSE (no match at {iou_match} but max IoU {summary['bestIoU_max']:.2f})"
#   else:
#       verdict = f"LOCALIZING ({n_matched} preds at IoU>{iou_match})"
#
# We need to insert a check on n_gt BEFORE the verdict block.

# Find the line number of the verdict block
LINE=$(grep -nE 'if n_matched == 0 and summary\["bestIoU_max"\] < iou_match' "$EVAL" | head -1 | cut -d: -f1)
if [ -z "$LINE" ]; then
  echo "[PATCH-2] ERROR: could not find verdict block in evaluate.py"
  exit 1
fi

# Patch: insert no-GT check before the existing block
# We use Python to do the in-place edit safely
python3 <<'PYEOF'
import re
from pathlib import Path
p = Path("/home/newadmin/swarm-bot/project/popw/working/code/industreal_improved/src/evaluation/evaluate.py")
src = p.read_text()

old = '''    if n_matched == 0 and summary["bestIoU_max"] < iou_match:
        verdict = f"TOTAL COLLAPSE (0 preds at IoU>{iou_match}, max={summary['bestIoU_max']:.2f})"
    elif n_matched == 0:
        verdict = f"NEAR-COLLAPSE (no match at {iou_match} but max IoU {summary['bestIoU_max']:.2f})"
    else:
        verdict = f"LOCALIZING ({n_matched} preds at IoU>{iou_match})"'''

new = '''    # V3_PATCH_NO_GT_VERDICT: distinguish "no GT in batch" from "model collapse"
    # Batches with 0 GT (e.g., 20_assy_0_1) were mislabelled as "TOTAL COLLAPSE",
    # which masks the real signal. A batch with 0 GT is vacuously collapse-immune.
    if summary.get("n_gt", 0) == 0:
        verdict = f"NO-GT (n_gt=0, max={summary['bestIoU_max']:.2f}, score_max={summary.get('score_max', 0.0):.3f})"
    elif n_matched == 0 and summary["bestIoU_max"] < iou_match:
        verdict = f"TOTAL COLLAPSE (0 preds at IoU>{iou_match}, max={summary['bestIoU_max']:.2f})"
    elif n_matched == 0:
        verdict = f"NEAR-COLLAPSE (no match at {iou_match} but max IoU {summary['bestIoU_max']:.2f})"
    else:
        verdict = f"LOCALIZING ({n_matched} preds at IoU>{iou_match})"'''

if old in src:
    src = src.replace(old, new, 1)
    p.write_text(src)
    print("[PATCH-2] evaluate.py patched: no-GT verdict added")
else:
    print("[PATCH-2] ERROR: verdict block not found")
    raise SystemExit(1)
PYEOF

echo "[PATCH-2] No-GT verdict patch applied"
grep -nE "V3_PATCH_NO_GT_VERDICT|TOTAL COLLAPSE|NO-GT" "$EVAL" | head -5
