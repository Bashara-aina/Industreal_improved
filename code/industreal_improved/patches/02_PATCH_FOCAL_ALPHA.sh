#!/bin/bash
# V3 Patch 1: FOCAL_ALPHA 0.25 → 0.5
# Reason: V2 confidence collapse (score_max ~0.09, never >0.30).
# Root cause: alpha=0.25 means positives contribute 0.25× to loss, so model
# is rewarded for predicting LOW confidence (focal loss is balanced toward
# easy negatives, which dominate the loss when alpha is low).
# Patch raises alpha to 0.5, giving equal weight to positive/negative examples.
# Expected effect: score_max should rise above 0.30; det_mAP50 should rise above 0.0021.
set -e

PROJECT_ROOT="/home/newadmin/swarm-bot/project/popw/working/code/industreal_improved"
CONFIG="${PROJECT_ROOT}/src/config.py"

if grep -qE "^FOCAL_ALPHA\s*=\s*0\.5\s*#\s*V3_PATCH" "$CONFIG"; then
  echo "[PATCH-1] FOCAL_ALPHA already at 0.5 (V3 patch applied) — no change"
  exit 0
fi

if ! grep -qE "^FOCAL_ALPHA\s*=\s*0\.25$" "$CONFIG"; then
  echo "[PATCH-1] ERROR: FOCAL_ALPHA not at 0.25 (already patched or modified)"
  grep -n "FOCAL_ALPHA" "$CONFIG"
  exit 1
fi

# Apply patch: 0.25 → 0.5 (V3_PATCH marker for idempotency)
sed -i 's/^FOCAL_ALPHA   = 0\.25$/FOCAL_ALPHA   = 0.5  # V3_PATCH: was 0.25 — confidence collapse fix/' "$CONFIG"

echo "[PATCH-1] FOCAL_ALPHA patched: 0.25 → 0.5"
grep -n "FOCAL_ALPHA" "$CONFIG" | head -3
