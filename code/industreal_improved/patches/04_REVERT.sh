#!/bin/bash
# V3 Revert script — undo V3 patches back to V2 baseline
set -e

PROJECT_ROOT="/home/newadmin/swarm-bot/project/popw/working/code/industreal_improved"
TS=$(date +%Y%m%d_%H%M%S)

# Find latest backup
LATEST_BACKUP=$(ls -td "${PROJECT_ROOT}/v3_patches"/backups_* 2>/dev/null | head -1)
if [ -z "$LATEST_BACKUP" ]; then
  echo "[REVERT] ERROR: no backups found in v3_patches/"
  exit 1
fi

echo "[REVERT] Restoring from: $LATEST_BACKUP"
cp "${LATEST_BACKUP}/config.py.PRE_V3" "${PROJECT_ROOT}/src/config.py"
cp "${LATEST_BACKUP}/evaluate.py.PRE_V3" "${PROJECT_ROOT}/src/evaluation/evaluate.py"

echo "[REVERT] Restored. Verifying:"
grep -nE "FOCAL_ALPHA" "${PROJECT_ROOT}/src/config.py" | head -2
grep -nE "V3_PATCH_NO_GT_VERDICT" "${PROJECT_ROOT}/src/evaluation/evaluate.py" | head -2 && echo "WARNING: V3 patch marker still found" || echo "  (no V3 markers — clean revert)"
