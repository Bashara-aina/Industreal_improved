#!/bin/bash
# V3 Patches: BACKUP script
# Backs up config.py and evaluate.py BEFORE applying patches
set -e

PROJECT_ROOT="/home/newadmin/swarm-bot/project/popw/working/code/industreal_improved"
TS=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="${PROJECT_ROOT}/v3_patches/backups_${TS}"

mkdir -p "$BACKUP_DIR"

echo "[BACKUP] Creating snapshot of files to be patched..."
cp "${PROJECT_ROOT}/src/config.py" "${BACKUP_DIR}/config.py.PRE_V3"
cp "${PROJECT_ROOT}/src/evaluation/evaluate.py" "${BACKUP_DIR}/evaluate.py.PRE_V3"

echo "[BACKUP] Snapshots created at: ${BACKUP_DIR}"
echo "  - config.py.PRE_V3"
echo "  - evaluate.py.PRE_V3"
ls -la "$BACKUP_DIR"
