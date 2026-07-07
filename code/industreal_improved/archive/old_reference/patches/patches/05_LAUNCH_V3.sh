#!/bin/bash
# V3 Launch script — to be run AFTER V2 smoke test completes
# Resumes from V2's final checkpoint with FOCAL_ALPHA=0.5 patches applied.
set -e

PROJECT_ROOT="/home/newadmin/swarm-bot/project/popw/working/code/industreal_improved"
V2_RUN="full_multi_task_tma_tbank_benchmark"
V2_CKPT_DIR="${PROJECT_ROOT}/src/runs/${V2_RUN}/checkpoints"
V2_LOG_DIR="${PROJECT_ROOT}/src/runs/${V2_RUN}/logs"
TS=$(date +%Y%m%d_%H%M%S)

# 1) Find V2's final resume point
echo "[V3-LAUNCH] Locating V2 final checkpoint..."
V2_RESUME=""
for ckpt in crash_recovery.pth latest.pth best.pth; do
  if [ -f "${V2_CKPT_DIR}/${ckpt}" ]; then
    V2_RESUME="${V2_CKPT_DIR}/${ckpt}"
    echo "[V3-LAUNCH] Found V2 ckpt: ${ckpt} ($(stat -c %s ${V2_CKPT_DIR}/${ckpt}) bytes)"
    break
  fi
done
if [ -z "$V2_RESUME" ]; then
  echo "[V3-LAUNCH] ERROR: no V2 checkpoint found in ${V2_CKPT_DIR}"
  exit 1
fi

# 2) Create V3 run dir
V3_RUN_DIR="${PROJECT_ROOT}/src/runs/full_multi_task_tma_tbank_v3_alpha05_${TS}"
mkdir -p "${V3_RUN_DIR}/checkpoints"
mkdir -p "${V3_RUN_DIR}/logs"
cp "${V2_RESUME}" "${V3_RUN_DIR}/checkpoints/loading_point.pth"
echo "[V3-LAUNCH] V3 run dir: ${V3_RUN_DIR}"

# 3) Verify patches are applied
echo "[V3-LAUNCH] Verifying V3 patches are applied..."
grep -qE "V3_PATCH" "${PROJECT_ROOT}/src/config.py" || {
  echo "[V3-LAUNCH] ERROR: FOCAL_ALPHA patch not applied. Run 02_PATCH_FOCAL_ALPHA.sh first."
  exit 1
}
grep -qE "V3_PATCH_NO_GT_VERDICT" "${PROJECT_ROOT}/src/evaluation/evaluate.py" || {
  echo "[V3-LAUNCH] ERROR: DET_PROBE no-GT patch not applied. Run 03_PATCH_DET_PROBE_NOGT.sh first."
  exit 1
}

# 4) Launch V3 — start at the same epoch as V2 finished, train 1 epoch as a sanity check
#    (full 23 epoch training can be triggered manually after smoke test passes)
cd "${PROJECT_ROOT}/src"
LOG_FILE="${V3_RUN_DIR}/logs/v3_smoke_test_${TS}.log"
echo "[V3-LAUNCH] Launching V3 smoke test. Log: ${LOG_FILE}"

nohup python3 -u training/train.py \
  --resume "${V3_RUN_DIR}/checkpoints/loading_point.pth" \
  --subset-ratio 1.0 \
  --max-epochs 24 \
  --seed 42 \
  --no-staged-training \
  --num-workers 0 \
  > "${LOG_FILE}" 2>&1 &

V3_PID=$!
echo "[V3-LAUNCH] V3 launched with PID ${V3_PID}"
echo "${V3_PID}" > "${V3_RUN_DIR}/v3.pid"
echo "[V3-LAUNCH] Done. Monitor with: tail -f ${LOG_FILE}"
