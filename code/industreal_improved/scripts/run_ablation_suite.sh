#!/usr/bin/env bash
# =============================================================================
# [F16 2026-07-02 Fable RF4 consult] AAIML Ablation Suite
# =============================================================================
# The mandatory ablation matrix for the paper's multi-task claim. Every run
# uses the IDENTICAL architecture + training hyperparameters as stage_rf4;
# only the trained task losses (and, where noted, one config knob) differ.
#
# Run each line ONE AT A TIME (they share the GPU), from the repo root:
#   bash scripts/run_ablation_suite.sh <name>
# 20-25 epochs each is enough for the ablation table (single-task heads
# converge much faster than the joint model). ABLATION_EPOCHS=<n> overrides.
#
# Each run writes to its own src/runs/<name>/ directory via
# OUTPUT_ROOT_OVERRIDE (config.py reads it at import; without it, all
# ablations would collide in the same feature-derived directory).
#
# Paper table produced: single-task vs multi-task per-head accuracy at matched
# parameter count and data distribution → the interference/synergy deltas that
# make the "fewer params, single pass" efficiency claim defensible.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/../src"

EPOCHS="${ABLATION_EPOCHS:-25}"
NAME="${1:-help}"
RUNS_ROOT="$(pwd)/runs"

run() {  # run <dirname> [ENV=VAL ...] -- <preset>
  local dir="$1"; shift
  local envs=()
  while [[ "$1" != "--" ]]; do envs+=("$1"); shift; done
  shift
  local preset="$1"
  echo "[ablation] preset=$preset -> $RUNS_ROOT/$dir (epochs=$EPOCHS)"
  env OUTPUT_ROOT_OVERRIDE="$RUNS_ROOT/$dir" "${envs[@]}" \
    python training/train.py --preset "$preset" --max-epochs "$EPOCHS"
}

case "$NAME" in
  det)
    # A1 — detection-only baseline (compare: det_mAP50_pc vs multi-task)
    run ablation_det_only -- ablation_det_only
    ;;
  act)
    # A2 — activity-only baseline (compare: act macro-F1 / top-1, grouped)
    run ablation_act_only -- ablation_act_only
    ;;
  psr)
    # A3 — PSR-only baseline. Every batch is a seq batch (nothing else trains).
    run ablation_psr_only PSR_SEQ_EVERY_N_BATCHES=1 -- ablation_psr_only
    ;;
  pose)
    # A4 — head-pose-only baseline (compare: forward angular MAE)
    run ablation_pose_only -- ablation_pose_only
    ;;
  kendall-fixed)
    # B — multi-task with FIXED lambda weights instead of learned Kendall
    run ablation_kendall_fixed KENDALL_FIXED_WEIGHTS=1 -- stage_rf4
    ;;
  grouping-none)
    # C — multi-task with RAW 75-class activity (no verb grouping)
    run ablation_grouping_none ACT_CLASS_GROUPING=none -- stage_rf4
    ;;
  *)
    echo "Usage: $0 {det|act|psr|pose|kendall-fixed|grouping-none}"
    echo "  ABLATION_EPOCHS=<n> to override the default 25 epochs."
    echo "Order of value for the paper: det, act, psr, pose (Ablation A),"
    echo "then kendall-fixed and grouping-none."
    exit 1
    ;;
esac
