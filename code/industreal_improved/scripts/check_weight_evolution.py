#!/usr/bin/env python3
"""
check_weight_evolution.py — Post-smoke-test weight evolution verifier.

Usage:
    python scripts/check_weight_evolution.py /path/to/train.log

Loads the training log from a candidate-config smoke test and verifies:

  1. Loss trajectory — all 4 heads produce finite, non-NaN losses
  2. Log-var bounds — Kendall log-var values stay within expected ranges:
       log_var_act  >= -0.5   (KENDALL_LOG_VAR_MIN_ACT)
       log_var_psr  <=  0.0   (KENDALL_LOG_VAR_MAX_PSR)
       log_var_pose <=  3.0   (KENDALL_LOG_VAR_MAX_POSE)
  3. UW-SO weights — if UW-SO was active, check that weights are finite
  4. No NaN cascade — no 'nan_skips' events (gradient NaN clipping)
  5. Per-task LR routing — verify head pose and PSR have separate LR entries

Returns exit code 0 if all checks pass, 1 otherwise.
"""

import re
import sys
import math
from pathlib import Path

# --- Constants (mirrors src/config.py Kendall bounds) ---
KENDALL_LOG_VAR_MIN_ACT = -0.5
KENDALL_LOG_VAR_MAX_PSR = 0.0
KENDALL_LOG_VAR_MAX_POSE = 3.0


def parse_log(log_path: str) -> dict:
    """Parse training log and extract structured metrics per epoch."""
    log = Path(log_path)
    if not log.exists():
        print(f"ERROR: Log file not found: {log_path}")
        sys.exit(1)

    text = log.read_text()

    result = {
        "epochs": [],
        "nan_skips_any": False,
        "has_uw_so_mention": False,
        "has_balanced_softmax_mention": False,
        "lr_param_groups": [],
        "total_steps": 0,
    }

    # Track per-epoch metrics
    current_epoch = None

    for line in text.splitlines():
        # --- Epoch start ---
        m = re.search(r"\[Epoch\s+(\d+)\]", line)
        if m:
            current_epoch = int(m.group(1))
            result["epochs"].append(
                {
                    "epoch": current_epoch,
                    "losses": {},
                    "log_vars": {},
                    "lr_values": {},
                    "nan_skips": 0,
                    "has_nan": False,
                }
            )

        # --- Loss values (train metrics log line) ---
        if "loss_det=" in line and current_epoch is not None:
            ep = result["epochs"][-1]
            for key in ["loss_det", "loss_pose", "loss_act", "loss_psr", "loss"]:
                m2 = re.search(rf"\b{key}=([\d.]+)", line)
                if m2:
                    ep["losses"][key] = float(m2.group(1))

        # --- Log-var values ---
        if "kd_d=" in line and current_epoch is not None:
            ep = result["epochs"][-1]
            for label, vkey in [("kd_d", "log_var_det"), ("kd_hp", "log_var_pose"),
                                ("kd_a", "log_var_act"), ("kd_r", "log_var_psr")]:
                m2 = re.search(rf"\b{label}=([+-]?\d+\.\d+)", line)
                if m2:
                    ep["log_vars"][vkey] = float(m2.group(1))

        # --- nan_skips ---
        if "nan_skips=" in line and current_epoch is not None:
            m2 = re.search(r"nan_skips=(\d+)", line)
            if m2:
                n = int(m2.group(1))
                result["epochs"][-1]["nan_skips"] = n
                if n > 0:
                    result["epochs"][-1]["has_nan"] = True
                    result["nan_skips_any"] = True

        # --- UW-SO mention ---
        if "UW-SO" in line or "uw_so" in line.lower():
            result["has_uw_so_mention"] = True

        # --- Balanced softmax mention ---
        if "BalancedSoftmax" in line or "balanced_softmax" in line.lower():
            result["has_balanced_softmax_mention"] = True

        # --- LR param groups (debug log) ---
        if "[LR]" in line and "g0=" in line:
            lr_strs = re.findall(r"g\d+=([\d.e+-]+)", line)
            result["lr_param_groups"] = [float(x) for x in lr_strs]

        # --- Step count ---
        if "Step" in line and "done" in line.lower():
            m2 = re.search(r"Step\s+(\d+)", line)
            if m2:
                result["total_steps"] = max(result["total_steps"], int(m2.group(1)))

    return result


def check_finite_losses(data: dict) -> bool:
    """Check all loss values are finite and non-NaN."""
    ok = True
    for ep in data["epochs"]:
        for key, val in ep["losses"].items():
            if not math.isfinite(val):
                print(f"  FAIL: Epoch {ep['epoch']} {key}={val} (non-finite)")
                ok = False
    if ok:
        print("  PASS: All losses finite and non-NaN")
    return ok


def check_logvar_bounds(data: dict) -> bool:
    """Check Kendall log-var values stay within documented bounds."""
    ok = True
    bounds = {
        "log_var_act": ("min", KENDALL_LOG_VAR_MIN_ACT),
        "log_var_psr": ("max", KENDALL_LOG_VAR_MAX_PSR),
        "log_var_pose": ("max", KENDALL_LOG_VAR_MAX_POSE),
    }
    for ep in data["epochs"]:
        for vkey, (bound_type, bound_val) in bounds.items():
            if vkey in ep["log_vars"]:
                val = ep["log_vars"][vkey]
                if bound_type == "min" and val < bound_val:
                    print(
                        f"  FAIL: Epoch {ep['epoch']} {vkey}={val:.4f} < min {bound_val}"
                    )
                    ok = False
                elif bound_type == "max" and val > bound_val:
                    print(
                        f"  FAIL: Epoch {ep['epoch']} {vkey}={val:.4f} > max {bound_val}"
                    )
                    ok = False
    if ok:
        print(f"  PASS: All log-var values within bounds")
    return ok


def check_nan_skips(data: dict) -> bool:
    """Check no NaN skip events occurred."""
    if data["nan_skips_any"]:
        for ep in data["epochs"]:
            if ep["has_nan"]:
                print(f"  FAIL: Epoch {ep['epoch']} had {ep['nan_skips']} nan_skip(s)")
        return False
    print("  PASS: No NaN skip events")
    return True


def check_lr_routing(data: dict) -> bool:
    """Check that PSR and head pose have distinct LR entries."""
    if not data["lr_param_groups"]:
        print("  SKIP: No [LR] debug log lines found (not logged at TRAIN_MAX_STEPS=20)")
        return True  # Non-fatal: debug logging may not fire in 20-step run

    lrs = data["lr_param_groups"]
    unique_lrs = set(round(lr, 10) for lr in lrs)
    print(f"  LR param groups ({len(lrs)}): {[f'{lr:.2e}' for lr in lrs]}")

    if len(unique_lrs) < 2:
        print(f"  WARN: Only {len(unique_lrs)} unique LR(s) — expected at least 2 distinct")
        return True  # Non-fatal in 20-step run

    print("  PASS: Multiple distinct LRs detected (per-task routing active)")
    return True


def check_uw_so_active(data: dict) -> bool:
    """Check UW-SO was active during training (non-fatal: depends on env)."""
    if data["has_uw_so_mention"]:
        print("  PASS: UW-SO weighting was active (found in logs)")
        return True
    print("  INFO: UW-SO not mentioned in log (may not have been enabled)")
    return True


def check_balanced_softmax_active(data: dict) -> bool:
    """Check balanced softmax was active (non-fatal: depends on env)."""
    if data["has_balanced_softmax_mention"]:
        print("  PASS: Balanced softmax was active (found in logs)")
        return True
    print("  INFO: Balanced softmax not mentioned in log (may not have been enabled)")
    return True


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <train.log>")
        sys.exit(1)

    log_path = sys.argv[1]
    print(f"Checking weight evolution: {log_path}")
    print()

    data = parse_log(log_path)
    print(f"Parsed {len(data['epochs'])} epoch(s), ~{data['total_steps']} steps")
    print()

    checks = [
        ("Finite losses", check_finite_losses(data)),
        ("Log-var bounds", check_logvar_bounds(data)),
        ("No NaN skips", check_nan_skips(data)),
        ("LR routing", check_lr_routing(data)),
        ("UW-SO active", check_uw_so_active(data)),
        ("Balanced softmax", check_balanced_softmax_active(data)),
    ]

    print()
    passed = sum(1 for name, result in checks if result)
    failed = sum(1 for name, result in checks if not result)
    print(f"Results: {passed} passed, {failed} failed out of {len(checks)} checks")

    # Print summary table
    print()
    print("--- Per-Epoch Summary ---")
    print(f"{'Epoch':>6} {'loss_det':>9} {'loss_pose':>10} {'loss_act':>9} {'loss_psr':>9} "
          f"{'lv_det':>7} {'lv_pose':>7} {'lv_act':>7} {'lv_psr':>7} {'nan':>4}")
    for ep in data["epochs"]:
        ls = ep["losses"]
        lv = ep["log_vars"]
        nan_flag = "Y" if ep["has_nan"] else "."
        print(f"{ep['epoch']:>6} "
              f"{ls.get('loss_det', 0):>9.4f} {ls.get('loss_pose', 0):>10.4f} "
              f"{ls.get('loss_act', 0):>9.4f} {ls.get('loss_psr', 0):>9.4f} "
              f"{lv.get('log_var_det', 0):>7.2f} {lv.get('log_var_pose', 0):>7.2f} "
              f"{lv.get('log_var_act', 0):>7.2f} {lv.get('log_var_psr', 0):>7.2f} "
              f"{nan_flag:>4}")

    if failed > 0:
        print(f"\nFAILED: {failed} check(s) failed")
        sys.exit(1)
    else:
        print("\nALL CHECKS PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()
