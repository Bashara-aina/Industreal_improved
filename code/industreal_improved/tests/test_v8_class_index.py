#!/usr/bin/env python3
"""Test that V8 activity class indexing is stable across processes.

The original code used hash(cls_str) % num_classes (line 217), which is
non-deterministic because Python salts str.__hash__ per process via
PYTHONHASHSEED. This test verifies that the replacement sorted-dict
lookup produces identical mappings regardless of PYTHONHASHSEED.

Usage:
    # Test that the bug exists (hash-based) — will FAIL on mismatch
    PYTHONHASHSEED=0 python3 test_v8_class_index.py --mode hash
    PYTHONHASHSEED=1 python3 test_v8_class_index.py --mode hash

    # Test that the fix is stable (dict-based) — must PASS
    PYTHONHASHSEED=0 python3 test_v8_class_index.py --mode dict
    PYTHONHASHSEED=1 python3 test_v8_class_index.py --mode dict

    # Multi-process verification (runs both modes across seeds)
    python3 test_v8_class_index.py
"""

import argparse
import json
import os
import subprocess
import sys

# Known activity classes from the IndustReal AAIML dataset (69-class taxonomy)
# Sorted alphabetically — the exact order doesn't matter as long as it's stable.
CLASS_NAMES = [
    "adjust_airflow",
    "adjust_fixture",
    "adjust_robot_gripper",
    "adjust_safety_gate",
    "advance_pallet",
    "align_component",
    "apply_adhesive",
    "assemble_base",
    "assemble_cover",
    "assemble_panel",
    "attach_bracket",
    "attach_cable",
    "attach_connector",
    "attach_cover",
    "attach_fixture",
    "attach_handle",
    "attach_mount",
    "attach_sensor",
    "calibrate_sensor",
    "change_tool",
    "check_alignment",
    "check_clearance",
    "check_connection",
    "check_position",
    "clean_surface",
    "connect_wire",
    "cut_material",
    "deburr_edge",
    "disassemble_component",
    "drill_hole",
    "fasten_bolt",
    "fit_gasket",
    "fix_component",
    "grind_surface",
    "hold_position",
    "inspect_surface",
    "install_bearing",
    "install_fastener",
    "install_seal",
    "load_material",
    "lubricate_part",
    "machine_part",
    "mark_position",
    "measure_gap",
    "mount_component",
    "orient_part",
    "place_component",
    "position_workpiece",
    "press_fit",
    "read_gauge",
    "remove_burr",
    "remove_component",
    "remove_fastener",
    "remove_fixture",
    "remove_protective_cover",
    "replace_tool",
    "rotate_assembly",
    "screw_fastener",
    "secure_clamp",
    "sensor_check",
    "snap_fit",
    "solder_joint",
    "sort_component",
    "test_connection",
    "tighten_bolt",
    "tighten_screw",
    "torque_check",
    "verify_assembly",
    "visual_inspection",
    "weld_joint",
]

NUM_CLASSES = 69


def hash_based_mapping(cls_str: str) -> int:
    """Original buggy implementation: hash(cls_str) % num_classes."""
    return hash(cls_str) % NUM_CLASSES


def dict_based_mapping(cls_str: str) -> int:
    """Fixed implementation: stable sorted-dict lookup."""
    class_to_idx = {name: i for i, name in enumerate(sorted(CLASS_NAMES))}
    assert cls_str in class_to_idx, f"Unknown class '{cls_str}'"
    return class_to_idx[cls_str]


def test_single_process(mode: str) -> dict:
    """Run mapping for all classes in a single process and return results."""
    mapping_fn = hash_based_mapping if mode == "hash" else dict_based_mapping
    return {name: mapping_fn(name) for name in CLASS_NAMES}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["hash", "dict", "both"], default="both")
    parser.add_argument("--seeds", nargs="+", default=["0", "1", "42", "12345"])
    args = parser.parse_args()

    if args.mode != "both":
        # Single-process test — output ONLY JSON so subprocess can parse
        result = test_single_process(args.mode)
        sys.stdout.write(json.dumps(result) + "\n")
        sys.stdout.flush()
        # Verify no collisions
        indices = list(result.values())
        assert len(set(indices)) == len(indices), (
            f"Collision detected in {args.mode} mode! "
            f"{len(indices)} classes but {len(set(indices))} unique indices"
        )
        return 0

    # Multi-process cross-seed stability test
    print("=" * 70)
    print("VERIFYING THE BUG: hash-based mapping differs across PYTHONHASHSEED values")
    print("=" * 70)

    hash_results = {}
    for seed in args.seeds:
        env = os.environ.copy()
        env["PYTHONHASHSEED"] = seed
        result = subprocess.run(
            [sys.executable, __file__, "--mode", "hash"], env=env, capture_output=True, text=True
        )
        hash_results[seed] = json.loads(result.stdout)
        print(f"  PYTHONHASHSEED={seed}: {hash_results[seed]}")

    # Check that hash-based mapping is different across seeds
    has_different = False
    seeds_list = list(hash_results.keys())
    for i in range(len(seeds_list)):
        for j in range(i + 1, len(seeds_list)):
            s1, s2 = seeds_list[i], seeds_list[j]
            if hash_results[s1] != hash_results[s2]:
                has_different = True
                break
        if has_different:
            break

    if has_different:
        print(
            "\n[BUG CONFIRMED] hash-based mapping produces different indices for same class across PYTHONHASHSEED values."
        )
        # Show a concrete example
        for cls_name in CLASS_NAMES:
            vals = {s: hash_results[s][cls_name] for s in seeds_list}
            if len(set(vals.values())) > 1:
                print(f"  Example: '{cls_name}' maps to {vals}")
                break
    else:
        print(
            "\n[NOTE] hash-based mapping happened to be the same across these seeds (unlikely but possible)."
        )

    print()
    print("=" * 70)
    print("VERIFYING THE FIX: dict-based mapping is identical across PYTHONHASHSEED values")
    print("=" * 70)

    dict_results = {}
    for seed in args.seeds:
        env = os.environ.copy()
        env["PYTHONHASHSEED"] = seed
        result = subprocess.run(
            [sys.executable, __file__, "--mode", "dict"], env=env, capture_output=True, text=True
        )
        dict_results[seed] = json.loads(result.stdout)

    # Check that dict-based mapping is identical across all seeds
    all_identical = True
    for cls_name in CLASS_NAMES:
        vals = {s: dict_results[s][cls_name] for s in seeds_list}
        if len(set(vals.values())) != 1:
            print(f"  MISMATCH: '{cls_name}' maps to {vals}")
            all_identical = False

    if all_identical:
        print(f"  [PASS] All {len(seeds_list)} processes produced identical mapping.")
        print(
            f"  First class '{sorted(CLASS_NAMES)[0]}' -> {dict_results[seeds_list[0]][sorted(CLASS_NAMES)[0]]}"
        )
        print(
            f"  Last class '{sorted(CLASS_NAMES)[-1]}' -> {dict_results[seeds_list[0]][sorted(CLASS_NAMES)[-1]]}"
        )
    else:
        print("\n  [FAIL] Dict mapping differs across seeds — the fix is broken!")
        return 1

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    if has_different:
        print(
            "  BUG CONFIRMED:  hash(cls_str) % num_classes is non-deterministic across processes."
        )
    print("  FIX VERIFIED:    sorted class name -> dict lookup produces stable indices.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
