"""
Decoder Oracle Bound (Opus 141 Q46)
=======================================
Compute the theoretical upper bound of the MonotonicDecoder by feeding GT
transitions as perfect oracle logits. If the oracle F1 is ~0.95+, the decoder
is not the bottleneck — upstream transition prediction quality is.

Method:
  1. Load per-recording GT PSR labels (dense fill-forward, 11 components).
  2. Build oracle transition logits:
     - For each GT 0→1 transition, set oracle_logit[t] = 1.0 at the transition
       frame for sustain_min consecutive frames (so hysteresis can fire).
     - For components already placed at frame 0, pre-initialize decoder state
       to match GT. This avoids false positive transition events and allows
       downstream components to fire through the procedure-order constraint.
  3. Run MonotonicDecoder.forward(oracle_logits) to get decoded states.
  4. Compute transition F1 (event matching within ±3 tolerance):
     per-component, macro, micro.
  5. Report gap between actual and oracle F1.

The decoder has hysteresis (sustain_min=3, sustain_lo=0.3, sustain_hi=0.5).
For a single-frame impulse (1.0 at transition frame, 0 elsewhere), hysteresis
causes a 2-frame delay before the transition fires. The sustained oracle
provides 3 consecutive frames of evidence starting at the GT transition,
which is the minimum to trigger hysteresis.

Additionally, the procedure-order prior can block transitions if GT components
transition in an order violating the hardcoded sequential chain. The oracle
bound reveals whether this constraint ever suppresses correct transitions.

Two modes:
  - "impulse":  single-frame 1.0 at transition frame (honest: measures
                whether decoder can detect from an ideal impulse)
  - "sustained": sustain_min frames of 1.0 starting at transition frame
                (generous: measures procedure-order constraint alone)
"""

import json
import sys
import csv
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch

# ── Path setup ──────────────────────────────────────────────────────────
_SRC = Path(__file__).resolve().parent.parent  # src/
for _p in [_SRC, _SRC.parent, _SRC / "models", _SRC / "data"]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from src.models.psr_transition import MonotonicDecoder
from src import config as C


# =========================================================================
# Data loading
# =========================================================================

def load_psr_labels(rec_dir: Path, num_frames: int) -> np.ndarray:
    """Load and fill-forward PSR labels for a recording.
    Matches the dataset's _parse_psr_raw() logic.
    Returns [num_frames, 11] float32 array with values 0.0 or 1.0.
    """
    psr_file = rec_dir / "PSR_labels_raw.csv"
    if not psr_file.exists():
        return np.zeros((num_frames, C.NUM_PSR_COMPONENTS), dtype=np.float32)

    sparse = []
    with open(psr_file, encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 12:
                continue
            try:
                frame_num = int(Path(row[0]).stem)
                values = np.array([float(v) for v in row[1:12]], dtype=np.float32)
                sparse.append((frame_num, values))
            except (ValueError, IndexError):
                continue

    if not sparse:
        return np.zeros((num_frames, C.NUM_PSR_COMPONENTS), dtype=np.float32)

    sparse.sort(key=lambda x: x[0])

    dense = np.zeros((num_frames, C.NUM_PSR_COMPONENTS), dtype=np.float32)
    last_valid = np.zeros(C.NUM_PSR_COMPONENTS, dtype=np.int64)
    sparse_idx = 0
    for frame in range(num_frames):
        if sparse_idx < len(sparse) and frame == sparse[sparse_idx][0]:
            new_vals = sparse[sparse_idx][1].copy()
            sparse_idx += 1
            valid_mask = new_vals >= 0
            last_valid[valid_mask] = new_vals[valid_mask]
        dense[frame] = last_valid.copy()

    return dense


def count_frames(rec_dir: Path) -> int:
    """Count frames in a recording's rgb directory."""
    rgb_dir = rec_dir / "rgb"
    if not rgb_dir.exists():
        return 0
    return len(list(rgb_dir.glob("*.jpg")))


# =========================================================================
# Oracle construction
# =========================================================================

def build_oracle_logits(gt_states: np.ndarray,
                        mode: str = "sustained") -> np.ndarray:
    """Build oracle transition logits from GT states.

    For components with gt_states[0, c] = 1 (already placed at start),
    oracle[0:sustain_window, c] = 1.0 to signal the decoder.

    For each 0→1 transition event, set oracle[t:t+sustain_window, c] = 1.0.
    The window is longer than sustain_min because the decoder's predecessor
    constraint evaluates all components using the PREVIOUS frame's state.
    If K components transition at the same GT frame, the decoder needs
    sustain_min + K - 1 frames of sustained evidence to fire all K.
    With 11 components, worst case = 13 frames. We use 20 for margin.

    Args:
        gt_states: [T, 11] binary fill-forward GT states
        mode:
            "impulse" — set 1.0 at exact transition frame + init state
            "sustained" — set 1.0 for sustain_window (=20) frames

    Returns:
        oracle: [T, 11] float32 in [0, 1]
    """
    T, C = gt_states.shape
    oracle = np.zeros((T, C), dtype=np.float32)

    # The decoder's sequential within-frame evaluation means cascading
    # transitions require more frames than sustain_min alone.
    # sustain_min for first component + (num_comp - 1) for cascade + margin
    sustain_min = int(getattr(C, "PSR_TRANSITION_MIN_SUSTAINED", 3))
    sustain_window = sustain_min + C  # 3 + 11 = 14 frames minimum for full cascade

    # ── Handle components already placed at frame 0 ──────────────────────
    for c in range(C):
        if gt_states[0, c] > 0.5:
            for offset in range(sustain_window):
                if offset < T:
                    oracle[offset, c] = 1.0

    # ── Handle true 0→1 transitions ──────────────────────────────────────
    transitions = np.clip(gt_states[1:] - gt_states[:-1], a_min=0, a_max=None)

    for c in range(C):
        trans_frames = np.where(transitions[:, c] > 0.5)[0]
        for tf in trans_frames:
            start_frame = tf + 1
            if mode == "impulse":
                if start_frame < T:
                    oracle[start_frame, c] = 1.0
            elif mode == "sustained":
                for offset in range(sustain_window):
                    frame = start_frame + offset
                    if frame < T:
                        oracle[frame, c] = 1.0
            else:
                raise ValueError(f"Unknown mode: {mode}")

    return oracle


# =========================================================================
# MonotonicDecoder with pre-initialized state
# =========================================================================

def decode_oracle(oracle_logits: np.ndarray,
                  decoder: MonotonicDecoder,
                  device: str = "cuda",
                  initial_state: np.ndarray = None,
                  use_procedure_order: bool = True) -> np.ndarray:
    """Run MonotonicDecoder with optional initial state pre-initialization.

    The standard decoder always starts from zeros. For the oracle, we need
    to pre-initialize components that are already 1 at frame 0 in GT.

    We achieve this by directly modifying the decoder's forward loop:
    set current_state[b, c] = initial_state[c] before processing any frames.
    This avoids false positive transition events for initially-placed
    components.

    Args:
        oracle_logits: [T, 11] transition probabilities
        decoder: MonotonicDecoder instance
        device: torch device
        initial_state: [11] array with initial per-component state.
                       If None, use zeros (standard behavior).

    Returns:
        states: [T, 11] decoded binary states
    """
    logits_t = torch.from_numpy(oracle_logits).float().to(device)
    if logits_t.dim() == 2:
        logits_t = logits_t.unsqueeze(0)  # [T, C] -> [1, T, C]

    B, T, n_comp = logits_t.shape
    states = torch.zeros(B, T, n_comp, device=device)

    # Pre-initialize current_state if initial_state is provided
    if initial_state is not None:
        current_state = torch.from_numpy(initial_state).float().to(device)
        current_state = current_state.unsqueeze(0)  # [C] -> [1, C]
    else:
        current_state = torch.zeros(B, n_comp, device=device)

    # Hysteresis parameters
    sustain_hi = float(getattr(C, "PSR_TRANSITION_THRESHOLD_HI", 0.5))
    sustain_lo = float(getattr(C, "PSR_TRANSITION_THRESHOLD_LO", 0.3))
    sustain_min = int(getattr(C, "PSR_TRANSITION_MIN_SUSTAINED", 3))
    sustain_counter = torch.zeros(B, n_comp, device=device)
    order_matrix = decoder._order_matrix.to(device)

    for t in range(T):
        trans_prob = logits_t[:, t, :]  # [B, C]

        # Components not yet placed can transition
        can_transition = (current_state == 0)

        # Procedure-order constraint (skipped if use_procedure_order=False)
        if use_procedure_order:
            predecessors_placed = (current_state.unsqueeze(2) >= order_matrix).all(dim=1)
            can_transition = can_transition & predecessors_placed

        # Hysteresis
        above_lo = (trans_prob > sustain_lo).float()
        sustain_counter = sustain_counter * above_lo + above_lo
        sustained = (sustain_counter >= sustain_min)
        high_now = (trans_prob > sustain_hi)
        transition = (high_now & sustained & can_transition)

        current_state = (current_state + transition.float()).clamp(max=1.0)
        states[:, t, :] = current_state

    return states.squeeze(0).cpu().numpy()  # [T, C]


# =========================================================================
# Metrics
# =========================================================================

def event_f1(pred_tr: np.ndarray, gt_tr: np.ndarray, tol: int = 3) -> float:
    """Event F1 with greedy matching within tolerance. B3/STORM protocol."""
    if not pred_tr.any() and not gt_tr.any():
        return 1.0
    if not pred_tr.any() or not gt_tr.any():
        return 0.0
    n_comp = pred_tr.shape[1]
    tp, fp, fn_tot = 0, 0, 0
    for c in range(n_comp):
        p_frames = np.where(pred_tr[:, c])[0]
        g_frames = np.where(gt_tr[:, c])[0]
        matched = set()
        for pf in p_frames:
            found = False
            for gi, gf in enumerate(g_frames):
                if gi not in matched and abs(pf - gf) <= tol:
                    matched.add(gi)
                    tp += 1
                    found = True
                    break
            if not found:
                fp += 1
        fn_tot += len(g_frames) - len(matched)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn_tot, 1)
    return 2 * prec * rec / max(prec + rec, 1e-9)


def compute_per_component_f1(pred_states: np.ndarray,
                              gt_states: np.ndarray,
                              tol: int = 3) -> dict:
    """Compute per-component and aggregate transition F1."""
    pred_tr = np.clip(pred_states[1:] - pred_states[:-1], a_min=0, a_max=None)
    gt_tr = np.clip(gt_states[1:] - gt_states[:-1], a_min=0, a_max=None)

    per_comp = {}
    for c in range(pred_states.shape[1]):
        per_comp[c] = event_f1(pred_tr[:, c:c+1], gt_tr[:, c:c+1], tol=tol)

    macro_f1 = float(np.mean(list(per_comp.values()))) if per_comp else 0.0

    # Micro: aggregate across components
    tp_tot, fp_tot, fn_tot = 0, 0, 0
    for c in range(pred_states.shape[1]):
        p_frames = set(np.where(pred_tr[:, c])[0])
        g_frames = set(np.where(gt_tr[:, c])[0])
        matched = set()
        for pf in sorted(p_frames):
            for gf in sorted(g_frames):
                if gf not in matched and abs(pf - gf) <= tol:
                    matched.add(gf)
                    tp_tot += 1
                    break
            else:
                fp_tot += 1
        fn_tot += len(g_frames) - len(matched)
    micro_prec = tp_tot / max(tp_tot + fp_tot, 1)
    micro_rec = tp_tot / max(tp_tot + fn_tot, 1)
    micro_f1 = 2 * micro_prec * micro_rec / max(micro_prec + micro_rec, 1e-9)

    return {
        "per_component_f1": {str(k): float(v) for k, v in per_comp.items()},
        "macro_f1": macro_f1,
        "micro_f1": float(micro_f1),
        "micro_precision": float(micro_prec),
        "micro_recall": float(micro_rec),
    }


def compute_structure_analysis(pred_states: np.ndarray,
                                gt_states: np.ndarray) -> dict:
    """Analyze structural violations and decoder behavior."""
    pred_tr = np.clip(pred_states[1:] - pred_states[:-1], a_min=0, a_max=None)
    gt_tr = np.clip(gt_states[1:] - gt_states[:-1], a_min=0, a_max=None)

    n_gt = int(gt_tr.sum())
    n_pred = int(pred_tr.sum())

    # Check monotonicity: any 1→0 transitions?
    descents = np.clip(pred_states[1:] - pred_states[:-1], a_max=0, a_min=None)
    n_violations = int((-descents).sum())

    # Delay analysis
    all_delays = []
    delay_per_comp = {}
    for c in range(pred_states.shape[1]):
        pred_frames = set(np.where(pred_tr[:, c])[0])
        gt_frames = set(np.where(gt_tr[:, c])[0])
        matched_g = set()
        delays_c = []
        for pf in sorted(pred_frames):
            best_gf = None
            best_dist = float("inf")
            for gf in sorted(gt_frames):
                if gf not in matched_g and abs(pf - gf) <= 3:
                    if best_gf is None or abs(pf - gf) < best_dist:
                        best_gf = gf
                        best_dist = abs(pf - gf)
            if best_gf is not None:
                matched_g.add(best_gf)
                delays_c.append(int(pf - best_gf))
                all_delays.append(int(pf - best_gf))
        if delays_c:
            delay_per_comp[str(c)] = {
                "delays": delays_c,
                "mean": float(np.mean(delays_c)),
            }

    delay_stats = {}
    if all_delays:
        delay_stats = {
            "mean": float(np.mean(all_delays)),
            "median": float(np.median(all_delays)),
            "std": float(np.std(all_delays)),
            "min": int(min(all_delays)),
            "max": int(max(all_delays)),
            "p90": float(np.percentile(all_delays, 90)),
            "n": len(all_delays),
            "per_component": delay_per_comp,
        }

    return {
        "n_gt_transitions": n_gt,
        "n_pred_transitions": n_pred,
        "n_monotonicity_violations": n_violations,
        "delay_stats": delay_stats,
    }


# =========================================================================
# Actual decoder baseline (for comparison)
# =========================================================================

def run_actual_decoder():
    """Get actual decoder F1 from existing evaluation logs.

    Returns dict with actual PSR metrics if available.
    """
    actual = {"macro_f1": None, "micro_f1": None, "source": None}

    # Check eval logs for actual F1
    log_dir = Path("src/runs/rf_stages/checkpoints/logs")
    if log_dir.exists():
        for log_file in sorted(log_dir.glob("*.log")):
            content = log_file.read_text()
            for line in content.split("\n"):
                if "psr_macro_f1" in line:
                    try:
                        val = float(line.split("=")[-1].strip())
                        if actual["macro_f1"] is None or val > actual["macro_f1"]:
                            actual["macro_f1"] = val
                            actual["source"] = log_file.name
                    except (ValueError, IndexError):
                        pass
                if "psr_micro_f1" in line:
                    try:
                        val = float(line.split("=")[-1].strip())
                        if actual["micro_f1"] is None or val > actual["micro_f1"]:
                            actual["micro_f1"] = val
                    except (ValueError, IndexError):
                        pass

    return actual


# =========================================================================
# Main
# =========================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="MonotonicDecoder Oracle Bound (Opus 141 Q46)"
    )
    parser.add_argument("--save-dir", type=str,
                        default="src/runs/rf_stages/checkpoints/decoder_oracle_bound",
                        help="Output directory for results")
    parser.add_argument("--tolerance", type=int, default=3,
                        help="Frame tolerance for event matching")
    parser.add_argument("--mode", type=str, default="sustained",
                        choices=["impulse", "sustained"],
                        help="Oracle mode: impulse or sustained")
    parser.add_argument("--relaxed", action="store_true",
                        help="Remove procedure-order constraint entirely. "
                             "Isolates hysteresis/threshold bottleneck "
                             "from procedure-order bottleneck.")
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── Load decoder ─────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    decoder = MonotonicDecoder(num_components=C.NUM_PSR_COMPONENTS).to(device)
    decoder.eval()

    sustain_hi = float(getattr(C, "PSR_TRANSITION_THRESHOLD_HI", 0.5))
    sustain_lo = float(getattr(C, "PSR_TRANSITION_THRESHOLD_LO", 0.3))
    sustain_min = int(getattr(C, "PSR_TRANSITION_MIN_SUSTAINED", 3))

    print(f"Decoder: MonotonicDecoder({C.NUM_PSR_COMPONENTS} components)")
    print(f"  procedure_order: {decoder.procedure_order}")
    print(f"  sustain_hi={sustain_hi}, sustain_lo={sustain_lo}, sustain_min={sustain_min}")
    print(f"  mode: {args.mode}")

    # ── Discover val recordings ──────────────────────────────────────────
    recordings_root = C.RECORDINGS_ROOT / "val"
    if not recordings_root.exists():
        print(f"ERROR: Validation recordings not found at {recordings_root}")
        sys.exit(1)

    rec_ids = sorted([d.name for d in recordings_root.iterdir() if d.is_dir()])
    print(f"Found {len(rec_ids)} val recordings")

    # ── Process each recording ───────────────────────────────────────────
    per_recording = {}

    n_total_frames = 0

    for rec_id in rec_ids:
        rec_dir = recordings_root / rec_id

        if not rec_dir.exists():
            print(f"  WARNING: {rec_dir} not found, skipping")
            continue

        num_frames = count_frames(rec_dir)
        if num_frames == 0:
            print(f"  WARNING: {rec_dir} has 0 frames, skipping")
            continue

        gt_states = load_psr_labels(rec_dir, num_frames)
        gt_states_bin = (gt_states > 0.5).astype(np.float32)
        n_total_frames += num_frames

        # Build oracle logits (with initial-state signaling)
        oracle_logits = build_oracle_logits(gt_states_bin, mode=args.mode)

        # Run decoder with pre-initialized current_state
        initial_state = gt_states_bin[0, :]  # [11]: what's already placed at frame 0
        pred_states = decode_oracle(oracle_logits, decoder, device,
                                     initial_state=initial_state,
                                     use_procedure_order=not args.relaxed)

        # ── Debug: log first few transitions ─────────────────────────────
        gt_transitions = np.clip(gt_states_bin[1:] - gt_states_bin[:-1], a_min=0, a_max=None)
        pred_transitions = np.clip(pred_states[1:] - pred_states[:-1], a_min=0, a_max=None)

        n_gt = int(gt_transitions.sum())
        n_pred = int(pred_transitions.sum())

        # ── Compute metrics ──────────────────────────────────────────────
        f1_metrics = compute_per_component_f1(pred_states, gt_states_bin,
                                               tol=args.tolerance)
        structure = compute_structure_analysis(pred_states, gt_states_bin)

        per_recording[rec_id] = {
            "num_frames": num_frames,
            "n_gt_transitions": structure["n_gt_transitions"],
            "n_pred_transitions": structure["n_pred_transitions"],
            "component_f1": f1_metrics["per_component_f1"],
            "macro_f1": f1_metrics["macro_f1"],
            "micro_f1": f1_metrics["micro_f1"],
            "n_monotonicity_violations": structure["n_monotonicity_violations"],
            "delay_stats": structure["delay_stats"],
        }

        print(f"  {rec_id}: {num_frames:5d} frames, "
              f"GT={n_gt:2d} transitions, Pred={n_pred:2d}, "
              f"macro F1={f1_metrics['macro_f1']:.4f}, "
              f"micro F1={f1_metrics['micro_f1']:.4f}")

        # For the first recording, print detailed transition info
        if rec_id == rec_ids[0]:
            print(f"    Initial state: {''.join(str(int(gt_states_bin[0, c])) for c in range(11))}")
            for c in range(11):
                gt_t = np.where(gt_transitions[:, c] > 0.5)[0]
                pred_t = np.where(pred_transitions[:, c] > 0.5)[0]
                if len(gt_t) > 0 or len(pred_t) > 0:
                    gt_str = f"GT={gt_t.tolist()}" if len(gt_t) > 0 else "GT=[]"
                    pred_str = f"Pred={pred_t.tolist()}" if len(pred_t) > 0 else "Pred=[]"
                    print(f"    comp{c}: {gt_str}, {pred_str}, F1={f1_metrics['per_component_f1'][str(c)]:.4f}")

    # ── Aggregate ────────────────────────────────────────────────────────
    if not per_recording:
        print("ERROR: No recordings processed!")
        sys.exit(1)

    # Macro: average of per-recording macro F1
    oracle_macro = float(np.mean([r["macro_f1"] for r in per_recording.values()]))
    oracle_micro = float(np.mean([r["micro_f1"] for r in per_recording.values()]))

    # Per-component macro across recordings
    per_comp_across_recs = defaultdict(list)
    for rec_data in per_recording.values():
        for c_str, f1_val in rec_data["component_f1"].items():
            per_comp_across_recs[c_str].append(f1_val)

    oracle_per_component = {}
    for c_str in sorted(per_comp_across_recs.keys(), key=int):
        vals = per_comp_across_recs[c_str]
        oracle_per_component[c_str] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }

    # Total transitions
    total_gt = sum(r["n_gt_transitions"] for r in per_recording.values())
    total_pred = sum(r["n_pred_transitions"] for r in per_recording.values())

    # Aggregate delay statistics
    all_delays_flat = []
    for rec_data in per_recording.values():
        ds = rec_data.get("delay_stats", {})
        for c_str, c_delays in ds.get("per_component", {}).items():
            all_delays_flat.extend(c_delays.get("delays", []))

    delay_summary = {}
    if all_delays_flat:
        delay_summary = {
            "mean": float(np.mean(all_delays_flat)),
            "median": float(np.median(all_delays_flat)),
            "std": float(np.std(all_delays_flat)),
            "min": int(min(all_delays_flat)),
            "max": int(max(all_delays_flat)),
            "p90": float(np.percentile(all_delays_flat, 90)),
            "n": len(all_delays_flat),
        }

    # Global F1
    from src.evaluation.psr_transition_f1 import event_f1 as global_event_f1
    global_pred_tr_list = []
    global_gt_tr_list = []

    # Recompute by iterating recordings (we already have pred_states available)
    # Actually let's recompute from scratch
    # (Using a fresh loop for clarity)
    all_pred_tr_global = []
    all_gt_tr_global = []

    # ── Actual decoder baseline ──────────────────────────────────────────
    actual = run_actual_decoder()

    # ═══════════════════════════════════════════════════════════════════════
    # BUILD RESULTS
    # ═══════════════════════════════════════════════════════════════════════

    results = {
        "description": (
            "MonotonicDecoder Oracle Bound (Opus 141 Q46). "
            "Oracle F1 = decoder F1 when fed perfect GT transition logits. "
            "This is the theoretical upper bound of the decoder. "
            "Gap between oracle and actual F1 = remaining headroom in decoder path."
        ),
        "mode": args.mode,
        "config": {
            "tolerance": args.tolerance,
            "num_components": C.NUM_PSR_COMPONENTS,
            "procedure_order": decoder.procedure_order,
            "use_procedure_order": not args.relaxed,
            "sustain_hi": sustain_hi,
            "sustain_lo": sustain_lo,
            "sustain_min": sustain_min,
        },
        "summary": {
            "n_recordings": len(per_recording),
            "n_total_frames": n_total_frames,
            "n_total_gt_transitions": int(total_gt),
            "n_total_pred_transitions": int(total_pred),
            "oracle_macro_f1": oracle_macro,
            "oracle_micro_f1": oracle_micro,
            "delay_stats": delay_summary,
            "per_component_oracle_f1": oracle_per_component,
            "initial_state_preinitialized": True,
        },
        "actual_decoder_baseline": actual,
        "per_recording": per_recording,
    }

    # ── Compute global F1 from pooled transitions ────────────────────────
    rec_ids_sorted = sorted(per_recording.keys())
    all_pred = []
    all_gt = []
    for rec_id in rec_ids_sorted:
        rec_dir = recordings_root / rec_id
        num_frames = count_frames(rec_dir)
        gt_states = load_psr_labels(rec_dir, num_frames)
        gt_states_bin = (gt_states > 0.5).astype(np.float32)
        oracle_logits = build_oracle_logits(gt_states_bin, mode=args.mode)
        initial_state = gt_states_bin[0, :]
        pred_states = decode_oracle(oracle_logits, decoder, device,
                                     initial_state=initial_state,
                                     use_procedure_order=not args.relaxed)
        all_pred.append(pred_states)
        all_gt.append(gt_states_bin)

    global_pred = np.concatenate(all_pred, axis=0)
    global_gt = np.concatenate(all_gt, axis=0)
    global_f1 = compute_per_component_f1(global_pred, global_gt, tol=args.tolerance)

    results["summary"]["global_macro_f1"] = global_f1["macro_f1"]
    results["summary"]["global_micro_f1"] = global_f1["micro_f1"]
    results["summary"]["global_per_component_f1"] = global_f1["per_component_f1"]

    # ── Save JSON ────────────────────────────────────────────────────────
    json_path = save_dir / "oracle_f1.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {json_path}")

    # ── Save Markdown report ─────────────────────────────────────────────
    gap_actual = ""
    if actual["macro_f1"] is not None:
        gap = oracle_macro - actual["macro_f1"]
        gap_actual = (
            f"\n## Gap Analysis\n\n"
            f"| Metric | Actual | Oracle | Gap |\n"
            f"|--------|--------|--------|-----\n"
            f"| Macro F1 | {actual['macro_f1']:.4f} | {oracle_macro:.4f} | +{gap:.4f} |\n"
        )
        if actual.get("micro_f1") is not None:
            gap_micro = oracle_micro - actual["micro_f1"]
            gap_actual += (
                f"| Micro F1 | {actual['micro_f1']:.4f} | {oracle_micro:.4f} | +{gap_micro:.4f} |\n"
            )
        gap_actual += f"\nSource: {actual.get('source', 'unknown')}\n"

    interpretation = ""
    if oracle_macro >= 0.95:
        interpretation = (
            f"**Oracle Macro F1 = {oracle_macro:.4f} >= 0.95.** "
            f"The decoder is NOT the bottleneck. Upstream transition prediction "
            f"is the binding constraint."
        )
    elif oracle_macro >= 0.85:
        interpretation = (
            f"**Oracle Macro F1 = {oracle_macro:.4f} (0.85-0.95).** "
            f"The decoder imposes moderate constraints. Some headroom exists "
            f"from upstream improvements, but decoder-level changes "
            f"(tuning hysteresis, relaxing procedure order) could also help."
        )
    else:
        interpretation = (
            f"**Oracle Macro F1 = {oracle_macro:.4f} < 0.85.** "
            f"The decoder IS a significant bottleneck. Procedure-order constraints "
            f"and/or hysteresis parameters are suppressing valid transitions."
        )

    delay_interpretation = ""
    if delay_summary:
        if delay_summary.get("median", 0) <= 2 and delay_summary.get("max", 0) <= args.tolerance:
            delay_interpretation = (
                f"Decoder delay (median {delay_summary['median']:.1f}f, "
                f"max {delay_summary['max']}f) is within the ±{args.tolerance}f tolerance. "
                f"Hysteresis is not a significant source of missed events."
            )
        else:
            delay_interpretation = (
                f"Decoder delay (median {delay_summary['median']:.1f}f, "
                f"max {delay_summary['max']}f) occasionally exceeds the "
                f"±{args.tolerance}f tolerance. Consider reducing sustain_min "
                f"or sustain_hi if transition recall is prioritized."
            )

    md_lines = [
        f"# MonotonicDecoder Oracle Bound (Opus 141 Q46)",
        f"",
        f"**Date:** 2026-07-06",
        f"**Mode:** `{args.mode}`",
        f"**Tolerance:** ±{args.tolerance} frames",
        f"",
        f"## Configuration",
        f"",
        f"| Parameter | Value |",
        f"|-----------|-------|",
        f"| Procedure Order | `{decoder.procedure_order}` |",
        f"| Use Procedure Order | `{results['config']['use_procedure_order']}` |",
        f"| `sustain_hi` | {results['config']['sustain_hi']} |",
        f"| `sustain_lo` | {results['config']['sustain_lo']} |",
        f"| `sustain_min` | {results['config']['sustain_min']} |",
        f"| Initial State Pre-initialized | True |",
        f"",
        f"## Summary",
        f"",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Recordings | {results['summary']['n_recordings']} |",
        f"| Total Frames | {results['summary']['n_total_frames']} |",
        f"| Total GT Transitions | {results['summary']['n_total_gt_transitions']} |",
        f"| Total Predicted Transitions | {results['summary']['n_total_pred_transitions']} |",
        f"| **Oracle Macro F1** | **{oracle_macro:.4f}** |",
        f"| **Oracle Micro F1** | **{oracle_micro:.4f}** |",
        f"| **Global Macro F1** | **{global_f1['macro_f1']:.4f}** |",
        f"| **Global Micro F1** | **{global_f1['micro_f1']:.4f}** |",
        f"",
        f"## Per-Component Oracle F1 (Macro Across Recordings)",
        f"",
        f"| Component | Mean F1 | Std | Min | Max |",
        f"|-----------|---------|-----|-----|-----|",
    ]
    for c_str in sorted(oracle_per_component.keys(), key=int):
        v = oracle_per_component[c_str]
        md_lines.append(
            f"| comp{c_str} | {v['mean']:.4f} | {v['std']:.4f} | "
            f"{v['min']:.4f} | {v['max']:.4f} |"
        )

    md_lines.extend([
        f"",
        f"## Per-Recording Results",
        f"",
        f"| Recording | Frames | GT Trans | Pred Trans | Macro F1 | Micro F1 | Violations |",
        f"|-----------|--------|----------|------------|----------|----------|------------|",
    ])
    for rec_id in sorted(per_recording.keys()):
        r = per_recording[rec_id]
        md_lines.append(
            f"| {rec_id} | {r['num_frames']} | {r['n_gt_transitions']} | "
            f"{r['n_pred_transitions']} | {r['macro_f1']:.4f} | "
            f"{r['micro_f1']:.4f} | {r['n_monotonicity_violations']} |"
        )

    if delay_summary:
        md_lines.extend([
            f"",
            f"## Decoder Delay Analysis",
            f"",
            f"| Stat | Value |",
            f"|------|-------|",
            f"| Mean Delay | {delay_summary['mean']:.2f} frames |",
            f"| Median Delay | {delay_summary['median']:.2f} frames |",
            f"| Std Delay | {delay_summary['std']:.2f} frames |",
            f"| P90 Delay | {delay_summary['p90']:.2f} frames |",
            f"| Min Delay | {delay_summary['min']} frames |",
            f"| Max Delay | {delay_summary['max']} frames |",
            f"| N Matched | {delay_summary['n']} |",
        ])

    md_lines.extend([
        f"",
        f"## Interpretation",
        f"",
        interpretation,
        f"",
        delay_interpretation,
    ])

    if gap_actual:
        md_lines.append(gap_actual)

    md_path = save_dir / "oracle_f1.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"Report saved to {md_path}")

    # ── Print summary ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"DECODER ORACLE BOUND RESULTS ({args.mode} mode)")
    print(f"{'='*60}")
    print(f"  Recordings:       {results['summary']['n_recordings']}")
    print(f"  Total frames:     {results['summary']['n_total_frames']}")
    print(f"  Total transitions: {results['summary']['n_total_gt_transitions']} GT, "
          f"{results['summary']['n_total_pred_transitions']} predicted")
    print(f"")
    print(f"  Oracle Macro F1:  {oracle_macro:.4f}")
    print(f"  Oracle Micro F1:  {oracle_micro:.4f}")
    print(f"  Global Macro F1:  {global_f1['macro_f1']:.4f}")
    print(f"  Global Micro F1:  {global_f1['micro_f1']:.4f}")
    print(f"")
    print(f"  Per-component oracle F1:")
    for c_str in sorted(oracle_per_component.keys(), key=int):
        v = oracle_per_component[c_str]
        print(f"    comp{c_str}: {v['mean']:.4f} ± {v['std']:.4f}")
    if delay_summary:
        print(f"")
        print(f"  Decoder delay: mean={delay_summary['mean']:.2f}f, "
              f"median={delay_summary['median']:.2f}f, "
              f"p90={delay_summary['p90']:.2f}f, "
              f"n={delay_summary['n']}")
    if actual["macro_f1"] is not None:
        print(f"")
        print(f"  Actual baseline:  macro F1={actual['macro_f1']:.4f}")
        print(f"  Gap (oracle - actual): {oracle_macro - actual['macro_f1']:.4f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
