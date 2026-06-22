#!/usr/bin/env python3
"""
diag_per_class_truth.py — Authoritative per-class detection truth from metrics.jsonl.

[Opus v11 §D] Resolves the per-class confusion (43_OPUS_MASTER_PROMPT_v11.md §5)
using data the training run ALREADY wrote. No torch, no GPU, no re-run — it just
parses runs/<run>/logs/metrics.jsonl (stdlib only).

It answers, from existing data:
  1. The REAL per-class AP + GT count, labeled by channel / category_id / name,
     so the channel-vs-category index ambiguity ("class 6 = 1739 GT") can't recur.
  2. det_mAP50 (COCO-24, diluted) vs det_mAP50_pc (present-class, honest), and the
     dilution gap caused by background + zero-GT channels.
  3. Which channels are background / zero-GT (UNMEASURABLE — not failures) vs the
     channels that genuinely have GT but score AP=0 (the real "stuck" classes).
  4. The class-6 question, directly.

Usage:
  python3 diag_per_class_truth.py [path/to/metrics.jsonl]
  python3 diag_per_class_truth.py --run runs/full_multi_task_tma_tbank_benchmark
  python3 diag_per_class_truth.py --epoch 18
"""
from __future__ import annotations

import argparse
import ast
import glob
import json
import os
from typing import Any, Dict, List, Optional

NUM_DET_CLASSES = 24


def load_class_names() -> Dict[int, str]:
    """Extract DET_CLASS_NAMES (1-indexed category_id -> name) from config.py source.

    Done via ast (not import) so this script never needs torch/config's heavy deps.
    Falls back to channel labels if config.py can't be found/parsed.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (os.path.join(here, 'config.py'), 'config.py'):
        try:
            src = open(cand, encoding='utf-8').read()
            i = src.index('DET_CLASS_NAMES = {')
            j = src.index('}', i) + 1
            d = ast.literal_eval(src[i + len('DET_CLASS_NAMES = '):j])
            return {int(k): str(v) for k, v in d.items()}
        except Exception:
            continue
    return {}


def resolve_metrics_path(args: argparse.Namespace) -> Optional[str]:
    if args.path and os.path.isfile(args.path):
        return args.path
    if args.run:
        p = os.path.join(args.run, 'logs', 'metrics.jsonl')
        if os.path.isfile(p):
            return p
    candidates = glob.glob(os.path.join('runs', '**', 'metrics.jsonl'), recursive=True)
    if not candidates:
        candidates = glob.glob(os.path.join('**', 'metrics.jsonl'), recursive=True)
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def read_records(path: str) -> List[Dict[str, Any]]:
    recs = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return recs


def get_val(rec: Dict[str, Any]) -> Dict[str, Any]:
    """A record may be {epoch, val:{...}} or flat metrics."""
    v = rec.get('val')
    return v if isinstance(v, dict) else rec


def has_per_class(val: Dict[str, Any]) -> bool:
    return bool(val.get('det_per_class')
                or val.get('det_per_class_ap')
                or val.get('det_per_class_gt'))


def normalize_keys(d: Any) -> Dict[int, float]:
    """JSON object keys are strings; coerce to int channel -> float."""
    out: Dict[int, float] = {}
    if isinstance(d, dict):
        for k, v in d.items():
            try:
                out[int(k)] = float(v)
            except (TypeError, ValueError):
                continue
    return out


def build_rows(val: Dict[str, Any], names: Dict[int, str]) -> List[Dict[str, Any]]:
    """Prefer the name-labeled det_per_class (v11+); else reconstruct from the two dicts."""
    labeled = val.get('det_per_class')
    if isinstance(labeled, list) and labeled:
        rows = []
        for r in labeled:
            ch = int(r.get('channel', -1))
            rows.append({
                'channel': ch,
                'category_id': int(r.get('category_id', ch + 1)),
                'name': r.get('name') or names.get(ch + 1, f'ch{ch}'),
                'gt': int(r.get('gt', 0)),
                'ap': float(r.get('ap', 0.0)),
                'is_background': bool(r.get('is_background', ch == 0)),
            })
        return sorted(rows, key=lambda r: r['channel'])

    ap = normalize_keys(val.get('det_per_class_ap', {}))
    gt = normalize_keys(val.get('det_per_class_gt', {}))
    channels = sorted(set(ap) | set(gt) | set(range(NUM_DET_CLASSES)))
    rows = []
    for ch in channels:
        rows.append({
            'channel': ch,
            'category_id': ch + 1,
            'name': names.get(ch + 1, f'ch{ch}'),
            'gt': int(gt.get(ch, 0)),
            'ap': float(ap.get(ch, 0.0)),
            'is_background': ch == 0,
        })
    return rows


def fmt_table(rows: List[Dict[str, Any]]) -> str:
    lines = [f"  {'ch':>2}  {'cat':>3}  {'name':<14} {'GT':>6}  {'AP':>6}  status",
             f"  {'--':>2}  {'---':>3}  {'-'*14} {'-'*6}  {'-'*6}  ------"]
    for r in rows:
        if r['is_background']:
            status = 'BACKGROUND (exclude from mAP)'
        elif r['gt'] == 0:
            status = 'zero-GT — UNMEASURABLE (not a failure)'
        elif r['ap'] == 0.0:
            status = '*** STUCK: has GT, AP=0 ***'
        elif r['ap'] >= 0.999:
            status = 'AP=1.0' + ('  <- likely artifact (few GT)' if r['gt'] < 50 else '')
        else:
            status = 'present'
        lines.append(
            f"  {r['channel']:>2}  {r['category_id']:>3}  {r['name']:<14} "
            f"{r['gt']:>6}  {r['ap']:>6.3f}  {status}"
        )
    return '\n'.join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('path', nargs='?', help='path to metrics.jsonl')
    ap.add_argument('--run', help='run dir (uses <run>/logs/metrics.jsonl)')
    ap.add_argument('--epoch', type=int, help='specific epoch (default: latest with det data)')
    args = ap.parse_args()

    path = resolve_metrics_path(args)
    if not path:
        print('ERROR: could not find metrics.jsonl. Pass a path or --run <dir>.')
        return 2
    print(f'# Source: {path}\n')

    recs = read_records(path)
    det_recs = [r for r in recs if has_per_class(get_val(r))]
    if not det_recs:
        print('No records with per-class detection data found.')
        print('(Per-class AP is written from the eval that runs every DET_METRICS_EVERY_N '
              'epochs — make sure a detection eval has completed since the v11 patch.)')
        return 1

    if args.epoch is not None:
        chosen = next((r for r in det_recs if r.get('epoch') == args.epoch), None)
        if chosen is None:
            print(f'No per-class record for epoch {args.epoch}; '
                  f'available: {[r.get("epoch") for r in det_recs]}')
            return 1
    else:
        chosen = det_recs[-1]

    val = get_val(chosen)
    names = load_class_names()
    rows = build_rows(val, names)

    epoch = chosen.get('epoch', '?')
    print(f'# Per-class detection truth @ epoch {epoch}')
    print(fmt_table(rows))

    # --- Honest vs diluted mAP -------------------------------------------------
    all_ap = [r['ap'] for r in rows]
    present = [r for r in rows if r['gt'] > 0 and not r['is_background']]
    present_ap = [r['ap'] for r in present]
    coco24 = sum(all_ap) / len(all_ap) if all_ap else 0.0
    pc = sum(present_ap) / len(present_ap) if present_ap else 0.0

    reported_map = val.get('det_mAP50')
    reported_pc = val.get('det_mAP50_pc')

    n_zero_gt = sum(1 for r in rows if r['gt'] == 0 and not r['is_background'])
    n_bg = sum(1 for r in rows if r['is_background'])
    n_stuck = sum(1 for r in present if r['ap'] == 0.0)

    print('\n# mAP@0.5 — diluted vs honest')
    if reported_map is not None:
        print(f'  det_mAP50      (logged, COCO-24)        = {float(reported_map):.4f}')
    print(f'  recomputed COCO-24 mean (incl. bg+zero-GT) = {coco24:.4f}')
    if reported_pc is not None:
        print(f'  det_mAP50_pc   (logged, present-class)   = {float(reported_pc):.4f}')
    print(f'  recomputed present-class mean (GT>0, no bg) = {pc:.4f}   <-- the honest number')
    print(f'  dilution gap (present - COCO24)            = {pc - coco24:+.4f}')
    print(f'  channels: {len(present)} present | {n_zero_gt} zero-GT | {n_bg} background | '
          f'{n_stuck} STUCK (GT>0, AP=0)')

    # --- Class-6 question, answered directly -----------------------------------
    print('\n# The "class 6" question (v11 §5)')
    ch6 = next((r for r in rows if r['channel'] == 6), None)
    if ch6:
        print(f'  channel 6 = category 7 = {ch6["name"]!r}: GT={ch6["gt"]}, AP={ch6["ap"]:.3f}')
        if ch6['gt'] == 0:
            print('  -> zero GT in this eval: UNMEASURABLE, not a "1739-GT mystery". '
                  'The v11 §5 count was an index/source error.')
        elif ch6['ap'] == 0.0:
            print('  -> genuinely STUCK (has GT, AP=0). Compare its GT count above to the '
                  'working classes before calling it data-starvation vs confusion.')
        else:
            print('  -> actually scoring; not stuck.')

    # --- The genuinely-stuck classes (the real worklist) -----------------------
    stuck = sorted((r for r in present if r['ap'] == 0.0), key=lambda r: -r['gt'])
    if stuck:
        print('\n# Genuinely stuck classes (GT>0, AP=0) — the real worklist, by GT count')
        for r in stuck:
            print(f'  ch {r["channel"]:>2} ({r["name"]}): GT={r["gt"]}, AP=0.000')
    else:
        print('\n# No genuinely-stuck classes: every channel with GT>0 scores AP>0.')

    print('\n# Read this as: judge subset-stage progress by the present-class mean, '
          'triage only the STUCK rows, and ignore zero-GT/background rows.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
