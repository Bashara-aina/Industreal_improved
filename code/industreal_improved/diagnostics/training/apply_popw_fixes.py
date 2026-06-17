#!/usr/bin/env python3
"""
apply_popw_fixes.py — Surgical patcher for the POPW/IndustReal codebase
========================================================================
Verified against the 2026-06-10 uploaded snapshot of:
    train.py (3733 lines), model.py (2167), losses.py (1505),
    evaluate.py (4004), eval_post_reinit.py (130)

Every patch is an EXACT-STRING replacement with an expected occurrence
count. If the anchor text is not found exactly N times, the patch is
SKIPPED and reported — nothing is ever fuzzily applied. This makes the
script safe to run against a live repo that may have drifted from the
snapshot: anything that doesn't match verbatim is surfaced for manual
review instead of silently corrupted.

Usage:
    python3 apply_popw_fixes.py --root /path/to/industreal_improved          # dry run (default)
    python3 apply_popw_fixes.py --root /path/to/industreal_improved --apply  # write changes (.bak saved)
    python3 apply_popw_fixes.py --self-test /path/to/snapshot_dir            # verify all patches apply + files still parse

Fix list (P0 = blocks the current retrain, P1 = correctness):

  FIX-1  P0  train.py    _check_per_class_activity_sanity passes the 1-D
             act_per_class_acc list into report_per_class_accuracy, which
             expects a 2-D confusion matrix -> numpy AxisError at
             evaluate.py:869 KILLS the run after every eval where
             epoch % 10 == 0, BEFORE latest.pth is saved (the train_digest
             shows this crash twice). Fix: pass act_confusion_matrix and
             wrap in try/except so a logging helper can never abort training.

  FIX-2  P0  evaluate.py report_per_class_accuracy hardened to also accept
             a 1-D per-class-accuracy vector (belt and suspenders for FIX-1).

  FIX-3  P0  model.py    forward() decides "this is a temporal sequence"
             from the persistent model._seq_len tag instead of the input
             tensor shape. With _seq_len=4 set once at startup, ANY 4-D
             batch whose size is divisible by 4 (train batch=4, val
             batch=16, eval BS=4) is regrouped into fake 4-frame sequences
             of UNRELATED frames. Fix: only treat dim()==5 inputs as
             sequences; read T from the tensor itself.

  FIX-4  P0  model.py    in the sequence path, PSR computes ONE prediction
             from the last position and .expand()s it across all T frames.
             Per-frame focal loss then compares one prediction to T
             different labels (optimum = window-average label = the exact
             constant-output collapse observed), and the temporal-smooth
             loss has IDENTICALLY ZERO gradient (pred diffs are diffs of
             the same tensor). Fix: apply the 11 output heads to every
             encoded position -> true per-frame causal predictions.

  FIX-5  P1  train.py    two of the three early-`continue` paths in the
             PSR sequence branch do NOT restore criterion.train_det/
             train_pose/train_act (saved+mutated for the PSR-only loss
             call). If either path fires once, every subsequent normal
             batch silently trains PSR ONLY for the rest of the run.

  FIX-6  P1  losses.py   the final `_safe()` NaN guard sets
             _nan_detected_this_step = True without `nonlocal`, so the
             assignment creates a dead local and loss_dict['__nan_detected__']
             underreports — train.py then backwards through NaN-replaced
             losses it was designed to skip.

  FIX-7  P1  losses.py   temporal-smooth loss negates the label change
             (diff_l = -1 * ...), i.e. it pushes the predicted transition
             in the OPPOSITE direction of the label transition. Dormant
             today only because FIX-4's expand bug zeroed the gradient;
             actively harmful once FIX-4 lands.

  FIX-8  P1  eval_post_reinit.py  typo 'USE_HEADPOSE_FIM' (missing L)
             builds the eval model with use_headpose_film=False while the
             checkpoint was trained with True — headpose_film weights are
             silently dropped at every re-eval.

Author: prepared for Bashara, 2026-06-10.
"""

from __future__ import annotations

import argparse
import ast
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Patch definitions
# ---------------------------------------------------------------------------

@dataclass
class Patch:
    fix_id: str
    priority: str            # 'P0' or 'P1'
    filename: str            # bare filename; located by search under --root
    summary: str
    old: str
    new: str
    expected_count: int = 1  # how many times `old` must occur in the file
    applied: bool = field(default=False, init=False)
    note: str = field(default='', init=False)


PATCHES: list[Patch] = [

    # ----------------------------------------------------------------- FIX-1
    Patch(
        fix_id='FIX-1',
        priority='P0',
        filename='train.py',
        summary=('_check_per_class_activity_sanity: pass the confusion matrix '
                 '(2-D) instead of the 1-D per-class-acc list; never let a '
                 'logging helper kill training'),
        old=(
            "    per_class_acc = val_metrics.get('act_per_class_acc', [])\n"
            "    if not per_class_acc:\n"
            "        return\n"
            "\n"
            "    from evaluate import report_per_class_accuracy\n"
            "    report_per_class_accuracy(per_class_acc, C.ACT_CLASS_NAMES, k=5)\n"
        ),
        new=(
            "    # [FIX 2026-06-10] report_per_class_accuracy expects a 2-D confusion\n"
            "    # matrix (it does cm.sum(axis=1)). Passing the 1-D act_per_class_acc\n"
            "    # list raised numpy AxisError at evaluate.py report_per_class_accuracy\n"
            "    # and KILLED the whole run after every eval at epoch % 10 == 0,\n"
            "    # before latest.pth was written (see train_digest.log tracebacks).\n"
            "    cm = val_metrics.get('act_confusion_matrix', None)\n"
            "    if cm is None:\n"
            "        cm = val_metrics.get('act_per_class_acc', [])  # 1-D fallback; evaluate.py now accepts it\n"
            "    if cm is None or len(cm) == 0:\n"
            "        return\n"
            "\n"
            "    try:\n"
            "        from evaluate import report_per_class_accuracy\n"
            "        report_per_class_accuracy(cm, C.ACT_CLASS_NAMES, k=5)\n"
            "    except Exception as exc:  # logging helper must NEVER abort training\n"
            "        logger.warning(f'  [SANITY] per-class activity report failed (non-fatal): {exc}')\n"
        ),
    ),

    # ----------------------------------------------------------------- FIX-2
    Patch(
        fix_id='FIX-2',
        priority='P0',
        filename='evaluate.py',
        summary='report_per_class_accuracy: accept 1-D per-class-acc input too',
        old=(
            "    row_sums = cm.sum(axis=1).clip(min=1.0)\n"
            "    per_class_acc = cm.diagonal() / row_sums\n"
        ),
        new=(
            "    # [FIX 2026-06-10] Accept either a 2-D confusion matrix or a 1-D\n"
            "    # per-class-accuracy vector (train.py historically passed the latter,\n"
            "    # producing AxisError: axis 1 is out of bounds for array of dimension 1).\n"
            "    if cm.ndim == 1:\n"
            "        per_class_acc = cm\n"
            "    else:\n"
            "        row_sums = cm.sum(axis=1).clip(min=1.0)\n"
            "        per_class_acc = cm.diagonal() / row_sums\n"
        ),
    ),

    # ----------------------------------------------------------------- FIX-3
    Patch(
        fix_id='FIX-3',
        priority='P0',
        filename='model.py',
        summary=('forward(): only dim()==5 inputs are temporal sequences; '
                 'stop regrouping ordinary 4-D batches via the persistent '
                 '_seq_len tag (batch=4 train, batch=16 val, BS=4 re-eval all '
                 'currently become fake sequences of unrelated frames)'),
        old=(
            "        # Handle [B, T, C, H, W] sequence input from PSR sequence mode\n"
            "        # _prepare_images flattens it to [B*T, C, H, W]; detect via _seq_len attribute\n"
            "        seq_len = getattr(self, '_seq_len', 1)\n"
            "        BT = B\n"
            "        if images.dim() == 5:\n"
            "            BT = images.shape[0] * images.shape[1]\n"
            "            images = images.reshape(BT, images.shape[2], images.shape[3], images.shape[4])\n"
        ),
        new=(
            "        # Handle [B, T, C, H, W] sequence input from PSR sequence mode.\n"
            "        # [FIX 2026-06-10] Sequence-ness is decided by the INPUT SHAPE, not by\n"
            "        # the persistent model._seq_len tag. The old logic regrouped ANY 4-D\n"
            "        # batch whose size was divisible by _seq_len (train batch=4, val\n"
            "        # batch=16, eval BS=4) into fake 4-frame 'sequences' of unrelated\n"
            "        # frames, then emitted ONE PSR prediction copied across the group —\n"
            "        # the direct mechanism behind 'unique_binary_patterns=1' at eval.\n"
            "        BT = B\n"
            "        if images.dim() == 5:\n"
            "            seq_len = images.shape[1]  # trust the tensor, not the global tag\n"
            "            BT = images.shape[0] * images.shape[1]\n"
            "            images = images.reshape(BT, images.shape[2], images.shape[3], images.shape[4])\n"
            "        else:\n"
            "            seq_len = 1  # 4-D input = independent frames; NEVER fake-sequence them\n"
        ),
    ),

    # ----------------------------------------------------------------- FIX-4
    Patch(
        fix_id='FIX-4',
        priority='P0',
        filename='model.py',
        summary=('sequence path: per-position PSR predictions instead of '
                 'last-position output expanded across T (restores per-frame '
                 'supervision and gives the temporal-smooth loss a real gradient)'),
        old=(
            "            # Use last position (causal -- only sees past)\n"
            "            last_out = encoded[:, -1, :]  # [B, hidden]\n"
            "            # Each head takes [B, hidden] → [B, 1]; cat gives [B, 12] (logits + confidence, Item 32)\n"
            "            psr_full = torch.cat([\n"
            "                head(last_out) for head in self.psr_head.output_heads\n"
            "            ], dim=-1)  # [B, 11]\n"
            "            confidence = torch.sigmoid(psr_full).max(dim=-1, keepdim=True)[0]  # [B, 1]\n"
            "            psr_logits = torch.cat([psr_full, confidence], dim=-1)  # [B, 12]\n"
            "            psr_logits = psr_logits.unsqueeze(1).expand(B_main, T_main, -1)  # [B, T, 12]\n"
            "            psr_logits = psr_logits.reshape(B_main * T_main, -1)  # [BT, 12]\n"
            "            psr_confidence = psr_logits[..., 11:]  # [BT, 1]\n"
        ),
        new=(
            "            # [FIX 2026-06-10] Per-position predictions. The old code took only\n"
            "            # encoded[:, -1, :] and .expand()ed one prediction across all T\n"
            "            # frames: the per-frame focal loss then compared ONE prediction to\n"
            "            # T different labels (optimum = window-average label = constant\n"
            "            # output collapse), and the temporal-smooth loss had identically\n"
            "            # zero gradient (differences of the same tensor). Causal masking\n"
            "            # already guarantees position t only sees frames <= t, so every\n"
            "            # position is a valid (and supervised) per-frame prediction.\n"
            "            enc_flat = encoded.reshape(B_main * T_main, -1)  # [BT, hidden]\n"
            "            psr_full = torch.cat([\n"
            "                head(enc_flat) for head in self.psr_head.output_heads\n"
            "            ], dim=-1)  # [BT, 11]\n"
            "            confidence = torch.sigmoid(psr_full).max(dim=-1, keepdim=True)[0]  # [BT, 1]\n"
            "            psr_logits = torch.cat([psr_full, confidence], dim=-1)  # [BT, 12]\n"
            "            psr_confidence = psr_logits[..., 11:]  # [BT, 1]\n"
        ),
    ),

    # ----------------------------------------------------------------- FIX-5
    Patch(
        fix_id='FIX-5',
        priority='P1',
        filename='train.py',
        summary=('PSR sequence branch: restore criterion train_* flags on the '
                 'two early-continue paths that currently leak train_det='
                 'train_pose=train_act=False into all subsequent batches'),
        old=(
            "                nan_skips += 1\n"
            "                optimizer.zero_grad(set_to_none=True)\n"
            "                del outputs_seq, loss_seq, loss_dict_seq, fake_outputs, fake_targets\n"
            "                torch.cuda.empty_cache()\n"
            "                continue\n"
        ),
        new=(
            "                nan_skips += 1\n"
            "                optimizer.zero_grad(set_to_none=True)\n"
            "                del outputs_seq, loss_seq, loss_dict_seq, fake_outputs, fake_targets\n"
            "                torch.cuda.empty_cache()\n"
            "                # [FIX 2026-06-10] Restore criterion flags — without this, an\n"
            "                # early exit here leaves train_det/pose/act=False PERMANENTLY\n"
            "                # and every later normal batch silently trains PSR only.\n"
            "                criterion.train_det  = _saved_train_det\n"
            "                criterion.train_pose = _saved_train_pose\n"
            "                criterion.train_act  = _saved_train_act\n"
            "                criterion.train_psr  = _saved_train_psr\n"
            "                continue\n"
        ),
        expected_count=2,  # the isfinite path and the requires_grad path share this block
    ),

    # ----------------------------------------------------------------- FIX-6
    Patch(
        fix_id='FIX-6',
        priority='P1',
        filename='losses.py',
        summary=("_safe(): add `nonlocal` so the final NaN guard actually sets "
                 "loss_dict['__nan_detected__'] (currently a dead local write)"),
        old=(
            "        def _safe(l, z):\n"
            "            if not torch.isfinite(l).all():\n"
            "                _nan_detected_this_step = True  # signal train.py to skip this step\n"
        ),
        new=(
            "        def _safe(l, z):\n"
            "            # [FIX 2026-06-10] Without `nonlocal`, this assignment created a\n"
            "            # NEW local inside the closure and the outer flag never changed —\n"
            "            # __nan_detected__ underreported, and train.py happily backwarded\n"
            "            # through NaN-replaced losses it was designed to skip.\n"
            "            nonlocal _nan_detected_this_step\n"
            "            if not torch.isfinite(l).all():\n"
            "                _nan_detected_this_step = True  # signal train.py to skip this step\n"
        ),
    ),

    # ----------------------------------------------------------------- FIX-7
    Patch(
        fix_id='FIX-7',
        priority='P1',
        filename='losses.py',
        summary=('temporal-smooth loss: remove the sign flip that pushes the '
                 'predicted transition OPPOSITE to the label transition'),
        old=(
            "                    diff_l = -1 * (l_i[1:] - l_i[:-1]).mean()\n"
        ),
        new=(
            "                    # [FIX 2026-06-10] The -1 made the loss push tanh(pred_diff)\n"
            "                    # toward the NEGATIVE of the label transition. Dormant while\n"
            "                    # seq-mode predictions were expanded copies (pred_diff == 0\n"
            "                    # exactly, zero gradient); actively harmful once per-position\n"
            "                    # predictions exist. Match the label change directly.\n"
            "                    diff_l = (l_i[1:] - l_i[:-1]).float().mean()\n"
        ),
    ),

    # ----------------------------------------------------------------- FIX-8
    Patch(
        fix_id='FIX-8',
        priority='P1',
        filename='eval_post_reinit.py',
        summary="typo USE_HEADPOSE_FIM -> USE_HEADPOSE_FILM (eval silently dropped the trained headpose_film)",
        old="use_headpose_film=bool(getattr(C, 'USE_HEADPOSE_FIM', False)),",
        new="use_headpose_film=bool(getattr(C, 'USE_HEADPOSE_FILM', False)),",
    ),
]


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

# Canonical locations for the 4 patched source files. Listed FIRST so the
# patcher always edits live code, never an audit/analysis snapshot copy.
# Order matters — the first existing entry wins.
CANONICAL_PATHS: dict[str, tuple[str, ...]] = {
    'train.py':           ('src/training/train.py',),
    'losses.py':          ('src/training/losses.py',),
    'evaluate.py':        ('src/evaluation/evaluate.py',),
    'model.py':           ('src/models/model.py', 'model.py'),  # model.py is symlinked at root
    'eval_post_reinit.py':('eval_post_reinit.py',),
}

# Folders that contain audit/analysis COPIES of the live source, never the
# live source itself. The patcher must skip them even if they are shallower
# than the canonical path.
_AUDIT_DIR_PARTS = (
    'opus_analysis', 'opus_analysis_package', 'opus_package_v2',
    'opus_consult_2026_06_10', 'files (1)', 'files',
)


def find_file(root: Path, filename: str) -> Path | None:
    """Locate `filename` under root, preferring canonical source paths.

    The repo keeps train.py/losses.py under src/training/, evaluate.py under
    src/evaluation/, model.py at src/models/ (symlinked at repo root), so the
    canonical locations are tried first. If none exist (e.g. the user has a
    flat layout), fall back to a recursive search that excludes audit
    snapshots and runs/ artifacts. Backup files (*.bak*), __pycache__, and
    the runs/ tree are always skipped.
    """
    # 1. Try the canonical paths first.
    for rel in CANONICAL_PATHS.get(filename, ()):
        cand = root / rel
        if cand.is_file():
            return cand

    # 2. Fall back to a recursive search, excluding audit/analysis copies,
    #    runs/, __pycache__, and any backup files.
    candidates = [
        p for p in root.rglob(filename)
        if p.is_file()
        and '.bak' not in p.name
        and 'runs' not in p.parts
        and '__pycache__' not in p.parts
        and not any(part.startswith(_AUDIT_DIR_PARTS) for part in p.parts)
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda p: len(p.parts))
    return candidates[0]


def apply_patches(root: Path, write: bool) -> int:
    """Apply (or dry-run) all patches. Returns a process exit code."""
    print(f'{"APPLY" if write else "DRY RUN"} against root: {root}\n')

    by_file: dict[str, list[Patch]] = {}
    for p in PATCHES:
        by_file.setdefault(p.filename, []).append(p)

    n_ok = n_skip = 0
    for filename, patches in by_file.items():
        path = find_file(root, filename)
        if path is None:
            for p in patches:
                p.note = f'file {filename} not found under {root}'
                n_skip += len(patches)
            print(f'[MISS] {filename}: not found under {root} — '
                  f'{len(patches)} patch(es) skipped')
            continue

        text = original = path.read_text(encoding='utf-8')
        file_changed = False

        for p in patches:
            count = text.count(p.old)
            if count != p.expected_count:
                p.note = (f'anchor found {count}x (expected {p.expected_count}) '
                          f'in {path} — file has drifted from the audited '
                          f'snapshot; apply manually per POPW_DEEP_AUDIT.md')
                n_skip += 1
                print(f'[SKIP] {p.fix_id} ({p.priority}) {filename}: {p.note}')
                continue
            text = text.replace(p.old, p.new)
            p.applied = True
            file_changed = True
            n_ok += 1
            print(f'[ OK ] {p.fix_id} ({p.priority}) {filename}: {p.summary}')

        if file_changed:
            # Syntax check the patched text before touching disk.
            try:
                ast.parse(text)
            except SyntaxError as exc:
                print(f'[FATAL] patched {path} fails to parse: {exc} — '
                      f'NOT writing. Please report this.')
                return 2
            if write:
                bak = path.with_suffix(path.suffix + '.bak_prefix_20260610')
                if not bak.exists():
                    shutil.copy2(path, bak)
                path.write_text(text, encoding='utf-8')
                print(f'       wrote {path} (backup: {bak.name})')
            else:
                assert original is not None  # readability

    print(f'\nSummary: {n_ok} patch(es) {"applied" if write else "would apply"}, '
          f'{n_skip} skipped.')
    if not write and n_ok:
        print('Re-run with --apply to write changes (originals saved as *.bak_prefix_20260610).')
    return 0 if n_skip == 0 else 1


def self_test(snapshot_dir: Path) -> int:
    """Verify every patch applies exactly once against the snapshot files and
    every patched file still parses. Intended for the uploaded snapshot."""
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        work = Path(td)
        for fname in {p.filename for p in PATCHES}:
            src = snapshot_dir / fname
            if not src.exists():
                print(f'[SELF-TEST] missing {src}')
                return 2
            shutil.copy2(src, work / fname)
        code = apply_patches(work, write=True)
        if code == 0:
            print('\n[SELF-TEST] all patches applied and all files parse. PASS')
        else:
            print('\n[SELF-TEST] FAIL — see skips above')
        return code


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', type=Path, default=Path('.'),
                    help='repo root to patch (searched recursively by filename)')
    ap.add_argument('--apply', action='store_true',
                    help='actually write changes (default: dry run)')
    ap.add_argument('--self-test', type=Path, metavar='SNAPSHOT_DIR',
                    help='verify patches against a flat snapshot dir and exit')
    args = ap.parse_args()

    if args.self_test:
        return self_test(args.self_test)
    return apply_patches(args.root, write=args.apply)


if __name__ == '__main__':
    sys.exit(main())
