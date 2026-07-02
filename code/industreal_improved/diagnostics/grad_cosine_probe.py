#!/usr/bin/env python3
"""Per-task backbone gradient cosine-similarity probe (offline, checkpoint-based).

[F12 2026-07-02 Fable RF4 consult — answers Q49/Q50 of file 95]

Measures, for each task head (detection, head_pose, activity, psr), the gradient
it induces on the SHARED parameters (backbone + FPN), then reports:
  - per-task backbone/FPN gradient norms (who is actually steering the trunk)
  - pairwise cosine similarity between task gradients (are tasks cooperating,
    orthogonal, or fighting?)

Interpretation:
  cos > 0.3   tasks reinforce each other on shared features (multi-task WIN)
  |cos| < 0.1 tasks are ~orthogonal (multi-task is free capacity-sharing)
  cos < -0.2  tasks actively conflict (consider gradient surgery / rebalancing)
  PSR is expected to show ZERO backbone gradient while DETACH_PSR_FPN=True —
  this script doubles as the verification of that isolation.

Runs entirely offline from a checkpoint — safe to execute on the idle GPU
while training continues on the other card:

    CUDA_VISIBLE_DEVICES=0 python diagnostics/grad_cosine_probe.py \
        --checkpoint src/runs/<run>/checkpoints/latest.pth \
        --preset stage_rf4 --num-batches 8

Results are averaged over --num-batches batches drawn from the train loader.
"""
import argparse
import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch


TASKS = ('det', 'head_pose', 'act', 'psr')


def _flag_map(task):
    """criterion train_* flags isolating a single task."""
    return {
        'train_det': task == 'det',
        # head_pose loss is computed when train_pose OR train_act is set; use
        # train_pose alone so the activity CE stays out of the head_pose probe.
        'train_pose': task == 'head_pose',
        'train_act': task == 'act',
        'train_psr': task == 'psr',
    }


def _shared_params(model):
    named = []
    for mod_name in ('backbone', 'fpn'):
        mod = getattr(model, mod_name, None)
        if mod is None:
            continue
        for pn, p in mod.named_parameters(prefix=mod_name):
            if p.requires_grad:
                named.append((pn, p))
    return named


def _grad_vector(loss, params):
    grads = torch.autograd.grad(
        loss, [p for _, p in params],
        retain_graph=True, allow_unused=True,
    )
    flat = []
    for (_, p), g in zip(params, grads):
        flat.append((g if g is not None else torch.zeros_like(p)).reshape(-1))
    return torch.cat(flat)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--preset', default='stage_rf4')
    ap.add_argument('--num-batches', type=int, default=8)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    from src import config as C
    C.apply_preset(args.preset)

    from src.models.model import POPWMultiTaskModel
    from src.training.losses import MultiTaskLoss
    from src.training.train import _build_loader  # reuse canonical loader config
    # train.py's import bootstraps sys.path with the src/ subdirectories, so the
    # dataset imports the same way train.py imports it.
    try:
        from data.industreal_dataset import IndustRealMultiTaskDataset
    except ImportError:
        from src.data.industreal_dataset import IndustRealMultiTaskDataset

    device = torch.device(args.device)
    model = POPWMultiTaskModel().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = ckpt.get('model_state_dict', ckpt.get('model', ckpt))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f'[probe] load_state_dict: {len(missing)} missing, {len(unexpected)} unexpected keys')
    model.train()  # need training-mode graph (eval mode changes head paths)

    criterion = MultiTaskLoss().to(device)
    epoch = int(ckpt.get('epoch', 0))
    criterion.set_epoch(epoch)

    train_ds = IndustRealMultiTaskDataset(
        split='train', img_size=C.IMG_SIZE, augment=False, seed=C.SEED,
    )
    train_loader = _build_loader(train_ds, 'train', C.BATCH_SIZE, num_workers=0)

    shared = _shared_params(model)
    n_shared = sum(p.numel() for _, p in shared)
    print(f'[probe] epoch={epoch} shared params (backbone+FPN): {n_shared/1e6:.1f}M, '
          f'batches={args.num_batches}, DETACH_PSR_FPN={getattr(C, "DETACH_PSR_FPN", None)}')

    norms = {t: [] for t in TASKS}
    cosines = {pair: [] for pair in itertools.combinations(TASKS, 2)}

    it = iter(train_loader)
    for b in range(args.num_batches):
        images, targets = next(it)
        images = images.to(device)
        for k, v in list(targets.items()):
            if isinstance(v, torch.Tensor):
                targets[k] = v.to(device)
        if 'detection' in targets:
            for t in targets['detection']:
                t['boxes'] = t['boxes'].to(device)
                t['labels'] = t['labels'].to(device)

        video_ids = [m['recording_id'] for m in targets['metadata']] if 'metadata' in targets else None
        outputs = model(images, video_ids=video_ids)

        vecs = {}
        for task in TASKS:
            saved = {k: getattr(criterion, k) for k in _flag_map(task)}
            for k, v in _flag_map(task).items():
                setattr(criterion, k, v)
            try:
                loss, _ = criterion(outputs, targets)
                vecs[task] = _grad_vector(loss, shared)
            except Exception as e:  # keep the probe running on odd batches
                print(f'[probe] batch {b} task {task}: skipped ({e})')
                vecs[task] = None
            finally:
                for k, v in saved.items():
                    setattr(criterion, k, v)

        for task, v in vecs.items():
            if v is not None:
                norms[task].append(v.norm().item())
        for (a, bb) in cosines:
            va, vb = vecs.get(a), vecs.get(bb)
            if va is None or vb is None:
                continue
            na, nb = va.norm(), vb.norm()
            if na > 1e-12 and nb > 1e-12:
                cosines[(a, bb)].append((va @ vb / (na * nb)).item())

        del outputs, vecs
        print(f'[probe] batch {b + 1}/{args.num_batches} done')

    print('\n=== Per-task shared-parameter gradient norms (mean over batches) ===')
    for t in TASKS:
        vals = norms[t]
        mean = sum(vals) / len(vals) if vals else float('nan')
        print(f'  {t:10s}: {mean:.4e}' + ('   <-- ZERO expected while DETACH_PSR_FPN=True'
                                           if t == 'psr' else ''))

    print('\n=== Pairwise gradient cosine similarity (mean over batches) ===')
    for (a, bb), vals in cosines.items():
        if vals:
            mean = sum(vals) / len(vals)
            verdict = ('COOPERATING' if mean > 0.3 else
                       'conflicting' if mean < -0.2 else 'orthogonal-ish')
            print(f'  {a:10s} vs {bb:10s}: {mean:+.4f}  ({verdict}, n={len(vals)})')
        else:
            print(f'  {a:10s} vs {bb:10s}: n/a (one side had zero/failed gradient)')


if __name__ == '__main__':
    main()
