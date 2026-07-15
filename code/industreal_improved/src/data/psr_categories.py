"""
PSR State Categories — Author-Faithful 24-Class Classification
================================================================

Implements the 24-category state space from the IndustReal paper's PSR pipeline
(copied verbatim from industreal_github/PSR/psr_utils.py lines 31-54).

Architecture (Path B):
  - Background (idx 0): no assembly in progress (state '00000000000')
  - States 1-22 (idx 1-22): the 22 valid 11-bit assembly states
  - Error_state (idx 23): invalid/uncertain state (rare in clean recordings)

Why this matters:
  Per-frame 11-component binary classification (old approach) treats each
  component as independent. The authors' pipeline instead treats the WHOLE
  ASSEMBLY STATE as a discrete variable that transitions through 22 valid
  configurations. This enforces structural constraints (e.g., a valid state
  string has bits 0-2 always set together once any of them is set, per the
  procedure_info.json order).

Conversion:
  Per-frame 11-bit dense label (from PSR_labels_raw.csv) -> single int class index.
  At inference: 24-class softmax -> argmax -> state string -> Naive/Accumulated
  Confidence PSR (psr_utils.py) -> step completion events -> POS/F1/delay.

This module is the SINGLE SOURCE OF TRUTH for the state space — both training
(labels) and inference (decoding) read from here.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


# =============================================================================
# 24 Categories (verbatim from psr_utils.py lines 31-54)
# =============================================================================
# Order matters: this is the class index used everywhere.
# - idx 0: 'background' (state '00000000000' — no assembly)
# - idx 1..22: the 22 valid 11-bit assembly states
# - idx 23: 'error_state' (invalid/uncertain)
CATEGORIES: List[str] = [
    "background",
    "10000000000",  # state 1
    "10010010000",  # state 2
    "10010100000",  # state 3
    "10010110000",  # state 4
    "11100000000",  # state 5
    "11110010000",  # state 6
    "11110100000",  # state 7
    "11110110000",  # state 8
    "11110111100",  # state 9
    "11110111110",  # state 10
    "11110110001",  # state 11
    "11110111101",  # state 12
    "11110111111",  # state 13
    "11110101111",  # state 14
    "11110011111",  # state 15
    "11110011110",  # state 16
    "11110101110",  # state 17
    "11100001110",  # state 18
    "11101101110",  # state 19
    "11101011110",  # state 20
    "11101111110",  # state 21
    "11101111111",  # state 22
    "error_state",
]

NUM_CATEGORIES: int = len(CATEGORIES)  # 24
BACKGROUND_IDX: int = 0
FIRST_VALID_STATE_IDX: int = 1  # state 1
LAST_VALID_STATE_IDX: int = 22  # state 22
ERROR_STATE_IDX: int = 23
NUM_VALID_STATES: int = 22  # states 1..22
IGNORE_INDEX: int = -1  # for cross-entropy ignore_index


# =============================================================================
# Lookup tables
# =============================================================================
# State string -> class index (for converting fill-forward labels to int targets)
_STRING_TO_IDX: Dict[str, int] = {s: i for i, s in enumerate(CATEGORIES)}

# 11-bit integer encoding -> class index (fast lookup for numpy arrays)
# Encode a state string as int: bit[i] = char[i] for i in 0..10
_BIT_TO_IDX: Dict[int, int] = {}
for _i, _s in enumerate(CATEGORIES):
    if _s in ("background", "error_state"):
        continue
    _bits = int(_s, 2)  # parse as binary
    _BIT_TO_IDX[_bits] = _i


def state_string_to_list(state_string: str) -> List[int]:
    """Convert an 11-bit state string to a list of 0/1 ints. (Mirrors psr_utils.py)"""
    return [int(c) for c in state_string]


def list_to_state_string(state_list: List[int]) -> str:
    """Convert a list of 0/1 ints to an 11-bit state string."""
    return "".join(str(int(v)) for v in state_list)


def bits_to_int(state_list) -> int:
    """Convert 11-element list/array of 0/1 to int bit encoding (MSB-first).

    Convention: matches `int(s, 2)` parsing of the 11-bit state string.
    Element at index 0 (leftmost char in '10000000000') maps to bit 10.
    Element at index 10 (rightmost char) maps to bit 0.

    So state_list = [1,0,0,0,0,0,0,0,0,0,0] (base installed) encodes as 2**10 = 1024.
    This matches `int("10000000000", 2) == 1024`.
    """
    out = 0
    for i, v in enumerate(state_list):
        if int(v) == 1:
            out |= (1 << (10 - i))
    return out


def int_to_bits(bits: int, n: int = 11) -> List[int]:
    """Convert int bit encoding back to n-element list of 0/1 (MSB-first).

    Inverse of bits_to_int. bit 10 of `bits` becomes element 0, bit 0 becomes element 10.
    """
    return [(bits >> (10 - i)) & 1 for i in range(n)]


# =============================================================================
# Conversion: 11-bit dense labels -> class index
# =============================================================================
def state_vector_to_class_idx(state_11bit) -> int:
    """Convert one 11-element array (0/1) to a class index.

    Returns:
        idx in [0, 23]:
          - 0 (background) for all-zero vector
          - idx in [1, 22] for any matching valid 11-bit state string
          - 23 (error_state) if vector doesn't match any valid state
          - -1 (IGNORE_INDEX) if vector contains -1 (error/aborted, sentinel)

    Raises:
        ValueError: if vector length != 11
    """
    state_11bit = np.asarray(state_11bit).flatten()
    if state_11bit.shape[0] != 11:
        raise ValueError(f"Expected 11-element vector, got {state_11bit.shape}")

    # Sentinel: any -1 means ignore
    if np.any(state_11bit < 0):
        return IGNORE_INDEX

    bits = bits_to_int(state_11bit)
    if bits == 0:
        return BACKGROUND_IDX  # all-zero = background
    return _BIT_TO_IDX.get(bits, ERROR_STATE_IDX)


def dense_labels_to_class_idx(dense: np.ndarray) -> np.ndarray:
    """Vectorized: per-frame [num_frames, 11] dense 0/1 labels -> per-frame int class index.

    Args:
        dense: np.ndarray of shape [num_frames, 11], values in {-1, 0, 1}.

    Returns:
        np.ndarray of shape [num_frames], dtype int64. Values in {-1, 0..23}.
        -1 means ignore (frame has -1 sentinel in any component).
    """
    if dense.ndim != 2 or dense.shape[1] != 11:
        raise ValueError(f"Expected [num_frames, 11], got {dense.shape}")

    # Check for -1 sentinel per frame (any -1 in the row → ignore)
    has_neg1 = np.any(dense < 0, axis=1)  # [num_frames]
    bits_per_frame = np.zeros(dense.shape[0], dtype=np.int64)
    for i in range(dense.shape[0]):
        bits_per_frame[i] = bits_to_int(np.clip(dense[i], 0, 1))

    # Build lookup table for [0, 2^11) → class index (cached)
    if not hasattr(dense_labels_to_class_idx, "_lookup"):
        lut = np.full(2 ** 11, ERROR_STATE_IDX, dtype=np.int64)
        lut[0] = BACKGROUND_IDX
        for bits, idx in _BIT_TO_IDX.items():
            lut[bits] = idx
        dense_labels_to_class_idx._lookup = lut  # type: ignore[attr-defined]

    lut = dense_labels_to_class_idx._lookup  # type: ignore[attr-defined]
    out = lut[bits_per_frame]
    out[has_neg1] = IGNORE_INDEX
    return out.astype(np.int64)


# =============================================================================
# State class distribution utilities (for class-balanced loss weighting)
# =============================================================================
def class_frequencies_from_indices(
    indices: np.ndarray,
    num_classes: int = NUM_CATEGORIES,
    ignore_index: int = IGNORE_INDEX,
) -> np.ndarray:
    """Count per-class frequency (ignoring ignore_index frames).

    Args:
        indices: np.ndarray of int class indices, may include ignore_index (-1).
        num_classes: total number of classes (default 24).
        ignore_index: value to ignore (default -1).

    Returns:
        np.ndarray of shape [num_classes], dtype int64. Counts of each class.
    """
    valid = indices[indices != ignore_index]
    valid = valid[(valid >= 0) & (valid < num_classes)]
    counts = np.bincount(valid, minlength=num_classes)
    return counts.astype(np.int64)


def class_weights_from_counts(
    counts: np.ndarray,
    beta: float = 0.999,
    eps: float = 1.0,
) -> np.ndarray:
    """Effective-number class-balanced weights (Cui et al. CVPR 2019).

    w_c = (1 - beta) / (1 - beta^n_c)
    where n_c is the per-class count.

    With beta=0.999 (long-tail), this down-weights very common classes by ~10x
    and up-weights rare classes by ~5x. Normalized to sum=num_classes.

    Args:
        counts: [num_classes] per-class counts.
        beta: effective-number decay (Cui et al.). 0.999 for severe imbalance.
        eps: minimum count to avoid divide-by-zero.

    Returns:
        [num_classes] float32 weights, normalized to sum = num_classes.
    """
    counts = counts.astype(np.float32)
    counts = np.maximum(counts, eps)
    eff_num = (1.0 - np.power(beta, counts)) / (1.0 - beta)
    eff_num = np.maximum(eff_num, eps)
    weights = 1.0 / eff_num
    weights = weights / weights.sum() * len(weights)
    return weights.astype(np.float32)


def get_default_class_weights() -> np.ndarray:
    """Uniform class weights as a safe fallback (use when prevalence unknown).

    Returns:
        [24] float32 array of all 1.0.
    """
    return np.ones(NUM_CATEGORIES, dtype=np.float32)


# =============================================================================
# Convenience: argmax from 24-class softmax -> state string (for inference)
# =============================================================================
def class_idx_to_state_string(class_idx: int) -> str:
    """Convert class index (0..23) back to state string."""
    if not 0 <= class_idx < NUM_CATEGORIES:
        raise ValueError(f"class_idx {class_idx} out of range [0, {NUM_CATEGORIES})")
    return CATEGORIES[class_idx]


def softmax_to_state_strings(probs: np.ndarray) -> List[str]:
    """Convert per-frame softmax [T, 24] -> list of T state strings (argmax).

    Useful for inference pipelines that want a list of state strings to feed
    into NaivePSR / AccumulatedConfidencePSR (psr_utils.py).
    """
    if probs.ndim != 2 or probs.shape[1] != NUM_CATEGORIES:
        raise ValueError(f"Expected [T, {NUM_CATEGORIES}], got {probs.shape}")
    argmax = probs.argmax(axis=1)
    return [CATEGORIES[i] for i in argmax]


# =============================================================================
# Torch helpers (used by losses.py / model.py)
# =============================================================================
def class_idx_to_one_hot(class_idx: torch.Tensor, num_classes: int = NUM_CATEGORIES) -> torch.Tensor:
    """[B] int -> [B, num_classes] float one-hot (0/1)."""
    return torch.nn.functional.one_hot(class_idx, num_classes=num_classes).float()


__all__ = [
    "CATEGORIES",
    "NUM_CATEGORIES",
    "BACKGROUND_IDX",
    "FIRST_VALID_STATE_IDX",
    "LAST_VALID_STATE_IDX",
    "ERROR_STATE_IDX",
    "NUM_VALID_STATES",
    "IGNORE_INDEX",
    "state_string_to_list",
    "list_to_state_string",
    "bits_to_int",
    "int_to_bits",
    "state_vector_to_class_idx",
    "dense_labels_to_class_idx",
    "class_frequencies_from_indices",
    "class_weights_from_counts",
    "get_default_class_weights",
    "class_idx_to_state_string",
    "softmax_to_state_strings",
    "class_idx_to_one_hot",
]