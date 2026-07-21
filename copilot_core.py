"""
copilot_core.py
===============
The one place the copilot's model and Stage-2 corrective-velocity controller are
defined. train_copilot.py (augmentation), evaluate_copilot.py, and
visualize_copilot.py all import `simulate_trajectory` from here so the controller
can never again disagree between training and evaluation.

Stage 1 — LSTM target classifier
    Input : per-tick features (see copilot_dataset.build_features)
    Output: logits over 8 target directions (predicted at every tick)

Stage 2 — Analytic corrective velocity (toward predicted target)
    direction  = normalize(UNIT_DIRS[pred] - cursor)     # from CURRENT cursor
    correction = direction * vel_mag * confidence
Magnitude `vel_mag` is resolved per trial from --copilot_vel_mag:
    'inv_ticks' -> 1 / T   (default; T = number of simulation ticks)
    <float>     -> that fixed value (e.g. 0.02 reproduces the V3 magnitude)
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from copilot_dataset import UNIT_DIRS, NormStats, per_tick_velocity

# UNIT_DIRS is float64 (8,2); cache a float32 copy for the sim loop
_UNIT = UNIT_DIRS.astype(np.float32)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class LSTMCopilot(nn.Module):
    """Two-layer LSTM + linear classifier. Predicts the target at every tick."""

    def __init__(self, input_size: int = 5, hidden_size: int = 64,
                 n_layers: int = 2, n_classes: int = 8, dropout: float = 0.2):
        super().__init__()
        self.input_size = input_size
        self.lstm = nn.LSTM(input_size, hidden_size, n_layers,
                            batch_first=True, dropout=dropout)
        self.classifier = nn.Linear(hidden_size, n_classes)

    def forward(self, x):                      # (B, T, F) -> (B, T, 8)
        out, _ = self.lstm(x)
        return self.classifier(out)


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# --------------------------------------------------------------------------- #
# Corrective-velocity controller (single definition)
# --------------------------------------------------------------------------- #
def resolve_vel_mag(spec, n_ticks: int) -> float:
    """Resolve --copilot_vel_mag into a scalar magnitude for a trial of n_ticks."""
    if isinstance(spec, str):
        if spec == "inv_ticks":
            return 1.0 / max(n_ticks, 1)
        return float(spec)
    return float(spec)


def corrective_velocity(cursor: np.ndarray, pred_label: int,
                        conf: float, vel_mag: float) -> np.ndarray:
    """Correction pointing from the CURRENT cursor toward the predicted target.

    cursor is in normalized (radius) units, so UNIT_DIRS[pred] is the predicted
    target's position on the unit target circle. Falls back to zero when the
    cursor already sits on the target point.
    """
    d = _UNIT[pred_label] - cursor
    dist = float(np.linalg.norm(d))
    if dist > 1e-6:
        d = d / dist
    else:
        d = np.zeros(2, dtype=np.float32)
    return d * vel_mag * conf


def angle_pred(cursor: np.ndarray) -> int:
    """argmax dot-product classification of a (normalized) cursor position."""
    norm = float(np.linalg.norm(cursor))
    if norm < 1e-6:
        return -1
    return int(np.argmax(_UNIT @ (cursor / norm)))


# --------------------------------------------------------------------------- #
# Feature construction for one tick (matches copilot_dataset.build_features order)
# --------------------------------------------------------------------------- #
def _tick_features(cursor_xy: np.ndarray, bci_v: np.ndarray, norm: NormStats,
                   raw_xy: Optional[np.ndarray], feature_set: str) -> np.ndarray:
    """[cursor_x, cursor_y, vx_unit, vy_unit, vel_mag_scaled] (+ raw_x, raw_y)."""
    mag = float(np.linalg.norm(bci_v))
    if mag > 1e-9:
        ux, uy = bci_v[0] / mag, bci_v[1] / mag
    else:
        ux, uy = 0.0, 0.0
    mag_scaled = (mag - norm.vel_mag_mean) / norm.vel_mag_std
    feat = [cursor_xy[0], cursor_xy[1], ux, uy, mag_scaled]
    if feature_set == "extensive":
        rp = cursor_xy if raw_xy is None else raw_xy
        feat += [rp[0], rp[1]]
    return np.asarray(feat, dtype=np.float32)


# --------------------------------------------------------------------------- #
# The single simulation used by augmentation AND evaluation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def simulate_trajectory(
    model: LSTMCopilot,
    bci_vel: np.ndarray,            # (T, 2) per-tick raw BCI velocity (normalized units)
    norm: NormStats,
    vel_mag_spec="inv_ticks",
    feature_set: str = "basic",
    device: str = "cpu",
    velocity_mode: str = "additive",
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Roll a copilot-in-the-loop trajectory from the origin.

    velocity_mode controls the per-tick cursor update (the two readings of the
    supervisor's "1/T"):
      'additive'       : cursor += bci_vel + correction
                         Magnitude-weighted integration; correction magnitude is
                         --copilot_vel_mag (default 1/T). A zero-correction run
                         exactly reproduces the recorded BCI endpoint.
      'combined_fixed' : cursor += unit(bci_vel + correction) * (1/T)
                         Equal-length steps; the cursor travels at a fixed speed
                         1/T and --copilot_vel_mag only bends direction. NOTE: a
                         zero-correction run does NOT reproduce the recorded BCI
                         endpoint (it re-integrates with equal steps), so the BCI
                         baseline is not a strict no-op of this mode.

    Returns (seq, final_cursor, final_pred).
    """
    model.eval()
    T = len(bci_vel)
    vel_mag = resolve_vel_mag(vel_mag_spec, T)
    step_fixed = 1.0 / max(T, 1)
    F = model.input_size
    seq = np.zeros((T, F), dtype=np.float32)

    cursor = np.zeros(2, dtype=np.float32)
    raw = np.zeros(2, dtype=np.float32)          # un-augmented (no-copilot) cursor
    h = c = None

    for t in range(T):
        bv = bci_vel[t].astype(np.float32)
        seq[t] = _tick_features(cursor, bv, norm, raw, feature_set)

        x_t = torch.tensor(seq[t][None, None, :], dtype=torch.float32, device=device)
        if h is None:
            out, (h, c) = model.lstm(x_t)
        else:
            out, (h, c) = model.lstm(x_t, (h, c))
        logits = model.classifier(out[:, -1, :])
        conf = float(torch.softmax(logits, dim=-1).max().item())
        pred = int(logits.argmax(dim=-1).item())

        corr = corrective_velocity(cursor, pred, conf, vel_mag)
        if velocity_mode == "additive":
            step = bv + corr
        elif velocity_mode == "combined_fixed":
            combined = bv + corr
            n = float(np.linalg.norm(combined))
            step = (combined / n) * step_fixed if n > 1e-9 else np.zeros(2, dtype=np.float32)
        else:
            raise ValueError(f"unknown velocity_mode {velocity_mode!r}")
        cursor = np.clip(cursor + step, -1.5, 1.5)
        raw = np.clip(raw + bv, -1.5, 1.5)

    return seq, cursor, angle_pred(cursor)


def build_sequence_raw(bci_vel: np.ndarray, cursor_pos: np.ndarray,
                       norm: NormStats, feature_set: str = "basic") -> np.ndarray:
    """Phase-1 features from the raw (un-augmented) trajectory.

    cursor_pos is the recorded BCI cursor path (T+1 positions); we align to the
    T velocity ticks by using positions[0..T-1] as the 'current cursor' per tick.
    """
    T = len(bci_vel)
    F = 7 if feature_set == "extensive" else 5
    seq = np.zeros((T, F), dtype=np.float32)
    for t in range(T):
        seq[t] = _tick_features(cursor_pos[t], bci_vel[t].astype(np.float32),
                                norm, cursor_pos[t], feature_set)
    return seq


# --------------------------------------------------------------------------- #
# Loss / tick weighting (ported from the greedy trainer)
# --------------------------------------------------------------------------- #
def make_tick_weights(T: int, weighting: str, exponent: float,
                      device: str = "cpu") -> torch.Tensor:
    """(T,) per-tick loss weights, normalized to sum to T (loss scale invariant)."""
    t = torch.arange(1, T + 1, dtype=torch.float32, device=device)
    if weighting == "constant":
        w = torch.ones(T, dtype=torch.float32, device=device)
    elif weighting == "linear":
        w = t / T
    elif weighting == "exponential":
        w = torch.exp(exponent * (t - 1) / (T - 1)) if T > 1 else torch.ones(T, device=device)
    else:
        raise ValueError(f"unknown tick_weighting {weighting!r}")
    return w * T / w.sum()


def masked_weighted_ce(logits: torch.Tensor, labels: torch.Tensor,
                       masks: torch.Tensor, weighting: str, exponent: float,
                       device: str = "cpu") -> torch.Tensor:
    """Cross-entropy over valid ticks, weighted by the tick-weighting scheme."""
    B, T, C = logits.shape
    tick_w = make_tick_weights(T, weighting, exponent, device)          # (T,)
    tick_w = tick_w.unsqueeze(0).expand(B, T).reshape(B * T)
    logits_flat = logits.reshape(B * T, C)
    labels_flat = labels.unsqueeze(1).expand(B, T).reshape(B * T)
    masks_flat = masks.reshape(B * T)
    combined = masks_flat * tick_w
    ce = nn.functional.cross_entropy(logits_flat, labels_flat, reduction="none")
    return (ce * combined).sum() / (combined.sum() + 1e-8)


# --------------------------------------------------------------------------- #
# Batched simulation (vectorized over trials; identical numerics to the loop)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def simulate_batch(
    model: LSTMCopilot,
    vel_list,                       # list of (T_i, 2) per-tick BCI velocities
    norm: NormStats,
    vel_mag_spec="inv_ticks",
    feature_set: str = "basic",
    device: str = "cpu",
    velocity_mode: str = "additive",
):
    """Roll many copilot trajectories at once, stepping the LSTM one tick at a
    time over the whole batch. Returns (seqs, finals, preds):
        seqs   : list of (T_i, F) feature arrays (trimmed to each trial length)
        finals : (B, 2) final cursor per trial
        preds  : (B,)  final-cursor classification per trial
    Padded ticks beyond a trial's length are frozen (cursor stops updating).
    """
    model.eval()
    B = len(vel_list)
    lengths = np.array([len(v) for v in vel_list])
    Tmax = int(lengths.max())
    F = model.input_size
    vel_mag = np.array([resolve_vel_mag(vel_mag_spec, int(L)) for L in lengths], dtype=np.float32)
    step_fixed = (1.0 / np.maximum(lengths, 1)).astype(np.float32)

    # (B, Tmax, 2) padded velocities
    V = np.zeros((B, Tmax, 2), dtype=np.float32)
    for i, v in enumerate(vel_list):
        V[i, :len(v)] = v

    cursor = np.zeros((B, 2), dtype=np.float32)
    raw = np.zeros((B, 2), dtype=np.float32)
    seqs = np.zeros((B, Tmax, F), dtype=np.float32)
    h = c = None
    idx = np.arange(B)

    for t in range(Tmax):
        active = t < lengths                              # (B,) bool
        bv = V[:, t, :]                                   # (B,2)
        mag = np.linalg.norm(bv, axis=1)
        unit = np.zeros_like(bv)
        nz = mag > 1e-9
        unit[nz] = bv[nz] / mag[nz, None]
        mag_scaled = (mag - norm.vel_mag_mean) / norm.vel_mag_std
        feat = np.concatenate([cursor, unit, mag_scaled[:, None]], axis=1)  # (B,5)
        if feature_set == "extensive":
            feat = np.concatenate([feat, raw], axis=1)                      # (B,7)
        seqs[:, t, :] = feat

        x_t = torch.tensor(feat[:, None, :], dtype=torch.float32, device=device)  # (B,1,F)
        if h is None:
            out, (h, c) = model.lstm(x_t)
        else:
            out, (h, c) = model.lstm(x_t, (h, c))
        logits = model.classifier(out[:, -1, :])          # (B,8)
        probs = torch.softmax(logits, dim=-1)
        conf = probs.max(dim=-1).values.cpu().numpy()
        pred = logits.argmax(dim=-1).cpu().numpy()

        # vectorized corrective velocity toward predicted target
        tgt = _UNIT[pred]                                  # (B,2)
        d = tgt - cursor
        dn = np.linalg.norm(d, axis=1)
        dd = np.zeros_like(d)
        ok = dn > 1e-6
        dd[ok] = d[ok] / dn[ok, None]
        corr = dd * (vel_mag * conf)[:, None]

        if velocity_mode == "additive":
            step = bv + corr
        elif velocity_mode == "combined_fixed":
            comb = bv + corr
            cn = np.linalg.norm(comb, axis=1)
            step = np.zeros_like(comb)
            good = cn > 1e-9
            step[good] = comb[good] / cn[good, None] * step_fixed[good, None]
        else:
            raise ValueError(f"unknown velocity_mode {velocity_mode!r}")

        upd = active[:, None]
        cursor = np.clip(cursor + step * upd, -1.5, 1.5)
        raw = np.clip(raw + bv * upd, -1.5, 1.5)

    finals = cursor
    preds = np.array([angle_pred(finals[i]) for i in range(B)])
    seqs_trim = [seqs[i, :lengths[i], :] for i in range(B)]
    return seqs_trim, finals, preds
