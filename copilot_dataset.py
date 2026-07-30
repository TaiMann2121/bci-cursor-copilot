"""
copilot_dataset.py
==================
Shared data + metric foundation for the arm-BCI copilot pipeline.

Everything downstream (train_copilot.py, evaluate_copilot.py, visualize_copilot.py,
surrogate_constructor.py, blend_constructor.py) is built on top of this module so
that the canonical constants, trajectory parsing, feature construction, metric, and
normalization are defined exactly once.

Data sources
------------
    old_decoder    data/online_arm_trajectories.csv                (int cursor, ~17 ticks/trial)
    eegk_sim       data/online_arm_trajectories_EEGK_simulation.csv (float cursor, 13 ticks/trial)
    eegk_real      data/OnlineArmTrajectoryEEGK/<Sxx>/**/online_arm_trajectories.csv  (+ typing_stats.npz)
    eegk_surrogate data/surrogate/  (produced by surrogate_constructor.py)   [not handled here]
    eegk_blend     data/blend/      (produced by blend_constructor.py)        [not handled here]

Canonical metric
----------------
For each trial, take the FINAL cursor position, take the dot product with each of the
8 target unit directions, argmax -> predicted label; compare to target_label.
(Verified against reference baselines: old_decoder 46.69%, eegk_sim 59.67%.)
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Geometry / constants (from README_online_arm_trajectories.md)
# --------------------------------------------------------------------------- #
RADIUS_PX: float = 432.0

# label -> target position on the 8-direction circle
TARGET_POS: Dict[int, Tuple[float, float]] = {
    0: (-305.47, 305.47),   # NW
    1: (0.0, 432.0),        # N
    2: (305.47, 305.47),    # NE
    3: (432.0, 0.0),        # E
    4: (305.47, -305.47),   # SE
    5: (0.0, -432.0),       # S
    6: (-305.47, -305.47),  # SW
    7: (-432.0, 0.0),       # W
}
DIR_NAMES = {0: "NW", 1: "N", 2: "NE", 3: "E", 4: "SE", 5: "S", 6: "SW", 7: "W"}

# label -> unit direction vector (shape (8, 2))
UNIT_DIRS: np.ndarray = np.array([TARGET_POS[i] for i in range(8)], dtype=np.float64)
UNIT_DIRS /= np.linalg.norm(UNIT_DIRS, axis=1, keepdims=True)

# per-trajectory identity columns
TRIAL_KEYS = [
    "subject_id",
    "session_number",
    "run_number",
    "trial_number",
    "inner_trial_number",
]

ALL_SUBJECTS = ["S01", "S02", "S03", "S04", "S05", "S06", "S07"]

# Default data file locations (relative to repo root)
DATA_PATHS = {
    "old_decoder": "data/online_arm_trajectories.csv",
    "eegk_sim": "data/online_arm_trajectories_EEGK_simulation.csv",
    "eegk_real": "data/OnlineArmTrajectoryEEGK",
    "eegk_surrogate": "data/surrogate",
    "eegk_blend": "data/blend",
}


# --------------------------------------------------------------------------- #
# Trajectory container
# --------------------------------------------------------------------------- #
@dataclass
class Trajectory:
    """One arm trial as ordered per-tick arrays (already normalized by RADIUS_PX)."""

    subject_id: str
    target_label: int
    pos: np.ndarray          # (T, 2) cursor position / RADIUS_PX
    arm_pred: np.ndarray     # (T,)  stored per-tick arm_prediction_label (raw, 0-7)
    keys: tuple              # full TRIAL_KEYS tuple (provenance)
    session_id: Optional[str] = None  # source session folder; disambiguates trials
                                      # whose TRIAL_KEYS collide across folders

    @property
    def n_ticks(self) -> int:
        return self.pos.shape[0]

    @property
    def final_pos(self) -> np.ndarray:
        return self.pos[-1]


# --------------------------------------------------------------------------- #
# Metric
# --------------------------------------------------------------------------- #
def label_from_position(xy: np.ndarray) -> int:
    """argmax dot-product of a position against the 8 target unit directions."""
    return int(np.argmax(UNIT_DIRS @ np.asarray(xy, dtype=np.float64)))


def angle_error_deg(xy: np.ndarray, target_label: int) -> float:
    """Absolute wrapped angular error (deg) between a position and its target direction."""
    v = np.asarray(xy, dtype=np.float64)
    if np.allclose(v, 0):
        return 180.0
    ang = np.arctan2(v[1], v[0])
    t = UNIT_DIRS[target_label]
    tang = np.arctan2(t[1], t[0])
    d = np.degrees(np.arctan2(np.sin(ang - tang), np.cos(ang - tang)))
    return abs(d)


def evaluate_final_positions(
    final_positions: np.ndarray, target_labels: np.ndarray
) -> Dict[str, float]:
    """Accuracy + mean angle error over a set of final positions.

    final_positions : (N, 2)   target_labels : (N,)
    """
    preds = np.array([label_from_position(p) for p in final_positions])
    acc = float((preds == target_labels).mean())
    err = float(
        np.mean([angle_error_deg(p, t) for p, t in zip(final_positions, target_labels)])
    )
    return {"accuracy": acc, "angle_error_deg": err, "n": int(len(target_labels))}


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _dataframe_to_trajectories(df: pd.DataFrame,
                               session_id: Optional[str] = None) -> List[Trajectory]:
    """Group a frame into trajectories. `session_id` (the source folder) is
    stamped on each trajectory so trials whose TRIAL_KEYS collide across session
    folders stay distinct. Pass one frame PER session file; concatenating files
    first and grouping by TRIAL_KEYS alone silently merges colliding trials."""
    trajs: List[Trajectory] = []
    for keys, g in df.groupby(TRIAL_KEYS, sort=False):
        g = g.sort_values("timestamp_seconds")
        pos = g[["cursor_pos_x", "cursor_pos_y"]].to_numpy(dtype=np.float64) / RADIUS_PX
        trajs.append(
            Trajectory(
                subject_id=str(keys[0]),
                target_label=int(g["target_label"].iloc[0]),
                pos=pos,
                arm_pred=g["arm_prediction_label"].to_numpy(dtype=np.int64),
                keys=tuple(keys),
                session_id=session_id,
            )
        )
    return trajs


def load_source(
    source: str,
    repo_root: str = ".",
    subjects: Optional[Sequence[str]] = None,
    clean: bool = False,
) -> List[Trajectory]:
    """Load a data source into a list of Trajectory objects.

    Handles CSV sources (old_decoder, eegk_sim) and the nested eegk_real folder.
    surrogate/blend are produced by their constructors and loaded via load_csv_file.

    clean : apply the 7/27 metric-safe cleaning (trim leading pre-onset dwell +
        per-session scale normalization). Only implemented for eegk_real, where
        per-session normalization is well defined; ignored for other sources.
        Default False, so clean-vs-raw stays an A/B on identical trials.

    NOTE: clean=False is NOT the pre-7/27 behavior. eegk_real is now always
    loaded per session folder to fix the TRIAL_KEYS collision described below,
    so eegk_real results from before that fix will not reproduce (they were
    computed on ~3k merged mixed-target trials). This is intended.
    """
    if source in ("old_decoder", "eegk_sim"):
        path = os.path.join(repo_root, DATA_PATHS[source])
        df = pd.read_csv(path)
        trajs = _dataframe_to_trajectories(df)
    elif source == "eegk_real":
        root = os.path.join(repo_root, DATA_PATHS[source])
        # ALWAYS load per session folder: TRIAL_KEYS collide across folders
        # (e.g. WordTyping vs SentenceTyping both have session0/run1/trial1), so
        # the old concat-then-group path silently merged ~3k distinct trials into
        # corrupted mixed-target sequences. Per-folder loading keeps them distinct.
        sessions = _load_eegk_real_sessions(root)
        trajs = clean_trajectory_sessions(sessions) if clean else \
            [t for s in sessions for t in s]
    else:
        raise ValueError(
            f"load_source does not handle '{source}'. "
            "Use surrogate_constructor / blend_constructor outputs via load_csv_file()."
        )

    if subjects is not None:
        keep = set(subjects)
        trajs = [t for t in trajs if t.subject_id in keep]
    return trajs


def load_csv_file(path: str, subjects: Optional[Sequence[str]] = None) -> List[Trajectory]:
    """Load an arbitrary trajectory CSV (e.g. a generated surrogate/blend file)."""
    df = pd.read_csv(path)
    if subjects is not None:
        df = df[df["subject_id"].isin(list(subjects))]
    return _dataframe_to_trajectories(df)


def _load_eegk_real_frame(root: str) -> pd.DataFrame:
    """Concatenate every online_arm_trajectories.csv under the nested EEGK folder tree.

    Expected layout (from README + repo screenshots):
        <root>/S01/WordTyping/S01_Typing_S2001/online_arm_trajectories.csv
        <root>/S01/WordTyping/S01_Typing_S2001/typing_stats.npz
        <root>/S02/.../online_arm_trajectories.csv
        ...
    NOTE: untested until the real EEGK zip is available on disk; coded to the README spec.
    """
    pattern = os.path.join(root, "**", "online_arm_trajectories.csv")
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        raise FileNotFoundError(
            f"No EEGK real trajectory CSVs found under {root!r}. "
            "Expected <root>/<Sxx>/.../online_arm_trajectories.csv"
        )
    return pd.concat((pd.read_csv(f) for f in files), ignore_index=True)


def _load_eegk_real_sessions(root: str) -> List[List["Trajectory"]]:
    """Load each online_arm_trajectories.csv as its own session (list of trials).

    The concatenating loader (_load_eegk_real_frame) drops the folder identity;
    per-session normalization needs it, so this keeps one list per session file.
    """
    pattern = os.path.join(root, "**", "online_arm_trajectories.csv")
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        raise FileNotFoundError(
            f"No EEGK real trajectory CSVs found under {root!r}. "
            "Expected <root>/<Sxx>/.../online_arm_trajectories.csv"
        )
    return [_dataframe_to_trajectories(pd.read_csv(f),
                                       session_id=os.path.basename(os.path.dirname(f)))
            for f in files]


# --------------------------------------------------------------------------- #
# Cleaning (7/27 supervisor items). Both steps are METRIC-SAFE: the endpoint
# metric is direction-only (argmax dot-product), so trimming leading dead ticks
# and per-session scalar rescaling of positions cannot change any label. Only
# the copilot's in-the-loop behavior (which reads the cleaned features and adds
# corrective velocity) can change. Kept opt-in (load_source(clean=True)).
# --------------------------------------------------------------------------- #
def trim_leading_dwell(traj: "Trajectory", eps: float = 1e-6) -> "Trajectory":
    """Drop leading ticks where the cursor has not yet left its start point.

    Keeps from the first tick whose displacement from pos[0] exceeds eps
    (movement onset). Positions stay absolute (position is a feature); only the
    dead pre-onset ticks that dilute the sequence are removed. Always keeps >=2
    ticks as a guard.
    """
    pos = traj.pos
    if len(pos) < 3:
        return traj
    d0 = np.linalg.norm(pos - pos[0], axis=1)
    moved = int(np.argmax(d0 > eps)) if np.any(d0 > eps) else 0
    onset = max(0, min(moved, len(pos) - 2))
    if onset == 0:
        return traj
    return Trajectory(
        subject_id=traj.subject_id,
        target_label=traj.target_label,
        pos=pos[onset:].copy(),
        arm_pred=traj.arm_pred[onset:].copy(),
        keys=traj.keys,
        session_id=traj.session_id,
    )


def _session_median_radius(trajs: Sequence["Trajectory"]) -> float:
    return float(np.median([np.linalg.norm(t.final_pos) for t in trajs]))


def clean_trajectory_sessions(
    sessions: List[List["Trajectory"]], do_trim: bool = True, do_scale: bool = True
) -> List["Trajectory"]:
    """Apply trim + per-session scale normalization; return flattened trials.

    Per-session scale = (global reference radius) / (this session's median final
    radius), where the reference is the median over all sessions' median final
    radius. Rescaling positions rescales velocity (their diff) by the same
    factor, fixing BOTH the velocity- and trajectory-scale drift in one scalar.
    Original trial keys are preserved, so split_real still partitions correctly.
    """
    target_radius = float(np.median([_session_median_radius(s) for s in sessions]))
    out: List["Trajectory"] = []
    for s in sessions:
        med = _session_median_radius(s)
        scale = (target_radius / med) if (do_scale and med > 1e-9) else 1.0
        for t in s:
            tt = trim_leading_dwell(t) if do_trim else t
            if scale != 1.0:
                tt = Trajectory(tt.subject_id, tt.target_label,
                                tt.pos * scale, tt.arm_pred, tt.keys, tt.session_id)
            out.append(tt)
    return out


def load_typing_stats(root: str) -> Dict[str, Dict[str, np.ndarray]]:
    """Load per-session typing_stats.npz files, keyed by subject_id.

    Returns {subject_id: {array_name: ndarray, ...}}. When a subject has multiple
    sessions, the first found is returned per subject (extend as needed).
    See README for array definitions (arm_confusion, angle_sigma_rad, ...).
    """
    stats: Dict[str, Dict[str, np.ndarray]] = {}
    for f in sorted(glob.glob(os.path.join(root, "**", "typing_stats.npz"), recursive=True)):
        # infer subject from path segment matching S\d\d
        subj = next((p for p in f.split(os.sep) if len(p) == 3 and p[0] == "S" and p[1:].isdigit()), None)
        if subj is None:
            continue
        if subj in stats:
            continue
        npz = np.load(f)
        stats[subj] = {k: npz[k] for k in npz.files}
    return stats


# --------------------------------------------------------------------------- #
# Baseline (no copilot)
# --------------------------------------------------------------------------- #
def baseline_metrics(trajs: Sequence[Trajectory]) -> Dict[str, float]:
    finals = np.array([t.final_pos for t in trajs])
    targets = np.array([t.target_label for t in trajs])
    return evaluate_final_positions(finals, targets)


# --------------------------------------------------------------------------- #
# Feature construction
# --------------------------------------------------------------------------- #
@dataclass
class NormStats:
    """Per-source normalization constants for the velocity-magnitude feature."""

    vel_mag_mean: float
    vel_mag_std: float

    def to_dict(self) -> dict:
        return {"vel_mag_mean": self.vel_mag_mean, "vel_mag_std": self.vel_mag_std}


def per_tick_velocity(pos: np.ndarray) -> np.ndarray:
    """Per-tick velocity in normalized (radius) units. v[0] = 0; v[t] = pos[t]-pos[t-1]."""
    v = np.zeros_like(pos)
    v[1:] = pos[1:] - pos[:-1]
    return v


def compute_norm_stats(trajs: Sequence[Trajectory]) -> NormStats:
    """Estimate velocity-magnitude mean/std over all non-initial ticks of a trajectory set."""
    mags: List[float] = []
    for t in trajs:
        v = per_tick_velocity(t.pos)[1:]
        mags.extend(np.linalg.norm(v, axis=1).tolist())
    mags_arr = np.asarray(mags, dtype=np.float64)
    return NormStats(float(mags_arr.mean()), float(mags_arr.std() + 1e-8))


def build_features(
    pos: np.ndarray,
    norm: NormStats,
    feature_set: str = "basic",
    raw_pos: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Per-tick feature matrix.

    basic (5): [cursor_x, cursor_y, vx_unit, vy_unit, vel_mag_scaled]
        - cursor_x/y are already normalized by RADIUS_PX (Trajectory.pos)
        - unit velocity has magnitude 1 (or 0 on the first tick / zero-velocity ticks)
        - vel_mag_scaled = (|v| - mean) / std

    extensive (7): basic + [raw_bci_x, raw_bci_y]
        - raw_pos is the un-augmented (no-copilot) cursor position / RADIUS_PX.
          When training on offline trajectories with no copilot yet applied,
          raw_pos == pos; the distinction matters once the copilot is in the loop.
    """
    v = per_tick_velocity(pos)
    mag = np.linalg.norm(v, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        unit = np.where(mag[:, None] > 1e-9, v / np.maximum(mag[:, None], 1e-9), 0.0)
    mag_scaled = (mag - norm.vel_mag_mean) / norm.vel_mag_std

    feats = np.concatenate(
        [pos, unit, mag_scaled[:, None]], axis=1
    )  # (T, 5)

    if feature_set == "extensive":
        rp = pos if raw_pos is None else raw_pos
        feats = np.concatenate([feats, rp], axis=1)  # (T, 7)
    elif feature_set != "basic":
        raise ValueError(f"unknown feature_set {feature_set!r} (expected basic|extensive)")
    return feats


# --------------------------------------------------------------------------- #
# Splits
# --------------------------------------------------------------------------- #
def split_subjects(mode: str) -> Dict[str, List[str]]:
    """Return {'train': [...], 'test': [...]} subject lists for a split mode.

    within_subject : per-subject train == test subject (handled by the trainer,
                     which builds one model per subject). Here we just return all.
    cross_subject  : S01-S05 train, S06-S07 test (legacy V3 convention).
    all_subject    : everyone in both (diagnostic only).
    """
    if mode == "within_subject":
        return {"train": ALL_SUBJECTS[:], "test": ALL_SUBJECTS[:]}
    if mode == "cross_subject":
        return {"train": ["S01", "S02", "S03", "S04", "S05"], "test": ["S06", "S07"]}
    if mode == "all_subject":
        return {"train": ALL_SUBJECTS[:], "test": ALL_SUBJECTS[:]}
    raise ValueError(f"unknown split mode {mode!r}")


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Self-test the data foundation.")
    ap.add_argument("--repo_root", default=".")
    args = ap.parse_args()

    for src in ("old_decoder", "eegk_sim"):
        trajs = load_source(src, repo_root=args.repo_root)
        base = baseline_metrics(trajs)
        norm = compute_norm_stats(trajs)
        ticks = np.array([t.n_ticks for t in trajs])
        feat = build_features(trajs[0].pos, norm, "basic")
        print(f"\n[{src}]")
        print(f"  trials={len(trajs)}  ticks/trial: {ticks.min()}-{ticks.max()} (mean {ticks.mean():.2f})")
        print(f"  baseline accuracy = {base['accuracy']*100:.2f}%  mean angle err = {base['angle_error_deg']:.2f} deg")
        print(f"  norm: vel_mag_mean={norm.vel_mag_mean:.4f} vel_mag_std={norm.vel_mag_std:.4f}")
        print(f"  default copilot_vel_mag (1/mean_ticks) = {1.0/ticks.mean():.4f}")
        print(f"  basic feature shape (trial 0) = {feat.shape}")


# --------------------------------------------------------------------------- #
# Real-data train/val/test split (for the properly-anchored source comparison)
# --------------------------------------------------------------------------- #
def split_real(trajs, val_frac: float = 0.15, test_frac: float = 0.2, seed: int = 0):
    """Per-subject split of real trajectories.

    TEST  : whole held-out (session,run) block(s), ~test_frac of trials, >=1 block.
            Leakage-free and deployment-realistic; the common eval set for all sources.
    VAL   : random stratified (by target) ~val_frac of the remaining trials; used only
            for per-source hyperparameter (magnitude) selection.
    TRAIN : the rest. Surrogates must be calibrated from TRAIN only.

    Returns {subject_id: {'train': [...], 'val': [...], 'test': [...]}}.
    """
    import numpy as _np
    from collections import defaultdict as _dd
    rng = _np.random.default_rng(seed)
    by_subj = _dd(list)
    for t in trajs:
        by_subj[t.subject_id].append(t)

    out = {}
    for s, ts in by_subj.items():
        blocks = _dd(list)
        for t in ts:
            # block = (session folder, session_number, run_number); the folder is
            # required because (session_number, run_number) collide across folders.
            blocks[(t.session_id, int(t.keys[1]), int(t.keys[2]))].append(t)
        bkeys = list(blocks.keys())
        bkeys.sort(key=lambda k: len(blocks[k]))   # smallest-first: minimize test size / train starvation
        n_total = len(ts)

        # greedily take whole blocks for TEST, ~test_frac of trials (>=1 block,
        # leave >=1 block for train), stopping before overshooting the target
        test, test_n, i = [], 0, 0
        target = test_frac * n_total
        while i < len(bkeys) - 1:
            bsize = len(blocks[bkeys[i]])
            if not test or test_n + bsize <= target:
                test += blocks[bkeys[i]]; test_n += bsize; i += 1
            else:
                break
        remaining = [t for k in bkeys[i:] for t in blocks[k]]

        # stratified VAL from the remaining trials
        by_tgt = _dd(list)
        for t in remaining:
            by_tgt[t.target_label].append(t)
        val, train = [], []
        for tgt, lst in by_tgt.items():
            lst = list(lst); rng.shuffle(lst)
            nv = int(round(val_frac * n_total * len(lst) / max(len(remaining), 1)))
            val += lst[:nv]; train += lst[nv:]
        out[s] = {"train": train, "val": val, "test": test}
    return out
