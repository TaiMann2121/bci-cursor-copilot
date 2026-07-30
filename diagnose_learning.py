"""
diagnose_learning.py
====================
The gating diagnostic the 7/27 supervisor meeting asked for: BEFORE concluding
"there is no recoverable pattern in the trajectory," prove the model is
functionally sound by watching the TRAINING and VALIDATION loss, not just a
downstream accuracy number.

It answers three questions, in order:

  Q1 (architecture sound?)  Does the per-tick training cross-entropy fall well
                            below chance = ln(8) = 2.079?  If not, the LSTM
                            cannot fit even the training set and the problem is
                            the model/features, not the data.
  Q2 (generalizes?)         Does validation CE track training CE, or diverge
                            (overfit)?  A big train/val gap means capacity/
                            regularization, not "no signal".
  Q3 (does cleaning help?)  Same run on RAW vs CLEANED trajectories, so the
                            effect of (a) trimming the ~5.5 leading dead ticks
                            and (b) per-session scale normalization is visible
                            as a shift in the loss/accuracy curves.

This deliberately strips away the copilot control law (additive corrective
velocity / DAgger). It trains the Stage-1 LSTM as a plain 8-way target
classifier on trajectory features. That isolates the only question that matters
here: is there decodable target information in the trajectory the model can
learn? The copilot control law is evaluated elsewhere (evaluate_copilot.py).

RUN
---
    python diagnose_learning.py                 # raw vs clean, all subjects
    python diagnose_learning.py --subjects S01 S04
    python diagnose_learning.py --epochs 40 --feature_set basic
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

# Windows consoles default to cp1252; force UTF-8 so table glyphs don't crash.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import copilot_core as core
import copilot_dataset as cd

DEVICE = "cpu"
CHANCE_CE = float(np.log(8))          # 2.0794 — per-tick CE of a uniform guess


# --------------------------------------------------------------------------- #
# Session-aware loading (each CSV file = one recording session)
# --------------------------------------------------------------------------- #
@dataclass
class Session:
    session_id: str          # folder name, the true session identity
    subject_id: str
    trajs: List[cd.Trajectory]


def load_sessions(root: str) -> List[Session]:
    """Load every online_arm_trajectories.csv as its own Session.

    The production loader (cd._load_eegk_real_frame) concatenates all files and
    drops the folder identity; per-session normalization needs it, so we keep it.
    """
    files = sorted(glob.glob(os.path.join(root, "**", "online_arm_trajectories.csv"),
                             recursive=True))
    if not files:
        raise FileNotFoundError(f"No EEGK CSVs under {root!r}")
    sessions: List[Session] = []
    for f in files:
        sid = os.path.basename(os.path.dirname(f))
        df = pd.read_csv(f)
        trajs = cd._dataframe_to_trajectories(df)
        subj = trajs[0].subject_id if trajs else "?"
        sessions.append(Session(sid, subj, trajs))
    return sessions


# --------------------------------------------------------------------------- #
# Cleaning (item 1 from the meeting) — both steps are metric-safe because the
# endpoint metric is direction-only (argmax dot-product), i.e. scale-invariant.
# --------------------------------------------------------------------------- #
# The cleaning primitives are the production ones in copilot_dataset (single
# source of truth); re-exported here for the diagnostic's Session wrapper.
trim_leading_dwell = cd.trim_leading_dwell


def clean_sessions(sessions: List[Session], do_trim: bool = True,
                   do_scale: bool = True) -> List[Session]:
    """Session-wrapper around cd.clean_trajectory_sessions (keeps session_id)."""
    target_radius = float(np.median([cd._session_median_radius(s.trajs)
                                     for s in sessions]))
    out: List[Session] = []
    for s in sessions:
        med = cd._session_median_radius(s.trajs)
        scale = (target_radius / med) if (do_scale and med > 1e-9) else 1.0
        new_trajs = []
        for t in s.trajs:
            tt = cd.trim_leading_dwell(t) if do_trim else t
            if scale != 1.0:
                tt = cd.Trajectory(tt.subject_id, tt.target_label,
                                   tt.pos * scale, tt.arm_pred, tt.keys, tt.session_id)
            new_trajs.append(tt)
        out.append(Session(s.session_id, s.subject_id, new_trajs))
    return out


# --------------------------------------------------------------------------- #
# Plain 8-way classifier training with train+val LOSS logging
# --------------------------------------------------------------------------- #
class ClsDataset(Dataset):
    def __init__(self, seqs: List[np.ndarray], labels: List[int], max_ticks: int, F: int):
        self.x, self.y, self.m = [], [], []
        for seq, lab in zip(seqs, labels):
            padded = np.zeros((max_ticks, F), dtype=np.float32)
            k = min(len(seq), max_ticks)
            padded[:k] = seq[:k]
            mask = np.zeros(max_ticks, dtype=np.float32)
            mask[:k] = 1.0
            self.x.append(padded); self.y.append(lab); self.m.append(mask)
        self.x = torch.tensor(np.stack(self.x))
        self.y = torch.tensor(self.y, dtype=torch.long)
        self.m = torch.tensor(np.stack(self.m))
        self.len_ = torch.tensor([min(len(s), max_ticks) for s in seqs], dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i], self.y[i], self.m[i], self.len_[i]


def _epoch_loss_acc(model, loader) -> Tuple[float, float]:
    """Mean per-tick CE (constant weighting) and FINAL-tick accuracy over a loader."""
    model.eval()
    tot_ce = tot_tok = 0.0
    correct = n = 0
    with torch.no_grad():
        for x, y, m, lens in loader:
            logits = model(x)                                  # (B,T,8)
            B, T, C = logits.shape
            ce = torch.nn.functional.cross_entropy(
                logits.reshape(B * T, C),
                y.unsqueeze(1).expand(B, T).reshape(B * T),
                reduction="none").reshape(B, T)
            tot_ce += (ce * m).sum().item(); tot_tok += m.sum().item()
            last = (lens - 1).clamp(min=0)
            final_logits = logits[torch.arange(B), last]       # (B,8)
            correct += int((final_logits.argmax(-1) == y).sum().item()); n += B
    return tot_ce / max(tot_tok, 1e-8), correct / max(n, 1)


def train_classifier(seqs, labels, F, epochs, seed, hidden=64, layers=2,
                     lr=1e-3, val_frac=0.2) -> dict:
    """Train the LSTM as a target classifier; log train/val CE + acc per epoch.

    Stratified-by-target random split within the given (subject) pool. Returns a
    history dict with per-epoch train_ce/val_ce/train_acc/val_acc."""
    rng = np.random.default_rng(seed)
    by_t = defaultdict(list)
    for i, lab in enumerate(labels):
        by_t[lab].append(i)
    val_idx, trn_idx = [], []
    for lab, idxs in by_t.items():
        idxs = list(idxs); rng.shuffle(idxs)
        nv = max(1, int(round(val_frac * len(idxs))))
        val_idx += idxs[:nv]; trn_idx += idxs[nv:]

    max_ticks = max(len(s) for s in seqs)
    trn = ClsDataset([seqs[i] for i in trn_idx], [labels[i] for i in trn_idx], max_ticks, F)
    val = ClsDataset([seqs[i] for i in val_idx], [labels[i] for i in val_idx], max_ticks, F)
    trn_loader = DataLoader(trn, batch_size=128, shuffle=True)
    val_loader = DataLoader(val, batch_size=256, shuffle=False)
    trn_eval = DataLoader(trn, batch_size=256, shuffle=False)

    torch.manual_seed(seed)
    model = core.LSTMCopilot(input_size=F, hidden_size=hidden, n_layers=layers).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5)

    hist = {"train_ce": [], "val_ce": [], "train_acc": [], "val_acc": []}
    for ep in range(1, epochs + 1):
        model.train()
        for x, y, m, lens in trn_loader:
            logits = model(x)
            B, T, C = logits.shape
            ce = torch.nn.functional.cross_entropy(
                logits.reshape(B * T, C),
                y.unsqueeze(1).expand(B, T).reshape(B * T),
                reduction="none").reshape(B, T)
            loss = (ce * m).sum() / (m.sum() + 1e-8)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        tr_ce, tr_acc = _epoch_loss_acc(model, trn_eval)
        va_ce, va_acc = _epoch_loss_acc(model, val_loader)
        hist["train_ce"].append(tr_ce); hist["val_ce"].append(va_ce)
        hist["train_acc"].append(tr_acc); hist["val_acc"].append(va_acc)
    hist["n_train"] = len(trn_idx); hist["n_val"] = len(val_idx)
    return hist


# --------------------------------------------------------------------------- #
# Feature building over a subject's trajectories
# --------------------------------------------------------------------------- #
def build_subject_arrays(trajs: List[cd.Trajectory], feature_set: str
                         ) -> Tuple[List[np.ndarray], List[int]]:
    norm = cd.compute_norm_stats(trajs)
    seqs = [cd.build_features(t.pos, norm, feature_set) for t in trajs]
    labels = [t.target_label for t in trajs]
    return seqs, labels


# --------------------------------------------------------------------------- #
# Orchestration: raw vs clean A/B
# --------------------------------------------------------------------------- #
def run(subjects, epochs, feature_set, seed, out_dir: Path):
    root = os.path.join(".", cd.DATA_PATHS["eegk_real"])
    raw_sessions = load_sessions(root)
    clean = clean_sessions(raw_sessions, do_trim=True, do_scale=True)

    def by_subject(sessions):
        d = defaultdict(list)
        for s in sessions:
            if subjects and s.subject_id not in subjects:
                continue
            d[s.subject_id].extend(s.trajs)
        return dict(sorted(d.items()))

    variants = {"raw": by_subject(raw_sessions), "clean": by_subject(clean)}
    F = 7 if feature_set == "extensive" else 5

    results = {}
    for vname, subj_map in variants.items():
        for subj, trajs in subj_map.items():
            seqs, labels = build_subject_arrays(trajs, feature_set)
            hist = train_classifier(seqs, labels, F, epochs, seed)
            results[(vname, subj)] = hist

    # ---- report ----
    print("\n" + "=" * 92)
    print(f"LEARNING DIAGNOSTIC  (feature_set={feature_set}, epochs={epochs}, seed={seed})")
    print(f"chance per-tick CE = ln(8) = {CHANCE_CE:.3f};  train CE must fall well below this")
    print("=" * 92)
    header = (f"{'subj':>5} {'variant':>7} {'ticks':>6} | {'train_CE':>9} {'val_CE':>8} "
              f"| {'train_acc':>9} {'val_acc':>8} | {'CE↓ vs chance':>13}")
    print(header); print("-" * 92)
    for subj in sorted({s for _, s in results}):
        for vname in ("raw", "clean"):
            h = results.get((vname, subj))
            if h is None:
                continue
            tr_ce, va_ce = h["train_ce"][-1], h["val_ce"][-1]
            tr_ac, va_ac = h["train_acc"][-1], h["val_acc"][-1]
            drop = CHANCE_CE - tr_ce
            print(f"{subj:>5} {vname:>7} {'':>6} | {tr_ce:>9.3f} {va_ce:>8.3f} "
                  f"| {tr_ac*100:>8.1f}% {va_ac*100:>7.1f}% | {drop:>+13.3f}")
        print("-" * 92)

    # verdict
    learned = [subj for subj in sorted({s for _, s in results})
               if results[("clean", subj)]["train_ce"][-1] < CHANCE_CE - 0.3]
    print(f"\nVERDICT: training CE fell >=0.3 below chance for {len(learned)}/"
          f"{len({s for _, s in results})} subjects (clean): {learned}")
    print("  -> if this list is full, the LSTM IS functionally sound and there IS")
    print("     learnable target signal in the trajectory. If train CE ~ 2.079,")
    print("     the architecture/features are the problem, not 'no recoverable pattern'.")

    # ---- save history + plot ----
    out_dir.mkdir(parents=True, exist_ok=True)
    ser = {f"{v}|{s}": h for (v, s), h in results.items()}
    (out_dir / "learning_history.json").write_text(json.dumps(ser, indent=2))
    try:
        _plot(results, feature_set, out_dir)
        print(f"\nsaved: {out_dir/'learning_curves.png'} and learning_history.json")
    except Exception as e:  # matplotlib optional
        print(f"\n(plot skipped: {e})")
        print(f"saved: {out_dir/'learning_history.json'}")


def _plot(results, feature_set, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    subjects = sorted({s for _, s in results})
    n = len(subjects)
    fig, axes = plt.subplots(2, n, figsize=(3.2 * n, 6), squeeze=False)
    for j, subj in enumerate(subjects):
        ax_ce, ax_ac = axes[0][j], axes[1][j]
        for vname, col in (("raw", "#c0392b"), ("clean", "#2471a3")):
            h = results.get((vname, subj))
            if h is None:
                continue
            ep = range(1, len(h["train_ce"]) + 1)
            ax_ce.plot(ep, h["train_ce"], color=col, ls="-", label=f"{vname} train")
            ax_ce.plot(ep, h["val_ce"], color=col, ls="--", label=f"{vname} val")
            ax_ac.plot(ep, h["train_acc"], color=col, ls="-")
            ax_ac.plot(ep, h["val_acc"], color=col, ls="--")
        ax_ce.axhline(CHANCE_CE, color="gray", ls=":", lw=1)
        ax_ce.set_title(f"{subj}  CE"); ax_ce.set_xlabel("epoch")
        ax_ac.axhline(1 / 8, color="gray", ls=":", lw=1)
        ax_ac.set_title(f"{subj}  final-tick acc"); ax_ac.set_xlabel("epoch")
        if j == 0:
            ax_ce.set_ylabel("cross-entropy"); ax_ac.set_ylabel("accuracy")
            ax_ce.legend(fontsize=7)
    fig.suptitle(f"Learning diagnostic (features={feature_set}) — solid=train, dashed=val, "
                 f"dotted=chance", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "learning_curves.png", dpi=130)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--feature_set", default="basic", choices=["basic", "extensive"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default="results/learning_diagnostic")
    args = ap.parse_args()
    run(args.subjects, args.epochs, args.feature_set, args.seed, Path(args.out_dir))


if __name__ == "__main__":
    main()
