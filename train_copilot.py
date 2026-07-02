"""
train_copilot.py
================
Unified, configurable copilot trainer for the Step-1 data-source/model search and
the Step-2 dimension tuning. Replaces train_supervised_copilot_greedy.py.

The SL path is the V3 LSTM + DAgger, ported into copilot_core (so the corrective
velocity is defined once and shared with evaluation). The RL path is a stub
pending the from-scratch velocity-based rewrite.

within_subject trains ONE model per subject (supervisor's call: a pooled model
can't capture subject-specific tendencies). cross_subject / all_subject train a
single pooled model. The authoritative test is always evaluate_copilot.py on the
held-out eval source (default EEGK real); during training we hold out a small
random validation fraction purely to pick the best epoch.

RUN (examples)
--------------
    # Step-1 candidate 1A: EEGK sim + SL, one model per subject
    python train_copilot.py --training_data eegk_sim --model_type sl --train_test within_subject

    # Reproduce a V3-style pooled run
    python train_copilot.py --training_data old_decoder --model_type sl \
        --train_test cross_subject --copilot_vel_mag 0.02 --tick_weighting linear
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import copilot_dataset as cd
import copilot_core as core

DEVICE = "cpu"
PHASE1_EPOCHS = 3
PHASE2_EPOCHS = 25


# --------------------------------------------------------------------------- #
# Trial -> (bci_vel, raw_pos, label) view used by the trainer
# --------------------------------------------------------------------------- #
def trial_view(traj: cd.Trajectory):
    bci_vel = traj.pos[1:] - traj.pos[:-1]          # (P-1, 2) per-tick velocity
    return {"vel": bci_vel.astype(np.float32),
            "pos": traj.pos.astype(np.float32),     # (P, 2) raw positions
            "label": traj.target_label}


class SeqDataset(Dataset):
    """Padded (seq, label, mask) tuples for a set of pre-built sequences."""

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

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i], self.y[i], self.m[i]


def evaluate_group(model, views, norm, cfg) -> dict:
    """Copilot vs BCI accuracy over a set of trials (used for epoch selection)."""
    cop = bci = 0
    for v in views:
        _, final, pred = core.simulate_trajectory(
            model, v["vel"], norm, cfg["copilot_vel_mag"], cfg["input_feature_set"],
            DEVICE, cfg["velocity_mode"])
        cop += int(pred == v["label"])
        bci += int(core.angle_pred(np.clip(np.cumsum(v["vel"], axis=0)[-1], -1.5, 1.5)) == v["label"])
    n = len(views)
    return {"copilot": cop / n, "bci": bci / n, "n": n}


def train_one_model(views: List[dict], norm: cd.NormStats, cfg: dict, tag: str,
                    out_dir: Path) -> dict:
    """Train a single model via DAgger on `views`. Returns a summary dict."""
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(views))
    n_val = max(1, int(round(cfg["val_frac"] * len(views))))
    val_views = [views[i] for i in idx[:n_val]]
    trn_views = [views[i] for i in idx[n_val:]]
    max_ticks = max(len(v["vel"]) for v in views)
    F = 7 if cfg["input_feature_set"] == "extensive" else 5

    model = core.LSTMCopilot(input_size=F, hidden_size=cfg["hidden_size"],
                             n_layers=cfg["num_lstm"]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=6, gamma=0.5)

    model_path = out_dir / "best_model.pt"
    best_val = -1.0
    total_epochs = PHASE1_EPOCHS + PHASE2_EPOCHS

    for epoch in range(1, total_epochs + 1):
        phase = 1 if epoch <= PHASE1_EPOCHS else 2
        t0 = time.time()

        if phase == 1:
            seqs = [core.build_sequence_raw(v["vel"], v["pos"], norm, cfg["input_feature_set"])
                    for v in trn_views]
        else:
            seqs = [core.simulate_trajectory(model, v["vel"], norm,
                    cfg["copilot_vel_mag"], cfg["input_feature_set"], DEVICE,
                    cfg["velocity_mode"])[0]
                    for v in trn_views]
        labels = [v["label"] for v in trn_views]

        ds = SeqDataset(seqs, labels, max_ticks, F)
        loader = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True)

        model.train()
        tot = toks = 0.0
        for x, y, m in loader:
            x, y, m = x.to(DEVICE), y.to(DEVICE), m.to(DEVICE)
            logits = model(x)
            loss = core.masked_weighted_ce(logits, y, m, cfg["tick_weighting"],
                                           cfg["weight_exponent"], DEVICE)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step()
            tot += loss.item() * m.sum().item(); toks += m.sum().item()
        sched.step()

        val = evaluate_group(model, val_views, norm, cfg)
        delta = (val["copilot"] - val["bci"]) * 100
        flag = ""
        if val["copilot"] > best_val:
            best_val = val["copilot"]
            torch.save(model.state_dict(), model_path)
            flag = " <- best"
        print(f"    [{tag}] ep {epoch:2d}/{total_epochs} P{phase} "
              f"loss={tot/(toks+1e-8):.4f} val_bci={val['bci']*100:.1f}% "
              f"val_cop={val['copilot']*100:.1f}% Δ{delta:+.2f}pp "
              f"({time.time()-t0:.0f}s){flag}")

    torch.save(model.state_dict(), out_dir / "model.pt")
    return {"tag": tag, "n_train": len(trn_views), "n_val": len(val_views),
            "best_val_copilot": best_val, "max_ticks": max_ticks,
            "norm": norm.to_dict()}


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def load_training_trajectories(cfg: dict) -> List[cd.Trajectory]:
    ds = cfg["training_data"]
    if ds in ("old_decoder", "eegk_sim", "eegk_real"):
        return cd.load_source(ds, repo_root=cfg["repo_root"])
    if ds in ("eegk_surrogate", "eegk_blend"):
        path = cfg["surrogate_csv"] if ds == "eegk_surrogate" else cfg["blend_csv"]
        if not path or not Path(path).exists():
            raise NotImplementedError(
                f"{ds} requires a generated CSV. Run "
                f"{'surrogate_constructor.py' if ds=='eegk_surrogate' else 'blend_constructor.py'} "
                f"first, then pass --{'surrogate_csv' if ds=='eegk_surrogate' else 'blend_csv'}.")
        return cd.load_csv_file(path)
    raise ValueError(f"unknown training_data {ds!r}")


def group_trajectories(trajs: List[cd.Trajectory], split: str) -> Dict[str, List[cd.Trajectory]]:
    if split == "within_subject":
        groups: Dict[str, List[cd.Trajectory]] = {}
        for t in trajs:
            groups.setdefault(t.subject_id, []).append(t)
        return dict(sorted(groups.items()))
    if split == "cross_subject":
        train_subj = cd.split_subjects("cross_subject")["train"]
        return {"pooled": [t for t in trajs if t.subject_id in train_subj]}
    if split == "all_subject":
        return {"pooled": list(trajs)}
    raise ValueError(f"unknown split {split!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--training_data", default="eegk_sim",
                    choices=["old_decoder", "eegk_sim", "eegk_real", "eegk_surrogate", "eegk_blend"])
    ap.add_argument("--model_type", default="sl", choices=["sl", "rl"])
    ap.add_argument("--train_test", default="within_subject",
                    choices=["within_subject", "cross_subject", "all_subject"])
    ap.add_argument("--copilot_vel_mag", default="inv_ticks",
                    help="'inv_ticks' (=1/T, default) or a float like 0.02")
    ap.add_argument("--velocity_mode", default="additive",
                    choices=["additive", "combined_fixed"],
                    help="additive: cursor+=bci+corr | combined_fixed: fixed 1/T step")
    ap.add_argument("--input_feature_set", default="basic", choices=["basic", "extensive"])
    ap.add_argument("--tick_weighting", default="exponential",
                    choices=["constant", "linear", "exponential"])
    ap.add_argument("--weight_exponent", type=float, default=3.0)
    ap.add_argument("--num_lstm", type=int, default=2)
    ap.add_argument("--hidden_size", type=int, default=64)
    # SL-specific
    ap.add_argument("--performance_metric", default="acc", choices=["acc", "err"])
    ap.add_argument("--learning_rate", type=float, default=1e-3)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--val_frac", type=float, default=0.15)
    # infra
    ap.add_argument("--repo_root", default=".")
    ap.add_argument("--surrogate_csv", default="")
    ap.add_argument("--blend_csv", default="")
    ap.add_argument("--out_root", default="runs")
    args = ap.parse_args()
    cfg = vars(args)

    if args.model_type == "rl":
        raise NotImplementedError(
            "RL path is the next build (from-scratch velocity-based PPO). "
            "SL path is ready; run with --model_type sl.")
    if args.performance_metric == "err":
        raise NotImplementedError(
            "angle-error (regression) objective not wired yet; use --performance_metric acc.")

    torch.manual_seed(42); np.random.seed(42)

    run_name = f"{args.training_data}_{args.model_type}_{args.train_test}"
    run_dir = Path(args.out_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 68)
    print(f"train_copilot | {run_name}")
    print(f"  vel_mag={args.copilot_vel_mag}  features={args.input_feature_set}  "
          f"tick_w={args.tick_weighting}(k={args.weight_exponent})")
    print("=" * 68)

    trajs = load_training_trajectories(cfg)
    groups = group_trajectories(trajs, args.train_test)
    print(f"Loaded {len(trajs)} trials -> {len(groups)} training group(s): "
          f"{list(groups.keys())}\n")

    manifest = {"config": {k: cfg[k] for k in cfg if k not in ('repo_root',)},
                "run_dir": str(run_dir), "models": {}}

    for tag, gtrajs in groups.items():
        gdir = run_dir / tag
        gdir.mkdir(exist_ok=True)
        norm = cd.compute_norm_stats(gtrajs)          # per-group (=per-subject) norm
        views = [trial_view(t) for t in gtrajs]
        print(f"── group {tag}: {len(views)} trials | "
              f"norm(mean={norm.vel_mag_mean:.4f}, std={norm.vel_mag_std:.4f})")
        summary = train_one_model(views, norm, cfg, tag, gdir)
        manifest["models"][tag] = {"path": str(gdir / "best_model.pt"), **summary}
        print()

    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("=" * 68)
    print(f"Done. {len(manifest['models'])} model(s) saved under {run_dir}/")
    print(f"Manifest: {run_dir / 'manifest.json'}")
    print(f"Next: python evaluate_copilot.py --run {run_dir} --eval_data eegk_real")
    print("=" * 68)


if __name__ == "__main__":
    main()
