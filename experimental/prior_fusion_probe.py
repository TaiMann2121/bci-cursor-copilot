"""
prior_fusion_probe.py  (experimental / decisive priors measurement)
==================================================================
Does fusing a LANGUAGE PRIOR over directions with the MOTOR evidence beat
motor-alone on the arm phase? Measured on the SentenceTyping EEGK trials, in
temporal (text) order.

Motor evidence  : von-Mises posterior over the 8 directions around the RAW BCI
                  endpoint angle (argmax == raw BCI, the strongest motor estimate
                  we established). kappa fit from the real endpoint angular error.
Language prior  : an English GROUP n-gram (char n-gram from wordfreq, mapped to
                  the 8 keyboard groups) giving P(dir_t | previous dirs).
                  - ORACLE context  : condition on the TRUE previous directions
                                      (upper bound on how much a prior can help).
                  - DECODED context : condition on the previously-DECODED directions
                                      (realistic; errors propagate).
Fusion          : P(dir) proportional to P_motor(dir) * P_prior(dir), then argmax.

Reports next-direction top-1 accuracy: motor-only vs fused (oracle & decoded).
Run:  python experimental/prior_fusion_probe.py
"""
from __future__ import annotations
import glob
from collections import defaultdict

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from wordfreq import top_n_list, word_frequency

import copilot_dataset as cd

# arm direction (group 0-7) -> letters
GROUPS = {0:['w','q','e','r'],1:['t','y','u'],2:['i','o','p'],3:['f','g','h','j'],
          4:['m','k','l'],5:['b','n',' '],6:['z','x','c','v'],7:['a','s','d']}
char2grp = {c:g for g,cs in GROUPS.items() for c in cs}
PHI = np.array([np.arctan2(cd.UNIT_DIRS[d,1], cd.UNIT_DIRS[d,0]) for d in range(8)])  # dir angles


def load_sentence_trials():
    """Ordered (endpoint_angle, true_dir) per SentenceTyping session.

    NOTE: timestamp_seconds is relative to each trial's onset (~0 for every trial),
    so trials MUST be ordered by (run, trial, inner_trial), not by timestamp.
    """
    files = sorted(glob.glob('data/OnlineArmTrajectoryEEGK/*/**/online_arm_trajectories.csv',
                             recursive=True))
    sessions = []
    for f in [x for x in files if 'Sentence' in x]:
        df = pd.read_csv(f)
        trials = []
        for keys, g in df.groupby(['run_number','trial_number','inner_trial_number']):
            g = g.sort_values('timestamp_seconds')
            end = g[['cursor_pos_x','cursor_pos_y']].to_numpy()[-1]
            trials.append((keys, np.arctan2(end[1], end[0]), int(g['target_label'].iloc[0])))
        trials.sort(key=lambda x: x[0])          # temporal order = (run, trial, inner_trial)
        sessions.append([(a, d) for _, a, d in trials])
    return sessions


def build_group_ngram(N=3, topk=40000):
    """English GROUP n-gram P(next group | up to N-1 previous groups), stupid-backoff."""
    ctx = [defaultdict(lambda: np.zeros(8)) for _ in range(N)]
    for w in top_n_list('en', topk):
        s = ' ' + w.lower() + ' '
        f = word_frequency(w, 'en')
        grps = [char2grp[c] for c in s if c in char2grp]
        for i in range(1, len(grps)):
            for o in range(N):
                if i - o < 0: continue
                key = tuple(grps[i-o:i])
                ctx[o][key][grps[i]] += f
    return ctx, N


def prior_dist(ngram, hist):
    """P(dir | history) with stupid backoff over the group n-gram."""
    ctx, N = ngram
    for o in range(N-1, -1, -1):
        key = tuple(hist[-o:]) if o > 0 else ()
        if key in ctx[o] and ctx[o][key].sum() > 0:
            v = ctx[o][key]; return v / v.sum()
    return np.ones(8) / 8


def motor_post(angle, kappa):
    p = np.exp(kappa * np.cos(angle - PHI)); return p / p.sum()


def fit_kappa(sessions):
    """MLE-ish kappa from circular concentration of endpoint errors vs true dir."""
    errs = [a - PHI[d] for s in sessions for a, d in s]
    R = np.abs(np.mean(np.exp(1j*np.array(errs))))       # mean resultant length
    return R*(2 - R**2)/(1 - R**2) if R < 0.99 else 50.0  # Fisher approx


def main():
    sessions = load_sentence_trials()
    n = sum(len(s) for s in sessions)
    kappa = fit_kappa(sessions)
    ngram = build_group_ngram()
    print(f"{len(sessions)} sessions, {n} trials | motor kappa={kappa:.2f}\n")

    m_ok = o_ok = d_ok = prior_only = 0
    for s in sessions:
        hist_true, hist_dec = [], []
        for angle, tgt in s:
            pm = motor_post(angle, kappa)
            m = int(pm.argmax())                                  # motor only (== raw BCI)
            # oracle context: true history
            po = prior_dist(ngram, hist_true)
            fo = int((pm * po).argmax())
            # decoded context: previously-decoded (fused) history
            pd_ = prior_dist(ngram, hist_dec)
            fd = int((pm * pd_).argmax())
            prior_only += int(po.argmax() == tgt)
            m_ok += int(m == tgt); o_ok += int(fo == tgt); d_ok += int(fd == tgt)
            hist_true.append(tgt); hist_dec.append(fd)

    print(f"{'method':30}{'top-1 dir acc':>14}")
    print("-"*44)
    print(f"{'prior only (oracle ctx)':30}{prior_only/n*100:>13.1f}%")
    print(f"{'motor only (= raw BCI)':30}{m_ok/n*100:>13.1f}%")
    print(f"{'motor x prior (ORACLE ctx)':30}{o_ok/n*100:>13.1f}%   <- ceiling")
    print(f"{'motor x prior (DECODED ctx)':30}{d_ok/n*100:>13.1f}%   <- deployable")
    print("-"*44)
    print(f"gain, oracle : {(o_ok-m_ok)/n*100:+.1f} pp   deployable: {(d_ok-m_ok)/n*100:+.1f} pp")


if __name__ == "__main__":
    main()
