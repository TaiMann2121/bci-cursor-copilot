"""
word_decode.py  (experimental / Problem 2: language-assisted word decoding)
==========================================================================
The first test of the typing system's real objective: recover intended WORDS from
noisy per-character motor evidence, using a language model.

Why this is the high-headroom axis
-----------------------------------
A character = (arm group, finger). Per-character motor accuracy is mediocre
(~55% arm x noisy finger), so whole-word motor-only accuracy is tiny
(0.55^5 ~ 5% for a 5-letter word). But a word is one of a few thousand valid
English strings, so spelling + word-frequency constraints can recover it far
above the per-character rate. That gap is the language-model win the brief asks
for, and it lives at the WORD level (not the arm-direction level, where a prior
only bought ~+0.7pp).

Method (real motor noise, sampled English words as ground truth)
----------------------------------------------------------------
For each sampled word, for each character c=(g,f):
  arm posterior   : sample a REAL arm endpoint angle for group g (real EEGK
                    trajectories) -> von-Mises posterior over the 8 groups.
  finger posterior: sample a predicted finger from the REAL pooled finger
                    confusion matrix (typing_stats.npz) -> posterior over 3 fingers.
  char posterior  : P(symbol) proportional to P(group)*P(finger), split across the
                    letters sharing a key. 27 symbols (26 letters + space).
Then decode three levels, no-language vs with-language:
  group (arm)   : motor argmax           vs  motor x LM group-prior
  character     : motor argmax           vs  motor x char n-gram (oracle context)
  whole word    : per-char argmax string vs  dictionary decode (spelling + freq)

Run:  python experimental/word_decode.py --n_words 600 --seed 0
"""
from __future__ import annotations
import argparse, glob
from collections import defaultdict

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import numpy as np
from wordfreq import top_n_list, word_frequency

import copilot_dataset as cd

# keyboard: group (arm dir 0-7) x finger (0-2) -> letters
GROUPS = {0:[['w','q'],['e'],['r']], 1:[['t'],['y'],['u']], 2:[['i'],['o'],['p']],
          3:[['f'],['g'],['h','j']], 4:[['m'],['k'],['l']], 5:[['b'],['n'],[' ']],
          6:[['z','x'],['c'],['v']], 7:[['a'],['s'],['d']]}
sym2pos, pos2sym = {}, {}
for g, fs in GROUPS.items():
    for f, syms in enumerate(fs):
        pos2sym[(g, f)] = syms
        for s in syms:
            sym2pos[s] = (g, f)
ALPHA = sorted(sym2pos)                                    # 27 symbols
PHI = np.array([np.arctan2(cd.UNIT_DIRS[d,1], cd.UNIT_DIRS[d,0]) for d in range(8)])


def build_arm_library(repo_root="."):
    """group -> array of real endpoint angles, plus a fitted von-Mises kappa."""
    trajs = cd.load_source("eegk_real", repo_root=repo_root)
    lib = defaultdict(list); errs = []
    for t in trajs:
        end = t.final_pos
        ang = np.arctan2(end[1], end[0])
        lib[t.target_label].append(ang)
        errs.append(np.arctan2(np.sin(ang-PHI[t.target_label]), np.cos(ang-PHI[t.target_label])))
    R = abs(np.mean(np.exp(1j*np.array(errs))))
    kappa = R*(2-R**2)/(1-R**2) if R < 0.99 else 50.0
    return {g: np.array(v) for g, v in lib.items()}, kappa


def build_finger_confusion(repo_root="."):
    C = np.ones((3, 3))
    for f in glob.glob(f"{repo_root}/data/OnlineArmTrajectoryEEGK/**/typing_stats.npz", recursive=True):
        d = np.load(f, allow_pickle=True)
        if 'finger_target_stats' in d and 'finger_pred_stats' in d:
            for a, b in zip(d['finger_target_stats'], d['finger_pred_stats']):
                C[int(a), int(b)] += 1
    return C


def build_char_lm(N=4, topk=25000):
    ctx = [defaultdict(lambda: defaultdict(float)) for _ in range(N)]
    for w in top_n_list('en', topk):
        w = w.lower()
        if not all(c in sym2pos for c in w):
            continue
        f = word_frequency(w, 'en'); s = ' '+w+' '
        for i in range(1, len(s)):
            for o in range(N):
                if i-o < 0: continue
                ctx[o][s[i-o:i]][s[i]] += f
    return ctx, N


def p_next(lm, context):
    ctx, N = lm
    for o in range(N-1, -1, -1):
        c = context[-o:] if o > 0 else ''
        if c in ctx[o] and sum(ctx[o][c].values()) > 0:
            d = ctx[o][c]; tot = sum(d.values())
            return np.array([d.get(s, 0)/tot for s in ALPHA])
    return np.ones(len(ALPHA))/len(ALPHA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_words", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--freq_weight", type=float, default=0.5)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    arm_lib, kappa = build_arm_library()
    C = build_finger_confusion(); Cr = C/C.sum(1, keepdims=True)
    lm = build_char_lm()

    vocab = [w.lower() for w in top_n_list('en', 15000)
             if all(c in sym2pos for c in w) and w.isalpha()]
    freq = {w: word_frequency(w, 'en') for w in vocab}
    bylen = defaultdict(list)
    for w in vocab: bylen[len(w)].append(w)
    cand = [w for w in vocab if 3 <= len(w) <= 8]
    wf = np.array([freq[w] for w in cand]); wf /= wf.sum()
    test = rng.choice(cand, args.n_words, p=wf, replace=True)

    def arm_post(g):
        a = float(rng.choice(arm_lib[g])); p = np.exp(kappa*np.cos(a-PHI)); return p/p.sum()

    gm=gf=cm=cf=ntot=0; wm=wl=0; cw_motor=cw_word=0
    for word in test:
        symposts = []
        for i, ch in enumerate(word):
            g0, f0 = sym2pos[ch]
            pg = arm_post(g0)                                   # real arm noise
            fp = rng.choice(3, p=Cr[f0]); pf = C[:, fp]/C[:, fp].sum()   # real finger noise
            # symbol posterior from motor only
            psym = np.zeros(len(ALPHA))
            for j, s in enumerate(ALPHA):
                gg, ff = sym2pos[s]; psym[j] = pg[gg]*pf[ff]/len(pos2sym[(gg, ff)])
            psym /= psym.sum(); symposts.append(psym)
            # group level
            gm += (pg.argmax() == g0)
            plm = p_next(lm, ' '+word[:i])
            gprior = np.array([plm[[ALPHA.index(s) for s in sum([pos2sym[(g,ff)] for ff in range(3)],[])]].sum()
                               for g in range(8)])
            gf += ((pg*gprior).argmax() == g0)
            # char level
            cm += (ALPHA[psym.argmax()] == ch)
            fused = psym*plm; fused /= fused.sum()
            cf += (ALPHA[fused.argmax()] == ch); ntot += 1
        # whole word: motor per-char argmax vs dictionary decode (spelling + freq)
        motor_word = ''.join(ALPHA[p.argmax()] for p in symposts)
        wm += (motor_word == word); cw_motor += sum(a==b for a,b in zip(motor_word, word))
        best, bs = None, -1e18
        for cwd in bylen[len(word)]:
            sc = sum(np.log(symposts[i][ALPHA.index(cwd[i])]+1e-12) for i in range(len(cwd)))
            sc += args.freq_weight*np.log(freq[cwd]+1e-12)
            if sc > bs: bs, best = sc, cwd
        wl += (best == word); cw_word += sum(a==b for a,b in zip(best, word))

    print(f"tested {args.n_words} freq-weighted English words ({ntot} chars) | arm kappa={kappa:.2f}\n")
    print(f"{'level':16}{'no language':>14}{'with language':>15}")
    print("-"*45)
    print(f"{'group (arm)':16}{gm/ntot*100:>13.1f}%{gf/ntot*100:>14.1f}%")
    print(f"{'character':16}{cm/ntot*100:>13.1f}%{cf/ntot*100:>14.1f}%")
    print(f"{'char-in-word':16}{cw_motor/ntot*100:>13.1f}%{cw_word/ntot*100:>14.1f}%")
    print(f"{'WHOLE WORD':16}{wm/args.n_words*100:>13.1f}%{wl/args.n_words*100:>14.1f}%")
    print("-"*45)
    print("'with language' at word level = dictionary decode (spelling + word freq).")


if __name__ == "__main__":
    main()
