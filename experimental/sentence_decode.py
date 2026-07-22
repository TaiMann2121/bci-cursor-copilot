"""
sentence_decode.py  (experimental / Problem 2: sentence-level decoding)
======================================================================
Extends word_decode.py from words to whole SENTENCES. The hard part is joint
SEGMENTATION: the space key is (group 5, finger 2), but group 5 also holds b/n,
so spaces are noisy -- the decoder must decide where words break AND what they are
at the same time. Word-level decoding assumes segmentation is given; a real system
does not have that.

Decoder: a beam-search noisy-channel over the full character sequence of KNOWN
length T. At each position it extends hypotheses with the top-k motor symbols,
scoring
    log P_motor(sym) + Bc * log P_charLM(sym | context) + Bw * word_validity_bonus
where the word bonus fires when a space (or the end) completes a token: + log
word-frequency if it is a real word, a penalty otherwise. This jointly handles
spelling, segmentation, lexical validity, and the known length.

Compared against:
  motor only      : per-character argmax (no language)
  word level      : segment on motor-argmax spaces, decode each word independently
                    (the ~41% approach from word_decode.py -- but now segmentation
                    is part of the problem, so its errors show)
  sentence level  : the joint beam decoder above

Motor noise is REAL (real arm endpoints + real finger confusion); the sentences are
a fixed set of natural English (self-contained, no punctuation/numbers).

Run:  python experimental/sentence_decode.py --seed 0
"""
from __future__ import annotations
import argparse
from collections import defaultdict

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import numpy as np
from wordfreq import top_n_list, word_frequency

import copilot_dataset as cd
from word_decode import (sym2pos, pos2sym, ALPHA, PHI,
                         build_arm_library, build_finger_confusion, build_char_lm, p_next)

SENTENCES = [
    "the quick brown fox jumps over the lazy dog",
    "she sells sea shells by the sea shore",
    "we hold these truths to be self evident",
    "all that glitters is not gold",
    "a journey of a thousand miles begins with a single step",
    "the early bird catches the worm",
    "actions speak louder than words",
    "practice makes perfect over time",
    "knowledge is power in the modern world",
    "the pen is mightier than the sword",
    "better late than never they always say",
    "an apple a day keeps the doctor away",
    "when in rome do as the romans do",
    "the grass is always greener on the other side",
    "every cloud has a silver lining somewhere",
    "honesty is the best policy in life",
    "look before you leap into the unknown",
    "the whole is greater than the sum of its parts",
    "birds of a feather flock together",
    "curiosity killed the cat but satisfaction brought it back",
    "the future belongs to those who prepare for it today",
    "science and reason light the path forward",
    "music can soothe even the most troubled mind",
    "the ocean covers most of our small blue planet",
    "children learn language faster than adults do",
    "a picture is worth a thousand spoken words",
    "the mountain trail was steep but the view rewarded us",
    "coffee in the morning helps many people focus",
    "history often repeats itself for those who forget",
    "the human brain remains a profound mystery to us",
]


def edit_distance(a, b):
    m, n = len(a), len(b)
    d = list(range(n+1))
    for i in range(1, m+1):
        prev, d[0] = d[0], i
        for j in range(1, n+1):
            cur = d[j]
            d[j] = min(d[j]+1, d[j-1]+1, prev+(a[i-1] != b[j-1]))
            prev = cur
    return d[n]


def dp_decode(posts, bylen, freq, fw=0.5, maxL=15):
    """Joint segmentation + word decode. DP over character positions: segment the
    length-T sequence into dictionary words separated by single spaces, maximizing
    sum of per-word (motor spelling + freq) + the space motor evidence at breaks.
    Keeps the hard dictionary constraint that made word-level decoding work, while
    solving the unknown word boundaries."""
    T = len(posts)
    logpost = [np.log(p + 1e-12) for p in posts]
    idx = {s: i for i, s in enumerate(ALPHA)}
    SP = idx[' ']
    cache = {}

    def best_word(q, L):
        if (q, L) in cache:
            return cache[(q, L)]
        lp = logpost[q:q+L]; best, bs = None, -1e18
        for w in bylen.get(L, ()):
            sc = fw * np.log(freq[w] + 1e-12)
            for i in range(L):
                sc += lp[i][idx[w[i]]]
            if sc > bs:
                bs, best = sc, w
        cache[(q, L)] = (bs, best); return cache[(q, L)]

    NEG = -1e18
    dp = [NEG]*(T+1); back = [None]*(T+1); dp[0] = 0.0
    for i in range(T):
        if dp[i] == NEG:
            continue
        base = dp[i] + (0.0 if i == 0 else logpost[i][SP])   # a space sits at position i
        start = i if i == 0 else i + 1
        for L in range(1, maxL+1):
            if start + L > T:
                break
            sc, w = best_word(start, L)
            if w is None:
                continue
            j = start + L
            if base + sc > dp[j]:
                dp[j] = base + sc; back[j] = (i, w)
    if dp[T] == NEG:
        return ''.join(ALPHA[p.argmax()] for p in posts)
    words, j = [], T
    while j > 0:
        i, w = back[j]; words.append(w); j = i
    return ' '.join(reversed(words))


def word_level_decode(symposts, bylen, freq, fw=0.5):
    """Segment on motor-argmax spaces, decode each segment independently."""
    motor = ''.join(ALPHA[p.argmax()] for p in symposts)
    out = []
    idx = 0
    for seg in motor.split(' '):
        L = len(seg)
        posts = symposts[idx:idx+L]; idx += L+1
        if L == 0 or L not in bylen:
            out.append(seg); continue
        best, bs = seg, -1e18
        for cwd in bylen[L]:
            sc = sum(np.log(posts[i][ALPHA.index(cwd[i])]+1e-12) for i in range(L)) + fw*np.log(freq[cwd]+1e-12)
            if sc > bs: bs, best = sc, cwd
        out.append(best)
    return ' '.join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    arm_lib, kappa = build_arm_library()
    C = build_finger_confusion(); Cr = C/C.sum(1, keepdims=True)
    vocab = [w.lower() for w in top_n_list('en', 20000)
             if all(c in sym2pos for c in w) and w.isalpha()]
    freq = {w: word_frequency(w, 'en') for w in vocab}
    bylen = defaultdict(list)
    for w in vocab: bylen[len(w)].append(w)

    def arm_post(g):
        a = float(rng.choice(arm_lib[g])); p = np.exp(kappa*np.cos(a-PHI)); return p/p.sum()

    def make_posts(text):
        posts = []
        for ch in text:
            g0, f0 = sym2pos[ch]
            pg = arm_post(g0)
            fp = rng.choice(3, p=Cr[f0]); pf = C[:, fp]/C[:, fp].sum()
            v = np.zeros(len(ALPHA))
            for j, s in enumerate(ALPHA):
                gg, ff = sym2pos[s]; v[j] = pg[gg]*pf[ff]/len(pos2sym[(gg, ff)])
            posts.append(v/v.sum())
        return posts

    agg = {k: [0, 0, 0] for k in ['motor', 'word', 'sent']}   # [char_ok, word_err, sent_ok]
    tot_chars = tot_words = 0
    examples = []
    for text in SENTENCES:
        posts = make_posts(text)
        T = len(text); tw = text.split(' ')
        tot_chars += T; tot_words += len(tw)
        motor = ''.join(ALPHA[p.argmax()] for p in posts)
        wl = word_level_decode(posts, bylen, freq)
        sl = dp_decode(posts, bylen, freq)
        for name, hyp in [('motor', motor), ('word', wl), ('sent', sl)]:
            agg[name][0] += sum(a == b for a, b in zip(text, hyp[:T].ljust(T)))
            agg[name][1] += edit_distance(tw, hyp.split(' '))
            agg[name][2] += int(hyp.strip() == text)
        examples.append((text, motor, wl, sl))

    print(f"{len(SENTENCES)} sentences, {tot_chars} chars, {tot_words} words | kappa={kappa:.2f}\n")
    print(f"{'decoder':16}{'char acc':>10}{'word acc':>10}{'sentence exact':>16}")
    print("-"*52)
    for name, label in [('motor', 'motor only'), ('word', 'word level'), ('sent', 'sentence level')]:
        ca = agg[name][0]/tot_chars*100
        wa = (1 - agg[name][1]/tot_words)*100
        se = agg[name][2]/len(SENTENCES)*100
        print(f"{label:16}{ca:>9.1f}%{wa:>9.1f}%{se:>15.1f}%")
    print("-"*52)
    print("word acc = 1 - word error rate (edit distance on word tokens).\n")
    print("Examples (truth / motor / word-level / sentence-level):")
    for text, motor, wl, sl in examples[:6]:
        print(f"  T : {text}")
        print(f"  M : {motor}")
        print(f"  W : {wl}")
        print(f"  S : {sl}\n")


if __name__ == "__main__":
    main()
