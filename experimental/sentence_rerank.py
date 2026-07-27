"""
sentence_rerank.py  (experimental / Problem 2: LLM sentence reranking)
=====================================================================
The next lever above the DP sentence decoder: use a language model to reconstruct
the sentence from the noisy character evidence + the DP's candidate decodes. The
DP (sentence_decode.py) uses only word-unigram frequency and gets ~25% word
accuracy; an LLM adds cross-word grammar and meaning — exactly the brief's
sentence-level suggestion ("use an LLM to directly infer a sentence from the
character probability sequence").

Pipeline per sentence:
  1. Build the same REAL motor posteriors as sentence_decode.py.
  2. Generate a small candidate set (motor-argmax, word-level decode, and the DP
     at a few frequency weights).
  3. Send the LLM: the known character length, the candidates, and the top-k
     symbol guesses per position. Ask for the single most likely English sentence.
  4. Score motor / dp / llm by char accuracy, word accuracy (1 - WER), exact match.

Credentials: the client picks up ANTHROPIC_API_KEY, or an `ant auth login`
profile. Nothing is hard-coded. Model: claude-opus-4-8 (adaptive thinking).

Run:
    export ANTHROPIC_API_KEY=sk-ant-...        # or: ant auth login
    python experimental/sentence_rerank.py
    python experimental/sentence_rerank.py --mock    # no API; plumbing check
"""
from __future__ import annotations
import argparse, hashlib, json
from collections import defaultdict

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))   # sibling imports

import numpy as np
from wordfreq import top_n_list, word_frequency

import copilot_dataset as cd
from word_decode import sym2pos, pos2sym, ALPHA, PHI, build_arm_library, build_finger_confusion
from sentence_decode import SENTENCES, edit_distance, dp_decode, word_level_decode

MODEL = "claude-opus-4-8"
CACHE_PATH = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".rerank_cache.json")


def load_cache():
    try:
        return json.load(open(CACHE_PATH))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_cache(cache):
    json.dump(cache, open(CACHE_PATH, "w"), indent=0)


# --- motor posteriors (same construction as sentence_decode / word_decode) ---
def make_poster(arm_lib, kappa, C, Cr, rng):
    def arm_post(g):
        a = float(rng.choice(arm_lib[g])); p = np.exp(kappa * np.cos(a - PHI)); return p / p.sum()

    def make_posts(text):
        posts = []
        for ch in text:
            g0, f0 = sym2pos[ch]
            pg = arm_post(g0)
            fp = rng.choice(3, p=Cr[f0]); pf = C[:, fp] / C[:, fp].sum()
            v = np.zeros(len(ALPHA))
            for j, s in enumerate(ALPHA):
                gg, ff = sym2pos[s]; v[j] = pg[gg] * pf[ff] / len(pos2sym[(gg, ff)])
            posts.append(v / v.sum())
        return posts
    return make_posts


def candidates(posts, bylen, freq):
    """A small, de-duplicated candidate set for the LLM to arbitrate."""
    cands = [''.join(ALPHA[p.argmax()] for p in posts),
             word_level_decode(posts, bylen, freq),
             dp_decode(posts, bylen, freq, fw=0.5),
             dp_decode(posts, bylen, freq, fw=0.2),
             dp_decode(posts, bylen, freq, fw=1.0)]
    seen, out = set(), []
    for c in cands:
        if c not in seen:
            seen.add(c); out.append(c)
    return out


def evidence_block(posts, topk=3):
    """Per-position top-k symbol guesses — the 'character probability sequence'."""
    lines = []
    for i, p in enumerate(posts):
        idx = np.argsort(p)[::-1][:topk]
        alts = " ".join(f"{('_' if ALPHA[j]==' ' else ALPHA[j])}:{p[j]:.2f}" for j in idx)
        lines.append(f"{i:2d} {alts}")
    return "\n".join(lines)


SYSTEM = (
    "You are decoding the output of a noisy EEG brain-computer-interface speller. "
    "Each character was typed by selecting an arm direction (1 of 8) then a finger "
    "(1 of 3); both are noisy, so per-character accuracy is only ~30%. You are given "
    "several candidate decodings of ONE English sentence, plus the top guesses for "
    "each character position. Reconstruct the single most likely intended sentence. "
    "It is lowercase English, words separated by single spaces, no punctuation or "
    "digits. Use grammar and meaning to resolve noise. Output ONLY the sentence."
)


def build_user(posts, cands, n_chars):
    return (f"Character length: {n_chars}\n\nCandidate decodings:\n"
            + "\n".join(f"  - {c}" for c in cands)
            + "\n\nPer-position top guesses (index: symbol:prob, _ = space):\n"
            + evidence_block(posts)
            + "\n\nMost likely intended sentence:")


def llm_decode(client, model, user, cache, use_cache=True):
    """Decode via the LLM, caching by (model, prompt) so re-runs don't re-bill.
    Only a genuine API call costs anything; a cache hit is free."""
    key = hashlib.sha256(f"{model}\n{user}".encode()).hexdigest()
    if use_cache and key in cache:
        return cache[key]
    msg = client.messages.create(
        model=model, max_tokens=1024,
        thinking={"type": "adaptive"},
        system=SYSTEM,
        messages=[{"role": "user", "content": user}],
    )
    text = next((b.text for b in msg.content if b.type == "text"), "")
    line = [l for l in text.strip().splitlines() if l.strip()][-1].lower()
    result = "".join(ch for ch in line if ch in sym2pos).strip()
    cache[key] = result
    save_cache(cache)                # persist after every paid call
    return result


def score(agg, name, text, truth, T):
    agg[name][0] += sum(a == b for a, b in zip(truth, text[:T].ljust(T)))
    agg[name][1] += edit_distance(truth.split(' '), text.split(' '))
    agg[name][2] += int(text.strip() == truth)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default=MODEL,
                    help="claude-opus-4-8 (default) | claude-sonnet-5 | claude-haiku-4-5 (cheapest)")
    ap.add_argument("--mock", action="store_true", help="skip the API (returns DP decode)")
    ap.add_argument("--no-cache", action="store_true", help="ignore the on-disk cache (force re-bill)")
    ap.add_argument("--limit", type=int, default=0, help="only first N sentences (cost control)")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    arm_lib, kappa = build_arm_library()
    C = build_finger_confusion(); Cr = C / C.sum(1, keepdims=True)
    make_posts = make_poster(arm_lib, kappa, C, Cr, rng)
    vocab = [w.lower() for w in top_n_list('en', 20000)
             if all(c in sym2pos for c in w) and w.isalpha()]
    freq = {w: word_frequency(w, 'en') for w in vocab}
    bylen = defaultdict(list)
    for w in vocab:
        bylen[len(w)].append(w)

    client, cache = None, load_cache()
    n_cached = 0
    if not args.mock:
        import anthropic
        client = anthropic.Anthropic()      # ANTHROPIC_API_KEY or ant profile

    sents = SENTENCES[:args.limit] if args.limit else SENTENCES
    agg = {k: [0, 0, 0] for k in ('motor', 'dp', 'llm')}
    tot_chars = tot_words = 0
    for text in sents:
        posts = make_posts(text); T = len(text)
        tot_chars += T; tot_words += len(text.split(' '))
        motor = ''.join(ALPHA[p.argmax()] for p in posts)
        dp = dp_decode(posts, bylen, freq)
        cands = candidates(posts, bylen, freq)
        if args.mock:
            llm = dp
        else:
            user = build_user(posts, cands, T)
            key = hashlib.sha256(f"{args.model}\n{user}".encode()).hexdigest()
            n_cached += (not args.no_cache and key in cache)
            llm = llm_decode(client, args.model, user, cache, use_cache=not args.no_cache)
        score(agg, 'motor', motor, text, T)
        score(agg, 'dp', dp, text, T)
        score(agg, 'llm', llm, text, T)
        print(f"  T  : {text}\n  DP : {dp}\n  LLM: {llm}\n")

    n = len(sents)
    tag = ("   [MOCK: llm==dp]" if args.mock
           else f"   [model={args.model} | {n_cached} cached (free), {n-n_cached} billed]")
    print(f"{n} sentences, {tot_chars} chars, {tot_words} words" + tag)
    print(f"{'decoder':10}{'char acc':>10}{'word acc':>10}{'sentence exact':>16}")
    print("-" * 46)
    for name, label in [('motor', 'motor'), ('dp', 'dp (unigram)'), ('llm', 'llm rerank')]:
        ca = agg[name][0] / tot_chars * 100
        wa = (1 - agg[name][1] / tot_words) * 100
        se = agg[name][2] / n * 100
        print(f"{label:10}{ca:>9.1f}%{wa:>9.1f}%{se:>15.1f}%")


if __name__ == "__main__":
    main()
