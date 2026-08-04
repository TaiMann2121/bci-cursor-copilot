"""
aggregate_null_control.py  (studies / companion to null_control.py)
===================================================================
Pools the per-seed `null_control.py` logs and answers the one question a single
seed cannot: **is the trained copilot's gain distinguishable from a
shuffled-label null?**

The headline is the PAIRED difference (trained - shuffled) computed within each
seed. Pairing matters because seed controls both the session-block split and the
init, so the two conditions share the same test trials and the same baseline;
comparing pooled means across seeds would throw that structure away and inflate
the spread.

Run:  python studies/aggregate_null_control.py results/null_control_seed*_clean_realonly.log
"""
from __future__ import annotations
import argparse
import glob
import re
import sys

import numpy as np

# ALL   2817   62.7%   63.6%   62.8%   63.0%   56.0%   60.8%   +0.85 +0.06 +0.28 -6.72
ALL_RE = re.compile(
    r"^ALL\s+(\d+)\s+([\d.]+)%\s+([\d.]+)%\s+([\d.]+)%\s+([\d.]+)%\s+([\d.]+)%\s+([\d.]+)%"
    r"\s+([+-][\d.]+)\s+([+-][\d.]+)\s+([+-][\d.]+)\s+([+-][\d.]+)",
    re.M)
CONF_RE = re.compile(r"trained=([\d.]+)\s+random=([\d.]+)\s+shuffled=([\d.]+)")


def parse(path: str):
    txt = open(path, encoding="utf-8", errors="replace").read()
    m = ALL_RE.search(txt)
    if not m:
        return None
    c = CONF_RE.search(txt)
    return {
        "n": int(m.group(1)),
        "raw": float(m.group(2)), "trained": float(m.group(3)),
        "random": float(m.group(4)), "shuffled": float(m.group(5)),
        "const": float(m.group(6)), "const_best": float(m.group(7)),
        "d_train": float(m.group(8)), "d_rand": float(m.group(9)),
        "d_shuf": float(m.group(10)), "d_const": float(m.group(11)),
        "c_tr": float(c.group(1)) if c else float("nan"),
        "c_sh": float(c.group(3)) if c else float("nan"),
    }


def fmt(vals) -> str:
    v = np.asarray(vals, dtype=float)
    return f"{v.mean():+.2f} +/- {v.std(ddof=1):.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    args = ap.parse_args()

    paths = sorted({p for pat in args.logs for p in glob.glob(pat)})
    rows = [(p, parse(p)) for p in paths]
    good = [(p, r) for p, r in rows if r]
    for p, r in rows:
        if not r:
            print(f"  (skipped, no ALL line: {p})", file=sys.stderr)
    if len(good) < 2:
        sys.exit("need at least 2 parsed seed logs")

    print("=" * 78)
    print(f"Null control pooled over {len(good)} seeds  (all values in pp vs raw)")
    print("=" * 78)
    print(f"{'seed log':46}{'d_train':>10}{'d_shuf':>10}{'paired':>10}")
    print("-" * 78)
    paired = []
    for p, r in good:
        d = r["d_train"] - r["d_shuf"]
        paired.append(d)
        print(f"{p.split('/')[-1][:44]:46}{r['d_train']:>+10.2f}{r['d_shuf']:>+10.2f}{d:>+10.2f}")
    print("-" * 78)

    cols = {k: [r[k] for _, r in good] for k in
            ("d_train", "d_rand", "d_shuf", "d_const", "raw", "c_tr", "c_sh")}
    print(f"  trained vs raw      : {fmt(cols['d_train'])} pp")
    print(f"  random-init vs raw  : {fmt(cols['d_rand'])} pp")
    print(f"  shuffled vs raw     : {fmt(cols['d_shuf'])} pp")
    print(f"  constant vs raw     : {fmt(cols['d_const'])} pp")
    print(f"  raw baseline        : {np.mean(cols['raw']):.2f}% "
          f"+/- {np.std(cols['raw'], ddof=1):.2f}")
    print(f"  mean confidence     : trained {np.mean(cols['c_tr']):.3f}, "
          f"shuffled {np.mean(cols['c_sh']):.3f}")

    pa = np.asarray(paired)
    n = len(pa)
    mean, sd = pa.mean(), pa.std(ddof=1)
    se = sd / np.sqrt(n)
    t = mean / se if se > 0 else float("inf")
    print("\n" + "-" * 78)
    print(f"PAIRED trained - shuffled : {mean:+.2f} +/- {sd:.2f} pp "
          f"(se {se:.2f}, t={t:.2f}, n={n})")
    print(f"  seeds where trained > shuffled: {int((pa > 0).sum())}/{n}")
    # 95% CI via the normal approximation; with n=5 this is indicative only.
    print(f"  ~95% CI (normal approx): [{mean - 1.96*se:+.2f}, {mean + 1.96*se:+.2f}] pp")
    print("-" * 78)
    if mean - 1.96 * se > 0:
        print("READ: the interval excludes 0 -- the copilot's gain is distinguishable")
        print("from a label-shuffled null. Target inference is contributing.")
    else:
        print("READ: the interval INCLUDES 0 -- at this sample size the copilot's gain")
        print("is NOT distinguishable from a label-shuffled null. Do not quote the")
        print("raw d_train as the copilot's benefit; the null-subtracted value above")
        print("is the honest estimate, and it is not yet separable from zero.")
    print(f"\nCaveat: the normal approximation is crude at n={n}; treat the interval")
    print("as indicative. Confidence is also unmatched between arms (see above) --")
    print("trained applies a proportionally larger correction, so part of the margin")
    print("is nudge strength rather than better aim.")


if __name__ == "__main__":
    main()
