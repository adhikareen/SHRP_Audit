"""Paired bootstrap CI for WER deltas between two systems on the same test set.

Table 6, cross-architecture MBR rows for CSL-Daily and PHX-2014T.

CTM:  fid 1 signer time gloss     (one row per gloss)      data/picks/
STM:  fid 1 signer 0.0 1e308 gloss1 ...   (one row per utt) work/refs/ (code/attach_refs.py)
"""
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_ctm(path):
    out = defaultdict(list)
    for line in open(path):
        p = line.split()
        if len(p) < 5:
            continue
        fid, g = p[0], p[4]
        if g == "[EMPTY]":
            out[fid] = []
        else:
            out[fid].append(g)
    return dict(out)


def parse_stm(path):
    out = {}
    for line in open(path):
        p = line.split()
        if len(p) < 5:
            continue
        out[p[0]] = p[5:]
    return out


def edit_distance(a, b):
    m, n = len(a), len(b)
    if m == 0:
        return n
    if n == 0:
        return m
    prev = list(range(n + 1))
    cur = [0] * (n + 1)
    for i in range(1, m + 1):
        cur[0] = i
        ai = a[i - 1]
        for j in range(1, n + 1):
            c = 0 if ai == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + c)
        prev, cur = cur, prev
    return prev[n]


def per_utt(refs, hyps):
    e = np.zeros(len(refs), dtype=np.int64)
    r = np.zeros(len(refs), dtype=np.int64)
    for i, (ref, hyp) in enumerate(zip(refs, hyps)):
        e[i] = edit_distance(ref, hyp)
        r[i] = max(len(ref), 1)
    return e, r


def bootstrap(be, br, ne, nr, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(be)
    bw = 100.0 * be.sum() / max(br.sum(), 1)
    nw = 100.0 * ne.sum() / max(nr.sum(), 1)
    d = np.empty(n_boot)
    for k in range(n_boot):
        idx = rng.integers(0, n, n)
        d[k] = 100.0 * ne[idx].sum() / max(nr[idx].sum(), 1) \
             - 100.0 * be[idx].sum() / max(br[idx].sum(), 1)
    return {
        "base_wer": bw,
        "new_wer": nw,
        "delta": nw - bw,
        "ci95_lo": float(np.percentile(d, 2.5)),
        "ci95_hi": float(np.percentile(d, 97.5)),
        "p_new_better": float((d >= 0).mean()),
    }


def evaluate(name, stm_path, base_ctm, new_ctm):
    stm = parse_stm(stm_path)
    bp = parse_ctm(base_ctm)
    np_ = parse_ctm(new_ctm)
    fids = sorted(set(stm) & set(bp) & set(np_))
    refs = [stm[f] for f in fids]
    be, br = per_utt(refs, [bp[f] for f in fids])
    ne, nr = per_utt(refs, [np_[f] for f in fids])
    bs = bootstrap(be, br, ne, nr)
    print(f"\n[{name}] n={len(fids)}")
    print(f"  base WER = {bs['base_wer']:.2f}")
    print(f"  new  WER = {bs['new_wer']:.2f}")
    print(f"  delta    = {bs['delta']:+.3f}   "
          f"95% CI [{bs['ci95_lo']:+.2f}, {bs['ci95_hi']:+.2f}]   "
          f"p(new<base) = {bs['p_new_better']:.3f}")
    return bs


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--picks-dir", type=Path, default=ROOT / "data" / "picks")
    ap.add_argument("--refs-dir", type=Path, default=ROOT / "work" / "refs")
    a = ap.parse_args()
    D, R = a.picks_dir, a.refs_dir
    evaluate("CSL-Daily   greedy vs MBR (4-bb pool, K=30)",
             f"{R}/csl_test.stm",  f"{D}/csl_greedy_test.ctm",  f"{D}/csl_mbr_test.ctm")
    evaluate("PHX-2014T   greedy vs MBR (5-bb pool, K=30)",
             f"{R}/phxT_test.stm", f"{D}/phxT_greedy_test.ctm", f"{D}/phxT_mbr_test.ctm")
