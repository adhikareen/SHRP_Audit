"""Sentence-overlap + NN edit-distance audit between dev and test gloss splits.

Table 2 (test exact-duplicate rate) and Table 3 (NN edit-distance buckets:
"nn_dist_test_to_dev_buckets" in the output JSON).
Reads the references from work/beams/ (code/attach_refs.py).
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np


def load_refs(path):
    d = pickle.load(open(path, "rb"))
    return [(s["fileid"], list(s["ref"])) for s in d["samples"]]


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


def nn_distances(src, tgt):
    out = np.full(len(src), 10**9, dtype=np.int32)
    for i, a in enumerate(src):
        best = len(a) + 1
        for b in tgt:
            d = edit_distance(a, b)
            if d < best:
                best = d
                if best == 0:
                    break
        out[i] = best
    return out


def analyze(name, dev_path, test_path, out):
    dev = load_refs(dev_path)
    tst = load_refs(test_path)
    print(f"[{name}] dev={len(dev)} test={len(tst)}", flush=True)

    dev_refs = [r for _, r in dev]
    test_refs = [r for _, r in tst]

    test_set = set(tuple(r) for r in test_refs)
    dev_set = set(tuple(r) for r in dev_refs)
    n_dev_in_test = sum(tuple(r) in test_set for r in dev_refs)
    n_test_in_dev = sum(tuple(r) in dev_set for r in test_refs)

    print("  computing NN edit-distances ...", flush=True)
    d_dt = nn_distances(dev_refs, test_refs)
    d_td = nn_distances(test_refs, dev_refs)

    buckets_dt = {}
    buckets_td = {}
    for k in (0, 1, 2, 3, 5, 10):
        buckets_dt[f"dist_<={k}"] = int((d_dt <= k).sum())
        buckets_td[f"dist_<={k}"] = int((d_td <= k).sum())
        buckets_td[f"frac_<={k}"] = float((d_td <= k).mean())

    print(f"  dev exact-dup in test: {n_dev_in_test}/{len(dev)} "
          f"({100*n_dev_in_test/len(dev):.2f}%)", flush=True)
    print(f"  test exact-dup in dev: {n_test_in_dev}/{len(tst)} "
          f"({100*n_test_in_dev/len(tst):.2f}%)", flush=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({
        "dataset": name,
        "n_dev": len(dev),
        "n_test": len(tst),
        "n_dev_exact_dup_with_test": n_dev_in_test,
        "frac_dev_exact_dup_with_test": n_dev_in_test / max(len(dev), 1),
        "n_test_exact_dup_with_dev": n_test_in_dev,
        "frac_test_exact_dup_with_dev": n_test_in_dev / max(len(tst), 1),
        "nn_dist_dev_to_test_buckets": buckets_dt,
        "nn_dist_test_to_dev_buckets": buckets_td,
    }, open(out, "w"), indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["csl", "phx", "phxT"], required=True)
    ROOT = Path(__file__).resolve().parent.parent
    ap.add_argument("--beams-dir", type=Path, default=ROOT / "work" / "beams")
    ap.add_argument("--out", type=Path, default=None,
                    help="default: work/diagnostics/overlap_<dataset>.json")
    a = ap.parse_args()
    if a.out is None:
        a.out = ROOT / "work" / "diagnostics" / f"overlap_{a.dataset}.json"
    analyze(a.dataset,
            a.beams_dir / f"{a.dataset}_combined_dev.pkl",
            a.beams_dir / f"{a.dataset}_combined_test.pkl",
            a.out)
