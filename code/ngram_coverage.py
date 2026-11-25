"""Test n-gram coverage by dev and by train, n = 1..4.

Table 11 and the 4-gram coverage rows of Table 2.

Coverage is the share (%) of distinct test n-grams that also occur in dev (or in train).
N-grams are taken within an utterance, over the gloss annotation split on whitespace.
The script reads the dataset annotation files directly; the arguments are those of
code/attach_refs.py, and any subset of the three datasets can be given.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from attach_refs import read_csl_daily, read_phoenix2014, read_phoenix2014t

ORDERS = (1, 2, 3, 4)


def ngrams(seqs, n):
    return {tuple(s[i:i + n]) for s in seqs for i in range(len(s) - n + 1)}


def coverage(info):
    """info: {split: [(fileid, signer, label)]} -> {"dev": [...], "train": [...]} in %, per n."""
    seqs = {split: [label.split() for _, _, label in rows] for split, rows in info.items()}
    out = {}
    for src in ("dev", "train"):
        out[src] = []
        for n in ORDERS:
            test = ngrams(seqs["test"], n)
            out[src].append(100.0 * len(test & ngrams(seqs[src], n)) / len(test))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csl-daily", type=Path)
    ap.add_argument("--phoenix2014", type=Path)
    ap.add_argument("--phoenix2014t", type=Path)
    a = ap.parse_args()
    jobs = [("CSL-Daily", a.csl_daily, read_csl_daily),
            ("PHX-2014", a.phoenix2014, read_phoenix2014),
            ("PHX-2014T", a.phoenix2014t, read_phoenix2014t)]
    if not any(root for _, root, _ in jobs):
        ap.error("give at least one of --csl-daily / --phoenix2014 / --phoenix2014t")
    for name, root, reader in jobs:
        if root is None:
            continue
        cov = coverage(reader(root))
        print(f"[{name}] test n-gram coverage (%), n = " + ", ".join(map(str, ORDERS)))
        for src in ("dev", "train"):
            print(f"  by {src:5s} " + "  ".join(f"{c:5.1f}" for c in cov[src]))


if __name__ == "__main__":
    main()
