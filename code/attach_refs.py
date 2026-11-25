"""Re-attach reference gloss sequences from your own licensed dataset copies.

The repository ships model outputs only. This script reads the official annotation
files and writes (under --out, default: work/, git-ignored):

  work/beams/*.pkl               beam pickles with "ref" and "wer" per sample and
                                 "gloss_dict" / "i2g" at top level
  work/refs/{csl,phx,phxT}_test.stm   test references, one utterance per line

Expected inputs
  --csl-daily     folder holding video_map.txt and split_1.txt (CSL-Daily sentence_label/)
  --phoenix2014   phoenix-2014-multisigner/ (holds annotations/manual/{train,dev,test}.corpus.csv)
  --phoenix2014t  PHOENIX-2014-T/ (holds annotations/manual/PHOENIX-2014-T.{train,dev,test}.corpus.csv)

Any subset of the three can be given; only the matching files are processed.
The result is checked against data/ref_checksums.json (SHA-256 of the references used
in the paper; the file holds digests only).
"""
import argparse
import hashlib
import json
import pickle
import sys
from itertools import groupby
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SPLITS = ("train", "dev", "test")
STM_FMT = "{fid} 1 {signer} 0.0 1.79769e+308 {label}\n"


# --------------------------------------------------------------------------- #
# Official annotation readers. Each returns {split: [(fileid, signer, label)]}
# in file order, label = the annotation string as stored in the dataset.
# --------------------------------------------------------------------------- #
def _find(root, rel):
    root = Path(root)
    if not root.is_dir():
        raise SystemExit(f"not a directory: {root}")
    for base in (root, *[p for p in root.iterdir() if p.is_dir()]):
        if (base / rel).is_file():
            return base
    raise SystemExit(f"cannot find {rel} under {root}")


def read_csl_daily(root):
    base = _find(root, "video_map.txt")
    split_of = {}
    for line in (base / "split_1.txt").read_text(encoding="utf-8").splitlines()[1:]:
        if line.strip():
            name, split = line.strip().split("|")
            split_of[name] = split
    out = {s: [] for s in SPLITS}
    for line in (base / "video_map.txt").read_text(encoding="utf-8").splitlines()[1:]:
        if not line.strip():
            continue
        _, name, _, gloss, _, _, _ = line.strip().split("|")
        # a video missing from split_1.txt goes to train, as in the usual preprocessing
        out[split_of.get(name, "train")].append((name, "unknown", gloss))
    return out


def _read_phoenix(root, pattern, id_col, signer_col, label_col, drop_train_rows=()):
    base = _find(root, pattern.format("test"))
    out = {}
    for split in SPLITS:
        lines = (base / pattern.format(split)).read_text(encoding="utf-8").split("\n")
        lines = [l.rstrip("\r") for l in lines if l.strip()]
        header = lines[0].split("|")
        cols = [header.index(c) for c in (id_col, signer_col, label_col)]
        rows = []
        for i, line in enumerate(lines[1:]):
            if split == "train" and i in drop_train_rows:
                continue
            f = line.split("|")
            rows.append(tuple(f[c] for c in cols))
        out[split] = rows
    return out


def read_phoenix2014(root):
    # train row 2390 is the sample with broken frames that the usual CSLR
    # preprocessing (VAC / CorrNet / SignGraph) removes
    return _read_phoenix(root, "annotations/manual/{}.corpus.csv",
                         "id", "signer", "annotation", drop_train_rows=(2390,))


def read_phoenix2014t(root):
    return _read_phoenix(root, "annotations/manual/PHOENIX-2014-T.{}.corpus.csv",
                         "name", "speaker", "orth")


# --------------------------------------------------------------------------- #
# Reference normalisation used for the beam pickles
# --------------------------------------------------------------------------- #
def _is_letter(tok):
    return len(tok) == 1 and tok.isalpha() and tok.isupper()


def phoenix_ref(label):
    """Whitespace split, join runs of single-letter tokens with '+', drop adjacent repeats."""
    toks, out, i = label.split(), [], 0
    while i < len(toks):
        j = i
        if _is_letter(toks[i]):
            while j + 1 < len(toks) and _is_letter(toks[j + 1]):
                j += 1
        out.append("+".join(toks[i:j + 1]))
        i = j + 1
    return [k for k, _ in groupby(out)]


def csl_ref(label):
    return label.split()


DATASETS = {
    # key: (file prefix, reader, label -> ref)
    "csl":  ("csl_",  read_csl_daily,    csl_ref),
    "phx":  ("phx_",  read_phoenix2014,  phoenix_ref),
    "phxT": ("phxT_", read_phoenix2014t, phoenix_ref),
}


def build_gloss_dict(info):
    """{gloss: [id, count]} over all splits, ids in sorted gloss order starting at 1."""
    count = {}
    for split in SPLITS:
        for _, _, label in info[split]:
            for g in label.split():
                count[g] = count.get(g, 0) + 1
    return {g: [i + 1, c] for i, (g, c) in enumerate(sorted(count.items(), key=lambda d: d[0]))}


def edit_distance(a, b):
    m, n = len(a), len(b)
    if m == 0:
        return n
    if n == 0:
        return m
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        cur = [i] + [0] * n
        ai = a[i - 1]
        for j in range(1, n + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1,
                         prev[j - 1] + (0 if ai == b[j - 1] else 1))
        prev = cur
    return prev[n]


# --------------------------------------------------------------------------- #
# Digests (so that a re-attachment can be compared with the references used in
# the paper without shipping them)
# --------------------------------------------------------------------------- #
def refs_digest(samples):
    h = hashlib.sha256()
    for s in samples:
        h.update((s["fileid"] + "\t" + " ".join(s["ref"]) + "\n").encode("utf-8"))
    return h.hexdigest()


def vocab_digest(gloss_dict):
    h = hashlib.sha256()
    for g, (i, _) in sorted(gloss_dict.items(), key=lambda d: d[1][0]):
        h.update(f"{i}\t{g}\n".encode("utf-8"))
    return h.hexdigest()


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --------------------------------------------------------------------------- #
def attach(pkl_in, pkl_out, info, to_ref, gloss_dict):
    d = pickle.load(open(pkl_in, "rb"))
    labels = {fid: label for fid, _, label in info[d["split"]]}
    for s in d["samples"]:
        if s["fileid"] not in labels:
            raise SystemExit(f"{pkl_in.name}: {s['fileid']} not found in the "
                             f"{d['split']} annotations")
        ref = to_ref(labels[s["fileid"]])
        s["ref"] = ref
        s["wer"] = np.asarray([100.0 * edit_distance(ref, list(c)) / max(len(ref), 1)
                               for c in s["cands"]], dtype=np.float32)
    d["gloss_dict"] = gloss_dict
    d["i2g"] = {v[0]: k for k, v in gloss_dict.items()}
    pkl_out.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(d, open(pkl_out, "wb"), protocol=4)
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csl-daily", type=Path)
    ap.add_argument("--phoenix2014", type=Path)
    ap.add_argument("--phoenix2014t", type=Path)
    ap.add_argument("--beams-in", type=Path, default=ROOT / "data" / "beams")
    ap.add_argument("--out", type=Path, default=ROOT / "work")
    ap.add_argument("--checksums", type=Path, default=ROOT / "data" / "ref_checksums.json")
    a = ap.parse_args()

    roots = {"csl": a.csl_daily, "phx": a.phoenix2014, "phxT": a.phoenix2014t}
    if not any(roots.values()):
        ap.error("give at least one of --csl-daily / --phoenix2014 / --phoenix2014t")
    expected = json.load(open(a.checksums)) if a.checksums.is_file() else {}
    got, n_bad = {}, 0

    def check(key, value):
        nonlocal n_bad
        got[key] = value
        if key not in expected:
            return "no checksum"
        if expected[key] == value:
            return "ok"
        n_bad += 1
        return "MISMATCH"

    for ds, (prefix, reader, to_ref) in DATASETS.items():
        if roots[ds] is None:
            continue
        info = reader(roots[ds])
        gloss_dict = build_gloss_dict(info)
        print(f"[{ds}] train/dev/test = "
              f"{'/'.join(str(len(info[s])) for s in SPLITS)}  vocab = {len(gloss_dict)}  "
              f"[{check(f'{ds}.vocab', vocab_digest(gloss_dict))}]")

        stm = a.out / "refs" / f"{ds}_test.stm"
        stm.parent.mkdir(parents=True, exist_ok=True)
        with open(stm, "w", encoding="utf-8") as f:
            for fid, signer, label in info["test"]:
                f.write(STM_FMT.format(fid=fid, signer=signer, label=label))
        print(f"  {stm.name:28s} {len(info['test']):5d} utterances  "
              f"[{check(stm.name, file_digest(stm))}]")

        for pkl in sorted(a.beams_in.glob(f"{prefix}*.pkl")):
            d = attach(pkl, a.out / "beams" / pkl.name, info, to_ref, gloss_dict)
            print(f"  {pkl.name:28s} {len(d['samples']):5d} utterances  "
                  f"[{check(pkl.name, refs_digest(d['samples']))}]")

    if n_bad:
        print(f"{n_bad} checksum mismatch(es): the annotations read here differ from the "
              f"ones used in the paper.", file=sys.stderr)
        sys.exit(1)
    print(f"wrote {a.out / 'beams'} and {a.out / 'refs'}")


if __name__ == "__main__":
    main()
