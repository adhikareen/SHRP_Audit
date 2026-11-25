"""MBR decoding over a beam pool under editdist / BLEU / chrF utilities.

Cross-architecture MBR on PHX-2014 (Tables 1 and 6) and the neural-utility comparison
of the Discussion. Uses no dev labels; references are read only to score the picks
(work/beams/, see code/attach_refs.py).

Source pools:
    --source combined  single pickle holding the full pool (<dir>/test_beams.pkl)
    --source rich      union of named per-backbone pickles (dataset-aligned by fid)
"""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

try:
    import sacrebleu
except ImportError:
    print("install: pip install sacrebleu", file=sys.stderr)
    raise


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


def corpus_wer(refs, hyps):
    e, r = per_utt(refs, hyps)
    return 100.0 * e.sum() / max(r.sum(), 1)


def load_combined(combined_dir, split="test"):
    d = pickle.load(open(combined_dir / f"{split}_beams.pkl", "rb"))
    samples = d["samples"]
    return (
        [s["fileid"] for s in samples],
        [list(s["ref"]) for s in samples],
        [[list(c) for c in s["cands"]] for s in samples],
    )


def _norm_fid(fid):
    return fid.split("|", 1)[0]


def load_rich(pickles, split="test"):
    """pickles: dict name->path. Returns (fids, refs, {name: beams}) intersected by fid."""
    by_arch = {}
    refs_by_fid = {}
    fid_order = []
    names = list(pickles)
    for name in names:
        d = pickle.load(open(pickles[name], "rb"))
        idx = {}
        for s in d["samples"]:
            fid = _norm_fid(s["fileid"])
            refs_by_fid.setdefault(fid, list(s["ref"]))
            idx[fid] = [list(c) for c in s["cands"]]
            if name == names[0] and fid not in set(fid_order):
                fid_order.append(fid)
        by_arch[name] = idx
    keep = [f for f in fid_order if all(f in by_arch[n] for n in names)]
    return (
        keep,
        [refs_by_fid[f] for f in keep],
        {n: [by_arch[n][f] for f in keep] for n in names},
    )


def pool_topk(per_arch, K_per_arch):
    names = list(per_arch)
    n = len(per_arch[names[0]])
    out = []
    for i in range(n):
        seen, kept = set(), []
        for name in names:
            for c in per_arch[name][i][:K_per_arch]:
                t = tuple(c)
                if t in seen:
                    continue
                seen.add(t)
                kept.append(c)
        out.append(kept)
    return out


def dedupe_topk(beams, K):
    out = []
    for cands in beams:
        seen, kept = set(), []
        for c in cands[:K]:
            t = tuple(c)
            if t in seen:
                continue
            seen.add(t)
            kept.append(c)
        out.append(kept)
    return out


def util_editdist(h, hp):
    L = max(len(h), len(hp), 1)
    return -edit_distance(h, hp) / L


def util_bleu(h, hp):
    return sacrebleu.sentence_bleu(" ".join(h), [" ".join(hp)],
                                    smooth_method="exp",
                                    use_effective_order=True).score


def util_chrf(h, hp):
    return sacrebleu.sentence_chrf(" ".join(h), [" ".join(hp)]).score


UTILS = {"editdist": util_editdist, "bleu": util_bleu, "chrf": util_chrf}


def mbr_pick(cands, utility):
    if not cands:
        return []
    if len(cands) == 1:
        return cands[0]
    u = UTILS[utility]
    n = len(cands)
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            v = u(cands[i], cands[j])
            M[i, j] = M[j, i] = v
    return cands[int(np.argmax(M.sum(axis=1)))]


def mbr_corpus(beams, utility):
    return [mbr_pick(b, utility) for b in beams]


def paired_bootstrap(be, br, ne, nr, n_boot=1000, seed=0):
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
        "base_wer": float(bw),
        "new_wer": float(nw),
        "delta": float(nw - bw),
        "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
        "p_one_sided_new_better": float((d >= 0).mean()),
    }


def greedy_best_top1(per_arch, beams_single, refs):
    if per_arch is not None:
        wer = {}
        for n, b in per_arch.items():
            wer[n] = corpus_wer(refs, [c[0] if c else [] for c in b])
        best = min(wer, key=wer.get)
        return wer[best], [b[0] if b else [] for b in per_arch[best]], best
    picks = [b[0] if b else [] for b in beams_single]
    return corpus_wer(refs, picks), picks, "combined-top1"


def run(*, dataset, source, combined_dir, pool_pickles, K, utilities,
        n_boot, seed, target_editdist, gate_tol, out_path):
    t0 = time.time()
    print(f"[{dataset}/{source}] loading beams", flush=True)
    if source == "combined":
        fids, refs, beams = load_combined(combined_dir, "test")
        beams = dedupe_topk(beams, K)
        per_arch = None
        label = f"combined (K={K}, n={len(refs)})"
    else:
        fids, refs, per_arch = load_rich(pool_pickles, "test")
        Kpa = max(1, K // len(pool_pickles))
        beams = pool_topk(per_arch, Kpa)
        label = f"rich {list(pool_pickles)} (K/arch={Kpa}, n={len(refs)})"

    sizes = [len(b) for b in beams]
    print(f"  pool size: mean={np.mean(sizes):.1f} "
          f"min={min(sizes)} max={max(sizes)}", flush=True)

    g_wer, g_picks, g_name = greedy_best_top1(per_arch, beams, refs)
    ge, gr = per_utt(refs, g_picks)
    print(f"  greedy ({g_name}) WER = {g_wer:.2f}", flush=True)

    out = {
        "dataset": dataset,
        "source": source,
        "pool_names": list(pool_pickles) if pool_pickles else None,
        "K": K,
        "label": label,
        "n_utt": len(refs),
        "n_boot": n_boot,
        "seed": seed,
        "greedy_best_wer": g_wer,
        "greedy_best_backbone": g_name,
        "utilities": {},
    }

    ed_cache = None
    for u in utilities:
        t = time.time()
        picks = mbr_corpus(beams, u)
        wer = corpus_wer(refs, picks)
        ue, ur = per_utt(refs, picks)
        bs = paired_bootstrap(ge, gr, ue, ur, n_boot=n_boot, seed=seed)
        out["utilities"][u] = {
            "wer": wer,
            "vs_greedy_best": bs,
            "minutes": (time.time() - t) / 60,
        }
        print(f"  MBR[{u:<8s}] WER={wer:.2f}  d={bs['delta']:+.2f}  "
              f"CI=[{bs['ci95'][0]:+.2f},{bs['ci95'][1]:+.2f}]  "
              f"p={bs['p_one_sided_new_better']:.3f}", flush=True)
        if u == "editdist":
            ed_cache = (ue, ur)

    if ed_cache is not None:
        be, br = ed_cache
        for u in utilities:
            if u == "editdist":
                continue
            ue, ur = per_utt(refs, mbr_corpus(beams, u))
            out["utilities"][u]["vs_editdist_mbr"] = paired_bootstrap(
                be, br, ue, ur, n_boot=n_boot, seed=seed)

    if target_editdist is not None:
        diff = out["utilities"]["editdist"]["wer"] - target_editdist
        out["gate"] = {
            "target": target_editdist,
            "reproduced": out["utilities"]["editdist"]["wer"],
            "abs_diff": abs(diff),
            "tol": gate_tol,
            "passed": abs(diff) <= gate_tol,
        }

    out["wallclock_min"] = (time.time() - t0) / 60
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(out_path, "w"), indent=2)
    print(f"  wrote {out_path}  ({out['wallclock_min']:.1f} min)", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--source", choices=["combined", "rich"], required=True)
    ap.add_argument("--combined-dir", type=Path)
    ap.add_argument("--pool-pickle", action="append", default=[], metavar="NAME=PATH")
    ap.add_argument("--K", type=int, default=50)
    ap.add_argument("--utilities", nargs="+", default=["editdist", "bleu", "chrf"])
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--paper-editdist", type=float, default=None)
    ap.add_argument("--gate-tol", type=float, default=0.05)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()

    pool_pickles = None
    if a.source == "rich":
        pool_pickles = {}
        for spec in a.pool_pickle:
            if "=" not in spec:
                raise SystemExit(f"--pool-pickle expects NAME=PATH, got: {spec}")
            name, path = spec.split("=", 1)
            pool_pickles[name] = Path(path)
        if not pool_pickles:
            raise SystemExit("--source rich requires at least one --pool-pickle")

    run(
        dataset=a.dataset, source=a.source,
        combined_dir=a.combined_dir, pool_pickles=pool_pickles,
        K=a.K, utilities=a.utilities,
        n_boot=a.bootstrap, seed=a.seed,
        target_editdist=a.paper_editdist, gate_tol=a.gate_tol,
        out_path=a.out,
    )


if __name__ == "__main__":
    main()
