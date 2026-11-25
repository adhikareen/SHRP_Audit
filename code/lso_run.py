"""
Leave-Sentences-Out (LSO) experiment: the sentence-disjoint evaluation, Table 5.

For a dataset, we:
  (1) take U = unique TEST gloss sequences; stratified 50/50 split (by test
      frequency) into U_seen / U_unseen with a fixed seed;
  (2) build the dev-conditioned fitting corpus = dev instances whose gloss is in
      U_seen ONLY (assert disjoint from U_unseen);
  (3) fit the two paper probes (Brants 4-gram dev-LM rescore, BoW frozen-emb
      20-seed) on that corpus and evaluate WER on full-test, TEST_seen, TEST_unseen;
  (4) report Delta vs greedy for seen and unseen separately.

Repeats over split seeds 0,1,2.  Writes lso_<dataset>.json and perutt_<dataset>.csv
to --out_dir (default work/sentence_disjoint/). Reads work/beams/ (code/attach_refs.py).
"""
import sys, json, argparse, time
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lso_probe import (load_beams, build_g2id, corpus_wer, per_utt_edits,
                       fit_probe, lm_rescore, edit_distance)

ROOT = Path(__file__).resolve().parent.parent
# K=50 beam of one SignGraph backbone per benchmark: <beams_dir>/<pool>_{dev,test}.pkl
POOLS = {
    'csl':  'csl_sg',
    'phx':  'phx_combined',
    'phxT': 'phxT_combined',
}
PAPER_GREEDY = {'csl': 25.67, 'phx': 19.62, 'phxT': 19.93}


def lso_split(U_list, freq, seed):
    """Stratified 50/50 split of unique test gloss seqs by test frequency."""
    rng = np.random.RandomState(seed)
    groups = defaultdict(list)
    for u in U_list:
        groups[freq[u]].append(u)
    seen, unseen = set(), set()
    for fval in sorted(groups):
        members = list(groups[fval]); rng.shuffle(members)
        flip = rng.randint(2)                      # randomise which half gets the odd one
        for i, u in enumerate(members):
            (seen if (i + flip) % 2 == 0 else unseen).add(u)
    return seen, unseen


def paired_bootstrap(edits_m, edits_g, reflen, n=1000, seed=0):
    """One-sided paired bootstrap: is method better (lower WER) than greedy?
    Returns (delta_wer, lo, hi, p_one_sided)."""
    rng = np.random.RandomState(seed)
    N = len(reflen)
    if N == 0:
        return float('nan'), float('nan'), float('nan'), float('nan')
    base = 100.0 * (edits_m.sum() / reflen.sum() - edits_g.sum() / reflen.sum())
    deltas = np.empty(n)
    for b in range(n):
        idx = rng.randint(0, N, N)
        r = reflen[idx].sum()
        deltas[b] = 100.0 * (edits_m[idx].sum() - edits_g[idx].sum()) / max(r, 1)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    p = float((deltas >= 0).mean())                # H1: method better (delta<0)
    return float(base), float(lo), float(hi), p


def wer_sub(samples, picks, idxs):
    refs = [samples[i]['ref'] for i in idxs]
    hyps = [samples[i]['cands'][picks[i]] for i in idxs]
    return corpus_wer(refs, hyps)


def run_dataset(ds, bow_seeds, device, out_dir, split_seeds=(0, 1, 2), beams_dir=ROOT / 'work' / 'beams'):
    pool = str(Path(beams_dir) / POOLS[ds])
    dev = load_beams(f'{pool}_dev.pkl')
    test = load_beams(f'{pool}_test.pkl')
    g2id, V = build_g2id(dev, test)
    dev_s, test_s = dev['samples'], test['samples']
    refs = [s['ref'] for s in test_s]
    greedy_pick = [0] * len(test_s)

    # ---- overlap / U ----
    test_keys = [tuple(s['ref']) for s in test_s]
    dev_keys = [tuple(s['ref']) for s in dev_s]
    freq = defaultdict(int)
    for k in test_keys:
        freq[k] += 1
    U = list(freq.keys())
    dev_key_set = set(dev_keys)
    n_exact = sum(1 for k in test_keys if k in dev_key_set)
    greedy_full = corpus_wer(refs, [s['cands'][0] for s in test_s])

    print(f"\n########## {ds.upper()}  pool={POOLS[ds]} ##########", flush=True)
    print(f"dev={len(dev_s)} test={len(test_s)} |U|={len(U)} "
          f"exact-dup={n_exact}/{len(test_s)}={100*n_exact/len(test_s):.2f}% "
          f"greedy={greedy_full:.2f} (Table 1 greedy-best {PAPER_GREEDY[ds]})", flush=True)

    # ---- reference: full-dev probes (non-LSO) ----
    lm_full = lm_rescore(dev_s, test_s, g2id)
    lm_full_wer = wer_sub(test_s, lm_full, range(len(test_s)))
    _, bow_full = fit_probe(dev_s, test_s, g2id, V, device, seeds=range(bow_seeds), freeze_emb=True)
    bow_full_wer = wer_sub(test_s, bow_full, range(len(test_s)))
    print(f"[full-dev]  greedy={greedy_full:.2f}  dev-LM={lm_full_wer:.2f}  "
          f"BoW-frozen({bow_seeds}s)={bow_full_wer:.2f}", flush=True)

    per_seed_results = []
    csv_rows = []
    for sseed in split_seeds:
        U_seen, U_unseen = lso_split(U, freq, sseed)
        # fitting corpus = dev whose gloss in U_seen ONLY
        fit_idx = [i for i, k in enumerate(dev_keys) if k in U_seen]
        fit_samples = [dev_s[i] for i in fit_idx]
        fit_keys = set(tuple(s['ref']) for s in fit_samples)
        assert fit_keys.isdisjoint(U_unseen), "LSO ASSERTION FAILED: fit corpus intersects U_unseen"

        seen_idx = [i for i, k in enumerate(test_keys) if k in U_seen]
        unseen_idx = [i for i, k in enumerate(test_keys) if k in U_unseen]
        assert len(seen_idx) + len(unseen_idx) == len(test_s)

        # ---- fit probes on LSO corpus ----
        lm_p = lm_rescore(fit_samples, test_s, g2id) if fit_samples else greedy_pick[:]
        if fit_samples:
            _, bow_p = fit_probe(fit_samples, test_s, g2id, V, device,
                                 seeds=range(bow_seeds), freeze_emb=True)
        else:
            bow_p = greedy_pick[:]

        res = {'split_seed': sseed,
               'n_U_seen': len(U_seen), 'n_U_unseen': len(U_unseen),
               'n_fit': len(fit_samples),
               'n_test_seen': len(seen_idx), 'n_test_unseen': len(unseen_idx)}
        for name, picks in [('greedy', greedy_pick), ('devLM', lm_p), ('BoW', bow_p)]:
            res[f'{name}_full'] = wer_sub(test_s, picks, range(len(test_s)))
            res[f'{name}_seen'] = wer_sub(test_s, picks, seen_idx) if seen_idx else float('nan')
            res[f'{name}_unseen'] = wer_sub(test_s, picks, unseen_idx) if unseen_idx else float('nan')
        per_seed_results.append(res)
        print(f"  [seed {sseed}] fit={len(fit_samples)} seen={len(seen_idx)} unseen={len(unseen_idx)} | "
              f"devLM seen={res['devLM_seen']:.2f} unseen={res['devLM_unseen']:.2f} | "
              f"BoW seen={res['BoW_seen']:.2f} unseen={res['BoW_unseen']:.2f}", flush=True)

        # per-utterance CSV (split seed 0 only, to keep size sane)
        if sseed == 0:
            for i, s in enumerate(test_s):
                sub = 'seen' if test_keys[i] in U_seen else 'unseen'
                rl = len(s['ref'])
                row = {'dataset': ds, 'split_seed': sseed, 'fileid': s['fileid'],
                       'subset': sub, 'ref_len': rl}
                for name, picks in [('greedy', greedy_pick), ('devLM', lm_p), ('BoW', bow_p)]:
                    row[f'{name}_edits'] = edit_distance(list(s['ref']), list(s['cands'][picks[i]]))
                csv_rows.append(row)

    # ---- significance on split seed 0 ----
    sseed0 = 0
    U_seen, U_unseen = lso_split(U, freq, sseed0)
    fit_samples = [dev_s[i] for i, k in enumerate(dev_keys) if k in U_seen]
    if fit_samples:
        lm_p = lm_rescore(fit_samples, test_s, g2id)
        _, bow_p = fit_probe(fit_samples, test_s, g2id, V, device, seeds=range(bow_seeds), freeze_emb=True)
    else:
        lm_p = greedy_pick[:]; bow_p = greedy_pick[:]
    seen_idx = [i for i, k in enumerate(test_keys) if k in U_seen]
    unseen_idx = [i for i, k in enumerate(test_keys) if k in U_unseen]
    sig = {}
    for name, picks in [('devLM', lm_p), ('BoW', bow_p)]:
        for subset, idxs in [('seen', seen_idx), ('unseen', unseen_idx)]:
            refs_s = [test_s[i]['ref'] for i in idxs]
            hyp_m = [test_s[i]['cands'][picks[i]] for i in idxs]
            hyp_g = [test_s[i]['cands'][0] for i in idxs]
            em, rl = per_utt_edits(refs_s, hyp_m)
            eg, _ = per_utt_edits(refs_s, hyp_g)
            d, lo, hi, p = paired_bootstrap(em, eg, rl, n=1000, seed=0)
            sig[f'{name}_{subset}'] = {'delta': d, 'ci_lo': lo, 'ci_hi': hi, 'p_onesided': p, 'n': len(idxs)}

    out = {'dataset': ds, 'pool': POOLS[ds], 'bow_seeds': bow_seeds,
           'n_dev': len(dev_s), 'n_test': len(test_s), 'n_U': len(U),
           'exact_dup_frac': n_exact / len(test_s),
           'greedy_full': greedy_full, 'paper_greedy': PAPER_GREEDY[ds],
           'lm_full_dev': lm_full_wer, 'bow_full_dev': bow_full_wer,
           'per_split_seed': per_seed_results, 'significance_seed0': sig}
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    json.dump(out, open(f'{out_dir}/lso_{ds}.json', 'w'), indent=2)
    if csv_rows:
        import csv
        with open(f'{out_dir}/perutt_{ds}.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys())); w.writeheader(); w.writerows(csv_rows)
    print(f"[saved] {out_dir}/lso_{ds}.json", flush=True)
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=list(POOLS))
    ap.add_argument('--bow_seeds', type=int, default=20)
    ap.add_argument('--gpu', default='cuda:0')
    ap.add_argument('--beams_dir', default=str(ROOT / 'work' / 'beams'))
    ap.add_argument('--out_dir', default=str(ROOT / 'work' / 'sentence_disjoint'))
    a = ap.parse_args()
    dev = torch.device(a.gpu if torch.cuda.is_available() else 'cpu')
    t0 = time.time()
    run_dataset(a.dataset, a.bow_seeds, dev, a.out_dir, beams_dir=a.beams_dir)
    print(f"[{a.dataset}] total {(time.time()-t0)/60:.1f} min", flush=True)
