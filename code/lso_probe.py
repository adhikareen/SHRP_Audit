"""
The two dev-conditioned probes of the paper (BoW frozen-emb + 4-gram LM), as
implemented for the sentence-disjoint (leave-sentences-out, LSO) evaluation, Table 5.

  - StupidBackoffLM: 4-gram LM with the backoff scheme of Brants et al. (alpha=0.4).
  - BoWReranker    : bag-of-glosses listwise reranker; emb dim 64, Adam lr=1e-3,
                     20 epochs, softmin-WER targets with tau=0.2 (Appendix A).
  - "bow-frozen-emb": embedding frozen at random init.
  - Fitting corpus  : LM, corpus features and listwise targets are all built from the
                      references of the fitting corpus (dev, or an LSO subset of dev).

Used both for the full-dev fit and for the fit restricted to U_seen sentences.
"""
import pickle, math, gc
from collections import Counter
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

SCALAR_KEYS = ['beam_logp', 'length', 'beam_rank', 'n_unique', 'ngram_match',
               'avg_freq', 'min_freq', 'self_repeat', 'lm_score']


# --------------------------------------------------------------------------- #
# WER
# --------------------------------------------------------------------------- #
def edit_distance(a, b):
    m, n = len(a), len(b)
    if m == 0: return n
    if n == 0: return m
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        cur = [i] + [0] * n
        ai = a[i - 1]
        for j in range(1, n + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1,
                         prev[j - 1] + (0 if ai == b[j - 1] else 1))
        prev = cur
    return prev[n]


def corpus_wer(refs, hyps):
    e = r = 0
    for a, b in zip(refs, hyps):
        e += edit_distance(list(a), list(b)); r += len(a)
    return 100.0 * e / max(r, 1)


def per_utt_edits(refs, hyps):
    """Return (edits[], ref_lens[]) for paired bootstrap."""
    E, R = [], []
    for a, b in zip(refs, hyps):
        E.append(edit_distance(list(a), list(b))); R.append(len(a))
    return np.array(E, dtype=float), np.array(R, dtype=float)


# --------------------------------------------------------------------------- #
# Brants stupid-backoff LM
# --------------------------------------------------------------------------- #
class StupidBackoffLM:
    def __init__(self, n_order=4, alpha=0.4):
        self.n_order = n_order; self.alpha = alpha
        self.counts = [Counter() for _ in range(n_order + 1)]
        self.total_uni = 0; self.vocab_size = 0

    def fit(self, sequences):
        for seq in sequences:
            T = len(seq)
            for k in range(1, self.n_order + 1):
                for i in range(T - k + 1):
                    self.counts[k][tuple(seq[i:i + k])] += 1
        self.total_uni = sum(self.counts[1].values())
        self.vocab_size = len(self.counts[1])

    def logp_word(self, history, word):
        h = tuple(history[-(self.n_order - 1):]); backoff_factor = 1.0
        for k in range(len(h), 0, -1):
            hh = h[-k:]; full = hh + (word,)
            c_full = self.counts[k + 1].get(full, 0)
            if c_full > 0:
                c_hh = self.counts[k].get(hh, 0)
                if c_hh > 0:
                    return math.log(backoff_factor) + math.log(c_full / c_hh)
            backoff_factor *= self.alpha
        c_w = self.counts[1].get((word,), 0)
        p_uni = max(c_w, 1) / max(self.total_uni + self.vocab_size, 1)
        return math.log(backoff_factor) + math.log(p_uni)

    def logp_seq(self, seq):
        s = 0.0
        for i, w in enumerate(seq):
            s += self.logp_word(tuple(seq[:i]), w)
        return s


# --------------------------------------------------------------------------- #
# BoW reranker (emb dim per Appendix A)
# --------------------------------------------------------------------------- #
class BoWReranker(nn.Module):
    def __init__(self, V, d_emb, scalar_dim, hidden, dropout, freeze_emb):
        super().__init__()
        self.emb = nn.Embedding(V, d_emb, padding_idx=0)
        if freeze_emb:
            self.emb.weight.requires_grad_(False)   # frozen random embedding
        self.scalar_proj = nn.Linear(scalar_dim, d_emb)
        self.head = nn.Sequential(
            nn.Linear(d_emb * 2, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, cand_ids, cand_mask, scalars):
        e = self.emb(cand_ids)
        m = cand_mask.unsqueeze(-1)
        pooled = (e * m).sum(1) / m.sum(1).clamp(min=1.0)
        s = self.scalar_proj(scalars)
        return self.head(torch.cat([pooled, s], dim=-1)).squeeze(-1)


# --------------------------------------------------------------------------- #
# Data / features
# --------------------------------------------------------------------------- #
def load_beams(path):
    d = pickle.load(open(path, 'rb'))
    return d


def build_g2id(*beam_objs):
    """Prefer the beams' own gloss_dict; else assign ids over all seen glosses."""
    for obj in beam_objs:
        if isinstance(obj, dict) and 'gloss_dict' in obj:
            gd = obj['gloss_dict']
            g2id = {k: v[0] for k, v in gd.items()}
            V = max(g2id.values()) + 2
            return g2id, V
    vocab = set()
    for obj in beam_objs:
        for s in obj['samples']:
            vocab.update(s['ref'])
            for c in s['cands']:
                vocab.update(c)
    g2id = {g: i + 1 for i, g in enumerate(sorted(vocab))}
    return g2id, len(g2id) + 2


def corpus_stats(fit_refs, g2id):
    """bigram set + gloss freq from the FITTING corpus refs."""
    bigram_set = set(); gloss_freq = Counter()
    seqs = [[g2id.get(g, 0) for g in r] for r in fit_refs]
    for seq in seqs:
        for g in seq: gloss_freq[g] += 1
        for i in range(len(seq) - 1):
            bigram_set.add((seq[i], seq[i + 1]))
    total = sum(gloss_freq.values())
    lm = StupidBackoffLM(4, 0.4); lm.fit(seqs)
    return bigram_set, gloss_freq, total, lm


def featurize(samples, g2id, bigram_set, gloss_freq, total, lm, MAXLEN=64):
    feats, wers, cand_ids = [], [], []
    for s in samples:
        K = len(s['cands'])
        f = np.zeros((K, len(SCALAR_KEYS)), dtype=np.float32)
        cid = np.zeros((K, MAXLEN), dtype=np.int64)
        for k, cand in enumerate(s['cands']):
            ids = [g2id.get(g, 0) for g in cand]
            L = min(len(ids), MAXLEN); cid[k, :L] = ids[:L]
            unique = len(set(ids))
            ngram_match = sum(1 for i in range(len(ids) - 1)
                              if (ids[i], ids[i + 1]) in bigram_set)
            freqs = [gloss_freq.get(g, 0) for g in ids] or [0]
            avg_f = float(np.mean(freqs)) / max(total, 1) * 1e6
            min_f = float(np.min(freqs)) / max(total, 1) * 1e6
            self_rep = (len(ids) - unique) / max(len(ids), 1)
            lm_s = lm.logp_seq(ids) / max(len(ids), 1) if ids else -1e6
            f[k] = [s['beam_logp'][k], len(ids), k, unique, ngram_match,
                    avg_f, min_f, self_rep, lm_s]
        feats.append(f)
        wers.append(np.asarray(s['wer'], dtype=np.float32))
        cand_ids.append(cid)
    return feats, wers, cand_ids


def normalize(feats, mu, std):
    return [(f - mu) / std for f in feats]


def fit_mu_std(fit_feats):
    allf = np.concatenate(fit_feats, axis=0)
    mu = np.median(allf, axis=0)
    std = np.std(np.clip(allf, mu - 1e3, mu + 1e3), axis=0) + 1e-6
    return mu, std


# --------------------------------------------------------------------------- #
# Probe fit / apply
# --------------------------------------------------------------------------- #
def _to_tensors(feats, cand_ids, wers, device):
    out = []
    for f, c, w in zip(feats, cand_ids, wers):
        mask = (c != 0).astype(np.float32)
        out.append((torch.from_numpy(c).to(device),
                    torch.from_numpy(mask).to(device),
                    torch.from_numpy(f).to(device),
                    torch.from_numpy(w).to(device)))
    return out


def train_one(fit_t, V, seed, device, epochs=20, d_emb=64, hidden=256,
              dropout=0.3, tau=0.2, lr=1e-3, freeze_emb=True):
    torch.manual_seed(seed); np.random.seed(seed)
    net = BoWReranker(V, d_emb, len(SCALAR_KEYS), hidden, dropout, freeze_emb).to(device)
    params = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    idx = np.arange(len(fit_t))
    for ep in range(epochs):
        net.train(); np.random.shuffle(idx)
        for i in idx:
            cid, cmask, sc, w = fit_t[i]
            logits = net(cid, cmask, sc)
            log_p = torch.log_softmax(logits, dim=0)
            target = torch.softmax(-w / tau, dim=0)
            loss = -(target * log_p).sum()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
    net.eval()
    return net


def fit_probe(fit_samples, test_samples, g2id, V, device,
              seeds=range(20), freeze_emb=True, verbose=False):
    """BoW probe fitted on fit_samples. Returns (per_seed_test_scores, ensemble_picks)."""
    fit_refs = [list(s['ref']) for s in fit_samples]
    bigram_set, gloss_freq, total, lm = corpus_stats(fit_refs, g2id)
    fit_f, fit_w, fit_c = featurize(fit_samples, g2id, bigram_set, gloss_freq, total, lm)
    test_f, test_w, test_c = featurize(test_samples, g2id, bigram_set, gloss_freq, total, lm)
    mu, std = fit_mu_std(fit_f)
    fit_f = normalize(fit_f, mu, std); test_f = normalize(test_f, mu, std)
    fit_t = _to_tensors(fit_f, fit_c, fit_w, device)
    test_t = _to_tensors(test_f, test_c, test_w, device)

    per_seed = []
    for seed in seeds:
        net = train_one(fit_t, V, seed, device, freeze_emb=freeze_emb)
        ts = []
        with torch.no_grad():
            for cid, cmask, sc, _ in test_t:
                ts.append(net(cid, cmask, sc).cpu().numpy())
        per_seed.append(ts)
        if verbose:
            picks = [int(np.argmax(x)) for x in ts]
            hyps = [test_samples[i]['cands'][p] for i, p in enumerate(picks)]
            refs = [test_samples[i]['ref'] for i in range(len(test_samples))]
            print(f"    seed {seed}: test={corpus_wer(refs, hyps):.2f}", flush=True)
        del net; gc.collect()
        if device.type == 'cuda': torch.cuda.empty_cache()

    # z-norm ensemble across seeds
    n = len(test_samples); picks = []
    Ksd = len(per_seed)
    for k in range(n):
        ss = np.stack([per_seed[c][k] for c in range(Ksd)], axis=0)
        ssn = (ss - ss.mean(1, keepdims=True)) / (ss.std(1, keepdims=True) + 1e-6)
        picks.append(int(np.argmax(ssn.mean(0))))
    return per_seed, picks


def lm_rescore(fit_samples, test_samples, g2id):
    """Standalone Brants 4-gram LM rescore (len-normalised argmax); LM fitted on fit_samples."""
    fit_refs = [[g2id.get(g, 0) for g in s['ref']] for s in fit_samples]
    lm = StupidBackoffLM(4, 0.4); lm.fit(fit_refs)
    picks = []
    for s in test_samples:
        best = -1e18; bi = 0
        for k, c in enumerate(s['cands']):
            ids = [g2id.get(g, 0) for g in c]
            lp = lm.logp_seq(ids) / max(len(ids), 1) if ids else -1e9
            if lp > best: best = lp; bi = k
        picks.append(bi)
    return picks
