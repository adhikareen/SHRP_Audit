# SHRP_Audit

Audit code for

> Nirmal Adhikari and Wookey Lee. *Auditing Dev-Fitted Sequence Selection in CSLR: A Cross-Benchmark Diagnostic.* Findings of EACL 2027.

The paper defines SHRP (Sentence-Held-out Reporting Protocol), a reporting standard for sequence
selection in continuous sign language recognition. This repository holds the audit code behind the
paper. It is not a new model. It covers:

- the dev/test overlap diagnostics;
- the two dev-conditioned probes (a 4-gram LM and a bag-of-glosses reranker) and the LM-fitting code;
- cross-architecture MBR;
- the paired bootstrap;
- the sentence-disjoint (leave-sentences-out, LSO) evaluation.

"Dev-conditioned" means fitted on dev labels. "Train-conditioned" means fitted on train labels.

```
code/
  attach_refs.py          rebuild the references from your dataset copies into work/
  nn_edit_distance.py     dev/test exact duplicates and nearest-neighbour edit-distance buckets
  ngram_coverage.py       test n-gram coverage by dev and by train
  neural_utility_mbr.py   MBR over a pool of beams (edit distance, BLEU, chrF) with paired bootstrap
  paired_bootstrap.py     paired bootstrap from per-utterance picks (CTM) and references (STM)
  lso_probe.py            the two probes: 4-gram LM, BoW frozen-embedding reranker
  lso_run.py              sentence-disjoint evaluation
data/
  beams/                  K=50 CTC beam candidates and beam scores, dev and test
  picks/                  per-utterance greedy and MBR hypotheses, CSL-Daily and PHX-2014T test
  diagnostics/            outputs of nn_edit_distance.py and neural_utility_mbr.py; nn_dist_*.npy
  sentence_disjoint/      lso_<dataset>.json written by lso_run.py
  ref_checksums.json      SHA-256 digests of the references used in the paper
figures_src/
  make_tripwire_figure.py Figure 1
```

## Requirements

Python 3.10, `pip install -r requirements.txt`. Only `lso_run.py` uses a GPU (it also runs on CPU).

## Datasets and references

This repository contains model outputs and statistics only. It contains no dataset annotations.
Obtain CSL-Daily, RWTH-PHOENIX-Weather 2014 and RWTH-PHOENIX-Weather 2014 T from their official
sources (links under Licences), then run from the repository root:

```
python code/attach_refs.py \
    --csl-daily    <CSL-Daily>/sentence_label \
    --phoenix2014  <phoenix-2014-multisigner> \
    --phoenix2014t <PHOENIX-2014-T>
```

It reads `video_map.txt` and `split_1.txt` (CSL-Daily) and `annotations/manual/*.corpus.csv`
(PHOENIX), and writes `work/beams/*.pkl` (the beams with `ref`, per-candidate `wer` and `gloss_dict`)
and `work/refs/*_test.stm`. Each output is checked against `data/ref_checksums.json`. `work/` is
git-ignored; all commands below read from it. Any subset of the three datasets can be given.

CSL-Daily references are the gloss annotation split on whitespace. For the two PHOENIX datasets,
runs of single-letter tokens are joined with `+` and adjacent repeats of a gloss are merged.

## Paper item to code

Commands are run from the repository root after `attach_refs.py`.

| Paper item | Command | Status |
|---|---|---|
| Table 1, greedy-best and cross-architecture MBR | `python code/paired_bootstrap.py` (CSL-Daily, PHX-2014T); PHX-2014: see "PHX cross-arch MBR" | included |
| Table 1, dev-conditioned BoW and dev-LM rows | | not included (*) |
| Table 1, train-conditioned BoW, train-LM and ROVER rows | | not included (*) |
| Table 2, test exact-duplicate rate | `python code/nn_edit_distance.py --dataset {csl,phx,phxT}` | included |
| Table 2, mean NN edit distance | `data/diagnostics/nn_dist_*.npy` | arrays included |
| Table 2, 4-gram coverage | `python code/ngram_coverage.py` with the dataset arguments (see "N-gram coverage") | included |
| Table 2, sentence-ID overlap | | not included (*) |
| Table 3 | `python code/nn_edit_distance.py --dataset {csl,phx,phxT}` | included |
| Table 4 | | not included (*) |
| Table 5 and the PHX-2014 / PHX-2014T paragraphs of the same section | `python code/lso_run.py --dataset {csl,phx,phxT}` | included |
| Table 6, MBR rows | `python code/paired_bootstrap.py` (CSL-Daily, PHX-2014T); PHX-2014: see "PHX cross-arch MBR" | included |
| Table 6, BoW rows | | not included (*) |
| Table 7 | rows repeat Table 1 | same-architecture MBR row not included |
| Table 8, Table 10 | | no code (text tables) |
| Table 9 | | not included (*) |
| Table 11 | `python code/ngram_coverage.py` with the dataset arguments (see "N-gram coverage") | included |
| Figure 1 | `python figures_src/make_tripwire_figure.py` | included |
| Figure 2 | | not included (*) |
| Discussion, neural-utility MBR | see "Neural-utility MBR" | included |

(*) Scripts for these items will be added to this repository.

## Paired bootstrap CI (CSL + PHX-T MBR rows)

```
python code/paired_bootstrap.py
```

Reproduces the paper's Table 6 MBR rows: CSL delta = +0.23, CI [-0.33, +0.81];
PHX-T delta = -1.36, CI [-2.05, -0.67]. It reads the released picks in `data/picks/`.

## Sentence overlap and NN edit-distance buckets

```
python code/nn_edit_distance.py --dataset csl
python code/nn_edit_distance.py --dataset phx
python code/nn_edit_distance.py --dataset phxT
```

CSL test exact-dup rate comes out at 99.66%. The Table 3 buckets are `nn_dist_test_to_dev_buckets`
in the output JSON (`work/diagnostics/overlap_<dataset>.json`).

`data/diagnostics/nn_dist_*.npy` hold one value per test utterance: the edit distance to the nearest
dev reference divided by the test reference length, capped at 1. Figure 1(b) reads them.

## N-gram coverage

```
python code/ngram_coverage.py \
    --csl-daily    <CSL-Daily>/sentence_label \
    --phoenix2014  <phoenix-2014-multisigner> \
    --phoenix2014t <PHOENIX-2014-T>
```

Prints the share of distinct test n-grams (n = 1..4) that also occur in dev and in train. It reads
the dataset annotation files directly and does not need `work/`.

## PHX cross-arch MBR (the 17.97 result)

```
B=work/beams
python code/neural_utility_mbr.py --dataset phx --source rich \
    --pool-pickle sg0=$B/phx_sg_seed0_test.pkl \
    --pool-pickle sg1=$B/phx_sg_seed1_test.pkl \
    --pool-pickle sg2=$B/phx_sg_seed2_test.pkl \
    --pool-pickle sg3=$B/phx_sg_seed3_test.pkl \
    --pool-pickle cn=$B/phx_corrnet_test.pkl  \
    --K 50 --utilities editdist \
    --out work/diagnostics/phx_xarch_mbr.json
```

This pipeline scores with raw edit-distance, which gives ~18.2 abs WER on PHX-2014
test (delta -1.36 vs the best single backbone, p < 0.001). The paper's 17.97
adds an sclite normalisation pass over the same picks; the sclite gain is ~0.3
abs and does not change the bootstrap delta sign or significance.

## Neural-utility MBR (editdist vs BLEU vs chrF)

```
python code/neural_utility_mbr.py --dataset csl --source rich \
    --pool-pickle teacher=work/beams/csl_combined_test.pkl \
    --K 50 --utilities editdist bleu chrf \
    --out work/diagnostics/neural_utility_csl.json
```

Swap `csl_combined_test.pkl` for `phx_combined_test.pkl` / `phxT_combined_test.pkl` for the other
two datasets. The head-to-head comparison with edit-distance MBR is under `vs_editdist_mbr` in the
output. `data/diagnostics/neural_utility_phx_rich.json` is the five-pickle PHX-2014 pool of the
previous section with `--K 125 --utilities editdist bleu chrf`.

## Sentence-disjoint evaluation

```
python code/lso_run.py --dataset csl
python code/lso_run.py --dataset phx
python code/lso_run.py --dataset phxT
```

Each Table 5 cell is the mean over the three split seeds of the matching field of `per_split_seed`
in `lso_csl.json` (`greedy_*`, `devLM_*`, `BoW_*` with `full`, `seen`, `unseen`). The seed-0 deltas
and confidence intervals are under `significance_seed0`. The experiment uses the beam of one
SignGraph backbone per benchmark and its own implementation of the two probes (`lso_probe.py`), as
described in Appendix A of the paper. `lso_run.py` also writes per-utterance edit counts
(`perutt_<dataset>.csv`); these are not released.

## Figure 1

```
python figures_src/make_tripwire_figure.py
```

Writes `figures_src/out/tripwire_figure.{pdf,png}`.

## Data files

`data/beams/<pool>_{dev,test}.pkl`:

```python
{"split": str, "beam_width": 50,
 "samples": [{"fileid": str, "cands": [[str, ...]] * 50, "beam_logp": float32[50]}, ...]}
```

| Pool | Content |
|---|---|
| `csl_sg` | CSL-Daily, one SignGraph backbone; used by `lso_run.py` |
| `csl_combined` | CSL-Daily pool used for the neural-utility comparison and the overlap diagnostics |
| `phx_combined`, `phxT_combined` | PHX-2014 / PHX-2014T, one SignGraph backbone; used by `lso_run.py`, the overlap diagnostics and the neural-utility comparison |
| `phx_sg_seed0..3`, `phx_corrnet` | PHX-2014, four SignGraph seeds and CorrNet; the MBR pool |

`data/picks/*.ctm`: one line per hypothesis gloss, `fileid 1 signer time gloss`.

The JSON files in `data/diagnostics/` and `data/sentence_disjoint/` were written by the commands
above on the released beams.

## Licences

- Code: MIT, see `LICENSE`.
- Released model outputs and statistics (`data/`): CC BY-NC-SA 4.0, see `data/LICENSE`.
- The underlying datasets are not redistributed here and are governed by their own licences. Obtain
  them from the official sources:
  - CSL-Daily (research-only agreement with USTC): https://ustc-slr.github.io/datasets/2021_csl_daily/
  - RWTH-PHOENIX-Weather 2014 (non-commercial): https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX/
  - RWTH-PHOENIX-Weather 2014 T (non-commercial): https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/

## Citation

```bibtex
@inproceedings{adhikari-lee-2027-auditing,
  title = {Auditing Dev-Fitted Sequence Selection in {CSLR}: A Cross-Benchmark Diagnostic},
  author = {Adhikari, Nirmal and Lee, Wookey},
  booktitle = {Findings of the Association for Computational Linguistics: EACL 2027},
  year = {2027}
}
```
