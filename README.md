# Preprocessing effort vs. learner choice — code and results

Replication code for a factorial study that crosses a **nine-level monotone
preprocessing ladder** with **eleven algorithm families** over **95 curated
tabular datasets**, evaluates all **9,405 cells** under matched ten-fold
cross-validation, and estimates the difference between the two factors' effect
sizes with a dataset-level bootstrap.

This repository contains **code, data and figures only**. The manuscript is not
included.

---

## For reviewers: start here

Everything reported in the paper can be checked in about two minutes on a
laptop. No GPU is needed — the expensive factorial has already run, and its
per-cell output is in `data/`.

```bash
git clone https://github.com/HAYDARKILIC/preprocessing.git
cd preprocessing
pip install -r requirements.txt
jupyter notebook notebooks/02_analysis_and_verification.ipynb   # Run All
```

The notebook ends with a block of assertions and raises if any fails. In this
repository expect **`245 / 245 passed`** — the full suite is 273, and the 28
manuscript-text checks are skipped automatically because the `.tex` is not
included. The notebook says so and continues.

| Layer | Question | Checks | In this repo |
|---|---|---:|---|
| A | Does the released data reproduce every reported number? | 234 | runs |
| B | Does that number appear in the manuscript source? | 28 | skipped (no `.tex`) |
| C | Is each derived ratio consistent with its own inputs? | 11 | runs |

If you would rather see the headline results without the assertions, run
`notebooks/00_quickstart.ipynb` — eight cells, one screen of output.

---

## What the study found

| | |
|---|---|
| Preprocessing effect | Kendall's *W* = 0.1177 |
| Algorithm effect | Kendall's *W* = 0.4580 |
| Contrast | Δ = −0.3403, 95 % CI [−0.4239, −0.2262] |

Learner choice dominates the marginal comparison. Three findings qualify that:

1. **Almost the entire preprocessing effect sits at one rung** — the introduction
   of feature scaling (P3 → P4, mean +0.0257, *p* < .001). The other seven
   transitions contribute nothing detectable in aggregate.
2. **Sensitivity varies roughly sixfold across learners**, and for SVM and *k*-NN
   the within-algorithm preprocessing effect (0.555 and 0.476) *exceeds* the
   pooled algorithm effect, reversing the ordering once the learner is fixed.
3. **The magnitude is fragile to portfolio composition.** Dropping the
   unregularised decision tree takes the ratio from 3.6× to 2.9×; dropping the two
   weakest learners takes it to 1.5×, with an interval spanning zero. The sign is
   robust across every specification; the size is not.

---

## Repository layout

```
notebooks/
  00_quickstart.ipynb                 8 cells — the results, nothing else
  01_pipeline.ipynb                   the full benchmark pipeline (needs a GPU)
  02_analysis_and_verification.ipynb  analyses + assertions (no GPU)
src/
  pmsp_rerun.py                       corrected predictive-MSP analysis
  patch_preprocessing.py              regenerates the patched pipeline
  check_bibliography.py               bibliography consistency checker
data/      12 files    per-cell results, three seeds, tuned sub-experiment,
                       variance audit, dataset descriptors and fingerprints
tables/    43 files    every result table reported in the paper, as CSV and as
                       LaTeX: omnibus tests, effect sizes, Nemenyi matrix,
                       multiplicity correction, MSP policy by regime, tuned vs
                       default, power, LODO sensitivity, predictive-MSP results,
                       GPU and implementation-equivalence audits
figures/   24 files    all 12 figures, vector (PDF) and raster (PNG)
docs/
  REPRODUCTION.md                     what runs where, and how long it takes
  CHANGELOG.md                        the bug found during review, and its fix
  DATA_DICTIONARY.md                  every column of every file
  PUBLISHING.md                       anonymity, GitHub, Zenodo DOI
```

Every table and figure that appears in the article is here, in the form it was
generated. The article itself is not.

---

## Key data files

| File | Rows | Contents |
|---|---:|---|
| `raw_results.csv` | 9,405 | One row per (dataset, algorithm, preprocessing level): fold mean, fold-level scores, dispersion, backend, status |
| `results_seed7.csv`, `results_seed13.csv` | 9,405 each | Independent cross-validation replications |
| `table1_dataset_characteristics.csv` | 95 | Descriptors of every benchmark problem |
| `tuned_results.csv` | 675 | Nested cross-validated tuned sub-experiment |
| `variance_stability_5rep.csv` | 693 | 5×10-fold repeated cross-validation audit |
| `dataset_fingerprints.csv` | 95 | SHA-256 of the canonicalised feature and target matrices |

No third-party dataset is redistributed. Benchmark problems are identified by
permanent OpenML identifier and served revision, with a fingerprint so identity
can be confirmed after retrieval from the original source.

---

## A bug found during review

The predictive component originally reported results identical to a baseline on
all 2,889 instances, with no confidence values. The cause: `mord.LogisticAT`
requires labels in a contiguous range 0..C−1, minimal-sufficiency labels are a
gapped subset of the nine rungs, and the resulting `ValueError` was caught by a
handler that substituted a baseline and let the run complete.

Fixed in `notebooks/01_pipeline.ipynb` (Cell 8d): labels are encoded to a
contiguous range before fitting and decoded after, and the handler now re-raises
instead of falling back. See `docs/CHANGELOG.md`.

The correction changed the finding rather than merely repairing the code: the
proposed ordinal model is **refuted** (0.6490 against a 0.7092 base rate), while
nearest-dataset retrieval **beats** the same baseline (0.7799). Both are reported.

---

## Requirements

Python 3.10+, and for `notebooks/01_pipeline.ipynb` a CUDA GPU. The analysis and
verification notebooks need neither.

```bash
pip install -r requirements.txt
```

`data/environment_lock.yml` records the exact versions resolved when the
factorial ran.

---

## Licence

Code MIT, data CC BY 4.0. See `LICENSE`.
