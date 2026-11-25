# Reproduction guide

Three notebooks, three levels of effort.

| Notebook | Needs | Time | Purpose |
|---|---|---|---|
| `00_quickstart.ipynb` | CPU | ~30 s | The headline results, eight cells |
| `02_analysis_and_verification.ipynb` | CPU | ~2 min | Every analysis, plus 273 assertions |
| `01_pipeline.ipynb` | CUDA GPU | many hours | The full factorial, from raw datasets |

Most readers want the middle one.

---

## The two-minute path

```bash
pip install -r requirements.txt
cd notebooks
jupyter notebook 02_analysis_and_verification.ipynb
```

Run all cells. The notebook loads `../data/raw_results.csv`, re-derives every
quantity reported in the paper, and asserts that each matches. It ends with a
summary table and raises if anything fails.

Manuscript-text checks (Layer B, 28 assertions) are skipped automatically here,
because the `.tex` is not in this repository. The notebook says so and continues.
To enable them, place `manuscript.tex` beside the notebook.

---

## The full path

`01_pipeline.ipynb` regenerates `data/raw_results.csv` from scratch. It requires:

- a CUDA GPU (the study used a single NVIDIA A100 80 GB) and the GPU-only
  dispatch policy described in the notebook;
- network access to OpenML, to retrieve the 95 benchmark datasets;
- the pinned dependency set in `data/requirements_pinned.txt`;
- several hours, with checkpointing every 25 cells so it can be resumed.

Cost: roughly 3.8 × 10⁵ model fits across the main factorial, two replication
seeds, the variance audit and the tuned sub-experiment.

Determinism: a single global seed (42) governs fold assignment, subsampling,
estimator initialisation and every bootstrap. GPU kernels are not bitwise
reproducible, so expect agreement to within the tolerance reported by the
implementation-equivalence audit (below 1 % on all but one audited pair).

---

## Adopting the fix without re-running the factorial

The bug documented in `CHANGELOG.md` affects only the predictive component. The
factorial output is unaffected, so:

1. Open `01_pipeline.ipynb`
2. Run Cells 1–3 (environment and globals)
3. Run Cell 8b onward

Expected after Cell 8d:

| Quantity | Value |
|---|---:|
| Instances / datasets | 963 / 92 |
| Majority-class base rate | 0.7092 |
| P-MSP exact accuracy | 0.6490 |
| B1 algorithm-conditional mode | 0.7092 |
| B3 meta-*k*NN | 0.7799 |
| Finite confidence values | 2889 of 2889 |

The last row is the tell: zero there means the fix has not taken effect and the
silent fallback has returned.

---

## Effective sample sizes

Denominators differ across tables, for a reason worth stating once:

| Stage | N |
|---|---:|
| Datasets loaded | 95 |
| Contributing at least one valid cell | 92 |
| Complete coverage of all nine ladder levels | 88 |
| Complete coverage of the ten Stage-A algorithms | 91 |
| Both conditions — the primary contrast | 87 |

Three datasets produced no usable cell. `notebooks/00_quickstart.ipynb` §6
diagnoses them: in two the rarest class holds fewer members than the ten folds,
so the stratified splitter raises in the outer loop; in the third the numeric
branch received a single column the transformer could not emit as a 2-D matrix.
Both causes are properties of the harness rather than the data, and both are
recoverable.
