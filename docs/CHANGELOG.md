# Changes made after the original benchmark run

## Fix — ordinal solver rejected non-contiguous labels

**Symptom.** The predictive component reported predictions identical to baseline
B1 on all 2,889 leave-one-dataset-out instances, and produced no confidence
values at all. Downstream, the confidence-aware abstention analysis could not run.

**Cause.** `mord.LogisticAT` requires the training labels to occupy a contiguous
range 0..C−1. Minimal-sufficiency labels are a *subset* of the nine ladder rungs,
with gaps wherever a level is never minimal — in this suite P5 never is. The fit
therefore raised:

```
ValueError: Values in y must be [0 1 2 3 4 5 6 7], instead got [0 1 2 3 5 6 7 8]
```

**Why it survived a full run.** The surrounding handler was written for a missing
optional dependency, but it caught a *rejected input*, substituted a strong
baseline and let the run complete. The output was indistinguishable from a
legitimate result, because a strong baseline is by construction plausible.

**Fix** (`notebooks/01_pipeline.ipynb`, Cell 8d):

1. encode labels to a contiguous range before fitting, decode predictions after;
2. call `predict_proba` so the abstention layer receives confidences;
3. **re-raise instead of falling back** — the meta-model is the object under test,
   so a failure must stop the analysis rather than disguise itself.

**Effect on the findings.**

| Quantity | Before | After |
|---|---:|---:|
| P-MSP exact accuracy | 0.7092 (= B1) | 0.6490 |
| B3 meta-*k*NN exact accuracy | 0.7103 | 0.7799 |
| Finite confidences | 0 of 2,889 | 2,889 of 2,889 |
| Abstention analysis | not evaluable | full operating curve |
| Verdict | inconclusive | ordinal model **refuted**; retrieval **succeeds** |

The correction produced a stronger result than the original non-finding: a clear
refutation of the proposed model, and a positive result for retrieval that the
analysis did not previously have.

---

## Analyses added during review

| Cell | Analysis | Why |
|---|---|---|
| 19 | Portfolio-composition sensitivity | The headline ratio depends on which learners are admitted; dropping the weakest takes it from 3.6× to 2.9×, and dropping two takes it to 1.5× with an interval spanning zero |
| 20 | ROC-AUC vs balanced accuracy at the class-weighting rung | The primary metric is threshold-free and cannot register reweighting; the reported null for that rung is a metric artefact (+0.0042, *p* = .17 on AUC against +0.0561, *p* < .001 on balanced accuracy) |
| 21 | Diagnosis of the three empty datasets | The original run recorded them as failures without a cause |
| 22 | Verification of the headline quantities | So the pipeline self-checks at the end of a run |

---

## Robustness layers delivered

Two layers were specified in the design but never produced by the original run,
because a meta-model that always returns the baseline makes them vacuous:

- a sweep over the tolerance τ ∈ {0.005, 0.01, 0.02} — the ordering is invariant;
- re-evaluation restricted to the 891 of 962 instances (92.6 %) whose label is
  identical under all three cross-validation seeds — the structure is unchanged,
  so label noise explains neither the refutation nor the retrieval advantage.
