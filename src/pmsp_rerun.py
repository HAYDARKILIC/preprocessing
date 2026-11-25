#!/usr/bin/env python3
"""
Re-execution of the Predictive Minimal Sufficient Preprocessing (P-MSP)
leave-one-dataset-out evaluation.

The originally released run took the fallback path: the ordinal solver was
unavailable, so the meta-model emitted no class probabilities and its
predictions coincided with baseline B1 on every instance. This script repeats
the analysis from the released per-cell artefacts with a working solver.

Inputs  : raw_results.csv, table1_dataset_characteristics.csv
Outputs : pmsp_rerun_predictions.csv, pmsp_rerun_summary.csv,
          pmsp_rerun_abstention.csv, pmsp_rerun_mcnemar.csv,
          pmsp_rerun_importance.csv
"""
import numpy as np
import pandas as pd
from mord import LogisticAT
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

LEVELS = ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8a", "P8b"]
K = len(LEVELS)
SEEDS = [42, 7, 13]
TAUS = [0.005, 0.01, 0.02]
TAU_PRIMARY = 0.01
THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60,
              0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

# ordinal a-priori sensitivity prior, as specified in the Methods
SENS = {"A2_DecTree": 0, "A3_RandForest": 0, "A4_GradBoosting_SKL": 0,
        "A6_XGBoost": 0, "A7_LightGBM": 0, "A8_CatBoost": 0,
        "A10_TabPFN": 1, "A1_Ridge_LogReg": 2, "A9_MLP": 3,
        "A5_SVM_SVR": 4, "A11_kNN": 4}


def build(tau):
    """Return the instance table: meta-features, MSP label and score profile."""
    r = pd.read_csv("raw_results.csv")
    r = r[r.fold_mean.notna()]
    t1 = pd.read_csv("table1_dataset_characteristics.csv").set_index("dataset")

    rows = []
    for (d, a), g in r.groupby(["dataset", "algo_id"]):
        s = g.set_index("pipeline")["fold_mean"].reindex(LEVELS)
        if s.notna().sum() == 0:
            continue
        best = s.max()
        # earliest available level within tolerance of the best available level
        k_star = int(np.argmax(((s >= best - tau) & s.notna()).values))
        m = t1.loc[d]
        rows.append(dict(
            dataset=d, algo_id=a, true_idx=k_star,
            log10_n=np.log10(m.n_samples), p=m.n_features,
            log10_np=np.log10(m.n_samples / max(m.n_features, 1)),
            miss=m.miss_rate_pct, cat_density=m.n_categorical / max(m.n_features, 1),
            has_hicard=float(m.n_high_card_cat > 0),
            ir=m.imbalance_ratio, n_classes=m.n_classes,
            is_cls=float(m.task == "classification"), sens=SENS[a],
            best=best, **{f"s_{L}": s[L] for L in LEVELS}))
    return pd.DataFrame(rows)


def _gap(row, p):
    """Performance forgone by the recommendation.

    If the recommended rung was not evaluable for this pair, a practitioner
    would climb to the next feasible rung, so the gap is measured there.
    """
    for j in range(p, K):
        v = row[f"s_{LEVELS[j]}"]
        if pd.notna(v):
            return float(row.best - v)
    return float("nan")


FEATS = ["log10_n", "p", "log10_np", "miss", "cat_density",
         "has_hicard", "ir", "n_classes", "is_cls", "sens"]


def regime(row):
    return (int(row.log10_n >= np.log10(5000)), int(row.miss > 5), int(row.ir >= 5))


def evaluate(df, seed):
    """One LODO sweep over all datasets, for the meta-model and five baselines."""
    rng = np.random.RandomState(seed)
    out = []
    datasets = sorted(df.dataset.unique())
    df = df.copy()
    df["regime"] = df.apply(regime, axis=1)

    for held in datasets:
        tr = df[df.dataset != held]
        te = df[df.dataset == held]
        if tr.dataset.nunique() < 3:
            continue

        sc = StandardScaler().fit(tr[FEATS].values)
        Xtr, Xte = sc.transform(tr[FEATS].values), sc.transform(te[FEATS].values)
        ytr = tr.true_idx.values

        # --- P-MSP: all-threshold ordinal logistic ---------------------------
        conf = np.full(len(te), np.nan)
        # mord.LogisticAT requires a CONTIGUOUS label range 0..C-1. The MSP
        # label set has gaps, which is what raised ValueError and triggered the
        # fallback in the originally released run. Encode, fit, decode.
        classes = np.unique(ytr)
        enc = {c: i for i, c in enumerate(classes)}
        ytr_enc = np.array([enc[v] for v in ytr])
        try:
            mdl = LogisticAT(alpha=1.0).fit(Xtr, ytr_enc)
            pred = classes[np.clip(mdl.predict(Xte).astype(int), 0, len(classes) - 1)]
            P = mdl.predict_proba(Xte)
            conf = P.max(axis=1)
        except Exception as e:
            mdl, pred = None, None
            print("   meta-model failed:", type(e).__name__, e)

        preds = {}
        if pred is not None:
            preds["P-MSP_meta"] = (np.clip(pred, 0, K - 1), conf)

        # --- baselines -------------------------------------------------------
        preds["B0_global"] = (np.full(len(te), int(np.median(ytr))), None)

        mode_by_algo = tr.groupby("algo_id").true_idx.agg(
            lambda s: int(s.value_counts().sort_index().idxmax()))
        preds["B1_algo_mode"] = (
            te.algo_id.map(mode_by_algo).fillna(int(np.median(ytr))).astype(int).values, None)

        mode_by_reg = tr.groupby("regime").true_idx.agg(
            lambda s: int(s.value_counts().sort_index().idxmax()))
        preds["B2_regime"] = (
            te.regime.map(mode_by_reg).fillna(int(np.median(ytr))).astype(int).values, None)

        kk = min(25, len(tr))
        nn = NearestNeighbors(n_neighbors=kk).fit(Xtr)
        idx = nn.kneighbors(Xte, return_distance=False)
        preds["B3_meta_knn"] = (
            np.array([int(pd.Series(ytr[i]).value_counts().sort_index().idxmax())
                      for i in idx]), None)

        # B4: oracle-anchored AutoML reference -- an upper bound by construction,
        # not a competing zero-knowledge predictor (see Methods)
        preds["B4_automl"] = (te.true_idx.values.astype(int), None)

        for meth, (pr, cf) in preds.items():
            for j, (_, row) in enumerate(te.iterrows()):
                p = int(pr[j])
                out.append(dict(
                    seed=seed, method=meth, dataset=held, algo_id=row.algo_id,
                    true_idx=int(row.true_idx), pred_idx=p,
                    abs_err=abs(p - int(row.true_idx)),
                    exact=int(p == row.true_idx),
                    off_by_one=int(abs(p - row.true_idx) <= 1),
                    confidence=(np.nan if cf is None else float(cf[j])),
                    perf_gap=_gap(row, p),
                    fit_fraction=(p + 1) / K))
    return pd.DataFrame(out)


def main():
    df = build(TAU_PRIMARY)
    print(f"instances: {len(df)}  datasets: {df.dataset.nunique()}  "
          f"algorithms: {df.algo_id.nunique()}")
    base = df.true_idx.value_counts(normalize=True).sort_index()
    print("label distribution (%):",
          {LEVELS[i]: round(100 * v, 2) for i, v in base.items()})
    print("majority-class base rate:", round(base.max(), 4))

    allp = pd.concat([evaluate(df, s) for s in SEEDS], ignore_index=True)
    allp.to_csv("pmsp_rerun_predictions.csv", index=False)

    per_seed = allp.groupby(["method", "seed"]).agg(
        exact=("exact", "mean"), off1=("off_by_one", "mean"),
        mae=("abs_err", "mean"), gap=("perf_gap", "mean"),
        fit=("fit_fraction", "mean")).reset_index()
    summ = per_seed.groupby("method").agg(["mean", "std"])
    summ.columns = ["_".join(c) for c in summ.columns]
    summ = summ.round(4).reset_index()
    summ.to_csv("pmsp_rerun_summary.csv", index=False)
    print("\n=== LODO summary (mean over 3 seeds) ===")
    print(summ[["method", "exact_mean", "exact_std", "off1_mean",
                "mae_mean", "gap_mean", "fit_mean"]].to_string(index=False))

    # --- McNemar of P-MSP against each baseline --------------------------------
    rows = []
    piv = allp.pivot_table(index=["seed", "dataset", "algo_id"],
                           columns="method", values="exact")
    for b in ["B0_global", "B1_algo_mode", "B2_regime", "B3_meta_knn", "B4_automl"]:
        if "P-MSP_meta" not in piv or b not in piv:
            continue
        a_, b_ = piv["P-MSP_meta"], piv[b]
        n01 = int(((a_ == 1) & (b_ == 0)).sum())
        n10 = int(((a_ == 0) & (b_ == 1)).sum())
        stat = 0.0 if (n01 + n10) == 0 else (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
        from scipy.stats import chi2
        p = 1.0 if (n01 + n10) == 0 else float(chi2.sf(stat, 1))
        rows.append(dict(comparison=b, pmsp_wins=n01, base_wins=n10,
                         chi2=round(stat, 3), p=p, pmsp_better=bool(n01 > n10 and p < .05)))
    for x, y_ in [("B3_meta_knn", "B1_algo_mode")]:
        a_, b_ = piv[x], piv[y_]
        n01 = int(((a_ == 1) & (b_ == 0)).sum()); n10 = int(((a_ == 0) & (b_ == 1)).sum())
        from scipy.stats import chi2 as _c
        stat = 0.0 if (n01+n10)==0 else (abs(n01-n10)-1)**2/(n01+n10)
        rows.append(dict(comparison=f"{x} vs {y_}", pmsp_wins=n01, base_wins=n10,
                         chi2=round(stat,3), p=float(_c.sf(stat,1)),
                         pmsp_better=bool(n01>n10 and _c.sf(stat,1)<.05)))
    mc = pd.DataFrame(rows)
    mc.to_csv("pmsp_rerun_mcnemar.csv", index=False)
    print("\n=== McNemar ===")
    print(mc.to_string(index=False))

    # --- abstention operating curve -------------------------------------------
    m = allp[(allp.method == "P-MSP_meta") & allp.confidence.notna()].copy()
    print(f"\nfinite confidence values: {len(m)} of "
          f"{(allp.method=='P-MSP_meta').sum()}")
    if len(m):
        term = df.set_index(["dataset", "algo_id"]).apply(
            lambda r: _gap(r, K - 1), axis=1)
        m["term_gap"] = list(zip(m.dataset, m.algo_id))
        m["term_gap"] = m.term_gap.map(term)
        rows = []
        for t in THRESHOLDS:
            ans = (m.confidence >= t).values
            gap = np.where(ans, m.perf_gap.values, m.term_gap.values)
            fit = np.where(ans, m.fit_fraction.values, 1.0)
            rows.append(dict(threshold=t, coverage=round(ans.mean(), 4),
                             acc_on_covered=round(m.exact[ans].mean(), 4) if ans.any() else np.nan,
                             mean_gap=round(float(np.nanmean(gap)), 4),
                             mean_fit_fraction=round(float(np.mean(fit)), 4)))
        ab = pd.DataFrame(rows)
        ab.to_csv("pmsp_rerun_abstention.csv", index=False)
        print("\n=== abstention operating curve ===")
        print(ab.to_string(index=False))

    # --- permutation importance on a single full-data fit ----------------------
    sc = StandardScaler().fit(df[FEATS].values)
    X, y = sc.transform(df[FEATS].values), df.true_idx.values
    classes = np.unique(y)
    enc = {c: i for i, c in enumerate(classes)}
    y_enc = np.array([enc[v] for v in y])
    mdl = LogisticAT(alpha=1.0).fit(X, y_enc)
    base_acc = (mdl.predict(X) == y_enc).mean()
    rng = np.random.RandomState(42)
    imp = []
    for i, f in enumerate(FEATS):
        drops = []
        for _ in range(20):
            Xp = X.copy()
            rng.shuffle(Xp[:, i])
            drops.append(base_acc - (mdl.predict(Xp) == y_enc).mean())
        imp.append((f, float(np.mean(drops))))
    ims = pd.DataFrame(imp, columns=["feature", "importance"])
    ims["importance"] = ims.importance.clip(lower=0)
    tot = ims.importance.sum()
    ims["normalised"] = (ims.importance / tot).round(4) if tot > 0 else 0.0
    ims = ims.sort_values("normalised", ascending=False)
    ims.to_csv("pmsp_rerun_importance.csv", index=False)
    print("\n=== permutation importance ===")
    print(ims.to_string(index=False))


if __name__ == "__main__":
    main()
