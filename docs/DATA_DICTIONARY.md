# Data dictionary

Every file in `data/`, its size, and its columns.

| File | Rows | Cols | First columns |
|---|---:|---:|---|
| `dataset_fingerprints.csv` | 95 | 11 | dataset, task, n_samples, n_features, miss_rate_pct, y_min, y_max, sha256 … |
| `moderator_data.csv` | 92 | 13 | dataset, eps2_prep, IR, IR_max_min, IR_max_median, cat_density, log10_n, miss_rate … |
| `multiseed_delta_eps2.csv` | 4 | 17 | seed, eps2_prep, eps2_algo_stageA, delta_eps2, n_prep, k_prep, n_algo, k_algo … |
| `pmsp_lodo_predictions.csv` | 17,334 | 13 | seed, method, model, dataset, algo_id, true_idx, pred_idx, abs_err … |
| `raw_results.csv` | 9,405 | 19 | dataset, task, algo_id, pipeline, ir, n, fold_mean, fold_std … |
| `results_seed13.csv` | 9,405 | 19 | dataset, task, algo_id, pipeline, ir, n, fold_mean, fold_std … |
| `results_seed7.csv` | 9,405 | 19 | dataset, task, algo_id, pipeline, ir, n, fold_mean, fold_std … |
| `table1_dataset_characteristics.csv` | 95 | 14 | dataset, n_samples, n_features, task, n_classes, imbalance_ratio, miss_rate_pct, n_numeric … |
| `table4_pivot.csv` | 95 | 13 | dataset, P1, P2, P3, P4, P5, P6, P7 … |
| `table6_algo_interaction.csv` | 11 | 6 | algo_id, P1, P4, P8a, P8b, delta |
| `table8_loo_sensitivity.csv` | 87 | 8 | dropped_dataset, eps2_prep_remaining, eps2_algo_remaining, delta_eps2, ci_lo_95, ci_hi_95, ci_excludes_zero, sign_flip_vs_full |
| `tuned_results.csv` | 675 | 10 | dataset, algo_id, pipeline, task, n_used, tuned_mean, tuned_std, n_folds … |
| `variance_stability_5rep.csv` | 693 | 7 | dataset, algo_id, pipeline, rep_mean, rep_std, n_folds, skip_reason |

## The central table: `raw_results.csv`

One row per (dataset, algorithm, preprocessing level) cell of the factorial.

| Column | Meaning |
|---|---|
| `dataset`, `algo_id`, `pipeline` | The cell's coordinates in the design |
| `task` | `classification` or `regression` |
| `fold_mean` | Mean score across the ten folds; **NaN marks a cell that did not produce a score** |
| `fold_std` | Standard deviation across folds |
| `fold_scores` | The ten individual fold scores |
| `bacc_mean` | Balanced accuracy, secondary metric (classification only) |
| `n_used` | Rows actually used, after any scope cap |
| `gpu_backend` | Implementation that executed the fit |
| `skip_reason` | Why the cell has no score, where a reason was recorded |
| `multi_output` | Whether the target is multi-output (no dataset in this suite is) |

Scores are macro one-vs-rest ROC-AUC for classification and R² for regression,
so `fold_mean` is not comparable across tasks in absolute terms. Every analysis
ranks within dataset before aggregating, for exactly that reason.

## Preprocessing levels

| Level | Adds or substitutes |
|---|---|
| `P1` | Numeric branch only: mean imputation; categorical columns dropped |
| `P2` | + one-hot encoding of all categorical features |
| `P3` | + ordinal encoding for high-cardinality categoricals (>10 levels) |
| `P4` | + standardisation of numeric features |
| `P5` | Standardisation → robust scaling (median / IQR) |
| `P6` | Mean → median imputation |
| `P7` | + interquartile clipping at 3×IQR |
| `P8a` | + variance thresholding at 0.01 |
| `P8b` | + balanced class weights when IR ≥ 5 |

Consecutive levels differ by exactly one transformation, so each adjacent
contrast identifies a single preprocessing decision.

## Algorithm identifiers

`A1_Ridge_LogReg` · `A2_DecTree` · `A3_RandForest` · `A4_GradBoosting_SKL` ·
`A5_SVM_SVR` · `A6_XGBoost` · `A7_LightGBM` · `A8_CatBoost` · `A9_MLP` ·
`A10_TabPFN` · `A11_kNN`

`A10_TabPFN` is evaluated only within its operating scope (n ≤ 3000, p ≤ 100),
so it is excluded from the primary Stage-A algorithm analysis.

## A note on `pmsp_lodo_predictions.csv`

This file is from the **original** run and is superseded: it was produced by the
fallback path documented in `CHANGELOG.md`, so its P-MSP rows are baseline
predictions rather than model output. It is retained for transparency. The
corrected analysis is in `notebooks/02_analysis_and_verification.ipynb`.

---

## `tables/` — every result table in the paper

CSV for analysis, LaTeX for the manuscript. Both are generated, not typed.

| File | Rows | Cols | First columns |
|---|---:|---:|---|
| `automl_anchor_flaml.csv` | 6 | 9 | dataset, algo_id, task, flaml_level, oracle_level, flaml_idx … |
| `bootstrap_BCa_vs_percentile.csv` | 2 | 6 | factor, W, ci_lo_percentile, ci_hi_percentile, ci_lo_BCa, ci_hi_BCa |
| `cuml_sklearn_equivalence_audit.csv` | 12 | 8 | algo, dataset, task, score_gpu_cuml, score_cpu_sklearn, abs_diff … |
| `delta_eps2_stageA_vs_stageB.csv` | 2 | 11 | stage, k_algo, N_datasets, eps2_prep, eps2_algo, delta_eps2 … |
| `fig4_kendall_w_values.csv` | 11 | 89 | algorithm, CalHous, Diabet, BreastC, Energy, Conc … |
| `friedman_iman_davenport.csv` | 2 | 7 | factor, chi2_F, p_chi2, F_Iman_Davenport, p_F, N … |
| `gpu_audit.csv` | 9,405 | 8 | dataset, algo_id, pipeline, task, gpu_policy, gpu_backend … |
| `internal_validity_checks.csv` | 5 | 2 | check, pass_rate |
| `interpretation_caveats.csv` | 3 | 2 | id, statement |
| `msp_policy_by_regime.csv` | 53 | 5 | algo_id, short_name, regime, n_datasets, recommended_pipeline_tau1% |
| `multiple_comparison_correction.csv` | 13 | 6 | Test, p_raw, q_Holm, q_BH, sig_Holm_0.05, sig_BH_0.05 |
| `multiseed_msp_stability.csv` | 11 | 7 | algo_id, short_name, 7, 13, 42, msp_modal … |
| `nemenyi_preprocessing.csv` | 9 | 10 | pipeline, P1, P2, P3, P4, P5 … |
| `pmsp_rerun_abstention.csv` | 14 | 5 | threshold, coverage, acc_on_covered, mean_gap, mean_fit_fraction |
| `pmsp_rerun_importance.csv` | 10 | 3 | feature, importance, normalised |
| `pmsp_rerun_mcnemar.csv` | 6 | 6 | comparison, pmsp_wins, base_wins, chi2, p, pmsp_better |
| `pmsp_rerun_predictions.csv` | 17,334 | 12 | seed, method, dataset, algo_id, true_idx, pred_idx … |
| `pmsp_rerun_summary.csv` | 6 | 13 | method, seed_mean, seed_std, exact_mean, exact_std, off1_mean … |
| `run_summary_main_experiment.csv` | 40 | 2 | metric, value |
| `table10_pmsp_lodo_performance.csv` | 6 | 13 | method, exact_acc, exact_acc_CIlo, exact_acc_CIhi, off_by_one_acc, level_MAE … , status |
| `table10b_pmsp_vs_external_baselines.csv` | 3 | 6 | vs, focus_only_wins, other_only_wins, stat, p, focus_better |
| `table12_pmsp_feature_importance.csv` | 10 | 2 | feature, importance |
| `table1_dataset_characteristics.csv` | 95 | 14 | dataset, n_samples, n_features, task, n_classes, imbalance_ratio … |
| `table2_algorithm_specification.csv` | 11 | 7 | algo_id, short_name, family, backend_cls, hyperparameters, prep_sensitivity_expected … |
| `table3_pipeline_ablation_map.csv` | 9 | 9 | pipeline, name, imputation, scaling, encoding, outlier_handling … |
| `table4_pivot.csv` | 95 | 13 | dataset, P1, P2, P3, P4, P5 … |
| `table4b_minimal_sufficient_preprocessing.csv` | 11 | 8 | algo_id, short_name, msp_tau0.5%, msp_tau1%, msp_tau2%, n_datasets … |
| `table6_algo_interaction.csv` | 11 | 6 | algo_id, P1, P4, P8a, P8b, delta |
| `table7_default_vs_tuned.csv` | 2 | 9 | Regime, W_prep, W_algo, delta_W, CI_lo, CI_hi … |
| `table8_loo_sensitivity.csv` | 87 | 8 | dropped_dataset, eps2_prep_remaining, eps2_algo_remaining, delta_eps2, ci_lo_95, ci_hi_95 … |
| `table9_empirical_power.csv` | 2 | 7 | factor, n_datasets, k_levels, observed_W, empirical_power_alpha05, parametric_power_alpha05 … |

LaTeX exports (12): `table10b_pmsp_vs_external_baselines.tex`, `table11_pmsp_rules.tex`, `table1_dataset_characteristics.tex`, `table2_algorithm_specification.tex`, `table3_pipeline_ablation_map.tex`, `table4_pivot.tex`, `table4b_minimal_sufficient_preprocessing.tex`, `table5_effect_sizes.tex`, `table6_algo_interaction.tex`, `table7_default_vs_tuned.tex`, `table8_loo_sensitivity.tex`, `table9_empirical_power.tex`

## `figures/` — 12 figures, two formats each

PDF for publication, PNG for viewing on GitHub. Figure numbers follow the
article: 1–2 critical-difference diagrams, 3 effect-size comparison,
4 algorithm-by-level heatmap, 4b minimal sufficient level, 4c multi-seed
contrast, 5 moderators, 6 lift over chance, 7 tuned vs default,
8 leave-one-dataset-out sensitivity, 11 predictive confusion, 12 Pareto.

## Known caveats of individual files

These are recorded here so that a reader reconciling the article against the
artefact does not have to rediscover them.

**`table10_pmsp_lodo_performance.csv` is superseded.** It comes from a fallback
run of `src/pmsp_rerun.py`: its `P-MSP_meta` row is byte-identical to its
`B1_algo_mode` row, and its confidence bounds equal the point estimates. The
authoritative values are in `pmsp_rerun_summary.csv`, which reports
`B3_meta_knn` exact accuracy 0.7799 and `P-MSP_meta` 0.6490, matching the
article. Every row now carries a `status` column saying so.

**`multiple_comparison_correction.csv` holds all thirteen declared tests.** The
eight consecutive-pair Wilcoxon p-values were recomputed on the per-dataset
algorithm-marginalised profiles from `raw_results.csv` and the adjusted columns
and significance flags were formed over the full pool; the file reproduces every
adjusted value in the article's Tables 7 and 9. The family of thirteen is the one declared in
`notebooks/01_pipeline.ipynb`: eight consecutive-pair Wilcoxon tests, three
two-tailed and one one-tailed Spearman correlation, and the high-missingness
subgroup test. The terminal contrast `P1->P8b` is not a member, which is why
Table 7 reports no adjusted value for that row. The
one-sided p-values for the imbalance ratios are stored in the pre-declared
positive direction (0.7097), not as half the two-sided value. Section S6 of the
Supplementary Material records the recomputation.

**`dataset_fingerprints.csv` has no provenance.** `source` reads `unknown` for all
95 rows and `openml_oid` and `openml_version` are empty throughout. These fields
were not captured at fetch time and cannot be reconstructed from the shipped
artefact; the SHA-256 column does identify the exact served revision of each
table, and `table1_dataset_characteristics.csv` carries the descriptors. Suite
membership is defined by the two published curated collections named in the
article.

**`delta_eps2_stageA_vs_stageB.csv`: `ci_excludes_zero` corrected.** Both rows
previously read `False` although both confidence intervals lie entirely below
zero ([-0.4239, -0.2262] and [-0.5090, -0.2178]). The flag is now `True` in both
rows, in agreement with the article. The `p_one_sided` value of 1.0 is correct
and not an error: the one-sided hypothesis was declared in the positive
direction and the realised contrast is negative.
