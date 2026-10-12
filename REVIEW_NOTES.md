# Notebook review notes (before treating metrics as final)

These notes were identified by reading the notebook source and saved cell outputs only. Notebook cells were **not executed**.

## High-priority issues

1. **Saved report counts disagree.** Several classifier reports have 131,741 holdout rows (support: class 0 = 108,657; class 1 = 23,084), while later reports have 133,733 rows (class 0 = 110,375; class 1 = 23,358). The saved notebook's initial dataset inspection reports 668,665 rows, for which an 80/20 split would ordinarily give 133,733 holdout rows. This suggests outputs may come from different notebook states or executions. Restart the kernel and run top-to-bottom before treating metrics as comparable.

2. **Hyperparameter search sees the holdout labels.** The code defines `x_train`/`x_test`, but the `RandomizedSearchCV` cell calls `.fit(x, y)` on the full labelled dataset. The final holdout results therefore should not be considered an uncontaminated final test estimate. Run the search on `x_train, y_train`, then evaluate once on `x_test, y_test`.

3. **A single scaler instance is refitted separately for several columns.** The notebook calls `fit_transform` on `Age`, income, commute distance, and then other individual columns using the same `StandardScaler` object. Only the statistics from the most recent fit remain in `scaler`; later, the test-set code reuses that same object for multiple different columns. Fit a separate scaler per feature or, preferably, use a `ColumnTransformer` inside a pipeline fitted only on training data.

4. **The tuning cell source and its saved output disagree.** The saved source for the `RandomizedSearchCV` cell passes `knn_model` as the estimator while searching XGBoost-specific parameters (`n_estimators`, `max_depth`, `objective`, and `eval_metric`). However, its saved output shows an `XGBClassifier` estimator. The tuning cells have no saved execution counts, another sign that source and outputs may not reflect the same run. Correct the estimator and re-run before reporting tuned results.

5. **Preprocessing happens before the holdout split.** Imputation/encoding/scaling should generally be learned from training data only, then applied to validation/test rows through a pipeline. Fitting preprocessing on all labelled rows can leak information into evaluation.

## Other notes

- The notebook's saved submission preview shows a `submission.csv` with `id` and `Will_Buy_EV`, where `Will_Buy_EV` is populated with predicted class-1 probabilities.
- Several imports are used for plotting, but this saved notebook contains no rendered image outputs detected during this review. Add selected plots only if you create/export them separately.
- The exact Kaggle competition URL is not recorded in the notebook text; add it manually to the README.
- Before publishing, search for secrets, private files, large embedded outputs, and any dataset material that the competition terms prohibit sharing.

## Suggested corrected order (future improvement)

1. Split raw labelled data into training and holdout sets.
2. Build preprocessing with `ColumnTransformer` (numeric scaling; categorical encoding) inside a pipeline.
3. Tune models with cross-validation on training data only.
4. Evaluate the chosen pipeline once on the holdout set, using accuracy plus per-class precision/recall/F1 and a confusion matrix.
5. Fit the final selected pipeline on all labelled data only when ready to generate the competition test predictions.
