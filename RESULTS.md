# Saved model results

Metrics below were read from the notebook's stored outputs; no code was executed during preparation. “EV-purchase” refers to target class `1` after the notebook maps `Will_Buy_EV` Yes/No to 1/0.

| Model | Accuracy | Precision (class 1) | Recall (class 1) | F1 (class 1) | Holdout rows in saved output |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.89 | 0.71 | 0.65 | 0.68 | 131,741 |
| Decision Tree | 0.89 | 0.72 | 0.65 | 0.68 | 131,741 |
| Gaussian Naive Bayes | 0.84 | 0.52 | 0.91 | 0.66 | 131,741 |
| K-Nearest Neighbors | 0.8898 | Not printed | Not printed | Not printed | Not printed in output |
| Random Forest | 0.89 | 0.71 | 0.64 | 0.68 | 131,741 |
| AdaBoost | 0.89 | 0.72 | 0.63 | 0.67 | 131,741 |
| Gradient Boosting | 0.90 | 0.73 | 0.65 | 0.69 | 131,741 |
| Hard-voting ensemble | 0.90 | 0.72 | 0.65 | 0.69 | 133,733 |
| XGBoost, initial settings | 0.90 | 0.73 | 0.66 | 0.69 | 133,733 |
| Bagging Classifier | 0.87 | 0.83 | 0.33 | 0.47 | 133,733 |
| XGBoost, tuned parameters used for final report | 0.90 | 0.72 | 0.66 | 0.69 | 131,741 |

## Caveats

The number of holdout rows differs across stored reports. The randomized-search source and saved output disagree about the estimator, and the saved search output reflects a search on full `x, y` before the final test report. The scaler is also refitted repeatedly on separate columns and then reused for the independent test file. Do not compare these results as a rigorously controlled benchmark until the notebook is corrected and run from a clean kernel.
