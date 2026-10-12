# Electric Vehicle Purchase Prediction

A supervised machine learning classification project exploring whether a customer is likely to purchase an electric vehicle (EV), using demographic, commuting, vehicle-ownership, charging-access, and attitude-related features.

> **Project type:** Binary classification · **Notebook:** `Electric-vehicle.ipynb`

## Problem statement

The target column is `Will_Buy_EV`, recorded as Yes/No in the original data and converted to a binary target during preprocessing. The notebook compares several classification approaches and prepares probability predictions for a competition-style `submission.csv` file.

## Dataset

The notebook reads these files from its original Kaggle/Colab environment:

- `train.csv` — labelled training data
- `test.csv` — rows for which predictions are generated
- `sample_submission.csv` — example submission format

The notebook's saved inspection output reports **668,665 rows and 15 columns** for the loaded training data before preprocessing. Dataset files are not included here. Add the exact competition/dataset URL when publishing, and download data from the original source according to its license and competition rules.

### Main features

The original columns include:

- Demographics: `Age`, `Gender`, `Annual_Income_USD`
- Mobility and vehicle ownership: `Daily_Commute_km`, `Number_of_Cars_Owned`, `Current_Car_Type`
- Charging access: `Charging_Stations_Near_Home`, `Charging_Stations_Near_Work`, `Home_Charging_Possible`
- Attitudes and context: `Environmental_Concern_Level`, `Range_Anxiety_Level`, `City_Type`, `Subsidy_Available`
- Identifier and target: `id`, `Will_Buy_EV`

## Workflow in the notebook

1. Load and inspect the training data with Pandas.
2. Review column types, descriptive statistics, and missing values.
3. Drop missing rows and encode categorical/binary values.
4. Apply scaling to selected numerical columns.
5. Split the labelled data into training and holdout sets.
6. Fit and compare classification models.
7. Tune XGBoost with `RandomizedSearchCV`.
8. Predict purchase probabilities for the separate `test.csv` and write `submission.csv` with `id` and `Will_Buy_EV` columns.

## Models explored

- Logistic Regression
- Decision Tree (`max_depth=8`)
- Gaussian Naive Bayes
- K-Nearest Neighbors (`n_neighbors=25`)
- Random Forest (`n_estimators=100`)
- AdaBoost (`n_estimators=100`)
- Gradient Boosting (`n_estimators=150`)
- Hard-voting ensemble
- Bagging Classifier
- XGBoost, including a randomized hyperparameter search

## Saved results

The notebook's saved outputs include the following metrics. These are transcribed from saved cell outputs; the notebook was not re-executed for this repository.

| Model | Saved accuracy | EV-purchase precision | EV-purchase recall | EV-purchase F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.89 | 0.71 | 0.65 | 0.68 |
| Decision Tree | 0.89 | 0.72 | 0.65 | 0.68 |
| Gaussian Naive Bayes | 0.84 | 0.52 | 0.91 | 0.66 |
| K-Nearest Neighbors | 0.8898 | — | — | — |
| Random Forest | 0.89 | 0.71 | 0.64 | 0.68 |
| AdaBoost | 0.89 | 0.72 | 0.63 | 0.67 |
| Gradient Boosting | 0.90 | 0.73 | 0.65 | 0.69 |
| Hard-voting ensemble | 0.90 | 0.72 | 0.65 | 0.69 |
| Bagging Classifier | 0.87 | 0.83 | 0.33 | 0.47 |
| Tuned XGBoost | 0.90 | 0.72 | 0.66 | 0.69 |

For the tuned XGBoost estimator, the saved best parameters are `n_estimators=400`, `max_depth=3`, `learning_rate=0.1`, `random_state=42`, with `objective='binary:logistic'` and `eval_metric='logloss'`.

**Interpret these results cautiously.** The saved outputs report different holdout support counts across model reports. The tuning-cell source also uses `knn_model` while the saved output indicates an XGBoost estimator, and the search is fitted on `x, y` before the selected model is evaluated against `x_test`. See [`REVIEW_NOTES.md`](REVIEW_NOTES.md) for the full reproducibility and evaluation caveats.

## Run the notebook

1. Install Python and Jupyter in a virtual environment.
2. Install dependencies from `requirements.txt`.
3. Download `train.csv`, `test.csv`, and `sample_submission.csv` from the original competition source.
4. Update the `/content/...` data paths in the notebook to point to your local dataset paths.
5. Open `Electric-vehicle.ipynb` and execute cells in order in a clean kernel.

The original notebook was retained as uploaded; paths and notebook logic have not been rewritten or executed for this package.

## Limitations and next improvements

- Move preprocessing into a train-fitted scikit-learn `Pipeline`/`ColumnTransformer` to prevent leakage and apply identical transformations consistently.
- Keep hyperparameter search inside the training data, then evaluate once on an untouched holdout set.
- Re-run all cells from a clean kernel so reported sample counts and results are consistent.
- Review class imbalance and decide whether EV-purchase recall, precision, F1, ROC-AUC, or another competition metric best reflects the goal.
- Record the source competition URL and final competition score, if applicable.

## Acknowledgements

Dataset/competition: **Add the exact Kaggle URL here**.

This is an educational portfolio project. It does not establish that the model is suitable for real-world customer targeting or purchasing decisions.
