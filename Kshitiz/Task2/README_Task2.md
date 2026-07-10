# Gray Interface '26 — Task 2: Regression & Classification

**Three separate notebooks, one README.**

---

## Part 1 · Linear Regression from Scratch

### What I did
Implemented Linear Regression using Gradient Descent from scratch using only NumPy — no sklearn LinearRegression allowed.

**Dataset:** Synthetic — `y = 4 + 3X + noise` (100 samples)

### Preprocessing
- Added a bias column (column of 1s) to the feature matrix so the intercept is learned automatically as part of the weight vector.
- Manual 80/20 train-test split using `np.random.permutation`.

### How Gradient Descent works
Start with weights = 0. Compute predictions, calculate the error, find the gradient (direction of steepest increase in loss), and move weights in the opposite direction by a small step (learning rate). Repeat for many epochs until loss stops decreasing.

### Learning Rate Experiments

| Learning Rate | Behaviour |
|---|---|
| 0.01 | Very slow — loss still decreasing at epoch 200 |
| 0.1 | Smooth convergence, reaches minimum ~epoch 150. Best choice. |
| 0.5 | Fast initially but risks overshooting the minimum |

### Metrics (Final Model — lr=0.1, epochs=1000)

| Set | MAE | MSE | RMSE | R² |
|---|---|---|---|---|
| Train | ~0.79 | ~0.98 | ~0.99 | ~0.91 |
| Test | ~0.82 | ~1.05 | ~1.02 | ~0.89 |

### Bonus: Polynomial Regression (Degree 2)
Added X² as an extra feature. Performance on training data improved slightly, but test metrics were nearly identical — the underlying relationship is linear, so the extra term doesn't help generalize.

---

## Part 2 · Ridge / Lasso / ElasticNet Regularization

**Dataset:** [Ames Housing Dataset](https://www.kaggle.com/datasets/shashanknecrothapa/ames-housing-dataset)

### Preprocessing Steps
1. **Dropped columns** with >40% missing values (too sparse to be useful).
2. **Filled numeric NaNs** with column median (robust to outliers).
3. **Filled categorical NaNs** with 'Unknown'.
4. **Label encoded** all categorical columns (simple, fast, works well with regularized linear models).
5. **StandardScaler** applied after splitting — fit on train only, transform both train and test (prevents data leakage).

### Why SGDRegressor instead of LinearRegression?
`LinearRegression` solves the Normal Equation **(XᵀX)⁻¹Xᵀy** — requires inverting a matrix, which is O(n³) and breaks down with many features or collinear inputs. `SGDRegressor` uses Stochastic Gradient Descent, scales to large datasets, and natively supports L1/L2 regularization via a single parameter change. For the Ames Housing dataset with many features, SGD is the more practical baseline.

### Regularization — What each does

| Model | Penalty | Effect |
|---|---|---|
| SGDRegressor | None | Baseline — may overfit with many features |
| Ridge | L2 (sum of squares of weights) | Shrinks all weights toward zero, keeps all features |
| Lasso | L1 (sum of absolute weights) | Drives some weights to exactly 0 — automatic feature selection |
| ElasticNet | L1 + L2 combined | Best of both — handles correlated features better than pure Lasso |

### Cross-validation for alpha selection
Used 5-fold CV on the training set to find the best regularization strength (alpha) for Ridge and Lasso. This avoids tuning on the test set, which would give an optimistic estimate of performance.

### Key Observations
- Ridge performed well overall — most features in Ames Housing are genuinely useful, so keeping all of them (but shrinking their weights) makes sense.
- Lasso zeroed out several features, which tells us those features aren't contributing much to predicting sale price.
- ElasticNet gave competitive results when features were correlated (e.g. overall quality and year built often move together).

---

## Part 3 · Logistic Regression

**Dataset:** [Santander Customer Transaction Prediction](https://www.kaggle.com/competitions/santander-customer-transaction-prediction)

### Preprocessing Steps
1. Dropped `ID_code` — not a feature.
2. Checked for missing values — none found.
3. **Stratified train-test split** — preserves the class ratio in both sets.
4. **StandardScaler** — essential for Logistic Regression; features with larger ranges otherwise dominate.

### Class Imbalance
The dataset has a ~10:1 imbalance (non-transactions vs transactions). Without handling this:
- Model predicts class 0 for everything → 90% accuracy but 0% recall on the minority class.

Fix: used `class_weight='balanced'` which automatically scales the loss penalty so misclassifying a positive sample costs more.

### Effect of C (Regularization Strength)

| C | Behaviour |
|---|---|
| 0.001 | Strong regularization — underfits, poor recall |
| 0.1 | Good balance — model generalizes well |
| 1.0 | Default — competitive ROC-AUC |
| 100.0 | Near-zero regularization — risk of overfitting |

### Metrics (C=1.0, balanced)

| Metric | Value |
|---|---|
| Accuracy | ~0.72 |
| Precision | ~0.20 |
| Recall | ~0.73 |
| F1 Score | ~0.32 |
| ROC-AUC | ~0.78 |

### Why accuracy is misleading here
With 10:1 imbalance, a model predicting all-zeros gets 90% accuracy but is completely useless. **ROC-AUC** is the real metric — it measures how well the model ranks positive cases above negative ones, regardless of threshold.

### Key Observations
- Feature scaling significantly improves convergence speed and final AUC.
- `class_weight='balanced'` dramatically improves Recall at the cost of some Precision — acceptable tradeoff when the goal is to catch as many positive customers as possible.
- ROC-AUC plateaus around C=0.1–1.0, confirming moderate regularization is optimal for this dataset.

---

## Repo Structure

```
Kshitiz/
├── Task2_Part1_LinearRegression.ipynb
├── Task2_Part2_Regularization.ipynb
├── Task2_Part3_LogisticRegression.ipynb
├── README.md   (Task 1)
└── README_Task2.md   (this file)
```
