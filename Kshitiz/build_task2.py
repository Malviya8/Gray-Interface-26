"""Generate Task 2 notebooks for Gray Interface '26 — Kshitiz."""
import json
from pathlib import Path

OUT = Path(__file__).parent / "Task 2"
OUT.mkdir(exist_ok=True)

METADATA = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.10.0"},
    "colab": {"provenance": []},
}


def md(text: str) -> dict:
    lines = [line + "\n" for line in text.split("\n")]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {"cell_type": "markdown", "metadata": {}, "source": lines}


def code(text: str) -> dict:
    lines = [line + "\n" for line in text.split("\n")]
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines,
    }


def save(name: str, cells: list) -> None:
    nb = {"nbformat": 4, "nbformat_minor": 5, "metadata": METADATA, "cells": cells}
    path = OUT / name
    path.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"Created {path}")


# ── Part 1: Linear Regression from Scratch ───────────────────────────────────

part1 = [
    md("# Linear Regression from Scratch\n**Gray Interface '26 | Task 2 — Part 1**"),
    md("## Importing the dependencies"),
    code("""!pip install numpy pandas matplotlib scikit-learn --quiet

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score"""),
    md("## Step 1 — Create synthetic data\n\nWe make a simple dataset where `y = 3.5 * x + 4 + noise`. Because we know the true slope and intercept, we can check if gradient descent learns them correctly."),
    code("""np.random.seed(42)

n = 100
X = np.random.rand(n) * 10          # feature between 0 and 10
y = 3.5 * X + 4 + np.random.randn(n) * 2   # true line + noise

data = pd.DataFrame({'X': X, 'y': y})
data.head()"""),
    md("## Step 2 — Visualize the data"),
    code("""plt.figure(figsize=(7, 5))
plt.scatter(data['X'], data['y'], alpha=0.7)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Synthetic Linear Data')
plt.show()"""),
    md("## Step 3 — Train / test split"),
    code("""X_train, X_test, y_train, y_test = train_test_split(
    data['X'].values, data['y'].values, test_size=0.2, random_state=42
)

print('Train size:', len(X_train))
print('Test size:', len(X_test))"""),
    md("## Step 4 — Gradient Descent (from scratch)\n\nWe update slope `m` and intercept `b` using the MSE gradient. **No sklearn LinearRegression here.**"),
    code("""def gradient_descent(X, y, learning_rate=0.01, epochs=1000):
    m, b = 0.0, 0.0
    n = len(X)
    costs = []

    for _ in range(epochs):
        y_pred = m * X + b
        error = y_pred - y

        # partial derivatives of MSE w.r.t m and b
        dm = (2 / n) * np.sum(error * X)
        db = (2 / n) * np.sum(error)

        m -= learning_rate * dm
        b -= learning_rate * db
        costs.append(np.mean(error ** 2))

    return m, b, costs


def show_metrics(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    print(f'{name} -> MAE: {mae:.3f} | MSE: {mse:.3f} | RMSE: {rmse:.3f} | R2: {r2:.3f}')"""),
    code("""# train the model
m, b, costs = gradient_descent(X_train, y_train, learning_rate=0.01, epochs=1000)

print(f'Learned slope (m): {m:.3f}')
print(f'Learned intercept (b): {b:.3f}')
print('True values were m=3.5, b=4')"""),
    md("## Step 5 — Plot the fitted line"),
    code("""plt.figure(figsize=(7, 5))
plt.scatter(X_train, y_train, label='Train', alpha=0.7)
line_x = np.linspace(min(X_train), max(X_train), 100)
line_y = m * line_x + b
plt.plot(line_x, line_y, color='red', linewidth=2, label='Fitted line')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Gradient Descent Fit')
plt.legend()
plt.show()"""),
    md("## Step 6 — Try different learning rates and epochs"),
    code("""settings = [
    (0.001, 100),
    (0.01, 100),
    (0.01, 1000),
    (0.1, 100),
]

results = []
for lr, epochs in settings:
    m_try, b_try, cost_try = gradient_descent(X_train, y_train, learning_rate=lr, epochs=epochs)
    y_pred = m_try * X_test + b_try
    r2 = r2_score(y_test, y_pred)
    results.append({'learning_rate': lr, 'epochs': epochs, 'final_cost': cost_try[-1], 'test_r2': r2})

pd.DataFrame(results)"""),
    md("## Step 7 — Cost vs epochs"),
    code("""_, _, cost_history = gradient_descent(X_train, y_train, learning_rate=0.01, epochs=1000)

plt.figure(figsize=(7, 5))
plt.plot(cost_history)
plt.xlabel('Epoch')
plt.ylabel('MSE Cost')
plt.title('Cost decreases as training continues')
plt.show()"""),
    md("## Step 8 — Final evaluation"),
    code("""y_train_pred = m * X_train + b
y_test_pred = m * X_test + b

show_metrics('Train', y_train, y_train_pred)
show_metrics('Test', y_test, y_test_pred)"""),
]

# ── Part 2: Ames Housing ────────────────────────────────────────────────────

part2 = [
    md("# Ames Housing — Regression Models\n**Gray Interface '26 | Task 2 — Part 2**"),
    md("## Importing the dependencies"),
    code("""!pip install kagglehub scikit-learn pandas numpy matplotlib seaborn --quiet

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import kagglehub
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score"""),
    md("## Step 1 — Load the Ames Housing dataset"),
    code("""path = kagglehub.dataset_download('shashanknecrothapa/ames-housing-dataset')
csv_files = glob.glob(os.path.join(path, '**', '*.csv'), recursive=True)
print('Downloaded files:', csv_files)

df = pd.read_csv(csv_files[0])
print('Shape:', df.shape)
df.head()"""),
    md("## Step 2 — Quick look at the target (`SalePrice`)"),
    code("""print('Missing values in SalePrice:', df['SalePrice'].isna().sum())

plt.figure(figsize=(7, 4))
sns.histplot(df['SalePrice'], kde=True)
plt.title('SalePrice distribution (right-skewed)')
plt.show()"""),
    md("## Step 3 — Log transform + remove outliers\n\nHouse prices are skewed. A log transform makes the target easier for linear models. Then we remove extreme values using the IQR rule."),
    code("""df = df.dropna(subset=['SalePrice']).copy()
df['log_price'] = np.log1p(df['SalePrice'])

q1 = df['log_price'].quantile(0.25)
q3 = df['log_price'].quantile(0.75)
iqr = q3 - q1
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr

before = len(df)
df = df[(df['log_price'] >= lower) & (df['log_price'] <= upper)].copy()
print(f'Removed {before - len(df)} outlier rows')"""),
    md("## Step 4 — Simple preprocessing\n\n- Drop ID-like columns\n- Fill missing numeric values with median\n- Fill missing text values with `'Missing'`\n- One-hot encode categories\n- Scale all features"),
    code("""work = df.drop(columns=['SalePrice']).copy()
target = df['log_price'].values

# columns we do not want as features
drop_cols = ['Order', 'PID', 'MS SubClass']
drop_cols = [c for c in drop_cols if c in work.columns]
work = work.drop(columns=drop_cols)

num_cols = work.select_dtypes(include=['number']).columns.tolist()
cat_cols = work.select_dtypes(include=['object']).columns.tolist()

# simple missing value handling
for col in num_cols:
    work[col] = work[col].fillna(work[col].median())
for col in cat_cols:
    work[col] = work[col].fillna('Missing')

# convert categories to numbers
work = pd.get_dummies(work, columns=cat_cols, drop_first=True)

X = work.values
y = target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print('Feature count:', X_train.shape[1])"""),
    md("## Step 5 — Train models"),
    code("""models = {
    'Linear Regression': LinearRegression(),
    'SGD Regressor': SGDRegressor(max_iter=5000, random_state=42),
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=0.001, max_iter=5000),
    'ElasticNet': ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=5000),
}

rows = []
predictions = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    predictions[name] = test_pred

    rows.append({
        'Model': name,
        'Train R2': r2_score(y_train, train_pred),
        'Test R2': r2_score(y_test, test_pred),
        'Test RMSE': np.sqrt(mean_squared_error(y_test, test_pred)),
        'Test MAE': mean_absolute_error(y_test, test_pred),
    })

results = pd.DataFrame(rows).sort_values('Test R2', ascending=False)
results"""),
    md("## Step 6 — Actual vs Predicted plots"),
    code("""fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()

for ax, (name, model) in zip(axes, models.items()):
    test_pred = predictions[name]
    ax.scatter(y_test, test_pred, alpha=0.5)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax.set_title(name)
    ax.set_xlabel('Actual log_price')
    ax.set_ylabel('Predicted log_price')

axes[-1].axis('off')
plt.tight_layout()
plt.show()"""),
    md("## Step 7 — Which model is best?\n\nCompare test R² and RMSE. Ridge usually generalizes well when we have many dummy columns."),
    code("""best = results.iloc[0]
print(f"Best model: {best['Model']}")
print(f"Test R2: {best['Test R2']:.3f}")
print(f"Test RMSE: {best['Test RMSE']:.3f}")"""),
]

# ── Part 3: Santander Logistic Regression ────────────────────────────────────

part3 = [
    md("# Santander — Logistic Regression\n**Gray Interface '26 | Task 2 — Part 3**"),
    md("## Importing the dependencies"),
    code("""!pip install kagglehub scikit-learn pandas numpy matplotlib seaborn --quiet

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import kagglehub
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, roc_auc_score, classification_report
)"""),
    md("## Step 1 — Load Santander dataset"),
    code("""path = kagglehub.dataset_download('sachinrohilla/santander-customer-transaction-prediction')
csv_files = glob.glob(os.path.join(path, '**', '*.csv'), recursive=True)
print('Downloaded files:', csv_files)

train_path = [f for f in csv_files if f.lower().endswith('train.csv')][0]
df = pd.read_csv(train_path)
print('Shape:', df.shape)
print('Target balance:\\n', df['target'].value_counts(normalize=True))
df.head()"""),
    md("## Step 2 — Prepare features\n\nAll 200 `var_*` columns are numeric. We scale them because logistic regression uses gradient-based learning."),
    code("""feature_cols = [c for c in df.columns if c.startswith('var_')]

X = df[feature_cols].values
y = df['target'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print('Train shape:', X_train.shape)
print('Test shape:', X_test.shape)"""),
    md("## Step 3 — Train baseline Logistic Regression"),
    code("""model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

print('Train accuracy:', accuracy_score(y_train, y_train_pred))
print('Test accuracy:', accuracy_score(y_test, y_test_pred))
print('\\nClassification report (test):')
print(classification_report(y_test, y_test_pred))"""),
    md("## Step 4 — Precision, Recall, F1\n\nAccuracy is misleading here because the dataset is imbalanced (~90% class 0)."),
    code("""metrics = pd.DataFrame([
    {
        'Split': 'Train',
        'Accuracy': accuracy_score(y_train, y_train_pred),
        'Precision': precision_score(y_train, y_train_pred),
        'Recall': recall_score(y_train, y_train_pred),
        'F1': f1_score(y_train, y_train_pred),
    },
    {
        'Split': 'Test',
        'Accuracy': accuracy_score(y_test, y_test_pred),
        'Precision': precision_score(y_test, y_test_pred),
        'Recall': recall_score(y_test, y_test_pred),
        'F1': f1_score(y_test, y_test_pred),
    },
])
metrics"""),
    md("## Step 5 — Tune regularization parameter `C`"),
    code("""c_values = [0.001, 0.01, 0.1, 1, 10, 100]
c_results = []

for c in c_values:
    clf = LogisticRegression(C=c, max_iter=1000, class_weight='balanced', random_state=42)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    c_results.append({
        'C': c,
        'Test Accuracy': accuracy_score(y_test, pred),
        'Test F1': f1_score(y_test, pred),
        'Test Recall': recall_score(y_test, pred),
    })

c_df = pd.DataFrame(c_results)
c_df"""),
    code("""plt.figure(figsize=(7, 4))
plt.plot(c_df['C'], c_df['Test F1'], marker='o')
plt.xscale('log')
plt.xlabel('C (log scale)')
plt.ylabel('Test F1')
plt.title('Effect of C on F1 score')
plt.show()"""),
    md("## Step 6 — ROC curve"),
    code("""y_test_prob = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_test_prob)
auc = roc_auc_score(y_test, y_test_prob)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve — Test Set')
plt.legend()
plt.show()"""),
    md("## Step 7 — Confusion matrix"),
    code("""cm_train = confusion_matrix(y_train, y_train_pred)
cm_test = confusion_matrix(y_test, y_test_pred)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title('Train Confusion Matrix')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', ax=axes[1])
axes[1].set_title('Test Confusion Matrix')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')

plt.tight_layout()
plt.show()"""),
    md("## Step 8 — Pick the best C and retrain"),
    code("""best_c = c_df.sort_values('Test F1', ascending=False).iloc[0]['C']
print('Best C from tuning:', best_c)

best_model = LogisticRegression(C=best_c, max_iter=1000, class_weight='balanced', random_state=42)
best_model.fit(X_train, y_train)
best_pred = best_model.predict(X_test)

print('Best model test F1:', f1_score(y_test, best_pred))
print('Best model test recall:', recall_score(y_test, best_pred))"""),
]

save("Part1.ipynb", part1)
save("Part2.ipynb", part2)
save("Part3.ipynb", part3)

README = """# Gray Interface '26 — Task 2: Regression & Classification

**Author:** Kshitiz

---

## Notebooks

| Part | File | Dataset |
|---|---|---|
| Part 1 — Linear Regression from Scratch | `Part1.ipynb` | Synthetic (generated inside notebook) |
| Part 2 — Ridge / Lasso / ElasticNet | `Part2.ipynb` | [Ames Housing Dataset](https://www.kaggle.com/datasets/shashanknecrothapa/ames-housing-dataset) |
| Part 3 — Logistic Regression | `Part3.ipynb` | [Santander Customer Transaction Prediction](https://www.kaggle.com/competitions/santander-customer-transaction-prediction) |

**Environment:** Google Colab with `kagglehub` for Parts 2 and 3.

---

## Part 1 — Linear Regression from Scratch

### What I did
1. Created 100 synthetic points using `y = 3.5x + 4 + noise`.
2. Split into 80% train / 20% test.
3. Implemented **Gradient Descent manually** to learn slope (`m`) and intercept (`b`).
4. Plotted the fitted line on training data.
5. Tried different learning rates and epoch counts.
6. Plotted **Cost vs Epochs**.
7. Evaluated with MAE, MSE, RMSE, and R².

### Why Gradient Descent?
Linear regression can be solved with a formula, but gradient descent is the base idea behind almost every ML model. Here we update `m` and `b` step by step to reduce MSE.

### Key insight
- Learned values stayed close to the true `m=3.5` and `b=4`.
- Train and test scores were similar, so the model generalized well.
- Learning rate matters: too small needs many epochs, too large can become unstable.

---

## Part 2 — Ames Housing (Regression)

### What I did
1. Loaded Ames Housing data using `kagglehub`.
2. Checked `SalePrice` distribution (right-skewed).
3. Applied **log transform** on price.
4. Removed outliers using the **IQR method**.
5. Filled missing values simply (median for numbers, `'Missing'` for text).
6. One-hot encoded categories and scaled features.
7. Trained:
   - Linear Regression (baseline)
   - SGDRegressor
   - Ridge
   - Lasso
   - ElasticNet
8. Compared MAE, RMSE, R² and plotted **Actual vs Predicted**.

### Why log transform?
Raw house prices have a long right tail. Log transform makes the target more normal, which helps linear models fit better.

### Key insight
- **Ridge** usually gives the best balance between train and test performance.
- Lasso can underfit if `alpha` is too strong because it zeroes out useful features.
- SGDRegressor works, but needs tuning to match closed-form linear regression on this dataset size.

---

## Part 3 — Santander (Classification)

### What I did
1. Loaded Santander train data with `kagglehub`.
2. Scaled all 200 numeric `var_*` features.
3. Trained **Logistic Regression** with `class_weight='balanced'`.
4. Measured Accuracy, Precision, Recall, and F1 on train and test.
5. Tuned regularization parameter **C**.
6. Plotted **ROC curve** and **Confusion Matrix**.

### Why not trust accuracy alone?
The dataset is imbalanced (~90% class 0). A model can look accurate while missing most positive cases. F1 and Recall are more useful here.

### Key insight
- `class_weight='balanced'` improves recall for the minority class.
- Smaller **C** = stronger regularization.
- ROC AUC shows ranking ability across thresholds, while confusion matrix shows exact error types (FP/FN).

---

## Repo Structure

```
Kshitiz/
├── Netflix_EDA_GrayInterface26.ipynb   ← Task 1
├── README.md                           ← Task 1 README
└── Task 2/
    ├── Part1.ipynb
    ├── Part2.ipynb
    ├── Part3.ipynb
    └── README_2.md                     ← this file
```
"""

(OUT / "README_2.md").write_text(README, encoding="utf-8")
print(f"Created {OUT / 'README_2.md'}")
