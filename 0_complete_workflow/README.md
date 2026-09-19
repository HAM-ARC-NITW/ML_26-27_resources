# Complete Workflow: From Problem Statement to Model

A practical guide to approaching a data science / machine learning problem end to end.

```
Problem Statement
      │
      ▼
1. Understand the problem & data
      │
      ▼
2. EDA (explore)
      │
      ▼
3. Preprocessing & feature engineering (prepare)
      │
      ▼
4. Modelling (learn)
      │
      ▼
5. Evaluation & error analysis
      │
      ▼
6. Communicate & deploy
      │
      └──► iterate back to any earlier stage
```

The process is **iterative**. Modelling results often send you back to EDA or feature engineering.

---

## Stage 1: Understand the Problem

Before touching data, answer these questions:

- **What is the goal?** (e.g. "predict which customers will churn")
- **What type of problem is it?**
  - Classification (categories): spam or not spam
  - Regression (numbers): house price
  - Clustering (no labels): customer segments
  - Time series: sales forecasting
- **What is the target variable?**
- **How will success be measured?** Choose the metric now, not after seeing results.
- **What are the costs of errors?** Missing a fraud case is not the same as a false alarm.
- **What data do we have, and is it legitimate to use?** (privacy, leakage)

**Output:** a one-paragraph problem statement, a target variable, and a success metric.

---

## Stage 2: Exploratory Data Analysis (EDA)

**Purpose:** understand and diagnose. You look at the data but do not yet transform it for modelling.

| Step | Question | Tools |
|------|----------|-------|
| Overview | How big is it? What types? | `shape`, `info()`, `describe()`, `head()` |
| Target analysis | Is it balanced? | `value_counts()`, bar chart |
| Data quality | Missing values? Duplicates? Wrong types? | `isna().sum()`, `duplicated()` |
| Univariate | How is each variable distributed? | histograms, KDE, boxplots, skewness |
| Outliers | Extreme values: errors or real? | IQR rule, Z-score, boxplot |
| Bivariate | How does each feature relate to the target? | grouped means, scatter plots, box plots |
| Statistical tests | Is a relationship significant? | chi-square, t-test, ANOVA |
| Correlation | Which features are redundant? | correlation heatmap, VIF |

**Output:** a list of findings and decisions, for example:

- "`deck` is 77% missing, so drop it or flag it."
- "`fare` is heavily skewed, so use a log transform."
- "Sex and class are the strongest predictors."
- "`pclass` and `fare` are correlated, so watch for redundancy."

---

## Stage 3: Preprocessing & Feature Engineering

**Purpose:** act on the EDA findings and turn raw data into model-ready input.

### 3.1 Split the data first

Split into train and test **before** fitting any transformation. This prevents data leakage.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)
```

### 3.2 Cleaning

| Problem | Options |
|---------|---------|
| Missing numeric | median / mean / group median / model-based (KNN, iterative) |
| Missing categorical | mode / "Unknown" category |
| Mostly missing column | drop, or create a "was missing" flag |
| Duplicates | drop (if truly duplicates) |
| Outliers | keep, cap (winsorize), transform, or remove if errors |
| Wrong types / inconsistent labels | convert, standardise text |

### 3.3 Transformations

| Technique | When to use |
|-----------|-------------|
| **Log / sqrt transform** | right-skewed features (fare, income) |
| **Standardisation** (z-score) | KNN, SVM, logistic/linear regression, neural nets, PCA |
| **Normalisation** (min-max) | neural nets, image data, when a bounded range is needed |
| **Robust scaling** | many outliers |
| No scaling | tree-based models (Random Forest, XGBoost) |

### 3.4 Encoding categorical variables

| Technique | When |
|-----------|------|
| One-hot encoding | nominal categories with few levels |
| Ordinal encoding | ordered categories (low < medium < high) |
| Target / frequency encoding | high-cardinality categories (careful about leakage) |

### 3.5 Feature engineering

Create new features from domain knowledge and EDA insights:

- **Combine:** `family_size = sibsp + parch + 1`
- **Bin:** `age_group` from `age`
- **Extract:** day, month, hour from dates; title from names
- **Flag:** `has_deck` from a mostly-missing column
- **Interact:** `price_per_sqft = price / area`
- **Aggregate:** customer's average purchase per month

### 3.6 Feature selection

- Drop near-constant and redundant (highly correlated) features.
- Use model-based importance, L1 regularisation, or recursive feature elimination.

**Output:** a reusable preprocessing pipeline and a clean feature matrix.

---

## Stage 4: Modelling

### 4.1 Start with a baseline

- Classification: predict the majority class, or use logistic regression.
- Regression: predict the mean, or use linear regression.

Any model you build must beat the baseline to be worth using.

### 4.2 Try a few model families

| Problem | Candidates |
|---------|-----------|
| Classification | Logistic Regression, Random Forest, Gradient Boosting (XGBoost, LightGBM), SVM, KNN |
| Regression | Linear / Ridge / Lasso, Random Forest, Gradient Boosting |
| Clustering | K-Means, DBSCAN, Hierarchical |

### 4.3 Validate properly

- Use **cross-validation** on the training set to compare models.
- Use **stratified** splits for imbalanced classification.
- Use **time-based splits** for time series (never shuffle).
- Tune hyperparameters with `GridSearchCV` or `RandomizedSearchCV`.
- **Touch the test set only once**, at the very end.

### 4.4 Handle class imbalance (if needed)

Class weights, oversampling (SMOTE), undersampling, or threshold tuning. Evaluate with precision, recall, and F1 rather than accuracy alone.

---

## Stage 5: Evaluation & Error Analysis

### Choose metrics that match the problem

| Task | Metrics |
|------|---------|
| Classification | accuracy, precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix |
| Regression | MAE, RMSE, R² |
| Clustering | silhouette score, Davies-Bouldin |

### Then look deeper

- Where does the model fail? Study misclassified examples.
- Does it perform equally well across groups (sex, region, age)? Check fairness.
- Which features matter most? Use feature importance or SHAP.
- Is it overfitting? Compare train and validation scores.

If results are poor, go back to Stage 2 or 3.

---

ons documented
