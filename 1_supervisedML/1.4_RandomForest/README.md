# Random Forest

---

*HAM ARC ML Sessions, 2026–27.*
*Group: [Nived](https://github.com/nived-m1), [Satya Ganesh](https://github.com/Sg3107)*

## 1. What is it?

Random Forest is a supervised learning algorithm that builds **many randomized decision trees** and combines their outputs into a single, more reliable prediction. It works for both:

* **Classification** — each tree votes, and the majority wins
* **Regression** — each tree gives a number, and we average them

Instead of trusting one tree (which can be unstable and prone to overfitting), it leans on the wisdom of the crowd.

Some examples where this shows up:

* Predicting tomorrow's maximum temperature from recent weather history (the example we used below)
* Classifying whether a transaction is fraudulent
* Ranking which features matter most in a dataset, using feature importances

## 2. Intuition

The core idea behind every ensemble method is simple: **different models make different mistakes**, so combining many of them tends to cancel individual errors out.

Say you train five decision trees for a classification task, and each one is individually 80% accurate. They won't all get the same examples wrong. When you let them vote, the majority is often right even when one or two trees are wrong.

Random Forest pushes this further by deliberately making the trees **different from each other** in two ways:

1. **Bootstrap sampling (bagging):** each tree trains on a different random subset of the data.
2. **Feature randomness:** each split only considers a random subset of the features, not all of them.

Together, this decorrelates the trees, so their errors don't line up — and when averaged or voted on, the noise cancels out while the real signal survives.

## 3. The algorithm

### Step 1: Bootstrap sampling

Each tree is trained on a dataset of the same size $n$, sampled **with replacement** from the original data. Some rows get picked multiple times, and some get left out entirely.

The probability that any one row is *not* selected in a single draw is $(1 - 1/n)$, so across $n$ draws:

$$
\mathbf{P}(\text{not selected})
=
\left(1 - \frac{1}{n}\right)^n
$$

As $n \to \infty$, this converges to a well known limit:

$$
\lim_{n \to \infty}
\left(1 - \frac{1}{n}\right)^n
=
\frac{1}{e}
\approx
0.3679
$$

So roughly **36.8% of the original data is left out** of each bootstrap sample. These leftover rows are called the **out-of-bag (OOB)** samples, and can be used as a free, built-in validation set for that tree.

### Step 2: Build each decision tree

For every split in every tree:

* **2a.** From all $p$ features, randomly select $m$ candidate features.
* **2b.** Among just those $m$ features, pick the best feature and threshold to split on (using Gini impurity or information gain).
* **2c.** Recurse on the left and right child nodes until they're pure, or a stopping condition (like minimum samples) is hit.

A common default for $m$ is:

$$
\mathbf{m} \approx \sqrt{\mathbf{p}} \quad \text{(classification)}
\qquad\qquad
\mathbf{m} \approx \frac{\mathbf{p}}{3} \quad \text{(regression)}
$$

**Gini impurity**, used to score how good a split is for classification, is:

$$
\mathbf{Gini}
=
1 - \sum_{i=1}^{C} \mathbf{p}_i^2
$$

where $C$ is the number of classes and $p_i$ is the proportion of class $i$ at that node. A pure node (only one class present) has a Gini impurity of $0$.

### Step 3: Repeat B times

Steps 1–2 are repeated to build $B$ independent trees, typically somewhere between 100 and 500.

### Step 4: Aggregate the predictions

* **Classification:** every tree votes, and the majority vote wins (**majority voting**).
* **Regression:** every tree's prediction is averaged (**averaging**):

$$
\mathbf{\hat{y}}
=
\frac{1}{B}
\sum_{b=1}^{B}
\mathbf{h}_b(\mathbf{x})
$$

where $h_b$ is the prediction from the $b$-th tree.

## 4. Key hyperparameters

| **hyperparameter**  | **what it controls**                                  |
| -------------------- | ------------------------------------------------------ |
| `n_estimators`        | Number of trees $B$ in the forest                      |
| `max_features`        | Number of candidate features $m$ considered per split  |
| `max_depth`           | Maximum depth allowed for each individual tree         |
| `min_samples_split`   | Minimum number of samples required to split a node     |
| `min_samples_leaf`    | Minimum number of samples allowed in a leaf node        |

More trees generally means more stable predictions, at the cost of more compute — unlike boosting, adding more trees here doesn't really risk overfitting, since each tree is trained independently.

## 5. Evaluating the model

For regression tasks, the usual metrics apply:

* **MAE (Mean Absolute Error):** average of the absolute differences between predictions and actual values.
* **MSE (Mean Squared Error):** average squared difference. Punishes large errors more than small ones.
* **RMSE (Root Mean Squared Error):** square root of MSE, back in the original units.
* **$R^2$ (R-squared):** fraction of the variance in $y$ explained by the model.

For classification, accuracy, precision/recall, and confusion matrices are more typical.

Random Forest also gives a free bonus: **feature importances**, which measure how much each feature reduces impurity across all the trees, giving a quick view of which inputs matter most.

## 6. Assumptions and things to watch out for

Random Forest is much more forgiving than linear regression, since it doesn't assume any particular shape for the relationship between $x$ and $y$. Still, a few things are worth knowing:

* **No linearity assumption:** trees split on thresholds, so non-linear relationships and feature interactions are picked up naturally.
* **Feature scaling isn't required:** since splits just compare a feature to a threshold, standardising $x$ makes no difference to the result.
* **Correlated features can dilute importances:** if two features carry the same information, the model may split importance between them, making both look individually less important than they really are.
* **Can still overfit:** with very deep trees and too few samples per leaf, individual trees can memorise noise, even though averaging many of them helps a lot.
* **Less interpretable than a single tree:** you gain accuracy but lose the easy "just follow the tree" explainability of a lone decision tree.

## 7. Summary

| **concept**          | **stuff to remember**                                                                              |
| ---------------------- | ----------------------------------------------------------------------------------------------------- |
| **Model**             | Average (regression) or majority vote (classification) over $B$ independently trained trees          |
| **Randomness source** | Bootstrap sampling of rows + random subsetting of features at every split                             |
| **OOB samples**       | ~36.8% of data left out of each bootstrap sample, usable as a built-in validation set                 |
| **Split criterion**   | Gini impurity or information gain (classification), variance reduction (regression)                    |
| **Key hyperparams**   | `n_estimators`, `max_features`, `max_depth`, `min_samples_split`, `min_samples_leaf`                  |
| **Evaluation**        | MAE / MSE / RMSE / $R^2$ (regression), accuracy / precision / recall (classification)                 |
| **Bonus**             | Feature importances, showing which inputs the forest relies on most                                    |

## 8. Resources we used

* [StatQuest - Random Forests Part 1: Building, Using and Evaluating](https://www.youtube.com/watch?v=J4Wdy0Wc_xQ) — covers bagging, OOB samples, and evaluation in one go.
* [StatQuest - Random Forests Part 2: Missing Data and Clustering](https://www.youtube.com/watch?v=sQ870aTKqiM) — a good follow-up on missing data handling and proximity matrices.
* [StatQuest - Decision Trees, Clearly Explained](https://www.youtube.com/watch?v=_L39rN6gz7Y) — worth watching first if the splitting logic in Section 3 feels unfamiliar, since Random Forest is built directly on top of it.
