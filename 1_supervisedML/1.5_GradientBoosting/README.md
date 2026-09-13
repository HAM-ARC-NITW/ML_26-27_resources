# Gradient Boosting

---

*HAM ARC ML Sessions, 2026–27.*
*Group: [Ayaan](https://github.com/Ayaan-Surkhi), [Subham](https://github.com/SubhamJM)*

## 1. What is it?

Gradient Boosting is a supervised learning algorithm that builds a strong predictive model out of many weak ones — usually small **decision trees** — added together **sequentially**. It can be used for both:

* **Regression** — predicting a continuous number
* **Classification** — predicting a class or probability

The key idea is:

> **Each new tree is trained to correct the mistakes of everything built so far.**

Some examples where this shows up:

* Predicting a house's price from its features, when the relationship isn't a clean straight line
* Ranking search results or recommendations (this is basically what powers XGBoost/LightGBM in most Kaggle-style competitions)
* Predicting a batsman's runs scored from balls faced, once the relationship stops being linear (the example we used below)

## 2. Intuition

Linear regression fits one line and stops. Gradient boosting instead fits a *sequence* of small trees, where:

1. The first tree makes a rough, imperfect prediction.
2. We look at where it was wrong (the **residuals**).
3. The next tree is trained specifically to predict those residuals, not the original target.
4. We add this new tree's prediction on top of the first, scaled down a bit.
5. Repeat, each time chipping away at whatever error is left over.

If you imagine sculpting a statue, the first tree is a rough block shape, and every following tree is a smaller, more careful correction on top of it.

## 3. The model

The final prediction after $M$ trees is just a sum of all of them:

$$
\mathbf{\hat{y}}
=
\mathbf{F}_0(\mathbf{x})
+
\alpha
\sum_{m=1}^{M}
\mathbf{h}_m(\mathbf{x})
$$

where $F_0$ is the initial prediction (usually just the mean of $y$), $h_m$ is the $m$-th tree, and $\alpha$ is the **learning rate**, which controls how much weight each new tree's correction gets.

Each individual tree $h_m$ is usually shallow — often just a **stump** (a single split) or a tree with a max depth of 2–5. Weak, high-bias learners are combined into one strong, low-bias model.

## 4. Measuring how good the model is: the loss function

Just like linear regression, we need a way to measure how wrong the current model is. For regression, the usual choice is again **mean squared error (MSE)**:

$$
\mathbf{L}(\mathbf{y}, \mathbf{\hat{y}})
=
\frac{1}{n}
\sum_{i=1}^{n}
\left(
\mathbf{y}_i - \mathbf{\hat{y}}_i
\right)^2
$$

For classification, this is usually swapped out for **log loss** instead, but the boosting procedure itself stays the same.

## 5. How the boosting actually works

This is where the "gradient" in gradient boosting comes from. At each step $m$, we want to find whatever correction reduces the loss the fastest — and it turns out that direction is just the **negative gradient of the loss** with respect to the current predictions.

For plain MSE, this negative gradient works out to simply be the residuals:

$$
\mathbf{r}_i
=
-\frac{\partial \mathbf{L}}{\partial \mathbf{\hat{y}}_i}
=
\mathbf{y}_i - \mathbf{\hat{y}}_i
$$

So, each round of training looks like:

1. Compute the residuals $r_i$ between the true values and the model's current predictions.
2. Fit a new small tree $h_m$ to predict these residuals from $x$.
3. Update the model:

$$
\mathbf{F}_m(\mathbf{x})
\leftarrow
\mathbf{F}_{m-1}(\mathbf{x})
+
\alpha \, \mathbf{h}_m(\mathbf{x})
$$

4. Repeat for $M$ rounds, or until the loss on a validation set stops improving.

This is essentially **gradient descent, but performed in "function space"** instead of parameter space — instead of nudging a slope and intercept, we're adding a whole new function (tree) each step.

## 6. The learning rate and number of trees

Two hyperparameters matter a lot here, and they trade off against each other:

* **Learning rate ($\alpha$):** shrinks the contribution of every tree. A small learning rate makes each individual correction gentle, so the model needs more trees to fit the data well, but tends to generalise better and overfit less.
* **Number of trees ($M$):** more trees mean more corrections, and hence lower training error — but too many, especially with too high a learning rate, will start fitting noise in the training data (overfitting).

In practice, a common recipe is: pick a small learning rate (e.g. 0.05–0.1) and use a validation set or early stopping to decide how many trees to add.

This is the boosting equivalent of the "learning rate too large" problem we ran into with gradient descent for linear regression — pick it badly, and the model won't converge well.

## 7. Evaluating the model

The same regression metrics from linear regression apply directly here:

* **MSE (Mean Squared Error):** average squared difference between predictions and actual values. Lower is better.
* **RMSE (Root Mean Squared Error):** square root of MSE, back in the original units. E.g. an RMSE of $1.2$ means the typical prediction error is around $1.2$ runs.
* **$R^2$ (R-squared):** the fraction of the variance in $y$ explained by the model. $1$ is a perfect fit, $0$ is no better than predicting the mean.

For classification tasks, accuracy, log loss, and AUC are used instead.

## 8. Assumptions and things to watch out for

Gradient boosting is much less strict about its assumptions than linear regression — since it isn't fitting a single straight line, it doesn't need linearity, and it handles non-linear relationships and feature interactions naturally. Still, a few things matter in practice:

* **Overfitting risk:** because boosting keeps reducing training error round after round, it can memorise noise if you use too many trees, too high a learning rate, or trees that are too deep. Cross-validation or early stopping is important.
* **Sensitivity to outliers:** because we're directly fitting residuals, MSE-based boosting can end up spending a lot of effort chasing a few extreme points.
* **Feature scaling isn't required:** unlike linear regression, tree-based splits don't care whether $x$ is standardised, since a split just picks a threshold.
* **Sequential, so slower to train:** unlike random forests, trees can't be built in parallel, since each one depends on the output of all previous trees.

## 9. Summary

| **concept**        | **stuff to remember**                                                                                    |
| ------------------- | ---------------------------------------------------------------------------------------------------------- |
| **Model**           | Sum of many small trees: $F_M(x) = F_0(x) + \alpha \sum_{m=1}^{M} h_m(x)$                                 |
| **Loss function**   | MSE for regression, log loss for classification                                                            |
| **Training goal**   | Sequentially fit each new tree to the negative gradient (residuals) of the loss so far                      |
| **Key hyperparams** | Learning rate $\alpha$ (shrinkage) and number of trees $M$, which trade off against each other              |
| **Weak learner**    | Usually a shallow decision tree (a stump, or depth 2–5)                                                     |
| **Evaluation**      | MSE or RMSE for error size, $R^2$ for variance explained (regression); accuracy/log loss/AUC (classification) |
| **Key risk**        | Overfitting if too many trees, too high a learning rate, or too-deep trees are used                        |

## 10. Resources

* [StatQuest - Gradient Boost, Part 1: Regression Main Ideas](https://www.youtube.com/watch?v=3CC4N4z3GJc) — the clearest intro to the residual-fitting idea.
* [StatQuest - Gradient Boost, Part 2: Regression Details](https://www.youtube.com/watch?v=2xudPOBz-vs) — walks through the math step by step.
* [StatQuest - AdaBoost, Clearly Explained](https://www.youtube.com/watch?v=LsK-xG1cLYA) — a good companion, since AdaBoost is boosting's older, simpler cousin.
