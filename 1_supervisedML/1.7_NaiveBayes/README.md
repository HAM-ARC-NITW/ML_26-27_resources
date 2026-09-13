# Naive Bayes

---

*HAM ARC ML Sessions, 2026–27.*
*Group: [Aadit](https://github.com/aadit-n), [Wahid](https://github.com/Abdul-Wahid2008), [Jibendra](https://github.com/Galaxyyus).*

## 1. What is it?

Naive Bayes is a supervised learning algorithm used for **classification**, meaning it predicts a category rather than a continuous number. It's built directly on **Bayes' theorem**, with one simplifying ("naive") assumption that makes the maths tractable: it treats every feature as independent of every other feature, given the class.

Some examples where this shows up:

* Classifying an email as spam or not spam from the words it contains (the example we used below)
* Sentiment analysis — deciding if a review is positive or negative
* Document/topic classification, and as a fast baseline for most text classification tasks in general

## 2. Intuition

Bayes' theorem lets us flip a probability around. We don't directly know $P(\text{class} \mid \text{features})$, but it's often much easier to estimate $P(\text{features} \mid \text{class})$ from training data — how often certain words show up in spam, for instance. Bayes' theorem converts one into the other:

$$
\mathbf{P}(\mathbf{y} \mid \mathbf{x})
=
\frac{
\mathbf{P}(\mathbf{x} \mid \mathbf{y}) \, \mathbf{P}(\mathbf{y})
}{
\mathbf{P}(\mathbf{x})
}
$$

In words: $\text{posterior} \propto \text{likelihood} \times \text{prior}$.

The "naive" part comes in when $x$ has many features (e.g. many words in an email). Computing $P(x_1, x_2, \dots, x_n \mid y)$ exactly would require modelling how every feature interacts with every other one, which needs a lot of data. Naive Bayes sidesteps this by simply **assuming the features are conditionally independent given the class** — an assumption that's usually false in the real world, but works surprisingly well in practice anyway.

## 3. The model

Given the independence assumption, the joint likelihood of all the features factorises into a simple product:

$$
\mathbf{P}(\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_n \mid \mathbf{y})
=
\prod_{i=1}^{n}
\mathbf{P}(\mathbf{x}_i \mid \mathbf{y})
$$

So the full classification rule becomes: pick whichever class $y$ maximises

$$
\mathbf{P}(\mathbf{y} \mid \mathbf{x})
\;\propto\;
\mathbf{P}(\mathbf{y})
\prod_{i=1}^{n}
\mathbf{P}(\mathbf{x}_i \mid \mathbf{y})
$$

We can drop the denominator $P(x)$ entirely, since it's the same for every class and doesn't affect which one wins.

In practice, we work in log-space to avoid multiplying together lots of small probabilities (which underflows quickly):

$$
\mathbf{\hat{y}}
=
\arg\max_{\mathbf{y}}
\left[
\log \mathbf{P}(\mathbf{y})
+
\sum_{i=1}^{n}
\log \mathbf{P}(\mathbf{x}_i \mid \mathbf{y})
\right]
$$

## 4. Flavours of Naive Bayes

The only thing that changes between variants is *how* $P(x_i \mid y)$ is modelled, depending on what kind of feature $x_i$ is:

* **Multinomial Naive Bayes:** for count data, like word counts in a document. Assumes each feature follows a multinomial distribution per class. The standard choice for text classification (used below).
* **Bernoulli Naive Bayes:** for binary features, like whether a word is present or absent (ignoring how many times it appears).
* **Gaussian Naive Bayes:** for continuous features, like height or temperature. Assumes each feature is normally distributed within each class, and estimates a mean and variance per feature, per class.

### A closer look at Gaussian Naive Bayes

Where multinomial NB estimates $P(x_i \mid y)$ by counting, Gaussian NB estimates it by fitting a **normal distribution** to each feature, separately for each class. For class $k$ and feature $j$, we estimate a mean $\mu_{kj}$ and variance $\sigma_{kj}^2$ from the training examples belonging to class $k$, and then plug a new value $x_j$ into the Gaussian density function:

$$
\mathbf{f}_{kj}(\mathbf{x}_j)
=
\frac{1}{\sqrt{2\pi\sigma_{kj}^{2}}}
\exp\!\left(
-\frac{(\mathbf{x}_j-\mu_{kj})^{2}}{2\sigma_{kj}^{2}}
\right)
$$

Just as before, the independence assumption means the $p$ predictors within class $k$ are treated as independent, so the class-conditional density is just the product of each feature's individual Gaussian density:

$$
\mathbf{f}_k(\mathbf{x})
=
\prod_{j=1}^{p}
\mathbf{f}_{kj}(\mathbf{x}_j)
$$

This is really the same "naive" trick as the multinomial case — we've just swapped out *how* $P(x_i \mid y)$ is computed, from a word-count ratio to a Gaussian density.

## 5. Training: estimating the probabilities

Training a Naive Bayes model just means estimating probabilities by counting, which is what makes it so fast compared to iterative methods like gradient descent.

For the **prior** $P(y)$, this is simply how often each class appears in the training data:

$$
\mathbf{P}(\mathbf{y})
=
\frac{\text{count of class } y}{\text{total training examples}}
$$

For **multinomial** features (e.g. word counts), the likelihood of feature $i$ given class $y$ is:

$$
\mathbf{P}(\mathbf{x}_i \mid \mathbf{y})
=
\frac{
\text{count of } x_i \text{ in class } y + \alpha
}{
\text{total feature count in class } y + \alpha n
}
$$

The $\alpha$ term here is **Laplace (additive) smoothing**. Without it, any word that never appeared in training for a given class would get a probability of exactly zero — which would zero out the entire product the moment that word shows up, no matter how confident every other word is. Adding a small $\alpha$ (usually $1$) prevents this.

## 6. Evaluating the model

Since this is a classification problem, the usual classification metrics apply:

* **Accuracy:** fraction of predictions that were correct overall.
* **Precision:** of everything predicted as the positive class (e.g. spam), how much actually was.
* **Recall:** of everything that actually was the positive class, how much did we catch.
* **F1 score:** the harmonic mean of precision and recall, useful when classes are imbalanced.

For spam detection specifically, recall matters if you don't want spam slipping into the inbox, while precision matters if you don't want real emails wrongly binned as spam — the two usually trade off against each other.

## 7. Assumptions

* **Conditional independence:** the defining assumption — every feature is assumed independent of every other feature, given the class. This is almost never exactly true (in text, word order and co-occurrence obviously matter), but the model is often still accurate enough in practice, especially for text.
* **Correct distributional choice:** Gaussian NB assumes features are normally distributed within each class; if that's a bad fit for the data, performance suffers.
* **Enough smoothing:** without Laplace smoothing, unseen feature/class combinations at prediction time can break the model outright (a zero probability multiplied through everything).
* **Feature relevance:** irrelevant or purely noisy features contribute equally to the product as informative ones, since the model doesn't learn feature weights the way linear regression does.

## 8. Code

For the dataset, we generated a small synthetic set of spam and non-spam ("ham") text messages, built from a handful of templates with light word-level noise added for variety.

We implemented it two ways to check that they agree:

1. **Multinomial Naive Bayes written from scratch**, using word counts, log-probabilities, and Laplace smoothing directly.
2. **scikit-learn's built-in `MultinomialNB`**, fed the same bag-of-words features from `CountVectorizer`.

### Output

```text
From scratch: acc=1.000 precision=1.000 recall=1.000 f1=1.000
Sklearn     : acc=1.000 precision=1.000 recall=1.000 f1=1.000

'free cash prize click now' -> from scratch: spam, sklearn: spam
'lets meet for the project review' -> from scratch: ham, sklearn: ham
```

Both implementations agree perfectly on the held-out test messages and on two brand-new example messages, which is a good sanity check that the from-scratch log-probability calculation matches sklearn's internals. (The dataset here is small and templated, so a perfect score is expected — the point is the two implementations matching, not that Naive Bayes is unbeatable on real-world spam.)

### Gaussian Naive Bayes, from scratch — Iris dataset

To check the Gaussian variant too, we implemented it from scratch on the classic Iris dataset (150 flowers, 4 measurements each, 3 species), using the mean/variance-per-class approach from Section 4:

```python
import numpy as np
from sklearn.datasets import load_iris

iris = load_iris()
X, y = iris.data, iris.target

np.random.seed(0)
indices = np.random.permutation(len(X))
split = int(0.8 * len(X))
train_idx, test_idx = indices[:split], indices[split:]
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

class_stats = {}
for cls in np.unique(y_train):
    cls_data = X_train[y_train == cls]
    class_stats[cls] = {
        "mean": np.mean(cls_data, axis=0),
        "var": np.var(cls_data, axis=0),
    }

def gaussian(x, mean, var):
    return (1 / np.sqrt(2 * np.pi * var)) * np.exp(-((x - mean) ** 2) / (2 * var))

def predict(sample):
    posteriors = {}
    for cls, stats in class_stats.items():
        likelihoods = gaussian(sample, stats["mean"], stats["var"])
        posteriors[cls] = np.prod(likelihoods)  # naive independence assumption
    return max(posteriors, key=posteriors.get)

predictions = [predict(x) for x in X_test]
accuracy = np.mean(np.array(predictions) == y_test)
```

**Output:**

```text
Number of test samples: 30
Accuracy: 0.9333

First 3 test predictions:
Actual: setosa    | Predicted: setosa
Actual: virginica | Predicted: virginica
Actual: setosa    | Predicted: setosa
```

### Gaussian Naive Bayes with sklearn — Breast Cancer dataset

We then compared against scikit-learn's built-in `GaussianNB`, this time on the (harder, higher-dimensional) Breast Cancer Wisconsin dataset — 30 features per sample, classifying tumours as malignant or benign:

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

data = load_breast_cancer()
x, y = data.data, data.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = GaussianNB()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

**Output:**

```text
Accuracy: 0.9737

              precision    recall  f1-score   support

   malignant       1.00      0.93      0.96        43
      benign       0.96      1.00      0.98        71

    accuracy                           0.97       114
   macro avg       0.98      0.97      0.97       114
weighted avg       0.97      0.97      0.97       114
```

Even with the strong (and technically false) independence assumption across 30 correlated features, Gaussian NB still gets a respectable accuracy here — a good illustration of why the "naive" assumption tends to work better in practice than the theory alone would suggest.

### Visualising the decision regions

Restricting to just two features (petal length and petal width) lets us actually plot what Gaussian NB's decision boundary looks like on the Iris dataset:

![Gaussian Naive Bayes decision regions on Iris petal length/width](images/gnb_decision_regions.png)

Because each class's features are modelled as independent Gaussians, the boundaries between regions come out as smooth curves rather than the straight lines you'd get from a linear model, but they're still fairly simple — Gaussian NB doesn't capture complex, twisting boundaries the way a tree-based method might.

## 9. Summary

| **concept**         | **stuff to remember**                                                                                     |
| --------------------- | -------------------------------------------------------------------------------------------------------- |
| **Model**            | Bayes' theorem plus a conditional independence assumption across features                                 |
| **Training goal**    | Estimate priors $P(y)$ and per-feature likelihoods $P(x_i \mid y)$ by counting, with smoothing            |
| **Key trick**        | Work in log-space and drop the shared denominator $P(x)$, since only the *relative* score across classes matters |
| **Flavours**         | Multinomial (counts), Bernoulli (binary presence), Gaussian (continuous)                                   |
| **Smoothing**        | Laplace smoothing ($\alpha$) prevents unseen feature/class pairs from zeroing out a prediction              |
| **Evaluation**       | Accuracy, precision, recall, F1 — precision/recall trade-off matters most for imbalanced classes           |
| **Key assumption**   | Features are conditionally independent given the class (rarely true, often still works)                    |

## 10. Resources we used

* [StatQuest - Naive Bayes, Clearly Explained](https://www.youtube.com/watch?v=O2L2Uv9pdDA) — the best starting point for the core idea and a worked-through example by hand.
* [StatQuest - Naive Bayes: Multinomial Naive Bayes](https://www.youtube.com/watch?v=temrxxaQjJk) — focuses specifically on the multinomial variant used in text classification.
* [StatQuest - Gaussian Naive Bayes, Clearly Explained](https://www.youtube.com/watch?v=H3EjCKtlVog) — covers the continuous-feature variant, and how it differs from the multinomial case.
* [Naïve Bayes Algorithm (Medium / Analytics Vidhya)](https://medium.com/analytics-vidhya/na%C3%AFve-bayes-algorithm-5bf31e9032a2) — a concise written walkthrough of the theory.
* [GeeksforGeeks - Naive Bayes Classifiers](https://www.geeksforgeeks.org/machine-learning/naive-bayes-classifiers/) — reference page covering the different variants and worked examples.
* *An Introduction to Statistical Learning* by Gareth James, Trevor Hastie et al. (2nd edition) — the textbook derivation of the Gaussian Naive Bayes classifier used in Section 4.
