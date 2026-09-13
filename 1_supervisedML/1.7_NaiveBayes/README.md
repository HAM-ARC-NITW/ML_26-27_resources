# Naive Bayes

---

*HAM ARC ML Sessions, 2026–27.*
*Group: [Pranav](https://github.com/falcon370), [Ashutosh].*

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

## 8. Summary

| **concept**         | **stuff to remember**                                                                                     |
| --------------------- | -------------------------------------------------------------------------------------------------------- |
| **Model**            | Bayes' theorem plus a conditional independence assumption across features                                 |
| **Training goal**    | Estimate priors $P(y)$ and per-feature likelihoods $P(x_i \mid y)$ by counting, with smoothing            |
| **Key trick**        | Work in log-space and drop the shared denominator $P(x)$, since only the *relative* score across classes matters |
| **Flavours**         | Multinomial (counts), Bernoulli (binary presence), Gaussian (continuous)                                   |
| **Smoothing**        | Laplace smoothing ($\alpha$) prevents unseen feature/class pairs from zeroing out a prediction              |
| **Evaluation**       | Accuracy, precision, recall, F1 — precision/recall trade-off matters most for imbalanced classes           |
| **Key assumption**   | Features are conditionally independent given the class (rarely true, often still works)                    |

## 9. Resources we used

* [StatQuest - Naive Bayes, Clearly Explained](https://www.youtube.com/watch?v=O2L2Uv9pdDA) — the best starting point for the core idea and a worked-through example by hand.
* [StatQuest - Naive Bayes: Multinomial Naive Bayes](https://www.youtube.com/watch?v=temrxxaQjJk) — focuses specifically on the multinomial variant used in text classification.
* [StatQuest - Gaussian Naive Bayes, Clearly Explained](https://www.youtube.com/watch?v=H3EjCKtlVog) — covers the continuous-feature variant, and how it differs from the multinomial case.
* [Naïve Bayes Algorithm (Medium / Analytics Vidhya)](https://medium.com/analytics-vidhya/na%C3%AFve-bayes-algorithm-5bf31e9032a2) — a concise written walkthrough of the theory.
* [GeeksforGeeks - Naive Bayes Classifiers](https://www.geeksforgeeks.org/machine-learning/naive-bayes-classifiers/) — reference page covering the different variants and worked examples.
* *An Introduction to Statistical Learning* by Gareth James, Trevor Hastie et al. (2nd edition) — the textbook derivation of the Gaussian Naive Bayes classifier used in Section 4.
