# Logistic Regression
---
*HAM ARC ML Sessions, 2026–27.*
*Group: Nawfal, [Rishi](https://github.com/cellrishi-code).*


## 1. Theory

Logistic Regression is a fundamental supervised learning algorithm used
primarily for binary classification tasks (where the output belongs to
one of two classes, e.g., 0 or 1, True or False, Spam or Not Spam).
Despite having "regression" in its name, it is a classification model.

Unlike Linear Regression, which outputs continuous values (which could
be less than 0 or greater than 1), Logistic Regression predicts the
probability that a given input point belongs to a certain class. To
ensure the output probability is always bounded between 0 and 1, it uses
a mathematical function called the Sigmoid function (or Logistic
function) to squash the output of a linear equation.

## 2. The Mathematics

The Hypothesis:

Logistic regression builds upon the linear regression equation:

$$
z = w \cdot x + b
$$

Where:

$w$ = weight vector

$x$ = feature vector

$b$ = bias term

To convert this continuous value $z$ into a probability (between 0 and
1), we pass it through the Sigmoid function:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

So, our final hypothesis (predicted probability) becomes:

$$
\hat{y} = P(y=1 \mid x) = \sigma(w \cdot x + b)
$$

Decision Boundary

To classify the output into distinct classes, we use a threshold
(usually 0.5):

If $\hat{y} \geq 0.5$, predict Class 1

If $\hat{y} < 0.5$, predict Class 0

Cost Function (Log Loss / Cross-Entropy)

We cannot use Mean Squared Error (MSE) like in linear regression because
the sigmoid function makes the loss landscape non-convex (full of local
minima). Instead, we use the Binary Cross-Entropy (or Log Loss) cost
function:

$$
J(w, b) = -\frac{1}{m} \sum \left[
y^{(i)} \log(\hat{y}^{(i)}) +
(1-y^{(i)}) \log(1-\hat{y}^{(i)})
\right]
$$

Where $m$ is the number of training examples, $y$ is the actual label,
and $\hat{y}$ is the predicted probability.

Gradient Descent

To minimize the cost function and find the optimal parameters ($w$ and
$b$), we use Gradient Descent. The update rules are derived by
taking the partial derivative of the cost function with respect to the
weights and bias:

$$
w = w - \alpha \frac{1}{m} \sum
\left[
(\hat{y}^{(i)} - y^{(i)})x^{(i)}
\right]
$$

$$
b = b - \alpha \frac{1}{m} \sum
\left[
\hat{y}^{(i)} - y^{(i)}
\right]
$$

Where $\alpha$ is the learning rate.

## Extra Learning Resources:
-- Google developer article: [link](https://developers.google.com/machine-learning/crash-course/logistic-regression)
-- StarQuest video: [link](https://www.youtube.com/watch?v=yIYKR4sgzI8)
-- sklearn [documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)