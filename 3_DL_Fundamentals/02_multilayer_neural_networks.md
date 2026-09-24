# Multilayer Neural Networks

Modern neural networks typically have more than one hidden layer, and often many units per layer.

In theory, a single hidden layer with a large number of units has the ability to approximate most functions. However, the learning task of discovering a good solution is made much easier with multiple layers, each of modest size.

---

## How Multiple Layers Work

The **first hidden layer** activations are computed as:

$$
A_k^{(1)} = h_k^{(1)}(X)
= g\left(w_{k0}^{(1)} + \sum_{j=1}^{p} w_{kj}^{(1)}X_j\right)
$$

for:

$$
k = 1, \ldots, K_1
$$

The **second hidden layer** treats the activations $A_k^{(1)}$ of the first hidden layer as inputs and computes new activations:

$$
A_\ell^{(2)} = h_\ell^{(2)}(X)
= g\left(w_{\ell0}^{(2)} + \sum_{k=1}^{K_1} w_{\ell k}^{(2)}A_k^{(1)}\right)
$$

for:

$$
\ell = 1, \ldots, K_2
$$

Notice that each of the activations in the second layer,

$$
A_\ell^{(2)} = h_\ell^{(2)}(X)
$$

is a function of the input vector $X$.

This is because while they are explicitly a function of the activations $A_k^{(1)}$ from layer $L_1$, these in turn are functions of $X$.

This would also be the case with more hidden layers.

Thus, through a **chain of transformations**, the network is able to build up fairly complex transformations of $X$ that ultimately feed into the output layer as features.

<img width="752" height="673" alt="Screenshot 2026-09-23 232553" src="https://github.com/user-attachments/assets/1954f54a-9809-4c82-a551-a33062f0586d" />

---

## Layer-Specific Notation

We have introduced additional superscript notation such as:

- $h_\ell^{(2)}(X)$ — activation function/output associated with layer 2
- $w_{\ell j}^{(2)}$ — weight associated with layer 2

The notation **$W^1$** represents the entire matrix of weights that feed from the input layer to the first hidden layer $L_1$.

For the MNIST example, this matrix will have:

$$
785 \times 256 = 200,960
$$

elements.

There are **785 rather than 784** because we must account for the intercept or **bias term**.

Each element $A_k^{(1)}$ feeds to the second hidden layer $L_2$ via the matrix of weights **$W^2$**, which has dimensions:

$$
257 \times 128 = 32,896
$$

---
## Output Layer

We now get to the output layer, where we have **ten responses rather than one**.

The first step is to compute ten different linear models:

$$ Z_m = \beta_{m0} + \sum_{\ell=1}^{K_2} \beta_{m\ell}h_\ell^{(2)}(X) $$

or equivalently:

$$ Z_m = \beta_{m0} + \sum_{\ell=1}^{K_2} \beta_{m\ell}A_\ell^{(2)} $$

for:

$$ m = 0,1,\ldots,9 $$

The matrix **$B$** stores all:

$$ 129 \times 10 = 1,290 $$

of these weights.

If these were all separate quantitative responses, we would simply set each:

$$ f_m(X) = Z_m $$

and be done.

However, we would like our estimates to represent **class probabilities**:

$$ f_m(X) = P(Y=m\mid X) $$

just like in multinomial logistic regression.

So we use the **softmax activation function**:

$$ f_m(X) = P(Y=m\mid X) = \frac{e^{Z_m}}{\sum_{\ell=0}^{9}e^{Z_\ell}} $$

for:

$$ m = 0,1,\ldots,9 $$

This ensures that the 10 numbers behave like probabilities:

- They are **non-negative**
- They **sum to one**

Even though the goal is to build a classifier, our model actually estimates a **probability for each of the 10 classes**.

The classifier then assigns the image to the class with the **highest probability**.

---

## Training the Network

To train this network, since the response is qualitative, we look for coefficient estimates that minimize the **negative multinomial log-likelihood**:

$$
-\sum_{i=1}^{n}\sum_{m=0}^{9}
y_{im}\log\left(f_m(x_i)\right)
$$

This is also known as **cross-entropy loss**.

If the response were quantitative, we would instead minimize **squared-error loss**.

<img width="750" height="237" alt="Screenshot 2026-09-23 232635" src="https://github.com/user-attachments/assets/414f980c-2d07-4d51-9629-82caba8129b4" />

---

## MNIST Example

We will illustrate a large dense network using the famous and publicly available **MNIST handwritten digit dataset**.

### What is MNIST?

The idea is to build a model to classify images into their correct digit class **0–9**.

Every image has:

$$
p = 28 \times 28 = 784
$$

pixels.

Each pixel is an **8-bit grayscale value between 0 and 255**, representing the relative amount of the written digit in that tiny square.

These pixels are stored in the input vector $X$.

The output is the class label, represented by a vector:

$$
Y=(Y_0,Y_1,\ldots,Y_9)
$$

with a **1 in the position corresponding to the label** and **0s elsewhere**.

In the machine learning community, this is known as **one-hot encoding**.

The original MNIST dataset contains:

- **60,000 training images**
- **10,000 test images**

---

# MNIST Classifier — Implementation

> The complete Python implementation will be added separately.
[`mnist_mlp.py`](mnist_mlp.py)
---

## Why do we divide by 255?

The original pixel values range from **0 to 255**.

Dividing by 255 scales them to the range:

$$
0 \leq X \leq 1
$$

This is called **normalization** and makes the input values easier for the neural network to work with.

---

## Train-Test Split

We use **80% of the data for training** and **20% for testing**.

---

## Building the MLP Classifier

Our network has the structure:

$$
784 \rightarrow 128 \rightarrow 64 \rightarrow 10
$$

where:

- **784** → input pixels
- **128** → first hidden layer
- **64** → second hidden layer
- **10** → output classes (digits 0–9)

Here:

- `activation="relu"` uses the **ReLU activation function**
- `solver="adam"` uses the **Adam optimizer**
- `max_iter=20` allows the model to train for up to 20 iterations

---

## Training the Model

The MLP is trained using the training data.

The trained model is then used to predict the digits in the test set.

---

## Evaluating the Model

The model's performance is evaluated using **accuracy**.

---

## Visualizing Predictions

The predictions can be visualized by displaying test images along with:

- **Predicted digit** → what the neural network predicted
- **True digit** → the actual label

---

## Dataset Dimensions

The loaded dataset contains:

- **70,000 images**
- **784 features per image**

After the 80/20 train-test split:

- **56,000 training instances**
- **14,000 evaluation instances**

So:

$$
56,000 + 14,000 = 70,000
$$

---

## Learning

An MLP connects multiple layers of neurons so that each layer can transform the representation produced by the previous layer.

For MNIST, the network transforms:

$$
784\text{ pixels}
\rightarrow
128
\rightarrow
64
\rightarrow
10\text{ digit probabilities}
$$

The final output gives a probability for each digit from **0 to 9**, and the class with the highest probability becomes the predicted digit.
