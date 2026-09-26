# Backpropagation

Backpropagation is a remarkably fast, efficient algorithm to untangle the massive web of
interconnected variables and equations in a neural network.

To illustrate backpropagation's efficiency, Michael Nielsen compares it to a simple and intuitive
alternative approach to computing the gradient of a neural network's loss function in his online
textbook, *"Neural Networks and Deep Learning"* (you can look it up, it's actually really helpful).

As Nielsen explains, one can easily estimate the impact of changes to any specific weight $w_j$ in
the network by simply completing a forward pass for two slightly different values of $w_j$, while
keeping all other parameters unchanged, and comparing the resulting loss for each pass. By
formalizing that process into a straightforward equation and implementing a few lines of code in
Python, you can automate that process for each weight in the network.

But now imagine that there are 1 million weights in your model, which would be quite modest for a
modern deep learning model. To compute the entire gradient, you'd need to complete **1,000,001
forward passes** through the network: 1 to establish a baseline, and then another pass to evaluate
changes to each of the million weights.

> **Backpropagation can achieve the same goal in just 2 passes: 1 forward pass and 1 backward pass.**

---

## Key Mathematical Concepts

To simplify an explanation of how backpropagation works, it will be helpful to first briefly review
some core mathematical concepts and terminology.

A **derivative** is the rate of change in an equation at a specific instant. In a linear equation,
the rate of change is a constant slope. In a nonlinear equation, like those used for activation
functions, this slope varies. **Differentiation** is the process of finding the derivative of a
specific function. By differentiating a nonlinear function, we can find the slope — its
instantaneous rate of change — at any specific point in the curve.

In functions with multiple variables, a **partial derivative** is the derivative of one variable
concerning the others. If we change one variable, but keep the others the same, how does the output
of the overall function change? The activation functions of individual nodes in a neural network
have many variables, including the many inputs from neurons in previous layers and the weights
applied to those inputs. When dealing with a specific node $n$, finding the partial derivatives of
the activation functions of neurons from the previous layer allows us to isolate the impact of each
on the overall output of $n$'s own activation function.

A **gradient** is a vector containing all the partial derivatives of a function with multiple
variables. It essentially represents all the factors affecting the rate at which the output of a
complex equation will change following a change in the input.

The **chain rule** is a formula for calculating the derivatives of functions that involve not just
multiple variables, but multiple functions. For example, consider a composite function
$f(x) = A(B(x))$. The derivative of the composite function $f$ is equal to the derivative of the
outer function $A$ multiplied by the derivative of the inner function $B$:

$$f'(x) = A'(B(x)) \cdot B'(x)$$

The chain rule is essential to calculating the derivatives of activation functions in neural
networks, which are composed of the outputs of activation functions of other neurons in previous
layers.

Though the logic behind backpropagation is relatively straightforward, the mathematics and notation
can become very complex, especially for those unfamiliar with variable calculus.

---

## Notation Reference

| Symbol | Meaning |
|--------|---------|
| $L$ | Number of layers (excluding input layer) |
| $n_l$ | Number of neurons in layer $l$ |
| $W^{(l)} \in \mathbb{R}^{n_l \times n_{l-1}}$ | Weight matrix for layer $l$ |
| $b^{(l)} \in \mathbb{R}^{n_l}$ | Bias vector for layer $l$ |
| $z^{(l)} \in \mathbb{R}^{n_l}$ | Pre-activation (weighted input) for layer $l$ |
| $a^{(l)} \in \mathbb{R}^{n_l}$ | Post-activation output of layer $l$ |
| $\sigma^{(l)}$ | Activation function for layer $l$ |
| $a^{(0)} = x$ | Input to the network |
| $\hat{y} = a^{(L)}$ | Network output |
| $\mathcal{L} = \mathcal{L}(a^{(L)}, y)$ | Loss for a single training example |

---

## Forward Pass

For each layer $l = 1, 2, \ldots, L$:

$$z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)}$$

$$a^{(l)} = \sigma^{(l)}\!\left(z^{(l)}\right)$$

In component form for neuron $j$ in layer $l$:

$$z_j^{(l)} = \sum_{k=1}^{n_{l-1}} W_{jk}^{(l)}\, a_k^{(l-1)} + b_j^{(l)}, \qquad a_j^{(l)} = \sigma^{(l)}\!\left(z_j^{(l)}\right)$$

---

## Goal: Compute Gradients of the Loss

We want to find:

$$\frac{\partial \mathcal{L}}{\partial W_{jk}^{(l)}} \quad \text{and} \quad \frac{\partial \mathcal{L}}{\partial b_j^{(l)}} \quad \text{for all } l, j, k$$

so we can perform **gradient descent**:

$$W_{jk}^{(l)} \leftarrow W_{jk}^{(l)} - \eta\, \frac{\partial \mathcal{L}}{\partial W_{jk}^{(l)}}, \qquad b_j^{(l)} \leftarrow b_j^{(l)} - \eta\, \frac{\partial \mathcal{L}}{\partial b_j^{(l)}}$$

---

## The Error Signal $\delta$

Define the **error signal** (local gradient) for neuron $j$ in layer $l$:

$$\delta_j^{(l)} \equiv \frac{\partial \mathcal{L}}{\partial z_j^{(l)}}$$

In vector form:

$$\delta^{(l)} = \nabla_{z^{(l)}} \mathcal{L}$$

This measures how sensitive the loss is to changes in the pre-activation of that neuron.

---

## The Four Fundamental Equations (BP1–BP4)

### BP1 — Output Layer Error

$$\boxed{\delta^{(L)} = \nabla_{a^{(L)}} \mathcal{L} \;\odot\; \sigma'^{(L)}\!\left(z^{(L)}\right)}$$

Component form:

$$\delta_j^{(L)} = \frac{\partial \mathcal{L}}{\partial a_j^{(L)}} \cdot \sigma'^{(L)}\!\left(z_j^{(L)}\right)$$

**Derivation sketch:**

$$\delta_j^{(L)} = \frac{\partial \mathcal{L}}{\partial z_j^{(L)}} = \sum_k \frac{\partial \mathcal{L}}{\partial a_k^{(L)}} \cdot \frac{\partial a_k^{(L)}}{\partial z_j^{(L)}}$$

Since $a_k^{(L)} = \sigma^{(L)}(z_k^{(L)})$ is applied elementwise:

$$\frac{\partial a_k^{(L)}}{\partial z_j^{(L)}} = \begin{cases} \sigma'^{(L)}(z_j^{(L)}) & \text{if } k = j \\ 0 & \text{otherwise} \end{cases}$$

So only $k = j$ survives:

$$\delta_j^{(L)} = \frac{\partial \mathcal{L}}{\partial a_j^{(L)}} \cdot \sigma'^{(L)}\!\left(z_j^{(L)}\right)$$

**Example (MSE loss):**

$$\mathcal{L} = \tfrac{1}{2}\left\|a^{(L)} - y\right\|^2 \implies \nabla_{a^{(L)}} \mathcal{L} = a^{(L)} - y$$

$$\therefore \quad \delta^{(L)} = \left(a^{(L)} - y\right) \odot \sigma'^{(L)}\!\left(z^{(L)}\right)$$

---

### BP2 — Error Backpropagation (layer $l$ from $l+1$)

$$\boxed{\delta^{(l)} = \left(\left(W^{(l+1)}\right)^{\!T} \delta^{(l+1)}\right) \odot \sigma'^{(l)}\!\left(z^{(l)}\right)}$$

Component form:

$$\delta_j^{(l)} = \left(\sum_{k=1}^{n_{l+1}} W_{kj}^{(l+1)}\, \delta_k^{(l+1)}\right) \cdot \sigma'^{(l)}\!\left(z_j^{(l)}\right)$$

**Derivation sketch:**

$$\delta_j^{(l)} = \frac{\partial \mathcal{L}}{\partial z_j^{(l)}} = \sum_k \frac{\partial \mathcal{L}}{\partial z_k^{(l+1)}} \cdot \frac{\partial z_k^{(l+1)}}{\partial z_j^{(l)}} = \sum_k \delta_k^{(l+1)} \cdot \frac{\partial z_k^{(l+1)}}{\partial z_j^{(l)}}$$

Now, since:

$$z_k^{(l+1)} = \sum_m W_{km}^{(l+1)}\, a_m^{(l)} + b_k^{(l+1)} = \sum_m W_{km}^{(l+1)}\, \sigma^{(l)}\!\left(z_m^{(l)}\right) + b_k^{(l+1)}$$

We get:

$$\frac{\partial z_k^{(l+1)}}{\partial z_j^{(l)}} = W_{kj}^{(l+1)}\, \sigma'^{(l)}\!\left(z_j^{(l)}\right)$$

Substituting back:

$$\delta_j^{(l)} = \sum_k \delta_k^{(l+1)}\, W_{kj}^{(l+1)}\, \sigma'^{(l)}\!\left(z_j^{(l)}\right) = \left(\sum_k W_{kj}^{(l+1)}\, \delta_k^{(l+1)}\right) \cdot \sigma'^{(l)}\!\left(z_j^{(l)}\right)$$

In matrix form, this is exactly BP2.

> **Key insight:** Error flows backward via the **transpose** of the weight matrix.

---

### BP3 — Weight Gradients

$$\boxed{\frac{\partial \mathcal{L}}{\partial W^{(l)}} = \delta^{(l)} \left(a^{(l-1)}\right)^{\!T}}$$

Component form:

$$\frac{\partial \mathcal{L}}{\partial W_{jk}^{(l)}} = \delta_j^{(l)}\, a_k^{(l-1)}$$

**Derivation:**

$$\frac{\partial \mathcal{L}}{\partial W_{jk}^{(l)}} = \frac{\partial \mathcal{L}}{\partial z_j^{(l)}} \cdot \frac{\partial z_j^{(l)}}{\partial W_{jk}^{(l)}} = \delta_j^{(l)} \cdot a_k^{(l-1)}$$

since $z_j^{(l)} = \sum_m W_{jm}^{(l)}\, a_m^{(l-1)} + b_j^{(l)}$.

---

### BP4 — Bias Gradients

$$\boxed{\frac{\partial \mathcal{L}}{\partial b^{(l)}} = \delta^{(l)}}$$

Component form:

$$\frac{\partial \mathcal{L}}{\partial b_j^{(l)}} = \delta_j^{(l)}$$

**Derivation:**

$$\frac{\partial \mathcal{L}}{\partial b_j^{(l)}} = \frac{\partial \mathcal{L}}{\partial z_j^{(l)}} \cdot \frac{\partial z_j^{(l)}}{\partial b_j^{(l)}} = \delta_j^{(l)} \cdot 1 = \delta_j^{(l)}$$

---

## Complete Backpropagation Algorithm

Given a single training example $(x, y)$:

### Step 1 — Forward Pass

Set $a^{(0)} = x$. For $l = 1, 2, \ldots, L$:

$$z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)}, \qquad a^{(l)} = \sigma^{(l)}\!\left(z^{(l)}\right)$$

Cache all $z^{(l)}$ and $a^{(l)}$.

### Step 2 — Output Error (BP1)

$$\delta^{(L)} = \nabla_{a^{(L)}} \mathcal{L} \;\odot\; \sigma'^{(L)}\!\left(z^{(L)}\right)$$

For MSE: $\nabla_{a^{(L)}} \mathcal{L} = a^{(L)} - y$.

### Step 3 — Backward Pass (BP2)

For $l = L-1,\; L-2,\; \ldots,\; 1$:

$$\delta^{(l)} = \left(\left(W^{(l+1)}\right)^{\!T} \delta^{(l+1)}\right) \odot \sigma'^{(l)}\!\left(z^{(l)}\right)$$

### Step 4 — Compute Gradients (BP3 & BP4)

For $l = 1, 2, \ldots, L$:

$$\frac{\partial \mathcal{L}}{\partial W^{(l)}} = \delta^{(l)} \left(a^{(l-1)}\right)^{\!T}, \qquad \frac{\partial \mathcal{L}}{\partial b^{(l)}} = \delta^{(l)}$$

---

## How Does Backpropagation Work?

Working **backward** from the model's output, backpropagation applies the **chain rule** to
calculate the influence of changes to each individual neural network parameter on the overall error
of the model's predictions.

Abstractly speaking, the purpose of backpropagation is to train a neural network to make better
predictions through supervised learning. More fundamentally, the goal of backpropagation is to
determine how model weights and biases should be adjusted to minimize error as measured by a
**loss function**.

On a technical, mathematical level, the goal of backpropagation is to calculate the gradient of
the loss function with respect to each of the individual parameters of the neural network. In
simpler terms, backpropagation uses the chain rule to calculate the rate at which loss changes in
response to any change to a specific weight (or bias) in the network.

Generally speaking, training neural networks with backpropagation entails the following steps:

1. A **forward pass**, making predictions on training data.
2. A **loss function** measures the error of the model's predictions during that forward pass.
3. **Backpropagation of error**, or a backward pass, to calculate the partial derivatives of the
   loss function.
4. **Gradient descent**, to update model weights.

---

## Forward Pass (Intuition)

Neural networks output predictions through **forward propagation**. Forward propagation is
essentially a long series of nested equations, with the outputs of the activation functions from
one layer of neurons serving as inputs to the activation functions of neurons in the next layer.

Model training typically begins with a **random initialization** of weights and biases. Model
hyperparameters — such as the number of hidden layers, the number of nodes in each layer, and
activation functions for specific neurons — are configured manually and are not subject to
training.

In each forward pass, an input is sampled from the training dataset. The nodes of the input layer
receive the input vector, and each passes their value multiplied by some random initial weight to
the nodes of the first hidden layer. The hidden units take the weighted sum of these output values
as input to an activation function, whose output value (conditioned by a random initial weight)
serves as input to the neurons in the next layer. This continues until the output layer, where a
final prediction occurs.

**Consider this simplified example of a neural network that classifies inputs into one of 5
categories:**

1. The **input layer** receives a numerical representation of an example sampled from the training
   data.
2. The input nodes pass their values to hidden units in the next layer. The hidden units use a
   **ReLU** activation function.
3. Data flows through the hidden layers, each progressively extracting key features, until it
   reaches the output layer.
4. The **output layer** contains **5 neurons**, each corresponding to a potential classification
   category.
5. The output neurons use a **softmax** activation function. The output value of each output
   neuron's softmax function corresponds to the probability, out of 1, that the input should be
   classified as the category that neuron represents.
6. The network predicts that the original input belongs to the category of whichever output neuron
   has the **highest softmax value**.

In a well-trained network, this model will consistently output a high probability value for the
correct classification and output low probability values for the other, incorrect classifications.
However, this neural network isn't yet trained — at this point, its weights and biases have random
initial values, so its predictions are generally inaccurate.

---

## Loss Function

After each forward pass, a **loss function** measures the difference (or "loss") between the
model's predicted output for a given input and the correct predictions (or "ground truth") for that
input. In other words, it measures how different the model's actual output is from the desired
output.

- In **supervised learning**, which uses labeled data, ground truth is provided by manual
  annotations.
- In **self-supervised learning**, which masks or transforms parts of unlabeled data samples and
  tasks models with reconstructing it, the original sample serves as ground truth.

The goal of the loss function is to quantify inaccuracy in a way that appropriately reflects both
the nature and magnitude of the error of the model's output for each input. Different mathematical
formulas for loss are best suited to specific tasks:

| Task | Common Loss Function |
|------|---------------------|
| Regression | Variants of **Mean Squared Error (MSE)** |
| Classification | Variants of **Cross-Entropy Loss** |

Because the loss function takes the output of a neural network as an input — and that neural
network output is a composite function comprising many nested activation functions of individual
neurons — differentiating the loss function entails differentiating the entire network. To do so,
backpropagation uses the **chain rule**.

> **"Loss function," "cost function," or "error function"?**
>
> In some contexts, the terms *cost function* or *error function* are used in place of *loss
> function*. Though some machine learning literature assigns unique nuance to each term, they're
> generally interchangeable. An **objective function** is a broader term for any evaluation
> function that we want to either minimize or maximize. Loss function, cost function, and error
> function refer specifically to functions we want to **minimize**.

---

## Backward Pass

Starting from the final layer, a **backward pass** differentiates the loss function to compute how
each individual parameter of the network contributes to the overall error for a single input.

Returning to our earlier example of the classifier model, we start with the 5 neurons in the final
layer, which we'll call layer $L$. The softmax value of each output neuron represents the
likelihood, out of 1, that an input belongs to their category. In a perfectly trained model, the
neuron representing the correct classification would have an output value close to 1 and the other
neurons would have an output value close to 0.

For now, we'll focus on the output unit representing the correct prediction, which we'll call
$L_c$. $L_c$'s activation function is a composite function, containing the many nested activation
functions of the entire neural network from the input layer to the output layer. Minimizing the
loss function would entail making adjustments throughout the network that bring the output of
$L_c$'s activation function closer to 1.

To do so, we'll need to know how any change in previous layers will change $L_c$'s own output. In
other words, we'll need to find the **partial derivatives** of $L_c$'s activation function.

The output of $L_c$'s activation function depends on the contributions it receives from neurons in
the penultimate layer, which we'll call layer $L-1$. One way to change $L_c$'s output is to change
the weights between the neurons in $L-1$ and $L_c$. By calculating the partial derivative of each
$L-1$ weight with respect to the other weights, we can see how increasing or decreasing any of
them will bring the output of $L_c$ closer to (or further away from) 1.

But that's not the only way to change $L_c$'s output. The contributions $L_c$ receives from $L-1$
neurons are determined not just by the weights applied to $L-1$'s output values, but by the actual
(pre-weight) output values themselves. The $L-1$ neurons' output values, in turn, are influenced by
weights applied to inputs they receive from $L-2$. So we can differentiate the activation functions
in $L-1$ to find the partial derivatives of the weights applied to $L-2$'s contributions. These
partial derivatives show us how any change to an $L-2$ weight will affect the outputs in $L-1$,
which would subsequently affect the output value of $L_c$ and thereby affect the loss function.

By that same logic, we could also influence the output values that $L-1$ neurons receive from $L-2$
neurons by adjusting the contributions that $L-2$ neurons receive from neurons in $L-3$. So we find
the partial derivatives in $L-3$, and so on, **recursively repeating this process until we've
reached the input layer**. When we're done, we have the **gradient of the loss function**: a vector
of its partial derivative for each weight and bias parameter in the network.

We've now completed a forward pass and backward pass for a single training example. However, our
goal is to train the model to generalize well to new inputs. To do so requires training on a large
number of samples that reflect the diversity and range of inputs the model will be tasked with
making predictions on post-training.

---

## Gradient Descent

Now that we have the gradients of the loss function with respect to each weight and bias parameter
in the network, we can minimize the loss function — and thus optimize the model — by using
**gradient descent** to update the model parameters.

Moving down — **descending** — the gradient of the loss function will decrease the loss. Since the
gradient we calculated during backpropagation contains the partial derivatives for every model
parameter, we know which direction to "step" each of our parameters to reduce loss.

Each step reflects the model "learning" from its training data. Our goal is to iteratively update
weights until we have reached the minimum gradient. The object of gradient descent algorithms is to
find the specific parameter adjustments that will move us down the gradient most efficiently.

### Learning Rate $\eta$

The size of each step is a tunable hyperparameter called the **learning rate**. Choosing the right
learning rate is important for efficient and effective training.

Recall that the activation functions in a neural network are nonlinear. Some gradients may be
approximately U-shaped: stepping in one direction moves down the gradient, but continuing to step
in that direction will eventually move up the gradient.
<img width="597" height="420" alt="Screenshot 2026-09-26 151919" src="https://github.com/user-attachments/assets/49e91a76-bfa5-4964-83f2-c868fb4c3dcd" />


| Learning Rate | Effect |
|---------------|--------|
| Too **low** | Always steps in the right direction, but calculating so many small changes is time-consuming and computationally expensive |
| Too **high** | Computationally efficient, but risks **overshooting** the minimum |

### Gradient Descent Variants

Another consideration in gradient descent is **how often to update weights**.

**Batch Gradient Descent**
Compute the gradients for every example in the training dataset, take an average of those gradients,
and use it to update parameters. The process is repeated iteratively in a series of training epochs
until the error rate stabilizes. When the training dataset is very large — as it typically is in
deep learning — batch gradient descent entails prohibitively long processing times. Calculating
gradients for millions of examples for each iteration of weight updates becomes inefficient.

**Stochastic Gradient Descent (SGD)**
Each epoch uses a **single training example** for each step. While loss might fluctuate on an
epoch-to-epoch basis, it quickly converges to the minimum throughout many updates.

**Mini-Batch Gradient Descent**
A middle-ground approach. Training examples are randomly sampled in **batches of fixed size**, and
their gradients are then calculated and averaged together. This mitigates the memory storage
requirements of batch gradient descent while also reducing the relative instability of SGD.

| Method | Batch Size | Pros | Cons |
|--------|-----------|------|------|
| **Batch GD** | $B = N$ | Accurate gradients | Slow; high memory usage |
| **SGD** | $B = 1$ | Fast updates; low memory | Noisy loss; slow convergence |
| **Mini-Batch GD** *(most common)* | $1 < B \ll N$ | Balances stability & efficiency | Requires tuning batch size |

---

## Resources

- [What is Backpropagation? — IBM](https://www.ibm.com/topics/backpropagation)
- [CS231n: Backpropagation — Stanford](https://cs231n.github.io/optimization-2/)
- *Neural Networks and Deep Learning* — Michael Nielsen
