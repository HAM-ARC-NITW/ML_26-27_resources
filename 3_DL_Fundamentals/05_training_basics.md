# Neural Network Training — Basics

Training is the process by which a neural network learns to map inputs to correct outputs by
adjusting its internal parameters (weights and biases) to minimize prediction errors.

---

## The Training Loop

Training follows this iterative process:

1. **Forward pass** — Input data flows through the network → predictions are made
2. **Loss computation** — Compare predictions to true labels → compute error (loss)
3. **Backward pass** — Compute gradients of loss w.r.t. all parameters using backpropagation
4. **Parameter update** — Adjust weights and biases using gradient descent
5. **Repeat** — Go back to step 1 with new data or next batch

---

## Mathematical View

**Given:**

- Training data: $\{(x_1, y_1),\ (x_2, y_2),\ \ldots,\ (x_N, y_N)\}$
- Network function: $f(x;\, \theta)$ where $\theta = \{W^{(l)},\, b^{(l)}\}$ for all layers
- Loss function: $\mathcal{L}\!\left(f(x;\,\theta),\, y\right)$

**Training solves:**

$$\theta^* = \underset{\theta}{\arg\min}\ \left[\frac{1}{N} \sum_{i=1}^{N} \mathcal{L}\!\left(f(x_i;\,\theta),\, y_i\right)\right]$$

In practice, we use **stochastic gradient descent (SGD)** or variants (Adam, RMSprop) to find
$\theta^*$ iteratively.

---

## Parameters (Learned from Data)

**Definition:** Internal variables that the model learns automatically during training from the
data.

**Examples:**

| Type | Notation |
|------|----------|
| Weights | $W^{(1)},\, W^{(2)},\, \ldots,\, W^{(L)}$ |
| Biases | $b^{(1)},\, b^{(2)},\, \ldots,\, b^{(L)}$ |
| Linear regression | slope $m$ and intercept $b$ in $y = mx + b$ |
| CNNs | filter/kernel values |
| RNNs | gating parameters |

**Key properties:**

- Initialized randomly (or with specific initialization schemes)
- Updated every training step via gradient descent
- Number can be huge (millions to billions in large models)
- Stored as part of the trained model file

---

## Hyperparameters (Set Before Training)

**Definition:** External configuration variables that you set before training begins. They control
how training happens and what model architecture you use.

**Examples:**

*Training algorithm hyperparameters:*

| Hyperparameter | Symbol/Notes |
|----------------|-------------|
| Learning rate | $\alpha$ or $\eta$ |
| Number of epochs | — |
| Batch size | — |
| Momentum (for SGD with momentum) | — |
| Beta1, Beta2 (for Adam optimizer) | $\beta_1,\, \beta_2$ |
| Dropout rate | — |
| Weight decay (L2 regularization strength) | — |

*Architecture hyperparameters:*

| Hyperparameter | Notation |
|----------------|----------|
| Number of hidden layers | $L$ |
| Number of neurons per layer | $n_1,\, n_2,\, \ldots,\, n_L$ |
| Activation function type | ReLU, sigmoid, etc. |
| Number of filters in CNN layers | — |
| Kernel size in convolutions | — |

**Key properties:**

- Set manually or via hyperparameter tuning (grid search, random search, Bayesian optimization)
- Remain **fixed** during training (not updated by gradient descent)
- Control the learning process and model capacity
- Different hyperparameter choices → different final parameter values

---

## Batch, Iteration & Epoch

These three terms describe how data is processed during training.

### Batch

**Definition:** A subset of the training data processed together before the optimizer updates the
weights.

**Batch size ($B$):** Number of training examples in one batch.

Common batch sizes: 16, 32, 64, 128, 256, 512

**Types of learning based on batch size:**

| Mode | Batch Size | Description |
|------|-----------|-------------|
| **Stochastic (online) learning** | $B = 1$ | Update after every single example. Very noisy gradients, slow convergence, low memory usage. |
| **Full-batch learning** | $B = N$ (entire dataset) | One update per epoch. Accurate gradient estimates, high memory usage, may not fit in GPU. |
| **Mini-batch learning** *(most common)* | $1 < B < N$ | Balance between stability and efficiency. Most common in practice (e.g. 32, 64, 128). Good GPU utilization. |

---

### Iteration (or Step)

**Definition:** One parameter update = one forward pass + one backward pass on one batch.

**Key point:** Each iteration updates the model's weights once.

**Example:**

$$\text{Dataset size} = 10{,}000 \text{ samples}, \quad \text{Batch size} = 100$$

$$\text{Iterations per epoch} = \frac{10{,}000}{100} = 100 \text{ iterations}$$

---

### Epoch

**Definition:** One complete pass through the entire training dataset. The model sees every
training example exactly once.

**Key points:**

- An epoch consists of multiple iterations (unless batch size = dataset size)
- Multiple epochs are needed because one pass is usually not enough to learn all patterns
- Too few epochs → **underfitting** (model hasn't learned enough)
- Too many epochs → **overfitting** (model memorizes training data)

---

## Relationship Formula

For a dataset with $N$ samples and batch size $B$:

$$\text{Iterations per epoch} = \frac{N}{B}$$

$$\text{Total iterations for } E \text{ epochs} = E \times \frac{N}{B}$$

**Example:**

| Variable | Value |
|----------|-------|
| Training images ($N$) | $50{,}000$ |
| Batch size ($B$) | $128$ |
| Epochs ($E$) | $20$ |

$$\text{Iterations per epoch} = \left\lfloor\frac{50{,}000}{128}\right\rfloor \approx 391 \text{ iterations}$$

$$\text{Total iterations} = 20 \times 391 = 7{,}820 \text{ weight updates}$$
