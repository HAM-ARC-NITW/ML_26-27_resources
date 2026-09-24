
### Introduction to Neural Networks

A neuron (also called a **unit** or **node**) is the basic computational building block of an artificial neural network. It receives one or more numeric inputs, combines them using learned weights and a bias, and then passes the result through a non-linear activation function to produce a single output that is sent to neurons in the next layer.ibm+3

Mathematically, for inputs *x*₁, *x*₂, ..., *x*ₙ*, weights *w*₁, *w*₂, ..., *w*ₙ*, bias *b*, and activation function *f*, a neuron computes:

**Weighted sum:**

*z* = *b* + ∑ *wᵢxᵢ*

**Activation / output:**

*a* = *f*(*z*)

This simple operation, repeated across many neurons in multiple layers, allows neural networks to approximate highly complex, non-linear functions.

---

## Core Components of a Neuron

### 1. Inputs

- Numeric values fed into the neuron.
- In early layers, these come from raw data (e.g., pixel intensities, sensor readings).
- In deeper layers, they come from outputs of previous neurons.

### 2. Weights (*wᵢ*)

- Each input has an associated weight.
- Weights control **how strongly** each input influences the neuron’s output.
- During training, the network **learns** optimal weight values to minimize prediction error.

### 3. Bias (*b*)

- A single extra parameter added to the weighted sum.
- Acts like a **threshold adjuster**: it lets the neuron activate even when all inputs are zero.
- Without bias, the neuron’s decision boundary would always pass through the origin, limiting flexibility.

### 4. Activation Function (*f*)

- A non-linear function applied to the weighted sum.
- Introduces **non-linearity**, enabling the network to model complex patterns (without it, stacking layers would just be equivalent to a single linear transformation).

| **Function** | **Formula** | **Output range** | **Typical use / notes** |
|---|---|---|---|
| **Sigmoid** | σ(x) = 1 / (1 + e⁻ˣ) | (0, 1) | Historically common; now rarely used in hidden layers due to saturation and vanishing gradients. [cs231n](https://cs231n.github.io/neural-networks-1/) |
| **Tanh** | tanh(x) | (−1, 1) | Zero-centered; better than sigmoid but still saturates |
| **ReLU** | max(0, x) |  |  |
| **Leaky ReLU** | max(αx, x), α ≈ 0.01 | (−∞, ∞) | Variant of ReLU that allows small negative activations to mitigate “dying ReLU” problem. [cs231n](https://cs231n.github.io/neural-networks-1/) |
| **Linear / Identity** | f(x) = x | (−∞, ∞) | Often used in output layers for regression tasks. [cs231n](https://cs231n.github.io/neural-networks-1/) |

---

## Layers in a Neural Network

Individual computational units are structured into distinct **layers**:

- **Input layer**: accepts raw feature data without executing mathematical transformations.
- **Hidden layers**: intermediate stages of nodes processing and transforming input signals.
- **Output layer**: yields ultimate model outputs, including predicted probabilities or target values.openstax+1

Within a **dense or fully connected** architecture, every unit connects to all units in the succeeding tier. A forward pass across a given layer is calculated via:

*h* = *f*(*Wx* + *b*)

where *W* denotes the weight matrix, *x* represents input vectors, *b* is the bias term, and *f* acts element-wise.

Cascading multiple layers yields a **deep network architecture**, capable of extracting hierarchical feature representations.
<img width="832" height="637" alt="Screenshot 2026-09-23 222824" src="https://github.com/user-attachments/assets/2ecd88ee-b3b5-42ee-a5b4-db2732e9c539" />


<img width="828" height="586" alt="Screenshot 2026-09-23 222932" src="https://github.com/user-attachments/assets/c89c0d26-5465-42bc-82e5-351873838d74" />

# RESOURCES

- **ISLR book (2nd ed)** by Gareth James, Daniela Witten, Trevor Hastie, and Robert Tibshirani
- [CS231n – Neural Networks](https://cs231n.github.io/neural-networks-1/)
- [IBM – Neural Networks](https://www.ibm.com/think/topics/neural-networks)
