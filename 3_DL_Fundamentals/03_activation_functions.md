# Activation Functions

Activation functions are the **source of non-linearity** in neural networks. Without them, no matter how many layers you stack, the entire network would collapse into a single linear transformation — making deep learning no more powerful than linear regression.

---

## The Non-Linearity Requirement

A neuron computes a weighted sum:

$$ z = w^T x + b $$

If you apply only linear operations across layers:

$$ h_1 = W_1x + b_1,\qquad h_2 = W_2h_1 + b_2,\qquad \ldots $$

the whole network simplifies to a single affine transformation:

$$ y = W_{\text{eff}}x + b_{\text{eff}} $$

Activation functions break this linearity, enabling the **universal approximation theorem**: a network with enough hidden units can approximate any continuous function arbitrarily well.

---

## Role in Training

During backpropagation, gradients flow through activation functions via their derivatives. The shape of $f'(x)$ directly controls:

- How fast weights update
- Whether gradients vanish or explode through deep layers
- Optimization stability and convergence speed

---

# Classical Activation Functions

## 3.1 Sigmoid (Logistic)

The sigmoid function is:

$$ \sigma(x) = \frac{1}{1 + e^{-x}} $$

**Range:** $(0,1)$

### Derivative

$$ \sigma'(x) = \sigma(x)(1-\sigma(x)) $$

The maximum value of the derivative is:

$$ \max \sigma'(x) = 0.25 \quad \text{at } x=0 $$

### Pros

- Smooth and differentiable
- Historically motivated by biological neuron firing probabilities
- Natural for binary probability outputs

### Cons

- **Not zero-centered** → can cause inefficient gradient dynamics
- **Saturates** for large $|x|$ → gradients approach 0
- **Maximum gradient is only 0.25** → repeated multiplication across layers can cause **vanishing gradients**

### Use Today

Almost never used in hidden layers of deep networks.

It is mainly used for **binary classification output layers**, where the output needs to represent a probability between 0 and 1.

---

## 3.2 Hyperbolic Tangent (Tanh)

The hyperbolic tangent function is:

$$
\tanh(x) = \frac{e^x-e^{-x}}{e^x+e^{-x}}
$$

**Range:** $(-1,1)$

### Derivative

$$
\tanh'(x) = 1-\tanh^2(x)
$$

The maximum value of the derivative is:

$$
\max \tanh'(x) = 1 \quad \text{at } x=0
$$

### Pros

- **Zero-centered**, which is better than sigmoid
- Larger maximum gradient ($1$ vs $0.25$) → better gradient flow than sigmoid

### Cons

- Still **saturates** for large $|x|$ → can cause vanishing gradients
- More computationally expensive than ReLU because it involves exponentials

### Use Today

Occasionally used in RNNs or shallow networks; rarely used in deep CNNs or transformers.

---

# Rectified Activation Functions

## 4.1 ReLU (Rectified Linear Unit)

The ReLU function is:

$$
\mathrm{ReLU}(x)=\max(0,x)
$$

or equivalently:

$$
\mathrm{ReLU}(x)=
\begin{cases}
x, & x\geq 0 \\
0, & x<0
\end{cases}
$$

### Derivative

$$
\mathrm{ReLU}'(x)=
\begin{cases}
1, & x>0 \\
0, & x<0
\end{cases}
$$

### Pros

- **Reduces vanishing gradients** for positive inputs because the gradient is 1
- **Computationally cheap**
- Encourages **sparse activations** because many neurons output exactly 0
- Became widely used in CNNs and MLPs after AlexNet (2012)

### Cons

- **Not zero-centered**
- **Dying ReLU problem:** if a neuron's pre-activation is always negative, its gradient is always 0, so the neuron may stop updating
- **Unbounded output** → can cause instability in some settings

---

## 4.2 Leaky ReLU

Leaky ReLU allows a small non-zero slope for negative inputs:

$$
\mathrm{LeakyReLU}(x)=
\begin{cases}
x, & x\geq 0 \\
\alpha x, & x<0
\end{cases}
$$

where typically:

$$
\alpha \approx 0.01
$$

### Derivative

$$
\mathrm{LeakyReLU}'(x)=
\begin{cases}
1, & x>0 \\
\alpha, & x<0
\end{cases}
$$

**Idea:** Allow a small, non-zero gradient for negative inputs to mitigate the dying ReLU problem.

### Pros

- Reduces dead neurons
- Still computationally cheap

### Cons

- $\alpha$ is a hyperparameter and may need tuning
- Still not zero-centered

---

## 4.3 Parametric ReLU (PReLU)

PReLU is similar to Leaky ReLU, but the negative slope $\alpha$ is **learnable**:

$$
\mathrm{ReLU}(x)=
\begin{cases}
x, & x\geq 0 \\
\alpha x, & x<0
\end{cases}
$$

Here, $\alpha$ is learned during backpropagation.

### Pros

- Network adapts the negative slope to the data
- Can improve over a fixed Leaky ReLU in some settings

### Cons

- Adds extra parameters → slight risk of overfitting
- Slightly more complex to implement

---

## 4.4 Other ReLU Variants

Some other ReLU-based activation functions include:

- **RReLU (Randomized ReLU):** $\alpha$ is sampled randomly during training
- **ELU (Exponential Linear Unit):** uses smooth negative saturation
- **SELU (Scaled ELU):** adds scaling for self-normalizing networks
- **BReLU (Bounded ReLU):** clamps output to a fixed range such as $[0,A]$

These aim to balance **gradient flow, non-linearity, and stability**.

---

# Exponential and Smooth Activations

## 5.1 ELU (Exponential Linear Unit)

The ELU function is:

$$
\mathrm{ELU}(x)=
\begin{cases}
x, & x>0 \\
\alpha(e^x-1), & x\leq 0
\end{cases}
$$

### Properties

- Smooth for negative inputs, unlike ReLU's hard cutoff
- Negative saturation can improve robustness to noise
- Helps push mean activations closer to zero

---

## 5.2 GELU (Gaussian Error Linear Unit)

The GELU function is:

$$
\mathrm{GELU}(x)=x\Phi(x)
$$

where $\Phi(x)$ is the **standard normal cumulative distribution function (CDF)**.

An approximation commonly used in practice is:

$$
\mathrm{GELU}(x)
\approx
0.5x
\left(
1+\tanh
\left(
\sqrt{\frac{2}{\pi}}
\left(x+0.044715x^3\right)
\right)
\right)
$$

### Properties

- Smooth and non-monotonic
- Interpolates between suppressing small inputs and passing larger inputs
- Commonly used in transformer architectures

### Why It Works Well

- Smooth gradients can provide a favorable optimization landscape
- Has a probabilistic interpretation: it scales the input according to its probability of being positive

---

## Swish / SiLU

The Swish function is:

$$
\mathrm{Swish}(x)=x\sigma(x)
$$

or:

$$
\mathrm{Swish}(x)=\frac{x}{1+e^{-x}}
$$

It is also called **SiLU (Sigmoid Linear Unit)** in some frameworks.

### Properties

- Smooth and non-monotonic
- Unbounded above
- Has a slightly negative minimum

### Use

Used in architectures such as **EfficientNet**, some modern CNNs, and some transformer variants.

It can perform comparably to or sometimes better than ReLU or GELU depending on the task.

---

# The Vanishing Gradient Problem

## What Happens?

During backpropagation, the gradient with respect to early-layer weights is a product of many terms:
## What Happens?

During backpropagation, the gradient with respect to early-layer weights is a product of many terms:

$$
\frac{\partial L}{\partial w^{(1)}} =
\frac{\partial L}{\partial a^{(L)}}
\prod_{l=2}^{L}
\frac{\partial a^{(l)}}{\partial a^{(l-1)}}
\frac{\partial a^{(1)}}{\partial w^{(1)}}
$$

If each layer's activation derivative is **less than 1** — for example, sigmoid has a maximum derivative of only $0.25$ — the product can shrink exponentially with depth.

If each layer's activation derivative is **less than 1** — for example, sigmoid has a maximum derivative of only $0.25$ — the product can shrink exponentially with depth.

## Consequences

- Early layers learn extremely slowly or not at all
- Deep networks can fail to converge or require careful initialization
- RNNs can especially suffer because gradients must also propagate through time

---

## How Activation Functions Affect Vanishing Gradients

### Sigmoid / Tanh

- Sigmoid derivative $\leq 0.25$
- Tanh derivative $\leq 1$
- Both saturate for large $|x|$
- Their derivatives approach 0 in the saturated regions
- Repeated multiplication can lead to **vanishing gradients**

### ReLU

- Derivative is 1 for positive inputs
- Gradients can pass through positive activations without shrinking
- No saturation on the positive side
- Main issue: **dead neurons** for consistently negative inputs

### Leaky ReLU / ELU / GELU / Swish

- Maintain non-zero gradients over wider input ranges
- Can reduce vanishing-gradient problems compared with sigmoid and tanh

---

# Comparative Summary

| Activation | Formula | Range | Zero-Centered? | Vanishing Gradient Risk | Typical Use |
|---|---|---|---|---|---|
| **Sigmoid** | $\sigma(x)=\frac{1}{1+e^{-x}}$ | $(0,1)$ | No | High | Binary output layer |
| **Tanh** | $\tanh(x)=\frac{e^x-e^{-x}}{e^x+e^{-x}}$ | $(-1,1)$ | Yes | High when saturated | RNNs, shallow networks |
| **ReLU** | $\max(0,x)$ | $[0,\infty)$ | No | Low on positive side | CNNs, MLPs |
| **Leaky ReLU** | $x$ if $x>0$, else $\alpha x$ | $(-\infty,\infty)$ | No | Low | When dying ReLU is observed |
| **PReLU** | $x$ if $x>0$, else $\alpha x$ | $(-\infty,\infty)$ | No | Low | Data-specific slope |
| **ELU** | $x$ if $x>0$, else $\alpha(e^x-1)$ | $(-\alpha,\infty)$ | Approximately | Low | When smooth negative saturation is useful |
| **GELU** | $x\Phi(x)$ | Approximately $(-0.17,\infty)$ | No | Low | Transformers |
| **Swish / SiLU** | $x\sigma(x)$ | Approximately $(-0.28,\infty)$ | No | Low | Modern CNNs, some transformers |

---

# Visual Comparison of Activations and Gradients

This shows how each activation transforms its input. Notice:

- **Sigmoid / Tanh:** S-shaped curves that saturate at the extremes
- **ReLU:** hard cutoff at 0
- **Leaky ReLU:** small slope for negative inputs
- **GELU / Swish:** smooth, curved transitions
- **ELU:** exponential behavior for negative inputs

The derivative curves show the **gradient each activation provides** at different input values:

- **Sigmoid:** maximum derivative of 0.25, quickly approaches 0
- **Tanh:** maximum derivative of 1, but approaches 0 at the extremes
- **ReLU:** derivative is 1 for positive inputs and 0 for negative inputs
- **Leaky ReLU:** small constant derivative for negative inputs
- **GELU / Swish:** smooth gradients
- **ELU:** exponential behavior for negative inputs

These derivative curves help explain why sigmoid and tanh can struggle in deep networks, while ReLU-family and other modern activations generally provide better gradient flow.

---

# Practical Guidelines

## Hidden Layers

- **General default for CNNs and MLPs:** ReLU or Leaky ReLU
- **Transformers:** GELU or SiLU in many architectures
- **If dead neurons occur:** consider Leaky ReLU, PReLU, or ELU
- **Very deep networks:** sigmoid and tanh are generally avoided in hidden layers because of their saturation behavior

## Output Layers

- **Binary classification:** Sigmoid — outputs a probability in $(0,1)$
- **Multi-class classification:** Softmax — converts output scores into class probabilities
- **Regression:** Linear / Identity — no activation, unless a bounded output is specifically required

---

# Key Takeaway

Activation functions introduce **non-linearity** into neural networks, allowing multiple layers to learn complex relationships.

Their derivatives also play a crucial role during **backpropagation**, because they determine how gradients flow through the network.

The most important ideas to remember are:

> **No activation → essentially a linear model.**

> **Activation functions → non-linearity and expressive power.**

> **Activation derivatives → influence gradient flow during training.**

<img width="826" height="332" alt="Screenshot 2026-09-26 150013" src="https://github.com/user-attachments/assets/84377756-7c7c-47e6-b069-ebfdb6151cbd" />
<img width="830" height="482" alt="Screenshot 2026-09-26 150006" src="https://github.com/user-attachments/assets/4d701e44-0d89-4130-974b-b95a44aab876" />
<img width="830" height="497" alt="Screenshot 2026-09-26 145956" src="https://github.com/user-attachments/assets/efabcfeb-53ad-44c3-a02f-7ce5116f4b6b" />
<img width="820" height="478" alt="Screenshot 2026-09-26 145947" src="https://github.com/user-attachments/assets/df3a7942-2c9e-4203-a893-d79bb1e98da1" />
<img width="821" height="473" alt="Screenshot 2026-09-26 145940" src="https://github.com/user-attachments/assets/aef0814a-f135-4dfa-a00b-5b2030fce587" />
<img width="817" height="482" alt="Screenshot 2026-09-26 145931" src="https://github.com/user-attachments/assets/af0f2c9c-1c94-4073-ba5b-91c8fe11706a" />

