# 3. GRU (Gated Recurrent Unit)


The LSTM works well but has many moving parts: a separate notebook, three gates, a candidate. The GRU asks: *can we keep the good idea (gates that protect memory) with less machinery?*

The answer: **drop the separate notebook, keep just one memory (the hidden state), and use two gates.**

| Gate | What it decides | Everyday picture |
|---|---|---|
| **Update gate** | How much of the old memory to keep versus replace with new content | one knob between "keep old" and "write new" |
| **Reset gate** | How much of the old memory to look at when proposing new content | "should I ignore what I knew before?" |

## One GRU step, broken down

**Step 1: compute the reset and update gates**, two 0-to-1 knobs.

![Reset and update gates](https://d2l.ai/_images/gru-1.svg)



**Step 2: propose new content** (the candidate), using the old memory only as much as the reset gate allows.

![Candidate hidden state](https://d2l.ai/_images/gru-2.svg)



**Step 3: blend old and new** using the update gate.

![New hidden state](https://d2l.ai/_images/gru-3.svg)



## Ze math mwahaha

$$h_t=(1-z_t)\odot h_{t-1}+z_t\odot\tilde h_t$$

- $h_{t-1}$ is the old memory, $\tilde h_t$ is the proposed new content, $z_t$ is the update gate.
- It is a **blend**: if $z_t$ is near 0, the memory is copied unchanged; if near 1, it is replaced by the new content.
- Some books write it with $z_t$ and $1-z_t$ swapped. Same idea, so check the convention before comparing code with a paper.

## GRU or LSTM?

| | LSTM | GRU |
|---|---|---|
| Gates | 3 (forget, input, output) | 2 (update, reset) |
| Separate long-term notebook | yes | no |
| Number of weights | more | about 25% fewer |
| Speed | slower | a bit faster |

Actually, there is no universal winner. The LSTM is older and a bit more powerful in principle. The GRU is simpler and faster to train and a good first try. Pick one, and switch only if your validation results say so.
- GRU = LSTM with fewer parts: one memory, two gates.
- The update gate blends old and new memory.
- Neither is always better; try GRU first for speed, LSTM if you need more power.

## Check yourself

1. What does a GRU remove compared to an LSTM?
2. What happens when the update gate is 0?
3. Name the two gates of a GRU.

<details><summary>Answers</summary>

1. The separate cell state (long-term notebook) and one of the three gates.
2. The old memory is copied forward unchanged (with the usual convention above).
3. Update gate and reset gate.
</details>

## Useful stuff

- Andrew Ng, [Sequence Models, Week 1: GRU](https://www.coursera.org/learn/nlp-sequence-models)
- Michael Phi, [Illustrated Guide to LSTM's and GRU's](https://towardsdatascience.com/illustrated-guide-to-lstms-and-gru-sa-step-by-step-explanation-44e9eb85bf21)
- Distill, [Visualizing memorization in RNNs](https://distill.pub/2019/memorization-in-rnns/) (compares GRU and LSTM on the same task)
- Paper: Cho et al. 2014, [Learning Phrase Representations using RNN Encoder-Decoder](https://arxiv.org/abs/1406.1078) (introduces the GRU)
- Paper: Chung et al. 2014, [Empirical Evaluation of Gated Recurrent Neural Networks](https://arxiv.org/abs/1412.3555)

