# 5. The Attention Model

In seq2seq the decoder sees only one summary of the sentence. Attention lets the decoder **look back at every word of the input at every step** and focus on the ones that matter right now.

![Encoder passing only one state](https://d2l.ai/_images/seq2seq-state.svg)

Without attention, the encoder's final state is the only thing the decoder gets.

## The idea in plain words

Think of an **open-book exam**. You do not memorise the whole textbook; you learn where to look. When you write each word of the translation, you glance back at the source sentence and put more focus on the words relevant to the word you are about to write.

## One decoder step, broken down

Say the decoder is about to write the next French word.

1. **Score:** compare the decoder's current state to *every* encoder state (one per input word). A bigger score means "this input word is more relevant right now."
2. **Turn scores into weights:** run the scores through **softmax**, which makes them positive and sum to 1. These are the **attention weights**, like percentages of focus.
3. **Mix:** add up the encoder states, each multiplied by its weight. The result is a fresh context vector that is mostly made of the relevant words.
4. **Write:** the decoder uses this context vector to produce the next word. Then repeat for the next word with a new set of weights.

**Tiny example:** the scores for three input words are 2.0, 1.0 and 0.1. After softmax the weights are about 0.66, 0.24 and 0.10. The new context is roughly two thirds the first word, a quarter the second, and a tenth the third.

![Seq2seq with Bahdanau attention](https://d2l.ai/_images/seq2seq-details-attention.svg)

The decoder queries all encoder outputs at every step.

## it's math time hehe

$$\text{weights}=\mathrm{softmax}(\text{scores}),\qquad \text{context}=\sum_i \text{weight}_i\times\text{encoder state}_i$$

That is all of attention: **softmax the scores, then take a weighted average.** Models differ only in how they compute the scores. The original (Bahdanau) version uses a tiny neural network; a later version (Luong) uses a simple dot product.

## What the model learns to look at

Plotting the weights for every output word shows which input words the model focused on. It learns word order differences between languages on its own. In this example the model correctly looked at "European Economic Area" even though French reverses the word order:

![Attention alignment for translation](https://jalammar.github.io/images/attention_sentence.png)

Lines show which source words each target word attended to.

## The query, key and value view

Modern descriptions of attention use three names, borrowed from searching:

- **Query:** what you are looking for (the decoder's current state).
- **Keys:** a label for each item you could look at (the encoder states, used for matching).
- **Values:** the actual content you take (the encoder states, used for mixing).

Match the query against all keys to get weights, then take a weighted mix of the values. This is exactly the four steps above, and it is the form the Transformer uses.

![Queries, keys and values](https://d2l.ai/_images/qkv.svg)

Attention is a weighted combination of values, where the weights come from how well the query matches each key.

- Attention = score, softmax, weighted average.
- It removes the bottleneck: the decoder can reach every input word at every step.
- Attention weights show which words the model used (useful, but not a perfect explanation of its decision).

## Check yourself

1. What problem of plain seq2seq does attention fix?
2. Why softmax?
3. If the weights are 0.9, 0.05, 0.05, which word dominates the context?
4. In the search analogy, what are the query, keys and values?

<details><summary>Answers</summary>

1. The fixed-size context vector bottleneck.
2. It turns raw scores into positive weights that sum to 1, so they behave like percentages of focus.
3. The first word.
4. Query = what you type; keys = page titles you match against; values = the page contents you read.
</details>

## Learn more
read the paper attention is all you need for an insight into how it works!
- Andrew Ng, [Sequence Models, Week 3: attention model](https://www.coursera.org/learn/nlp-sequence-models)
- Jay Alammar, [Visualizing a Neural Machine Translation Model](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)
- Distill, [Attention and Augmented Recurrent Neural Networks](https://distill.pub/2016/augmented-rnns/)
- Lilian Weng, [Attention? Attention!](https://lilianweng.github.io/posts/2018-06-24-attention/)
- Video: [StatQuest, Attention for Neural Networks](https://www.youtube.com/results?search_query=StatQuest+Attention+for+Neural+Networks+Clearly+Explained) (search link)
- Paper: Bahdanau et al. 2014, [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)
- Paper: Luong et al. 2015, [Effective Approaches to Attention-based NMT](https://arxiv.org/abs/1508.04025)

