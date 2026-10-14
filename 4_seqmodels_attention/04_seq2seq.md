# 4. Seq2seq (Encoder-Decoder), Beam Search and BLEU

Translation, summarisation and chatbots all need a **sequence in and a different-length sequence out**. "The cat sleeps" is 3 words; "Le chat dort" happens to be 3 too, but "I am hungry" becomes "J'ai faim" (2 words). A simple RNN that outputs one thing per input cannot do this.

## The idea in plain words

Split the job between two RNNs, like a translator who **reads the whole sentence first, then writes the translation from memory:**

1. **Encoder:** reads the input sentence word by word and ends with a summary of the whole sentence, called the **context vector** (just a list of numbers).
2. **Decoder:** starts from that summary and writes the output sentence one word at a time, stopping when it writes a special end-of-sentence token.

![Encoder-decoder](https://d2l.ai/_images/seq2seq.svg)

The left RNN reads, the right RNN writes. `<bos>` and `<eos>` mark the beginning and end.

The context vector really is just numbers:

![Context vector](https://jalammar.github.io/images/context.png)


## Training versus using it

- **Training:** at every step, give the decoder the **correct previous word** from the training data, even if it would have guessed wrong. This keeps learning stable and is called **teacher forcing**.
- **Using it (inference):** the correct answer is unknown, so the decoder feeds **its own previous output** back in, one word at a time.

![Predicting token by token](https://d2l.ai/_images/seq2seq-predict.svg)


## Choosing the words: greedy and beam search

At each step the decoder gives a probability for every word in the vocabulary. How do we pick?

- **Greedy search:** always take the single most likely word. Fast, but a locally best word can lead to a worse sentence overall.
- **Beam search:** keep the best few candidate sentences at every step (the number kept is the **beam width**, say 3) and extend all of them. At the end, choose the best complete sentence. Better results, a little slower.

![Greedy search](https://d2l.ai/_images/s2s-prob1.svg)



![Beam search](https://d2l.ai/_images/beam-search.svg)



## Judging a translation: BLEU

**BLEU** is a score that measures how much the machine's translation overlaps (in words and short phrases) with one or more human reference translations. Higher is better. It is quick and widely used, but crude: a correct translation phrased differently from the reference scores lower.

## The problem

The decoder only ever sees **one** context vector. For a 5-word sentence that is fine. For a 50-word sentence, cramming everything into one list of numbers loses information, and quality drops as sentences get longer. Humans do not translate that way: a translator glances back at the original while writing. Giving the decoder that ability is exactly **attention** (next file).

- Encoder reads, decoder writes, the context vector connects them.
- Teacher forcing when training; own outputs when generating.
- Beam search keeps several candidates; greedy keeps one.
- One fixed context vector is a bottleneck for long inputs.

## Check yourself

1. What are the two parts of a seq2seq model, and what links them?
2. Why use teacher forcing during training?
3. What does a beam width of 1 correspond to?
4. Why does translation quality drop for long sentences in a basic seq2seq model?

<details><summary>Answers</summary>

1. An encoder and a decoder, linked by the context vector.
2. It stops early mistakes from derailing learning, because the decoder always sees the correct previous word.
3. Greedy search.
4. The whole sentence has to be compressed into one fixed-size vector, which cannot hold enough detail.
</details>

## Useful Links

- Andrew Ng, [Sequence Models, Week 3](https://www.coursera.org/learn/nlp-sequence-models) (basic models, beam search, BLEU)
- Jay Alammar, [Visualizing a Neural Machine Translation Model](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/) (animations; first half is plain seq2seq)
- Stanford CS224n, [course site](http://web.stanford.edu/class/cs224n/) (lecture on translation, seq2seq and attention; videos are on YouTube)
- Video: [StatQuest, Sequence-to-Sequence Encoder-Decoder](https://www.youtube.com/results?search_query=StatQuest+Sequence-to-Sequence+Encoder-Decoder+Clearly+Explained) (search link)
- Paper: Sutskever et al. 2014, [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215)

