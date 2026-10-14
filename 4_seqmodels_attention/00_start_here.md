## What is a sequence?

A sequence is data where **order matters**: the words of a sentence, the samples of an audio clip, daily stock prices, the notes of a song, the letters of a DNA strand.

"Dog bites man" and "man bites dog" contain the same words and mean very different things. Any model for sequences has to respect order.

## Why not just use a normal neural network?

A plain feedforward network struggles with sequences for three reasons:

1. **Inputs have different lengths.** One sentence has 5 words, the next has 50. A normal network expects a fixed-size input.
2. **It ignores order.** If you feed words in as one big fixed vector, the network does not know which word came first.
3. **It does not share what it learns.** If the network learns that "Harry" is a name at position 1, it would have to relearn that for position 7. A word should mean the same thing wherever it appears.

Sequence models fix all three: they process one step at a time, remember what came before, and reuse the same weights at every step.

## Kinds of sequence tasks

Basically "different types of RNNs". Knowing which type your problem is tells you how to set up the model.

| Type | Input to output | Example |
|---|---|---|
| Many to one | a sequence, one answer | Is this movie review positive or negative? |
| One to many | one input, a sequence | Generate music from a genre; caption an image |
| Many to many (same length) | each input has an output | Tag each word as a name, place or other |
| Many to many (different length) | a sequence in, a different-length sequence out | Translate English to French |

## Turning words into numbers

A network only understands numbers, so every word first becomes a list of numbers called an **embedding**.

- The simplest way is **one-hot**: a huge list with a single 1 at the word's position. It is wasteful and tells the model nothing about meaning.
- An **embedding** is a short list of numbers (say 300) learned from data. Think of it as coordinates on a map of meaning: "king" and "queen" end up close together, "king" and "banana" far apart.

![Words become vectors](https://jalammar.github.io/images/embedding.png)

Each word becomes a vector (a list of numbers) before the model sees it.

There are ready made embeddings like Word2Vec, GloVe etc. All you need to know for now is, a word becomes a vector and the model works on these vectors.

Each model exists because the one before it had a specific weakness.

1. **RNN**: remembers by passing a note forward. *Weakness:* the note fades over long sentences.
2. **LSTM and GRU**: add gates so important information survives. *Weakness:* they still read one word at a time, and when translating, squeezing a whole sentence into one note is too much.
3. **Seq2seq**: one RNN reads the sentence, another writes the translation. *Weakness:* one note for a whole sentence is a bottleneck.
4. **Attention**: let the writer look back at any word of the original. This fixes the bottleneck.
5. **Self-attention and the Transformer**: use attention everywhere and drop the step-by-step reading, so everything can run in parallel. *Weakness:* it no longer knows word order.
6. **Positional encoding**: add the position of each word as extra numbers.

## Words you will meet

| Word | Plain meaning |
|---|---|
| Token | one piece of the input, usually a word or part of a word |
| Time step | one position in the sequence (word 1, word 2, ...) |
| Hidden state | the model's running memory, a list of numbers updated at every step |
| Gradient | the signal that tells the model how to adjust its weights while learning |
| Gate | a number between 0 and 1 that acts like a volume knob: 0 blocks, 1 lets everything through |
| Context vector | a summary of the input, handed from the reader part to the writer part |
| Query, key, value | the three roles a word plays in attention|
