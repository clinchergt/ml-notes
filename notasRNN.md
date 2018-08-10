# Sequence models

Where are they useful?

## Examples

* Speech recognition
* Music generation
* Sentiment classification
* DNA sequence analysis
* Machine translation
* Video activity recognition
* Name entity recognition

## Notation

`x<1>` would be the first word.

`T_x` would be the length of X.



## Representing words

One option could be one hot encoding.

## Recurrent Neural Network Model

Problems with using a regular forward neural network would be how to handle
the input and outputs, since they differ across examples and tasks. There's
also the more serious fact that the network doesn't share features across
different positions of text.

#### What is a recurrent network?

You get the use the weights learned from the last example in order to predict
the next training example. So the network feeds itself across the training
phase. There's a problem with this is that the prediction at a certain time
information earlier in the sequence but not later. Bidirectional RNNs are for
this and we'll learn about them later.

Sometimes information from the past isn't enough to predict accurately what
the current example should be.

Usually for forward propagation, `tanh` or `ReLU` is used for the activation
function. Mostly `tanh` in RNNs.

`a<t> = g(W_aa*a<t-1> + W_ax*x<t> + b_a)`

Which could be summed up by concatenating `W_aa` and `W_ax` and getting a new
matrix and you end up with:

`a<t> = g(W_a[a<t-1>, x<t>] + b_a)`

#### Backprop

Define loss function over all the examples.


## Different types of RNNs

* One to many: output each step *and* feed that to the next step, the next
output.
* Many to many: if the output and input differ in size, you can use this and
the architecture would end up reading the input completely and then outputting
the output. When they're the same, you can output as you read and feed to the
next one.
* Many to one: you read the whole input and output at the end.

## Language modelling

Tokenize, get a vocabulary. Vector representation of words.

You might also want to have an EOS (end of sentence) token, as well as an UNK
(unknown word).

### Sampling novel sequences

You feed the output of the previous one to the current one and keep sampling.

NOTE: You can use character language models can be used too, but they're more
expensive and you don't get some sort of information. As computers get
faster people have been using more and more character-based models.

## Vanishing gradient

The RNNs described above are not very good at catching long term dependencies
like when you conjugate a verb but there's a subordinate clause between the
subject and the verb.

This means we're running into the vanishing gradient problem. In practice,
the network is gonna take too much effort to realize about the relationship
between e.g. subject and verb.

The basic RNNs has many local influences. Close values influence a lot.

You could also get exploding gradient, you could do some gradient clipping.
This means rescale the gradients once a threshold is met.

## Gated Recurrent Unit (GRU)

You introduce a memory cell. And you get a `~c`, which is the candidate to
replace the memory cell `c`. TO decide this, you have an activation function
represented by `gamma_u` and its values are always between 0 and 1.

`c<t> = gamme_u * ~c<t> + (1-gamma_u) + c<t-1>`

GRUs and LSTMs are the most robust and most commonly used methods to deal
with the problem the previous section describes.

## Long Term Short Memory (LSTM)

This is even more powerful than the GRU. These networks use three different
gates in the process of updating the memory cells. In concrete it uses two
to determine whether to update the memory cell and another one to get the new
output of the layer.

This makes it so it's more powerful when deciding whether to remember a value
or keep the current one. You can keep one value for long periods of time and
have it help with far along in the future predictions.

## Bidirectional RNN

You need to be able to look at the future training examples in some cases.
But, how?

You have a forward recurrent component (could call it e.g. `a<1>`)

Disadvantage is that you need the entire sequence input before you can make
predictions. Like, if you're doing speech recognition, you need to wait
for the person to stop talking and in order to process it all and make a
prediction.

## Deep RNNs


## Word embeddings

Featurized representations are very powerful because you are able to conclude
stuff from similar embedding vectors (i.e. words) and generalize. You could
even apply some arithmetic operators to vectors and get a logical result.

You can also continue to finetune word embeddings with new data.

Word embeddings make the most sense when your specific task has few training
examples. If you have a lot of data, maybe you could try start with that and
not try to attempt some transfering of word embeddings.

To find similarities you can use different measures:

* Cosine similarity
* Euclidean distance (actually a dissimilarity)
* etc.

### Embedding matrix

This the vector representations for each of the words in the vocabulary. This
means that it's a matrix of e.g. 300 x 10,000. Assuming you want a 300
dimensional word embeddings and your vocabulary is composed of 10k words.

You then use special operations to look up words in the matrix at a specific
index. One method would be multiplying a one hot vector with the index of the
word. E * O_j, where O is a one hot vector and j is the index of the word.

You then initialize this E matrix randomly and learn it using e.g. gradient
descent.

## Sentiment classification model

If you just average word embedding vectors, you ignore word order and other
subtleties about language. You could instead use an RNN to classify.

## Biases in word embeddings

Some embeddings can reflect some social biases. So you could have e.g.
father is to doctor as mother is to... nurse. Which is wrong from a language
point of view and from a social point of view. We want to eliminate this
uneccessary biases.

1. Identify bias direction
1. Neutralize: for every word that's not definitional, project to get rid of
bias.
1. Equalize pairs.

## Beam search

You evaluate different first word candidate and generate the string and
evaluate. You do this for all possible `B` candidates. `B` being a parameter.
Steps are as follows:

1. Pick `b` candidates.
1. Get second word candidates
1. Evaluate the probability for the first two word choices and pick the
top `b`.
1. Eliminate every other that didn't make the cut.

This entails that you make `b` copies of the network at each evaluation
of `b` candidates.

### Improvements to beam search

* Use the max of the sum of logs instead of multiplying the probabilities,
not doing so could result in numerical underflowing.
* Normalize it by the number of word in the translation, i.e. dividing the
sum of log probabilities by `T_y` to the power of alpha, where alpha is a 
number between 0 and 1.
* Larger values of `b` you get better results but it's slower. As you go
up you get diminishing returns.

## Error analysis

Let's say that:

* Human translation is `y*`
* Alg translation is `y^`

We also have `y*` is a better translation and `y^` is worse

You can have case 1 where the `P(y*|x) > P(y^|x)`

Beam serach is at fault since there's no way it should be the case since
`y*` is a better translation.

You can have the opposite case where `P(y^|x) >= P(y*|x)`

The RNN is at fault since it's not finding that `y*` is a better translation.

## Bleu Score

An evaluation for the possible translations. The closer it is to human
reference translations, the higher the score.

### Modificed precision

You give each word credit up to a max. For instance, if you have two reference
translations and one of them contains a 'the' and another one contains two.
So you have the max score for the word 'the' of two. This is not only done with
unigrams, it can be done with bigrams, etc.

**BP**: brevity penalty.

bp = 1 if MT_out_length > reference_out_length
bp = exp(1-mt_out_length/refrence_outlength) otherwise

## Attention model intuition

Instead of attempting to use an encoder/decoder for the whole sentence (which
isn't great for long sentences), it uses a different approach.

## Speech recognition

## Trigger word detection

You could output 0 for everything and after the trigger word is said you can
output 1 once or several times (a bit of a hack).





