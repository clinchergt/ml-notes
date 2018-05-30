# DL

antes: 70/30 o 60/20/20
hasta incluso unos años
para 100-10,000 puntos de datos

para la nueva epoca, big data:
(1,000,000 de puntos tal vez)
tal vez solo 10k para el test set es suficiente para evaluar
y otro 10k para test

hay nuevos cocientes como 98/1/1


mismatched train/test distribution

e.g. pictures from the web to train, but pictures from an app you run where people upload pics to test

in this case, make sure dev and test set come from the same distribution

get more data by being creative: web crawling, etc.

not having a test set might be okay. when?

bias/variance

high bias -> le falta
high variance -> overfitting

compare these two to figure out what's going on.
* train set error
* dev set error

low train and high dev, high variance
high train, high dev, high bias
high train, even higher dev, high bias and high variance
low on both you have low bias and variance

"high" error on train can be subjective, depending on what a perfect system can achieve
high variance *and* high bias means it doesn't generalize well but overfits specific training examples
worse possible case scenario tbh ^

Steps to validate model:

does it have high bias? (training dataset performance)
if it does, you could try a bigger network, train for longer, other opt algs, etc.
there are different NN architectures, could maybe try out another one.

once you reduce the bias, do you have high variance?
does your model generalize well for more data. if you have high variance, get more data, try regularization,
try another NN architecture as well.

Regularization:

Reduces overfitting.

Add a term to multiply the norm of the W parameters usually the L2 regularization:

J(W,b) = 1/m sum(L (y_, y)) + lambda/(2m) * || w ||^2

In NN you do this for every layer

and you add it to the dW's, the regularization expression

also called weight decay.

regularization punishes large values in W. Which means they are penalized in the activation
function as well. the larger lambda is and the smaller weights are, the more linear the function will be. it just makes
that every layer is roughly linear, which makes the NN more linear

Dropout regularization:

flip a coin and kill nodes at random while training each example. the coin is flipped
for every step, and you train with only the surviving on, backpropagation, and then
train the next example with different nodes, etc.

d3 = np.random.randn(a3.shape[0], a3.shape[1]) < keep_prob
a3 = np.multiply(a3, d3)
a3 /= keep_prob  # restores the previous (before dropout) expected value

at test time, you don't drop out.

Why does dropout work?

You can't rely on a single feature, you have to spread out the importance i.e. the weights.

You can up the probability of dropout if the layer is really big, and set it to be smaller for smaller layers,
kinda like upping lambda for L2 regularization.

downside is you have more hyperparameters. another option is to have dropout only for certain layers.


Other methods:

data augmentation: e.g. vision, you can rotate images and distort them slightly and zoom in etc. and have your
model understand that it's the same thing should be the same output.

early stopping: stop once the error stops going down and starts going up.

NOTE: Concept: orthogonalization, work on problems independently. early stopping breaks this

----

### Normalize data:

Cost function gets more symmetric. unnormalized elongates the function cause of the vastly different ranges.
speeds up training, less iterations for optimization.

### Vanishing/exploding gradients

the more layers they gradients explode or vanish depending on if they're slighly above or below 1

#### Weight initialization for DNNs: 

this kinda solves the above problem, being careful when initializing.
the larger n (number of features in layer) is, the smaller you want wi

the variance of wi = 1/n actually 2/n

W[l] = np.random.randn(shape) * np.sqrt(2/(n[l-1]))

this doesn't solve exploding/vanishing gradients but it helps a lot

this version assumes ReLU, there's other variants.

if you use tanh you could use sqrt(1/n[l-1]) there's also sqrt(2/(n[l-1] + n[l])

this variance parameter could be another hyperparameter for the NN

### Numerical approximation of gradients

checking derivative cmputation, you nodge theta both forwards and backwards. and using the difference between
those points. (f(theta + epsilon) - f(theta - epsilon)) / 2epsilon

### Gradient checking

Take all W's and b's and reshape them into a single big vector
same thing for dW's and db's

now J(giant_theta)

use the previous concept and add/sub epsilon to giant_theta and get dTheta_aprox and compare with dTheta

check: norm(dTheta_aprox - dTheta)/(norm(dTheta_aprox) + norm(dTheta))

DO NOT use this in training, only to debug
Remember to include regularization if there's any
Doesn't work with dropout


## Batch vs mini-batch gradient descent

stochastic(?)

X{n} nth batch of X
Y{n} same

batch gradient descent is the whole dataset per step
mini-batch a mini batch per step

the cost gets adjusted to the new sizes of the batches, so e.g. 1/1000 and not 1/m
same goes for the regularization term

1 epoch is a whole pass thru the whole dataset
you want more than 1 epoch usually

#### Intuition

you train on a different set each step so the progress on cost will be less stable, it should
still trend downwards. but it's gonna be noisy

another parameter would be the mini-batch size
if the size == m, then you just end up with regular batch gradient descent
if the size == 1, this would be the stochastic gradient descent(!)

you need to be inbetween, unless your dataset is small, then batch gradient descent is good enough
stochastic doesn't take advantage of vectorization
typical mini-batch sizes are: 64, 128, 256, 512


## More optimizers that are better than gradient descent

### exponentially weight (moving) averages
v_t = beta * v_t-1 + (1 - beta)*theta_t

v_t is approximately averaging over 1/(1 - beta) day's temperature.

the bigger beta, the slower it adapts to the changes in v_t
the smaller beta, the faster you adapt but the noisier it gets

### Bias correction in exponentially weighted averages

the first few days are gonna be really low, cause of 1 - beta, since that multiplies and at first
it's just 0

better estimate: v_t/(1-beta^t)

note: not a lot of people bother with implementing this, maybe not worth it

### Momentum

Almost always works faster than standard gradient descent

basically you get a weight average for he gradients and update with those. you take the oscillation
of the first iterations and move in the general direction of the average.

w = w - alpha * VdW
b = b - alpha * Vdb

beta = 0.9 means last 10 iterations
VdW and Vdb are initialized with 0s

### RMSprop

grouped mean square prop

Sdw = beta*SdW + (1-beta)dW^2 # element wise squared
Sdb = beta*Sdb + (1-beta)db^2

w = w - alpha(dW/sqrt(Sdw))
b = b - alpha(db/sqrt(Sdb))

if you're overshooting on an axis this will scale that up and smoothen out the motion due to the
dividing term. the larger the jump, the more you divide by when updating so it smoothens out and
hopefully don't make gradient descent as zig-zaggy and maybe use a bigger learning rate


you can combine RMSprop with momentum. this would make this beta beta2, simply to not confuse
both betas

### Adam optimization alg

Vdw = 0, SdW = 0, Vdb = 0 Sdb = 0

you iterate over t and use a mini batch

VdW = beta1 + (1-beta1)dW
Vdb = beta1 + (1-beta1)db

Sdw = beta2*SdW + (1-beta2)dW^2
Sdb = beta2*Sdb + (1-beta2)db^2

Vdw[corrected] = Vdw/(1-beta1^t)
Vdb[corrected] = Vdb/(1-beta1^t)

Same for SdW and Sdb

W = W - alpha* VdW[corrected]/(sqrt(SdW[corrected]) + epsilon)
b = b - alpha* Vdb[corrected]/(sqrt(Sdb[corrected]) + epsilon)

alpha needs to be tuned
beta1 usually 0.9 (dW)
beta2 usually 0.999 (dW^2)
epsilon usually 10e-08

### Learning rate decay

As you go, you can take smaller steps by having alpha be slower

recall what epoch is; going thru the whole training set

alpha = (1/(1+decay_rate * epoch_num)) * alpha

decay_rate is another parameter

There's also exponential learning decay: alpha = 0.95^epochnum * alpha
Also there's some constant variant: alpha = k/sqrt(epochnum) * alpha
Also there's a staircase method, update after a while

And finally there's also manual learning decay

so many hyperparameters! how to optimize for them? try them out and stuff

## Local optima

saddle point, they're not really a problem.
local optima don't really translate to high dimensional spaces

plateaus are a problem tho, they slow down stuff since you go
in their directions until you find your way out
you probably won't get stuck there tho


## Hyperparameters

alpha 
beta
beta1, beta2, epsilon
\# layers
\# hidden units
learning rate decay
mini-batch size

_most important_: alpha
next would be: momentum, 0.9 is a good start
mini batch size and the number of hidden units

third in importance: number of layers and the learning
rate decay

you pretty much never tune adam parameters

Before: you used a grid and tried out all the points to map out the scenario

Nowadays in deep learning: you pick at random and test on those points
this works well because it allows you to vary the more important variables
more than when you do a grid.

You then zoom into the area where the best were found and try out more random
points but in a more dense way.

### Use an appropriate scale for hyperparameters

Some hyperparameters allow for a true random search or hell even a grid

but e.g. alpha could not behave this way, you could avoid a linear scale
and use a log scale and sample uniformly at random on the log scale

r = -4 * np.random.rand()
alpha = 10^r

another tricky case could be the hyperparameters for exponentially weighted avgs.

set the range to 1-beta = 0.1 ... 0.001
and use the same method described above

### Tuning in practice

babysitting model: gradually babysit the model as it's training

training many models in parallel: 


## Batch normalization

is this basically normalizing the A parameters before plugging it into the next
layer and have it train faster and have the whole network converge faster?

before or after the activation function? the default choice is to normalize Z

given some intermediate values in NN z1, ... , zm

get mean and variance
znorm = (zi - mean)/ (sqrt(variance + epsilon))

instead we compute
`zi = gamma zi + beta,

where gamma and beta are learnable parameters of the model and you update
them as you would upate W and b

setting gamma and beta to reasonable values you can get something that's
really good at normalizing

you may not want the intermediate values to be have mean 0 and variance 1

### Fitting batch norm into a neural network

X ----> Z -------> `Z ---> a(Z) ---> etc.
   W,b     beta,
          gamma 

any constant you add is gonna be negated by substracting the mean,
so you usually just don't use b
to use gradient descent you have to get dGamma, dBeta and dW and then
update the same way with a learning rate, etc.

#### WHy does it work

From the perspective of each layer, they get features and try to improve the
answer

But this hidden unit values that are features for the next one suffer from
_covariance shift_

batch norm eases the problem that the input values change all the time

batch norm also has a sligh regularization effect. this is due to the fact
that the mean/variance is computed on each minibatch which has a similar
effect as dropout. as you increase the mini batch size this effect decreases
however, don't turn to batch norm as a regularization alg

### what happens at test time?

you come up with an estimate for mean and variance. how?
usually you use exponentially weighted average across the mini batch

keep track of the averages during training etc.

## Multi-class classification

### Softmax regression

C = # of classes

the output layer has the number of classes

n[L] = C

so you have it output the probability for each class

after computing z[L]

the activation function for the softmax classifier

t = e^(z[L]) # element wise

a_i[L] = e^z[L]/(sum(t_i))

so you do e^x and normalize the sum of those so it's 1


### Train a softmax classifier

NOTE: it's softmax in contrast to hardmax. hard max doesn't map
probabilities to each class, but usually just something like
[1,0,0,0]

if C = 2, then softmax ends up being a simple regression

Loss(y', y) = -sum from 1 to C of (y_j log(y'_j))

y being a vector with a similar shape as hardmax. one hot vector
for the classes

J(W,b,...) = 1/m* sum(L(y', y))

*Backpropagation*

dZ[L] = y' - y


## Frameworks

- Ease of programming (including deploying)
- Running speed
- Truly open


### Tensorflow

`tf` stuff


# ML Project structure and approaches

## ML Strategy

Fit training set well on cost function
Fit dev set well on cost function
Fit test set well on cost function
Performs well in real world

This is the chain of assumptions in ML

There's optimizing metrics
and there are satisficing metrics

If you have N metrics, it's reasonable to make one optimizing
and the rest of them satisficing (i.e. reach a threshold)


dev and test set must have the same distribution
otherwise you're gonna be optimizing for the wrong thing

sizes for dev and test sets

1M+ 98% train, 10k dev set, 10k test set


size for test set: should be big enough to give
high confidence

having no test set is fine.

### Changing metrics

when calculating the error, add a weight term that
shoots up the error if something you really don't want to happen
happens

to sum up for now

1. define a metric to evaluate classifiers, and try optimize for it
2. worry separately how to do well on this metric

if you're doing well on metric and dev set but not in your application
then you have to change your metric and/or your dev set

bayes optimal error: best possible error. no way to achieve

#### Avoidable bias

If humans outperform your training error by a lot then you need to work on your bias

Human error is a proxy for bayes error.

the avoidable bias is the differnece between the human/base error and your model

if someone or some group of humans or a system is capable of achieving a certain
accuracy tha'ts better than the rest, you use that as an estimate for bayes error.

## Surpassing human level performance

You don't really know what bayes error is now. not sure what to focus on.
this slows down progress

Reduce avoidable bias: train bigger model, train longer/better opt. algs
											(momentum, adam, rmsprop)
											NN architecture/hyperparameter
											or try RNN and CNN

Reduce variance: more data, regularization, other architectures and
				 hyperparameters


## Error analysis

analyze your data and see if your approach will have a significant
impact or if it's just gonna be marginal.

for instance you optimize for something in your data that only represents
5% of your dev/test data. performance will barely improve

## Incorrectly labeled examples

So long as the errors are reasonably random, the alg can handle them,
they are quite robust. maybe not worth it to correct and waste time
on it.

if they are systematic errors, that's a problem.

look at overall dev set error and out of this check:
- then look at errors due to incorrect labels
- then errors due to other causes

### Correcting incorrect dev/test set

- apply the same process for both
- consider examples both that the alg got right and the ones it got wrong
- train and dev/test come from different distributions


### Build system fast and then iterate

- Quickly set up a dev/test set and a metric
- Build initial system quickly
- Use bias/variance analysis and error analysis
- 

### Training and testing on different distributions

Some teams shove all kinds of data to the training set. more and more
teams train and test on different distributions.

e.g. pics of cats from the web (professional pics, in focus, etc.)
vs pics uploaded by people (out of focus, cropped cats, etc.)

the latter being the target distribution because you want to id
pics taken by your users.

option 1: merge train and dev sets and then randomly shuffle into
train and dev and test set. huge disdvantage: most of the test set
doesn't represent your target distribution. this is not acceptable

option 2: add a portion of your target data to the training set and
dev and test set represents solely your target distribution. over
the long term, this will give a better performance.

### Bias and variance on mismatched datasets

The way you analyze is different if you have mismatched datasets

It's not as easy to conclude that you have high variance if the
training error is low and the dev error is high.

What we do is shuffle the training set and set apart a small fraction
to know if you have high variance. this new train-dev set has the same
distribution as the training set. so you cross validate on the new
train-dev set. You train on the train set, and test on both the
train-dev set and on the dev set. So now you can compare. e.g.:

train error: 1%
train-dev error: 9%  ---> same distribution, so you can conclude variance
dev error: 10%

train error: 1%
train-dev error: 1.5% --> low variance
dev error: 10%  ---> data mismatch problem

Comparing with human error is still valid to determine high avoidable
bias, using the human error as a proxy for bayes error.

*What happens if the dev/test error is lower than train/train-dev error?*

### How to address data mismatch

Carry out manual error analysis. try to understand the differences, only
on dev sets, not on test sets.

collect more data similar to the dev/test set. you can even simulate
noise and stuff.

Careful with how you add noise synthesis, your model could overfit to the
specific noise you add. even if they look/sound fine to humans.


## Transfer learning

If you have a network you want to adapt a NN to another task e.g. you have
a vision NN and you want to adapt it to an x-ray application, you delete
the last layers and the weights calculated for it.

Initialize randomly the weights you removed and train again. that way you
have transferred the knowledge you had.

a lot of the lower levels of the network could help in another problem.
which is why it's useful to be able to transfer learning.

when retraining you could also add more layers to replace the last layer.

Tranferring learning is useful when the new problem has less data than
the original data. So for instance going from speech recognition to
wakeword detection.

## Multi taks learning

If you have multiple outputs, the loss function needs to account for them.

Unlike softmax regression, one image can have multiple labels, not just one.

If the dataset is incomplete wrt some labels. That is, if someone didn't label
every field for every example. And if there's some missing values, you can
still train. How? You only sum over the values that are present, and omit the
rest.

When do you benefit from multi-task leraning:

* Shared lower-level features
* Amount of data for each task is quite similar
* Can train a big enough NN to do well on *all* tasks


## End to end deep learning

You skip the different stages. So e.g., instead of defining features for audio
stuff, then phonemes, words, etc. and then arriving at the transcript, you
only just go from audio to transcript.

You need *a lot* of data. If you don't have a lot of data, you should probably
just stick to the usual pipeline.

#### When to use it?

Pros:

* Let the data speak for itself
* Less hand designing of components needed

Cons:

* You need *a lot* of data
* Excludes potentially useful hand-designed components

**In short: Do you have enough data to learn a function complex enough
to map x to y?**

FIN







