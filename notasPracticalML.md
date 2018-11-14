# Practical Machine Learning

Data is *very* important.

Garbage in garbage out concept. If you don't have good data to feed the alg,
you're not gonna be getting good results no matter how good the alg is.

Features matter!

Good features do the following:

* Lead to data compression
* retain relevant information
* are created based on expert application knowledge

Some common mistakes people make:

* Trying to automate feature selection
* Not paying attention to data specific quirks
* Throwing away information unnecessarily

Algorithms matter less than you think, most of the time, if you get more data
you'll get better results than if you used the 'best' alg.

A good predictor complies with the following:

* Interpretable
* Simple
* Accurate (this might be a tradeoff between the others and this one)
* Fast (to train and test)
* Scalable

Prediction is about accuracy tradeoffs. You might trade speed or simplicity for it, etc.

Not understanding features might lead you to get stuck once the alg starts failing or
something.

Accuracy might not be the most important thing about a model. Speed might matter a lot.

### In sample versus out of sample errors

In sample error is from the same dataset you used to build the predictor.

Out of sample error is the error rate on new data. Also called generalization
error.

**Key points:**

* Out of sample error is what we care about
* In sample error < out of sample error
* This is because of overfitting

## Prediction study design

1. Define your error rate
1. Split data into training testing and validation (Dev)
1. On the training set pick features (using CV)
1. On the training set pick the prediction function (using CV)
1. If there's a validation set, apply to test and refine and then apply
1x to validation

Avoid small smaple sizes. Just so you can rule out being right randomly or
by chance.

** Dataset splits **

Backtesting. Sample in time chunks if predictions evolve with time

If you're predicting a rare event you could use a positive predictive value
instead of just using whatever else in order to take into account how
rare the event is.

**ROC analysis**

Since your predictors are gonna output a probability you have to pick a cutoff
point. And you're gonna wanna do this right. ROC curves allow for an nice
analysis on this. You plot and then you pick whatever is closer to the perfect
classifier (top left corner)

## Cross Validation

Split the training set into a subtraining set and a subtesting set. Test stuff
on the subtesting set and repeat that process a few times getting different
sets every time and then averaging. This is done to estimate what will happen
once you get new data.

You can do this randomly or use k-folds

For time series you have to use chunks. You might be ignoring rich data

if you're using k-folds, if you use a large k you get less bias and more
variance and if you pick a smaller k you get more bias and less variance

## What data should you use?

Be very careful about why your prediction works or doesn't work. Especially
wary of unrelated data and how it can lead you to wrong conclusions.





