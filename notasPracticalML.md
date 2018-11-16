# Practical Machine Learning

Data is *very* important.

"Garbage in; garbage out" concept. If you don't have good data to feed the alg,
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

**Dataset splits**

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

## **Week 2**

## Caret package

You can use it to clean and preprocess data. You can also slice and prepare
the data like doing stuff like CV and train/test splits. There's also useful
functions like ConfusionMatrix

It also includes regresions, naives bayes, svm, regression trees, random trees,
etc.

### Training options

Weights: useful if your data is unbalanced

**Metrics**:

* RMSE and RSquared (for continuous)
* Accuracy and Kappa (for categorical)

There's also a preprocessing option (will go in depth later)

You can specify if you want CV, bootstrapping, return options.

Since some processes dependon on random draws. If you set a seed, you will
ensure you'll get the results on different runs. Repeatable results.

## Plotting predictors

(And data)

Density plots contrasting two variables

Graph over the training set, you shouldn't look at the test set (does this
make sense?)

You're looking for:

* Outliers
* Imbalances
* Skewed variables
* Groups of points not explained by a predictor

## Preprocessing

Standardize variables. Take the value, substract the mean and then divide
by the std.

Remember to use the train mean and the train std to standardize the test set,
even tho they're not the actual values for the test set, it should work and
should standardize fairly well.

**BoxCox** transforms (take a look into this)

**Imputation**: caret has a preprocess method that can impute values using
knnImpute as an argument for the `method` parameter.


**CHECK OUT THE PREPROCESS CARET FUNCTION**

## Covariate creation

1. From raw data to covariate (create useful features that summarize data)
"Summarization vs feature loss" the game basically. You need to explain most
of what's going on.
1. Transforming tidy covariates (e.g. squaring a variable) This should be done
only on the training set. More necessary for regressions and svms. Spend some
time on the EDA. New covariates should be added to the dataset

Common covariates to add

* Dummy variables for categorical variables
* Removing zero covariates (`nearZeroVar`, e.g. vars that only have one val)
Variables that don't have any variability in them and you can throw them away

**You have to create the same covariates in the test set using the same
method**

Be careful about overfitting. If you overcreate features apply some filtering
before feeding them into your model

## Preprocessing with principal components analysis

You leave out the outcome column.

You get the correlation of each var to every other var.

Which of them have a high correlation. Higher than 0.8 e.g.

How can we take those variables and make them one variable. We should pick a
combination that captures the most information possible.

* Find a new set of multivariate variables that are uncorrelated and explain
as much variance as possible
* If you put all hte variables together in one matrix, find the matrix created
with the fewer variables

### SVD

Singular value decomposition, if a X is a matrix with each variable then:

`X = UD(V^-1)`

where **U** are orthogonal left singular vectors and **V** are orthogonal right
vectors and **D** is a diagonal matrix(singular values)

### PCA

The right singular values if you first scale (standardize with mean and std)

`prcomp` in R.

Or in caret `preProcess` with the method `pca` and specify the number of comps

Watch out for outliers *before* you do PCA. You should also plot the predictors

This can make it harder to interpret predictor stuff. It's mostly useful for
linear models.

## Predicting with different methods

### Predicting with trees

**Pros:**

* Easy to interpret
* Better performance in non-linear settings

**Cons:**

* Without CV it can lead to overfitting
* Harder to esimate uncertainty
* Results may be variable

You basically do

1. Start with all variables
1. Find the variable that best separates the outcomes
1. Divide the data into two groups/leaves
1. Within each split find the best divide
1. Continue until groups are too small

**Gini index**:

0 = perfect purity  
0.5 = no purity

Check out **Deviance/information** gain too.

In caret, use `train` and use the method `"rpart"`

The package `rattle` and its function `fancyRpartPlot` can plot the three
in a fancy way.

### Bagging

Bootstrap aggregating.

1. Resample cases and recalculate in predictions
1. Average or majority vote

You get similar bias as with trees and reduced variance. It's useful for
non linear functions.

in `caret` you can set method to `bagEarth`, `treeBag`, `bagFDA`.

You can create your own bagging method with caret. You set the predictors,
then you set the outcome. You set the number of subsamples and then you set
a control which specifies how you're gonna fit, how you're gonna predict, and
finally how you put the results together e.g. averaging.

Bagging is very useful for non linear models. OFten used with trees, an
extension of this is random forests.

### Random forests

An extension to bagging using regression trees.

1. Bootstrap samples
1. At each split, bootstrap variables
1. Grow multiple trees and vote

Second step is key as it allows for different trees

Some cons are

* Speed
* Interpretability
* Overfitting

But it's accurate.

In `caret` you can use the method `rf`

You can use `getTree` to explore the resulting trees.

### Boosting

Accurate out of the box.

Take a lot of weak predictors and weight them and add them up. This results in
a stronger predictor.

1. Start with a set of classifiers
1. Create a classifier that combines these classifier functions
	* Goal is to minimize error
	* Iterative
	* Calculate weights

So you basically classify and you upweight the points you missclassified

### Model based prediction

1. You assume the data follow a probabilistic model
1. Uses bayes' theorem to identify optimal classifiers.

More common models use this

* Linear discriminant analysis
* Quadratic discriminant analysis
* Model based prediction, allows for complicated versions of the covariance
matrix
* Naive bayes: assumes indpendence between predictors




