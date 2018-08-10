# Convolutional networks

Mostly used for computer vision. (Why?)

Inputs can get very large. This means that there's gonna be a lot of weights
to learn and to train. A small image can result in having to learn 1B
weights.

### Detect edges

You define a filter or a kernel. You convolve the input data with the
filter. For instance you do a grayscale image 6x6

`6x6 * 3x3  ->  4x4
    ^
    |_____ convolve`

Specific filters can detect edges, vertical, horizontal, negative, etc.
You don't get this manually tho, you get this thru backprop, you learn
the filters basically.

To convolve you multiply element wise and then add up everything.
You then shift the filter step by step.

### Padding

When using filters, the image gets smaller tho, this is a problem.
Also, information at the edges of the image only belong to a few of
the windows for the convolution operation. So you lose information
there since it belongs to only a small fraction of the output.

To solve both of these you add a padding, an extra frame around the image
or pixels you don't care about. You usually pad with zeros and you pad
`p` number of pixels.

THere's two types: Valid and Same convolutions

* Valid: no padding
* Same: the output size is the same as the input size.

Usually you have odd-numbered filters. e.g. 3x3 or 5x5

### Strided convolution

The stride amount defines how big the step is when moving the filter around
the image to get the result. You end up with a different output size btw.

Mathematicians might call the convolution we use in DL they call it cross
correlation. Convolution involves mirroring by both axes the filter before
multiplying and adding.

## Convolutions on Images

Images have a height, a width and a number of channels. Your filter will
have the same. You can represent the filter and the image as a cube and
you can perform the intuition for the convolution in that way. You shift the
cube as you go. This is also called **convolving on volumes**.

## Convolutional layers in NN

The image input is what we could call a[0] and the filters can be interpreted
as the W[1] parameters. You also add a constant to the output of applying
the filters and then you apply e.g. ReLU. The output of that is the output of
one layer of a convolutional network.

The dimensionality of the output of the layer depends on the dimensions of the
filter and the number of filters.

### Pooling layers

There's pooling layers. You can have max pooling.

* Max pooling: you divide the input into sections, and you get the max number
of that section and assign that to that position of the output. You can use
stepping in this as well.

## Example ConvNet

Usually:

Input image -> conv layer -> conv layer -> ... -> FC layer -> FC -> softmax

## Why convolutions?

Relatively fewer parameters to train compared to FC layers. This is due to:

* *parameter sharing*: this means that once you learn a specific filter, you
can apply it at any position of the image. You can share those parameters.
* *sparsity of connections*: this means that each layer and each output
value depends on a small number of inputs.

## Case Studies

### LeNet-5

First layer consists of 6 filters of 5x5 each, and **avg** pool (nowadays
you'd use a max pool). Second layer consists of 16 5x5 filters and another
avg pool. After that you get two FC layers and finally the output layer.

Lots of outdated stuff in this. They used sigmoid instead of ReLU and stuff.

### AlexNet

First layer has 11x11 96 filters with a stride of 4. Then you have a
max pool, stride 2, 3x3. Next layer is a same convolution, 5x5, 256
filters and another max pool, 3x3 and stride of 2. Next layer is
another same pool, 3x3, 384 filters. and then again, and then again,
and then again, and then a max pool of 3x3 stride 2 which results in
6x6x256 and then FC layers.

Input is much larger than LeNet but it could handle this. It was using
ReLU. Training was done on multiple GPUs.

### VGG - 16

First two layers are convolutions 3x3 filters, 64 on the first layer
then pooling, and then 128 filters with a max pool, then more conv
and pooling layers. You end up with 7x7x512. Then you do FC layers
4096 -> 4096 -> Softmax 1000.

Roughly doubling the number of filters on every layer till they
ended up with 512 filters.

### ResNet

#### Residual block

Shortcut or skip connection, you skip a layer, in order to pass
information deeper into the network.

`a[l+2] = g(z[l+2] + a[l])`

This allows you to train much deeper networks as it helps with
vanishing and exploding gradients.

### 1x1 Convolutions

Doesn't seem very useful when there's only 1 channel. If there's
more channels, then it's a lot more useful. You get the equivalent as
having 32 channels, multiplying by certain weights and then applying
ReLU to it. This also provides you with a way to reduce the number of
channels.

Previously you could shrink nH and nW with pooling layers, but now you can
reduce the number of channels as well.

### Inception Network

You use multiple filters, but stack them up and as a result
you end up with a much larger volume. You can even stack max pools.

It's important to note that in order to stack up this different filters you
need to use 'same' convolutions, in order to keep the dimensions compatible,
even for max pooling.

You can end up with a sort of bottleneck section in the network cause of its
dimensionality in terms of number of channels. This results in having to use
a fraction of calculations, reducing calculations. If you are careful about
this, you can shrink without hurting performance too much.

You can add side branches. You can have some hidden layers and try to make
predictions with it using softmax. What this attempts is that even middle
layers have a good idea of how to predict. This adds a regularization aspect
to the network.

## Transfer learning

## Data Augmentation

Common methods include:

* Mirroring
* Random cropping
* Rotation, shearing, local warping, etc. (less useful)
* Color shifting

You cna do distortions as you train.

## State of CV

The less data you have for complex problems, the more you're gonna have to
manually label stuff, hand engineering, hand pick features, etc.

This is why CV has generated complex models and stuff to be able to perform
well on this task.

### Common techniques

* Ensembling: this means average y_ of several networks.
* Multi-crop: Run classifier on multiple versions of test images, you can
even get 10 crops for each image. 5 crops and then mirror.

## Object Localization

Labels for the location of the object.

## Landmark Localization

You need a dataset that has a ton of landmark labels, laboursome work tbh,
so it could be rare to apply this too often.

You could have several face landmarks for like AR stuff, and stuff. You add
another output unit per landmark.

NOTE: Landmark `n` is always consistent across different images.

## Object Detection

###  Sliding Window Detection

You closely crop the image. You get a lot of images of both positive and
negative cases and feed it to the network to train. The objective of the net
would then be to process a crop and output whether it's positive or not.

You shift across the image and feed it to the network with a window. Once this
is done you make the window bigger and shift again thru the image, and so on
making the window larger.

#### Convolutional Implementation of Sliding Windows

Way described above is slow. So how do we do this convolutionally.

Use filters and 1x1 filters instead of FC layers. This way, you can evaluate
the whole image instead of doing passes using different filters.

Sometimes filters don't perfectly align with the image, the cutout isn't
perfect and you don't sometimes it doesn't even have to be a perfect square
cutout, maybe the object you're finding is long in width but not tall.

#### YOLO Algorithm

You add a grid to the image (in practice something like 19x19) and you run
the first alg on each grid area. You have a label for each grid, where you
have whether there's an object or not. You assign each landmark to the grid
where the center of it is located, even if there's a bit of it in another
bounding box.

#### Intersection of union

You take the label area of the image, i.e. the correct answer of the object
detection process and then answer of your network and you take the area of
both and you divide the intersection over the union of the areas.

The higher the area is the better. You set a threshold (usually 0.5) to
accept an answer as correct.

#### Non-max Suppression

When more than one cell thinks it has found the object center. You might end
up with different detections. Non-max suppression cleans up these detections.
It finds rectangles that overlap a lot with others, and remove them. It finds
the squares with the highest pc and erase any others that have a high iou with
them and then just keeps those.

It basically removes the ones that aren't maximal, hence its name.

* Discard all boxes with a pc below a threshold (usually 0.6)
* Pick the box with the largest pc and output that as a prediction
* Discard any remaining boxes with an iou >= threshold(0.5) with
the output box.

##### Anchor boxes

You define anchor boxes of a certain shape and whatever is detected then
gets compared with the anchor boxes and calculates the iou and picks
whichever it has more overlap with.


## Face recognition

* Face recognition
* Face verification

### One Shot Learning

You can't just feed pictures and output a label of which person it is, you'd
have to retrain every time you get a new person, and you only get one example
per person so it's not gonna work well. Makes no sense.

You don't learn to recognize directly thru pictures, you learn a similarity
function. That is, if you input two images, you need to output the degree
of difference and you compare that with a threshold called tau.

### Siamese Network

You get the FC last layer of the CNN you are using and you can say that this
last layer is an encoding of the image in an N-dimensional space.

You then define the difference of them as the norm_2. If the norm is small
you can conclude it's the same person; if it's large you conclude they are
not.

But how to define the objective function?

### Triplet Loss Function

You look at three images:

* **A**nchor image, reference image
* **P**ositive image, a similar image
* **N**egative image, a dissimilar image

So you want:

`||f(A) - f(P)||^2  - ||f(A) - f(N)||^2 <= 0`

Trivially, we could make f() always output `0` for the condition to be true
so we don't compare to zero but to -alpha so we can add it on the left side
instead of substracting it from the right side. We call alpha a _margin_.

`||f(A) - f(P)||^2  - ||f(A) - f(N)||^2 + alpha <= 0`

The loss function you then define as the max between 0 and the expression
above.

So you can have `J` be the sum of the losses over `A(i)`, `P(i)`, `N(i)`

**Note**: You want your training set to be structured such that images are
hard to be trained on, i.e. you need to make the images similar enough so it's
hard to train on them.

### Binary Classification Alternative

You output with siamese networks and then you feed that into a logistic
regression unit and train on it. You compare the images and determine whether
they're the same `1` or different `0` people. You could also feed the
difference between the values/features output by each of the siamese networks.

Note: You can precompute whatever image is not new thru one side of the
siamese network.

## Neurol Style Transfer

You can take the style of an image and apply it to another image.

* Take a unit and see what kind of image patches maximizes its activation.
* Repeat for other units

### Content Cost Function

If you have `a[l](C)` and `a[l](G)` and they have similar activations then
you can conclude that the style is similar.

### Style Cost Function

How correlated channels are. Grab the activations matrix and multiply it
by its transposed version. That way you get the correlations matrix.



