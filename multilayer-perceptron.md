# Multilayer Perceptron

![](.gitbook/assets/254.png)

## Adaptive Radial Base

![](.gitbook/assets/255.png)

![](.gitbook/assets/256.png)

![](.gitbook/assets/257.png)

## Sigmoid Bases

![](.gitbook/assets/258.png)

![](.gitbook/assets/259.png)

### Adaptive sigmoid bases

![](.gitbook/assets/260.png)

## Multilayer perceptron MLP

![](.gitbook/assets/261.png)

![](.gitbook/assets/262.png)

## Regression

![](.gitbook/assets/263.png)

![](.gitbook/assets/264.png)

## Classification

![](.gitbook/assets/265.png)

![](.gitbook/assets/266.png)

## !!! Activation function

### Logistic function

### Hyperbolic tangent

![](.gitbook/assets/267.png)

### ReLU

### Leaky ReLU

### Softplus

![](.gitbook/assets/268.png)

## Network architecture

Feedforward network aka multilayer perceptron

![](.gitbook/assets/269.png)

![](.gitbook/assets/270.png)

## Depth vs Width

![](.gitbook/assets/271.png)

![](.gitbook/assets/272.png)

![](.gitbook/assets/273.png)

## Multilayer perceptron

![](.gitbook/assets/274.png)

![](.gitbook/assets/275.png)

![](.gitbook/assets/276.png)

![](.gitbook/assets/277.png)

![](.gitbook/assets/278.png)

![](.gitbook/assets/279.png)

![](.gitbook/assets/280.png)

fully-connected: all outputs of one layer's units are input to all the next units

![](.gitbook/assets/281.png)

## Regularization strategies

### Overfit: variance reduction

![](.gitbook/assets/282.png)

Data augmentation \(增大\)

![](.gitbook/assets/283.png)

![](.gitbook/assets/284.png)

## Noise robustness

![](.gitbook/assets/285.png)

## Early stopping

![](.gitbook/assets/286.png)

## Bagging

![](.gitbook/assets/287.png)

## Dropout

![](.gitbook/assets/288.png)

![](.gitbook/assets/289.png)

Let's start with **normal dropout**, i.e. dropout only at training time. Here dropout serves as a regularization to **avoid overfitting**. During test time, dropout is not applied; instead, all nodes/connections are present, but the weights are adjusted accordingly\(e.g. multiply the dropout ratio\). Such a model during test time can be understood as a average of an ensemble of neural networks.

Notice that for normal dropout, at test time the prediction is **deterministic**. _Without other source of randomness, given one test data point, the model will always predict the same label or value_.

For **Monte Carlo dropout**, the dropout is applied at both training and test time. At test time, the prediction is **no longer** **deterministic**, but depending on which nodes/links you randomly choose to keep. Therefore, _given a same datapoint, your model could predict different values each time._

So the primary goal of Monte Carlo dropout is to generate random predictions and **interpret them as samples from a probabilistic distribution**. In the authors' words, they call it Bayesian interpretation.

Example: suppose you trained an dog/cat image classifier with Monte Carlo dropout. If you feed a same image to the classifier again and again, the classifier may be predicting dog 70% of the times while predicting cat 30% of the time. Therefore you can interpret the result in a probabilistic way: with 70% probability, this image shows a dog.

## Summary

* Deep feed-forward networks learn adaptive bases
* more complex bases at higher layers
* increasing depth is often preferable to width
* various choices of activation function and architecture
* universal approximation power
* their expressive power often necessitates using regularization schemes

