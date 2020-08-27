# Perceptron & SVM

## Perceptron

### Objective

![](.gitbook/assets/145.png)

![](.gitbook/assets/146.png)

so perceptron tries to minimize the distance of misclassiﬁed points from the decision boundary and push them to the right side

![](.gitbook/assets/147.png)

### Optimization

![](.gitbook/assets/148.png)

### Codes

![](.gitbook/assets/149.png)

Example

![](.gitbook/assets/150.png)

**observations**: after ﬁnding a linear separator no further updates happen; the ﬁnal boundary depends on the order of instances \(diﬀerent from all previous methods\)

### Issues

![](.gitbook/assets/151.png)

cyclic updates if the data is not linearly separable?

* try make the data separable using additional features?
* data may be inherently noisy

even if linearly separable convergence could take many iterations

the decision boundary may be suboptimal

## Margin

![](.gitbook/assets/152.png)

### Max margin classifier

![](.gitbook/assets/153.png)

![](.gitbook/assets/154.png)

![](.gitbook/assets/155.png)

### Hard margin SVM objective

![](.gitbook/assets/156.png)

![](.gitbook/assets/157.png) ![](.gitbook/assets/158.png)

### Soft margin SVM constraints

allow points inside the margin and on the wrong side but penalize them

![](.gitbook/assets/159.png) ![](.gitbook/assets/160.png)

![](.gitbook/assets/161.png)

![](.gitbook/assets/162.png)

## Hinge loss

Why hinge loss:

We will punish the misclassified data points, which are located either inside the margin or wrong side of the margin, the margin is _distance from boundary._ If classify correctly, then no punishment.

![](.gitbook/assets/163.png)

![](.gitbook/assets/164.png)

![](.gitbook/assets/165.png)

In hard margin SVM there are, by definition, no misclassifications

### Perceptron vs SVM

![](.gitbook/assets/166.png)

* The Perceptron does not try to optimize the separation "distance". As long as it finds a hyperplane that separates the two sets, it is good. SVM on the other hand tries to maximize the "support vector", i.e., the distance between two closest opposite sample points.
* The SVM typically tries to use a "kernel function" to project the sample points to high dimension space to make them linearly separable, while the perceptron assumes the sample points are linearly separable.

The major practical difference between a \(kernel\) perceptron and SVM is that **perceptrons can be trained online \(i.e. their weights can be updated as new examples arrive one at a time\)** whereas SVMs cannot be. See this question for information on whether SVMs can be trained online. So, even though a SVM is usually a better classifier, perceptrons can still be useful because they are cheap and easy to re-train in a situation in which fresh training data is constantly arriving.

### ![](.gitbook/assets/167.png)

### SVM codes

![](.gitbook/assets/168.png)

![](.gitbook/assets/169.png)

![](.gitbook/assets/170.png)

![](.gitbook/assets/171.png)

Hard SVM vs soft SVM vs Perceptron

![](.gitbook/assets/172.png)

![](.gitbook/assets/173.png)

## SVM recap

![](.gitbook/assets/174.png)

## SVM vs. logistic regression

![](.gitbook/assets/175.png)

![](.gitbook/assets/176.png)

![](.gitbook/assets/177.png)

Multiclass classification

![](.gitbook/assets/178.png)

![](.gitbook/assets/179.png)



