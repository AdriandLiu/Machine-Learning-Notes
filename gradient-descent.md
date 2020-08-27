# Gradient Descent

## Optimization in ML

![](.gitbook/assets/98.png)

## Gradient

Partial derivate ![](.gitbook/assets/99.png)

![](.gitbook/assets/100.png)

## Gradient descent

Iterative algorithm for optimization

![](.gitbook/assets/101.png)

## Convex function

### Definition

![](.gitbook/assets/102.png)

Any two points can be connected by at most one line

![](.gitbook/assets/103.png)

### Why convex?

Convex is easier to minimize:

* Critical pts are the global minimum
* Gradient descent can ﬁnd it ![](.gitbook/assets/104.png)

![](.gitbook/assets/105.png)

Concave function

## Recognizing convex functions

![](.gitbook/assets/106%20%281%29.png)

## Gradient for linear and logistic

## ![](.gitbook/assets/107%20%281%29.png)

Partial derivative w.r.t m

![](.gitbook/assets/108%20%281%29.png)

### Time complexity

![](.gitbook/assets/109%20%281%29.png)

### Codes

![](.gitbook/assets/110%20%281%29.png)

![](.gitbook/assets/111.png)

Example: ![](.gitbook/assets/112%20%281%29.png)

### Learning rate alpha

Learning rate has a signiﬁcant eﬀect on GD

* **too small**: may take a long time to converge
* **too large**: it overshoots

![](.gitbook/assets/113%20%281%29.png)

## Stochastic gradient descent

![](.gitbook/assets/114.png)

Use average \(expected value\)

| **Batch gradient update** | **Stochastic gradient update** |
| :--- | :--- |
| With small learning rate: **guaranteed** improved at each step | The **steps** are “on average” in the right direction |
| ![](.gitbook/assets/115.png) | ![](.gitbook/assets/116%20%281%29.png) |
| computes the gradient using the **whole dataset** | Stochastic gradient descent \(SGD\) computes the gradient using a **single sample**. Most applications of SGD actually use a **minibatch** of several samples |
| Slower | Faster |
| Directly towards an optimum solution, either local or global | SGD works well for error manifolds that have **lots of local maxima/minima**. In this case, the somewhat noisier gradient calculated using the reduced number of samples tends to jerk the model out of local minima into a region that hopefully is more optimal. |
| ![](.gitbook/assets/117.png) | ![](.gitbook/assets/118.png) |
| [https://stats.stackexchange.com/questions/49528/batch-gradient-descent-versus-stochastic-gradient-descent](https://stats.stackexchange.com/questions/49528/batch-gradient-descent-versus-stochastic-gradient-descent) |  |

### Convergence of SGD

Grdients will not reach 0 at optimum, how to guarantee convergence: **schedule** to have a smaller learning rate over time: learning rate is getting smaller while iterating.

![](.gitbook/assets/119.png)

## Minibatch SGD

![](.gitbook/assets/120.png)

### Codes

![](.gitbook/assets/121.png)

![](.gitbook/assets/122.png)

## Momentum

收窄SGD振幅：

* use a running average of gradients
* more recent gradients should have higher weights

![](.gitbook/assets/123.png)

Average moving:

![](.gitbook/assets/124%20%281%29.png)

### Codes

![](.gitbook/assets/125.png)

![](.gitbook/assets/126.png)

## Adagrad \(Adaptive gradient\)

* use diﬀerent learning rate for each parameter
* make the learning rate adaptive

![](.gitbook/assets/127.png)

useful when parameters are updated at diﬀerent rates \(e.g., NLP\)

![](.gitbook/assets/128.png)

**problem: the learning rate goes to zero too quickly**

## RMSprop \(Root Mean Squared Propagation\)

![](.gitbook/assets/129.png)

## Adam \(Adaptive Moment Estimation\)

two ideas so far:

1. use momentum to smooth out the oscillations

2. adaptive per-parameter learning rate

both use exponential moving averages

![](.gitbook/assets/130.png)

![](.gitbook/assets/131.png)

The authors propose default values of 0.9 for β1

, 0.999 for β2, and 10^−8 for ϵ.

## Adding L2 Regularization

![](.gitbook/assets/132.png)

## Adding L1 Regularization

![](.gitbook/assets/133.png)

**Adding regularization can also help with optimization**

