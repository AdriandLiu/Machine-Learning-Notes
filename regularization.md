# Regularization

**Avoid overfitting**

**When overfitting, we often see large weights**

**!!! Idea: penalize large parameter values**

why: larger weight/parameter values usually indicate overfitting, because the model pay too much attention on some specific features, what we are trying to do is reduce/penalize the large parameter values, as long as the parameter values remain in a reasonable range, the model is no longer overfitting, same logic/idea for both linear and neural nets.&#x20;

Neural net regularization:&#x20;

* dropout: randomly deactivate some neurons in hidden layer
* batch normalization: normalize the batch of data based on its mean and standard deviation, which will lower the influence of some large values to the gradient update, so that the weight wouldn't be very large (pay too much attention)



**The main intuitive difference between the L1 and L2 regularization is that L1 regularization tries to estimate the median of the data while the L2 regularization tries to estimate the mean of the data to avoid overfitting**.



### Regularization in DL

by adding a parameter norm penalty Ω(θ) to the objective function J of the DL model

{% embed url="https://medium.com/analytics-vidhya/regularization-understanding-l1-and-l2-regularization-for-deep-learning-a7b9e4a409bf" %}

## How regularization work, L2 e.g.

![](<.gitbook/assets/image (87).png>)

when $$\lambda$$ is very large, we penalize the weights and they become close to zero

![](<.gitbook/assets/image (88).png>)

![](<.gitbook/assets/image (89).png>)

why $$\lambda$$ is large, weight become close to zero:

![](<.gitbook/assets/image (90).png>)

take the partial derivate of w (calculate gradient), and we can see that when **lambda is very large, w - lambda\*w will make the w close to zero**

## Why regularization work

During high bias, weights will be very small. During high variance, weights will be high. Similarly, during regularisation....if lambda is near infinity or high, our weights will tend to go down, because the function (gradient decent) will always try to minimize the overall value. If there's less lambda, weights will increase and model will try to fit each data point.......that also creates overfitting problems. So by tuning lambda in such a way that; both bias and variance should be in a acceptable range.

## Ridge regression

**L2 regularized** linear least squares regression:

![](.gitbook/assets/76.png)

Side note L2 norm:

![](.gitbook/assets/77.png)

* regularization parameter  λ > 0 controls the strength of regularization
* a good practice is to not penalize the intercept

## Ridge weight formula

![](.gitbook/assets/78.png)

## Ridge with data normalization

**Without regularization**: ![](.gitbook/assets/79.png)

![](.gitbook/assets/80.png)

**With regularization**: ![](.gitbook/assets/81.png)

Diff features will be penalized differently

![](.gitbook/assets/82.png)

**Instead of maximize log-likelihood, we maximize the posterior**

![](.gitbook/assets/83.png)

## Maximum a Posteriori (MAP)

![](.gitbook/assets/84.png)

## Gaussian prior

![](.gitbook/assets/85.png)

## Laplace prior

Lasso ![](.gitbook/assets/86.png)

## L1 vs L2 regularization

![](.gitbook/assets/87.png)

![](.gitbook/assets/88.png)

## Diff regularization subset selection

![](.gitbook/assets/89.png)

optimizing this is a diﬃcult combinatorial problem:

* search over all    2^D     subsets

## L1 VS L0

It’s just like LASSO but has a little difference. LASSO has a limit:

the L1 norm of the parameters < t (some constant threshold)

For L0 regularization. The constraint is the number of parameters < t (some constant threshold)

Most people never heard about it because LASSO is good enough in the cases that people want to punish some parameters to zero. L0 regularization shares the same function with it. **The difference is L0 is more extreme than L1. The parameters are much easier to be punished to zero.**

If you have 500 features in the pool and you want 10 of them left, you can try LASSO. However, if you have 10k features in the pool and you want 10 of them left, you probably want to try L0 regularization.

## Bias-variance decomposition

![](.gitbook/assets/90.png)

![](.gitbook/assets/91.png)

![](.gitbook/assets/92.png)

![](.gitbook/assets/93.png)

Larger regularization penalty -> high bias – low variance

high variance in more complex models means that test and training error can be very diﬀerent

high bias in simplistic models means that training error can be high

![](.gitbook/assets/94.png)

## Cross validation

k-fold CV

![](.gitbook/assets/95.png)

leave-one-out CV:extreme case of k=N

Test data:

once the hyper-parameters are selected, we can use the whole set for training use test set for the **ﬁnal** assessment

## Evaluation

![](.gitbook/assets/96.png)

ROC receiver operating characteristic

![](.gitbook/assets/97.png)

How to graph: [https://acutecaretesting.org/en/articles/roc-curves-what-are-they-and-how-are-they-used](https://acutecaretesting.org/en/articles/roc-curves-what-are-they-and-how-are-they-used)

##
