# Logistics

  
**Logistics is a linear classifier**

more than one class:  y ∈ {0,1,…,C} \(**multi-class**: logistics fit C number of classifier and make prediction based on the most confident result\), fit a linear model to each class, turn y into one-hot encoding: ![](file:///C:/Users/ldhan/AppData/Local/Temp/msohtmlclip1/01/clip_image002.jpg)

![](.gitbook/assets/image%20%2821%29.png)

Linear regression is sensitive to the **outliers**

L2 loss **problem**: correct prediction can have higher loss than the incorrect one! ![](file:///C:/Users/ldhan/AppData/Local/Temp/msohtmlclip1/01/clip_image004.jpg)

![](.gitbook/assets/image%20%2866%29.png)

Solution: Squash the loss function: ![](file:///C:/Users/ldhan/AppData/Local/Temp/msohtmlclip1/01/clip_image006.jpg) 

![](.gitbook/assets/image%20%2830%29.png)

Logistics:

![](.gitbook/assets/image%20%2854%29.png)

Decision boundary:

![](.gitbook/assets/image%20%2860%29.png)

## Logistic Regression Model:

![](.gitbook/assets/image%20%2877%29.png)

## Loss functions for linear classifier

![](.gitbook/assets/image%20%2838%29.png)

Simplifying the cost function

![](.gitbook/assets/image%20%2828%29.png)

## Find the optimal weights

### Gradient

![](.gitbook/assets/image%20%2855%29.png)

![](.gitbook/assets/image%20%2886%29.png)

Logit function: ![](file:///C:/Users/ldhan/AppData/Local/Temp/msohtmlclip1/01/clip_image019.png)

![](.gitbook/assets/image%20%2842%29.png)

## Likelihood

![](.gitbook/assets/image%20%2822%29.png)

![](.gitbook/assets/image%20%2848%29.png)

### Maximum likelihood & logistic regression

Cross Entropy loss deviated from log-likelihood of Bernoulli PDF

Minimizing logistic loss corresponds to maximizing **Bernoulli** likelihood. Minimizing squared-error loss corresponds to maximizing **Gaussian** likelihood \(it's just OLS regression; for 2-class classification it's actually equivalent to LDA\).

Log likelihood: why? Likelihood value blows up for large N, work with log-likelihood instead \(same maximum\) cross entropy

![](.gitbook/assets/image%20%2810%29.png)

### Maximum likelihood & linear regression

Squared error loss:

![](.gitbook/assets/image%20%2817%29.png)

Minimizing squared-error loss corresponds to maximizing **Gaussian** likelihood \(it's just OLS regression; for 2-class classification it's actually equivalent to LDA\).

![](.gitbook/assets/image%20%2893%29.png)

## Multiclass classification

**Binary** classification: **Bernoulli** likelihood

![](.gitbook/assets/image%20%2875%29.png)

C classes: categorical likelihood

**Softmax**

Softmax做事情就是对**最大值**进行强化 \(for the reason why exp\(z\_i\)\)

![](file:///C:/Users/ldhan/AppData/Local/Temp/msohtmlclip1/01/clip_image031.png) ![](file:///C:/Users/ldhan/AppData/Local/Temp/msohtmlclip1/01/clip_image033.jpg)

![](.gitbook/assets/image%20%2849%29.png)

![](.gitbook/assets/image%20%2862%29.png)

![](.gitbook/assets/image%20%2871%29.png)

if input values are large, **softmax** becomes similar to **argmax**

![](.gitbook/assets/image%20%2844%29.png)

### Softmax likelihood & one-hot encoding

**Why one-hot encoding**: The integer values have a natural ordered relationship between each other and machine learning algorithms may be able to understand and harness this relationship

![](.gitbook/assets/image%20%2841%29.png)

we can also use this encoding for **categorical inputs features**  one-hot encoding for input features

![](.gitbook/assets/image%20%2876%29.png)

**Problem:** these features are not linearly independent, why? might become an issue for linear regression. why

**Solution:**

remove one of the one-hot encoding features

![](.gitbook/assets/image%20%2868%29.png)

## Optimization

### Gradient descent!

![](.gitbook/assets/image%20%2880%29.png)

![](.gitbook/assets/image%20%2825%29.png)

