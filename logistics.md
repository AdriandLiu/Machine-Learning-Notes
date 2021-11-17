# Logistics

\
**Logistics is a linear classifier**

more than one class:  y ∈ {0,1,…,C} (**multi-class**: logistics fit C number of classifier and make prediction based on the most confident result), fit a linear model to each class, turn y into one-hot encoding: ![](file:///C:/Users/ldhan/AppData/Local/Temp/msohtmlclip1/01/clip\_image002.jpg)

![](<.gitbook/assets/image (32).png>)

Linear regression is sensitive to the **outliers**

L2 loss **problem**: correct prediction can have higher loss than the incorrect one! ![](file:///C:/Users/ldhan/AppData/Local/Temp/msohtmlclip1/01/clip\_image004.jpg)

![](<.gitbook/assets/image (33).png>)

Solution: Squash the loss function: ![](file:///C:/Users/ldhan/AppData/Local/Temp/msohtmlclip1/01/clip\_image006.jpg)&#x20;

![](<.gitbook/assets/image (34).png>)

Logistics:

![](<.gitbook/assets/image (35).png>)

Decision boundary:

![](<.gitbook/assets/image (36).png>)

## Logistic Regression Model:

![](<.gitbook/assets/image (37).png>)

## Loss functions for linear classifier

![](<.gitbook/assets/image (38).png>)

Simplifying the cost function

![](<.gitbook/assets/image (39).png>)

## Find the optimal weights

### Gradient

![](<.gitbook/assets/image (40).png>)

![](<.gitbook/assets/image (85).png>)

Logit function: ![](file:///C:/Users/ldhan/AppData/Local/Temp/msohtmlclip1/01/clip\_image019.png)

![](<.gitbook/assets/image (41).png>)

## Likelihood

![](<.gitbook/assets/image (42).png>)

![](<.gitbook/assets/image (43).png>)

### Maximum likelihood & logistic regression

Cross Entropy loss deviated from log-likelihood of Bernoulli PDF

Minimizing logistic loss corresponds to maximizing **Bernoulli** likelihood. Minimizing squared-error loss corresponds to maximizing **Gaussian** likelihood (it's just OLS regression; for 2-class classification it's actually equivalent to LDA).

Log likelihood: why? Likelihood value blows up for large N, work with log-likelihood instead (same maximum) cross entropy

![](<.gitbook/assets/image (44).png>)

### Maximum likelihood & linear regression

Squared error loss:

![](<.gitbook/assets/image (45).png>)

Minimizing squared-error loss corresponds to maximizing **Gaussian** likelihood (it's just OLS regression; for 2-class classification it's actually equivalent to LDA).

![](<.gitbook/assets/image (82).png>)

## Multiclass classification

**Binary** classification: **Bernoulli** likelihood

![](<.gitbook/assets/image (46).png>)

C classes: categorical likelihood

**Softmax**

Softmax做事情就是对**最大值**进行强化 (for the reason why exp(z\_i))

![](file:///C:/Users/ldhan/AppData/Local/Temp/msohtmlclip1/01/clip\_image031.png) ![](file:///C:/Users/ldhan/AppData/Local/Temp/msohtmlclip1/01/clip\_image033.jpg)

![](<.gitbook/assets/image (47).png>)

![](<.gitbook/assets/image (48).png>)

![](<.gitbook/assets/image (49).png>)

if input values are large, **softmax** becomes similar to **argmax**

![](<.gitbook/assets/image (50).png>)

### Softmax likelihood & one-hot encoding

**Why one-hot encoding**: The integer values have a natural ordered relationship between each other and machine learning algorithms may be able to understand and harness this relationship

![](<.gitbook/assets/image (51).png>)

we can also use this encoding for **categorical inputs features**  one-hot encoding for input features

![](<.gitbook/assets/image (52).png>)

**Problem:** these features are not linearly independent, why? might become an issue for linear regression. why

**Solution:**

remove one of the one-hot encoding features

![](<.gitbook/assets/image (53).png>)

## Optimization

### Gradient descent!

![](<.gitbook/assets/image (54).png>)

![](<.gitbook/assets/image (55).png>)
