# Naive Bayes

  
**Discriminative**: conditional distribution p\(y\|x\) eg: linear and logistics

**Generative**: joint distribution p\(x,y\) = p\(y\)p\(x\|y\) eg:

 Naïve bayes ![](file:///C:/Users/ldhan/AppData/Local/Temp/msohtmlclip1/01/clip_image003.jpg)

![](.gitbook/assets/image%20%2813%29.png)

Example:

![](.gitbook/assets/image%20%2857%29.png)

**in a generative classiﬁer likelihood & prior class probabilities are learned from data**

Some generative classiﬁers:

·         Gaussian Discriminant Analysis: the likelihood is multivariate Gaussian

·         Naive Bayes: decomposed likelihood

## Naïve bayes model

NB **assumption**: ![](file:///C:/Users/ldhan/AppData/Local/Temp/msohtmlclip1/01/clip_image007.jpg)

![](.gitbook/assets/image%20%2823%29.png)

when features are **conditionally independent** given the label

## Max joint likelihood

Joint distribution: p\(x,y\) = p\(y\)p\(x\|y\)

![](.gitbook/assets/image%20%2861%29.png)

During training of NB:

![](.gitbook/assets/image%20%288%29.png)

During testing of NB: ![](file:///C:/Users/ldhan/AppData/Local/Temp/msohtmlclip1/01/clip_image013.jpg)

![](.gitbook/assets/image%20%2843%29.png)

## Naïve bayes formula details:

## Class prior

**Choice of class prior depends on the type of classes**

![](.gitbook/assets/image%20%2873%29.png)

**Binary classification:**

Bernoulli distribution: ![](file:///C:/Users/ldhan/AppData/Local/Temp/msohtmlclip1/01/clip_image019.jpg)

![](.gitbook/assets/image%20%2818%29.png)

Max log-likelihood & get the max point: ![](file:///C:/Users/ldhan/AppData/Local/Temp/msohtmlclip1/01/clip_image021.jpg)

![](.gitbook/assets/image%20%2812%29.png)

**Multiclass classification:**

Categorical distribution: ![](file:///C:/Users/ldhan/AppData/Local/Temp/msohtmlclip1/01/clip_image023.jpg)

![](.gitbook/assets/image%20%2811%29.png)

Optimal parameters: ![](file:///C:/Users/ldhan/AppData/Local/Temp/msohtmlclip1/01/clip_image025.jpg)

![](.gitbook/assets/image%20%2845%29.png)

## Likelihood

**Choice of likelihood distribution depends on the type of features**

![](.gitbook/assets/image%20%2874%29.png)

·         Bernoulli:  binary features

·         Categorical: categorical features

·         Gaussian: continuous distribution

**each feature may use a diﬀerent likelihood**

**and separate max-likelihood estimate for each feature** ![](file:///C:/Users/ldhan/AppData/Local/Temp/msohtmlclip1/01/clip_image029.jpg)

![](.gitbook/assets/image%20%2856%29.png)

## Binary features: Bernoulli NB

MLE: ![](file:///C:/Users/ldhan/AppData/Local/Temp/msohtmlclip1/01/clip_image031.jpg)

![](.gitbook/assets/image%20%2824%29.png)

Implementation:

![](.gitbook/assets/image%20%2879%29.png)

## Multinomial features:

Multinomial likelihood: ![](file:///C:/Users/ldhan/AppData/Local/Temp/msohtmlclip1/01/clip_image035.jpg)

![](.gitbook/assets/image%20%2834%29.png)

MLE estimates: ![](file:///C:/Users/ldhan/AppData/Local/Temp/msohtmlclip1/01/clip_image037.jpg)

![](.gitbook/assets/image%20%2850%29.png)

## Gaussian features:

![](.gitbook/assets/image%20%284%29.png)

MLE![](file:///C:/Users/ldhan/AppData/Local/Temp/msohtmlclip1/01/clip_image041.jpg)

![](.gitbook/assets/image%20%2816%29.png)

## Decision Boundary

two classes have the same probability: ![](file:///C:/Users/ldhan/AppData/Local/Temp/msohtmlclip1/01/clip_image043.jpg)

![](.gitbook/assets/image%20%2847%29.png)

![](.gitbook/assets/image%20%2872%29.png)

### Discriminative vs Generative Classification

![](.gitbook/assets/image%20%2878%29.png)

