# Naive Bayes

\
**Discriminative**: conditional distribution p(y|x) eg: linear and logistics

**Generative**: joint distribution p(x,y) = p(y)p(x|y) eg:

&#x20;Naïve bayes ![](file:///C:/Users/ldhan/AppData/Local/Temp/msohtmlclip1/01/clip\_image003.jpg)

![](<.gitbook/assets/image (56).png>)

Example:

![](<.gitbook/assets/image (57).png>)

**in a generative classiﬁer likelihood & prior class probabilities are learned from data**

Some generative classiﬁers:

·         Gaussian Discriminant Analysis: the likelihood is multivariate Gaussian

·         Naive Bayes: decomposed likelihood

## Naïve bayes model

NB **assumption**: ![](file:///C:/Users/ldhan/AppData/Local/Temp/msohtmlclip1/01/clip\_image007.jpg)

![](<.gitbook/assets/image (58).png>)

when features are **conditionally independent** given the label

## Max joint likelihood

Joint distribution: p(x,y) = p(y)p(x|y)

![](<.gitbook/assets/image (60).png>)

During training of NB:

![](<.gitbook/assets/image (61).png>)

During testing of NB: ![](file:///C:/Users/ldhan/AppData/Local/Temp/msohtmlclip1/01/clip\_image013.jpg)

![](<.gitbook/assets/image (62).png>)

## Naïve bayes formula details:

## Class prior

**Choice of class prior depends on the type of classes**

![](<.gitbook/assets/image (63).png>)

**Binary classification:**

Bernoulli distribution: ![](file:///C:/Users/ldhan/AppData/Local/Temp/msohtmlclip1/01/clip\_image019.jpg)

![](<.gitbook/assets/image (64).png>)

Max log-likelihood & get the max point: ![](file:///C:/Users/ldhan/AppData/Local/Temp/msohtmlclip1/01/clip\_image021.jpg)

![](<.gitbook/assets/image (65).png>)

**Multiclass classification:**

Categorical distribution: ![](file:///C:/Users/ldhan/AppData/Local/Temp/msohtmlclip1/01/clip\_image023.jpg)

![](<.gitbook/assets/image (66).png>)

Optimal parameters: ![](file:///C:/Users/ldhan/AppData/Local/Temp/msohtmlclip1/01/clip\_image025.jpg)

![](<.gitbook/assets/image (67).png>)

## Likelihood

**Choice of likelihood distribution depends on the type of features**

![](<.gitbook/assets/image (68).png>)

·         Bernoulli:  binary features

·         Categorical: categorical features

·         Gaussian: continuous distribution

**each feature may use a diﬀerent likelihood**

**and separate max-likelihood estimate for each feature** ![](file:///C:/Users/ldhan/AppData/Local/Temp/msohtmlclip1/01/clip\_image029.jpg)

![](<.gitbook/assets/image (69).png>)

## Binary features: Bernoulli NB

MLE: ![](file:///C:/Users/ldhan/AppData/Local/Temp/msohtmlclip1/01/clip\_image031.jpg)

![](<.gitbook/assets/image (70).png>)

Implementation:

![](<.gitbook/assets/image (71).png>)

## Multinomial features:

Multinomial likelihood: ![](file:///C:/Users/ldhan/AppData/Local/Temp/msohtmlclip1/01/clip\_image035.jpg)

![](<.gitbook/assets/image (72).png>)

MLE estimates: ![](file:///C:/Users/ldhan/AppData/Local/Temp/msohtmlclip1/01/clip\_image037.jpg)

![](<.gitbook/assets/image (73).png>)

## Gaussian features:

![](<.gitbook/assets/image (74).png>)

MLE![](file:///C:/Users/ldhan/AppData/Local/Temp/msohtmlclip1/01/clip\_image041.jpg)

![](<.gitbook/assets/image (75).png>)

## Decision Boundary

two classes have the same probability: ![](file:///C:/Users/ldhan/AppData/Local/Temp/msohtmlclip1/01/clip\_image043.jpg)

![](<.gitbook/assets/image (76).png>)

![](<.gitbook/assets/image (77).png>)

### Discriminative vs Generative Classification

![](<.gitbook/assets/image (78).png>)
