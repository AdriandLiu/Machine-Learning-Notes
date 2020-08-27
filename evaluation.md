# Evaluation

## Evaluation and comparison

Each model with same cost functions: report loss

Eg. Least squares or cross entropy

Each model with different cost functions:

Standard evaluation measures/metrics

## Performance metrics for classification

**False positive \(type 1\)**

**False negative \(type 2\)**

Eg: patient does not have disease but received positive diagnostic \(Type I error\)

patient has disease but it was not detected \(Type II error\)  

a message that is not spam is assigned to the spam folder \(Type I error\)

a message that is spam appears in the regular folder \(Type II error\)

confusion matrix

![](.gitbook/assets/134.png)

![](.gitbook/assets/135.png)

Accuracy, error rate, precision, recall, F1 score:

![](.gitbook/assets/136.png)

Fbeta score

![](.gitbook/assets/137.png)

### Example precision recall

![](.gitbook/assets/138.png)

### Less common metrics:

![](.gitbook/assets/139.png)

## Performance metrics for multi-class classiﬁcation

Report average metrics per class, eg. Average precision

![](.gitbook/assets/140.png)

## Trade-oﬀ between precision and recall ROC&AUC

![](.gitbook/assets/141.png)

**To compare classiﬁcation algorithms compare their Area Under the Curve \(AUC\)**

Note that higher AUC doesn’t mean all performance measures are better

![](.gitbook/assets/142.png)

**Also important when comparing ranking algorithms e.g. search results**

**Intuition**: AUC is equivalent to the probability of ranking a random positive example higher than a random negative example

## Cross validation in Evaluation

![](.gitbook/assets/143.png)

Over-ﬁtting in Model Selection

more severe on small dataset and when having too many hyper-parameters but present even with few hyperparameters \(小数据更易，很多hyper结果跟很少hyper相似\)

### Nested CV

![](.gitbook/assets/144.png)



