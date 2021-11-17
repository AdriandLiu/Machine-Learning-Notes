# Linear

## Representing Data

![](<.gitbook/assets/image (13).png>)

![](<.gitbook/assets/image (14).png>)

we assume N instances in the dataset each instance has D features indexed by d

## Linear Model

![](<.gitbook/assets/image (16).png>)

![](<.gitbook/assets/image (15).png>)

## Loss Function

Objective:  find parameters to fit the data

![](<.gitbook/assets/image (17).png>)

Residual: ![](file:///C:/Users/ldhan/AppData/Local/Temp/msohtmlclip1/01/clip\_image012.jpg)

![](<.gitbook/assets/image (25).png>)

Linear least square (L2 loss) cost function:

![](<.gitbook/assets/image (24).png>)

L1 vs L2 Loss Function

L1 and L2 are two loss functions in machine learning which are used to minimize the error.

L1 Loss function stands for **Least Absolute Deviations**. Also known as LAD.

![](<.gitbook/assets/image (23).png>)

L2 Loss function stands for **Least Square Errors**. Also known as LS.

![](<.gitbook/assets/image (22).png>)

**AIMS TO MIN LOSS FUNTION:**

![](<.gitbook/assets/image (26).png>)

Derivative:

![](<.gitbook/assets/image (27).png>)

![](<.gitbook/assets/image (28).png>)

### Direct Solution

![](<.gitbook/assets/image (29).png>)

![](<.gitbook/assets/image (30).png>)

Linear for **large dataset:**

Stochastic gradient descent

what if X^T X is **not invertible**? (columns of X are not linearly independent, either redundant features or numFeature > numInstance; D> N)

![](<.gitbook/assets/image (31).png>)
