# Linear

## Representing Data

![](.gitbook/assets/image%20%2839%29.png)

![](.gitbook/assets/image%20%2851%29.png)

we assume N instances in the dataset each instance has D features indexed by d

## Linear Model

![](.gitbook/assets/image%20%2867%29.png)

![](.gitbook/assets/image%20%2820%29.png)

## Loss Function

Objective:  find parameters to fit the data

![](.gitbook/assets/image%20%2863%29.png)

Residual: ![](file:///C:/Users/ldhan/AppData/Local/Temp/msohtmlclip1/01/clip_image012.jpg)

![](.gitbook/assets/image%20%282%29.png)

Linear least square \(L2 loss\) cost function:

![](.gitbook/assets/image%20%2835%29.png)

L1 vs L2 Loss Function

L1 and L2 are two loss functions in machine learning which are used to minimize the error.

L1 Loss function stands for **Least Absolute Deviations**. Also known as LAD.

![](.gitbook/assets/image%20%2858%29.png)

L2 Loss function stands for **Least Square Errors**. Also known as LS.

![](.gitbook/assets/image%20%2869%29.png)

**AIMS TO MIN LOSS FUNTION:**

![](.gitbook/assets/image%20%2881%29.png)

Derivative:

![](.gitbook/assets/image%20%2865%29.png)

![](.gitbook/assets/image%20%2833%29.png)

### Direct Solution

![](.gitbook/assets/image%20%2837%29.png)

![](.gitbook/assets/image%20%2852%29.png)

Linear for **large dataset:**

Stochastic gradient descent

what if X^T X is **not invertible**? \(columns of X are not linearly independent, either redundant features or numFeature &gt; numInstance; D&gt; N\)

![](.gitbook/assets/image%20%289%29.png)

