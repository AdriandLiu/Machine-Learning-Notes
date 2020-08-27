# Gradient Computation & Automatic Differentiation

## Landscape of the cost function

![](.gitbook/assets/290.png)

![](.gitbook/assets/291.png)

![](.gitbook/assets/292.png)

## Jacobian matrix

![](.gitbook/assets/293.png)

## Chain rule

![](.gitbook/assets/294.png)

## Training a two layer MLP

![](.gitbook/assets/295.png)

## Gradient calculation

![](.gitbook/assets/296.png)

### For regression

![](.gitbook/assets/297.png)

### For binary classification

![](.gitbook/assets/298.png)

### For multiclass classification

![](.gitbook/assets/299.png)

Softmax: ![](.gitbook/assets/300.png)

Code:

def softmax\( u, \# N x K \): u\_exp = np.exp\(u - np.max\(u, 1\)\[:, None\]\)

return u\_exp / np.sum\(u\_exp, axis=-1\)\[:, None\]

xs = np.array\(\[-1, 0, 3, 5\]\)

print\(softmax\(xs\)\) \# \[0.0021657, 0.00588697, 0.11824302, 0.87370431\]

Cross entropy: ![](.gitbook/assets/301.png)

def cross\_entropy\(p, q\):

 return -sum\(\[p\[i\]\*log2\(q\[i\]\) for i in range\(len\(p\)\)\]\)

![](.gitbook/assets/302.png)

### Example:

![](.gitbook/assets/303.png)

Codes:

![](.gitbook/assets/304.png)

![](.gitbook/assets/305.png)

![](.gitbook/assets/306.png)

![](.gitbook/assets/307.png)

![](.gitbook/assets/308.png)

![](.gitbook/assets/309.png)

## Automating gradient computation

![](.gitbook/assets/310.png)

## Automatic differentiation

![](.gitbook/assets/311.png)

## Backpropagation

## Forward mode

![](.gitbook/assets/312.png)

### Computational graph

![](.gitbook/assets/313.png)

## Reverse mode

![](.gitbook/assets/314.png)

### Computational graph

![](.gitbook/assets/315.png)

## Forward vs Reverse mode

![](.gitbook/assets/316.png)

## Summary

![](.gitbook/assets/317.png)



