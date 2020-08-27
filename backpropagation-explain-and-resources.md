# Backpropagation Explain and Resources

Suppose two hidden layers MLP:

![](.gitbook/assets/318.png)

**Backpropagation aims to find which weight/bias/activation function has the relatively larger influence on minimize the cost and update them correspondingly.**

For example:

![](.gitbook/assets/319.png)

This is the cost – weight/bias of hidden layer 2, to calculate the ratio between the weights \(and biases\) and the cost function. The ones with the largest ratio will have the greatest impact on the cost function and will give us 'the most bang for our buck'.

Ex. We want to increase the prob to classify it into 2, we would like to know which neuron’s weight/bias/activation has larger influence so that we can adjust them to get our desired output - 2, such as the yellow line.

![](.gitbook/assets/320.png)

![](.gitbook/assets/321.png)![](.gitbook/assets/322.png)![](.gitbook/assets/323.png)

We will have to average the changes

![](.gitbook/assets/324.png)![](.gitbook/assets/325.png)

Math formula \(output – the last hidden layer\):

![](.gitbook/assets/326.png)

![](.gitbook/assets/327.png)

![](.gitbook/assets/328.png)

With multiple layers and neurons:

![](.gitbook/assets/329.png)

Calculating the gradient

![](.gitbook/assets/330.png)

Formula for the hidden layer 1:

![](.gitbook/assets/331.png)

Extra hidden layer:

![](.gitbook/assets/332.png)

**Summarization**

![](.gitbook/assets/333.png)

Resources:

[https://mlfromscratch.com/neural-networks-explained/\#backpropagation](https://mlfromscratch.com/neural-networks-explained/#backpropagation)

[https://www.youtube.com/watch?v=Ilg3gGewQ5U](https://www.youtube.com/watch?v=Ilg3gGewQ5U)

