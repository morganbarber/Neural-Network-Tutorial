# Creating a Neural Network from Scratch using Numpy

In this notebook, we will create a neural network from scratch using only Numpy. We will then use this neural network to perform a binary classification task on a simple dataset. We will begin by explaining the mathematics behind neural networks and then implement the neural network using Python and Numpy.

## The Mathematics of Neural Networks

A neural network consists of layers of interconnected neurons. Each neuron computes a weighted sum of its inputs, adds a bias term, and then applies an activation function to the result. Mathematically, the output of a neuron can be represented as:

$$
y = f(w^T x + b)
$$

where:

- $x$ is the input vector,
- $w$ is the weight vector,
- $b$ is the bias term, and
- $f$ is the activation function.

For a binary classification task, we can use the sigmoid activation function, defined as:

$$
f(z) = \frac{1}{1 + e^{-z}}
$$

## Neural Network Architecture

In this example, we will create a simple feedforward neural network with one input layer, one hidden layer, and one output layer. The input layer will have two neurons (for two input features), the hidden layer will have three neurons, and the output layer will have one neuron (for binary classification).

## Initializing Weights and Biases

We will initialize the weights and biases randomly using a normal distribution with mean 0 and standard deviation 1. We will use the following function to initialize the weights and biases:

```python
def initialize_parameters(input_size, hidden_size, output_size):
    W1 = np.random.randn(hidden_size, input_size)
    b1 = np.random.randn(hidden_size, 1)
    W2 = np.random.randn(output_size, hidden_size)
    b2 = np.random.randn(output_size, 1)

    return W1, b1, W2, b2
```

## Forward Propagation

Next, we will implement the forward propagation step, where we compute the outputs of each layer given the inputs.

```python
def forward_propagation(X, W1, b1, W2, b2):
    Z1 = np.dot(W1, X) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    return Z1, A1, Z2, A2
```

## Backward Propagation

After the forward propagation step, we need to compute the gradients of the weights and biases with respect to the loss function. We will use the binary cross-entropy loss function, defined as:

$$
L(y, \hat{y}) = -y \log(\hat{y}) - (1 - y) \log(1 - \hat{y})
$$

To compute the gradients, we will use the chain rule, which states that the gradient of a composed function is the product of the gradients of its constituent functions:

$$
\frac{\partial L}{\partial w} = \frac{\partial L}{\partial \hat{y}} \frac{\partial \hat{y}}{\partial z} \frac{\partial z}{\partial w}
$$

We will implement the backward propagation step as follows:

```python
def backward_propagation(X, Y, Z1, A1, Z2, A2, W1, b1, W2, b2):
    m = X.shape[1]

    dZ2 = A2 - Y
    dW2 = (1/m) * np.dot(dZ2, A1.T)
    db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.dot(W2.T, dZ2) * sigmoid_derivative(Z1)
    dW1 = (1/m) * np.dot(dZ1, X.T)
    db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)

    return dW1, db1, dW2, db2
```

## Updating Weights and Biases

Now that we have the gradients, we can update the weights and biases using gradient descent:

$$
w = w - \alpha \frac{\partial L}{\partial w}
$$

where $\alpha$ is the learning rate.

We will implement this step as follows:

```python
def update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate):
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    return W1, b1, W2, b2
```

## Training the Neural Network

Finally, we can train the neural network using the following function:

```python
def train_nn(X, Y, epochs, learning_rate):
    input_size = X.shape[0]
    hidden_size = 3
    output_size = 1

    W1, b1, W2, b2 = initialize_parameters(input_size, hidden_size, output_size)

    for epoch in range(epochs):
        Z1, A1, Z2, A2 = forward_propagation(X, W1, b1, W2, b2)
        dW1, db1, dW2, db2 = backward_propagation(X, Y, Z1, A1, Z2, A2, W1, b1, W2, b2)
        W1, b1, W2, b2 = update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate)

    return W1, b1, W2, b2
```

We can now train the neural network on our dataset and use the trained weights and biases to make predictions.
