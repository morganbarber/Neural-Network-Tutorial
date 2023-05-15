# Creating a Neural Network from Scratch in Python

In this tutorial, we will create a simple neural network from scratch using Python and NumPy. We will cover the basic concepts of neural networks and the required mathematical background.

## Introduction to Neural Networks

A neural network is a computational model inspired by the human brain. It consists of interconnected nodes or neurons that process/input data and produce/output results. Neural networks can learn from data and are particularly good at recognizing patterns, making them ideal for various applications like image recognition, natural language processing, and game playing.

## Mathematical Background

A neural network typically consists of layers: an input layer, one or more hidden layers, and an output layer. Each layer contains nodes or neurons, which are connected to the neurons of the previous and next layers through weights. The input for each neuron is a weighted sum of the outputs of the neurons in the previous layer. The neuron applies an activation function to this weighted sum and produces an output.

The most critical components of a neural network are:

1. **Weights**: The strength of the connections between neurons.
2. **Activation function**: A non-linear function applied to the weighted sum of the inputs.
3. **Loss function**: A measure of the difference between the predicted output and the actual output.
4. **Backpropagation**: An algorithm to minimize the loss function by adjusting the weights.

Let's start by defining the activation function and its derivative.

### Activation Function

The activation function introduces non-linearity in the neural network, allowing it to learn complex patterns. A popular choice for the activation function is the sigmoid function:

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

The sigmoid function squashes the input into the range (0, 1), which can be interpreted as a probability.

### Derivative of the Activation Function

During backpropagation, we need the derivative of the activation function. The derivative of the sigmoid function is:

$$
\sigma'(x) = \sigma(x) * (1 - \sigma(x))
$$

## Building the Neural Network

We will create a simple neural network with one hidden layer.

### Imports

First, let's import the necessary libraries:

```python
import numpy as np
```

### Activation Function and Its Derivative

Now, let's define the sigmoid activation function and its derivative:

```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))
```

### Initializing the Network

We initialize weights randomly for each layer, with a bias term to shift the activation function:

```python
def initialize_network(input_nodes, hidden_nodes, output_nodes):
    hidden_layer = {'weights': np.random.randn(input_nodes, hidden_nodes), 'bias': np.random.randn(hidden_nodes)}
    output_layer = {'weights': np.random.randn(hidden_nodes, output_nodes), 'bias': np.random.randn(output_nodes)}
    return (hidden_layer, output_layer)
```

### Forward Propagation

Forward propagation calculates the output of the neural network given the input:

```python
def forward_propagation(input_data, hidden_layer, output_layer):
    hidden_layer_input = np.dot(input_data, hidden_layer['weights']) + hidden_layer['bias']
    hidden_layer_output = sigmoid(hidden_layer_input)
    
    output_layer_input = np.dot(hidden_layer_output, output_layer['weights']) + output_layer['bias']
    output_layer_output = sigmoid(output_layer_input)
    
    return hidden_layer_output, output_layer_output
```

### Backward Propagation

Backward propagation optimizes the weights by minimizing the loss function using gradient descent:

```python
def backward_propagation(input_data, hidden_layer, output_layer, hidden_layer_output, output_layer_output, actual_output, learning_rate):
    output_error = actual_output - output_layer_output
    output_gradient = output_error * sigmoid_derivative(output_layer_output)
    
    hidden_error = np.dot(output_gradient, output_layer['weights'].T)
    hidden_gradient = hidden_error * sigmoid_derivative(hidden_layer_output)
    
    output_layer['weights'] += learning_rate * np.dot(hidden_layer_output.T, output_gradient)
    output_layer['bias'] += learning_rate * np.sum(output_gradient, axis=0)
    
    hidden_layer['weights'] += learning_rate * np.dot(input_data.T, hidden_gradient)
    hidden_layer['bias'] += learning_rate * np.sum(hidden_gradient, axis=0)
```

### Training the Network

Now, let's create a function to train the network using forward and backward propagation:

```python
def train_network(input_data, actual_output, input_nodes, hidden_nodes, output_nodes, learning_rate, epochs):
    hidden_layer, output_layer = initialize_network(input_nodes, hidden_nodes, output_nodes)
    
    for _ in range(epochs):
        hidden_layer_output, output_layer_output = forward_propagation(input_data, hidden_layer, output_layer)
        backward_propagation(input_data, hidden_layer, output_layer, hidden_layer_output, output_layer_output, actual_output, learning_rate)
    
    return hidden_layer, output_layer
```

### Testing the Network

Once the network is trained, we can use it to make predictions:

```python
def predict(input_data, hidden_layer, output_layer):
    _, output_layer_output = forward_propagation(input_data, hidden_layer, output_layer)
    return output_layer_output
```

## Conclusion

This tutorial showed you how to create a simple neural network from scratch using Python and NumPy. You learned about the mathematical concepts behind neural networks, such as the activation function, its derivative, and the backpropagation algorithm. You can now experiment with different architectures, activation functions, and optimization techniques to improve the performance of your neural network.
