import numpy as np

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of sigmoid (for backpropagation)
def sigmoid_derivative(x):
    return x * (1 - x)

# Training data (XOR problem)
# Input: 4 samples with 2 features each
inputs = np.array([[0, 0], 
                   [0, 1], 
                   [1, 0], 
                   [1, 1]])

# Output: The expected output for each input (XOR)
outputs = np.array([[0], 
                    [1], 
                    [1], 
                    [0]])

# Set seed for reproducibility
np.random.seed(1)

# Initialize weights randomly with mean 0
input_layer_neurons = 2    # Number of neurons in input layer (2 features)
hidden_layer_neurons = 2   # Number of neurons in hidden layer
output_neuron = 1          # Number of neurons in output layer (1 output)

# Weights for layers
weights_input_hidden = np.random.uniform(size=(input_layer_neurons, hidden_layer_neurons))
weights_hidden_output = np.random.uniform(size=(hidden_layer_neurons, output_neuron))

# Biases
bias_hidden = np.random.uniform(size=(1, hidden_layer_neurons))
bias_output = np.random.uniform(size=(1, output_neuron))

# Learning rate
learning_rate = 0.1

# Number of training iterations
epochs = 10000

# Training loop
for epoch in range(epochs):
    # Feedforward
    hidden_layer_activation = np.dot(inputs, weights_input_hidden) + bias_hidden
    hidden_layer_output = sigmoid(hidden_layer_activation)

    output_layer_activation = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
    predicted_output = sigmoid(output_layer_activation)

    # Backpropagation
    # Calculate the error (difference between predicted and actual)
    error = outputs - predicted_output
    d_predicted_output = error * sigmoid_derivative(predicted_output)

    # Error of the hidden layer
    error_hidden_layer = d_predicted_output.dot(weights_hidden_output.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)

    # Update weights and biases
    weights_hidden_output += hidden_layer_output.T.dot(d_predicted_output) * learning_rate
    weights_input_hidden += inputs.T.dot(d_hidden_layer) * learning_rate
    bias_output += np.sum(d_predicted_output, axis=0, keepdims=True) * learning_rate
    bias_hidden += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate

# After training is done, print the predicted output
print("Final predicted output after training:")
print(predicted_output)
