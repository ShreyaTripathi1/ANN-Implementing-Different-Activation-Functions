# ANN

Artificial Neural Networks (ANNs) are computational models inspired by biological neural networks. They consist of layers of artificial neurons (nodes) connected by weights, where information flows through these layers, getting processed and transformed by activation functions.

</br>

## ANNs have the following key components:

  **Input Layer:** Takes input features.
  
  **Hidden Layers:** Intermediate layers that perform computations.
  
  **Output Layer:** Produces the final prediction or classification.
  
  **Weights and Biases:** Parameters adjusted during training.
  
  **Activation Functions:** Non-linear functions (e.g., ReLU, Sigmoid) that introduce non-linearity.

During training, ANNs use an algorithm called ```backpropagation``` to minimize the loss function by adjusting weights using an optimization method like ```gradient descent```.

</br>

## Common types of ANNs:

*Feedforward Neural Network (FNN):* Data flows in one direction from input to output.

*Recurrent Neural Network (RNN):* Designed for sequential data, having loops that allow information to persist.

*Convolutional Neural Network (CNN):* Specially designed for image data, using convolutional layers to extract features.

</br>
  
## Algorithm (for a simple Feedforward Neural Network):

- Step 1: Initialize the network by specifying the number of layers, neurons, and activation functions.
  
- Step 2: Feed the input data through the input layer to the hidden layers, calculating the weighted sum at each node: </br>

        z=∑(wi​⋅xi​)+b
    
    Apply the activation function:  </br>
    
        a=f(z)

- Step 3: Pass the output of the hidden layers to the output layer to generate the prediction.
  
- Step 4: Calculate the error (loss) using a loss function (e.g., Mean Squared Error for regression, Cross-Entropy Loss for classification).
  
- Step 5: Use backpropagation to compute the gradients and update the weights using an optimization algorithm (e.g., gradient descent).
  
- Step 6: Repeat Steps 2-5 for several epochs until the loss is minimized.




  
