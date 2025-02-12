import numpy as np
from part1_nn_lib import *

if __name__ == "__main__":

    #############################################################
    #                    Testing LinearLayer                    #
    #############################################################

    # Define test parameters
    batch_size = 10  # Number of samples in the batch
    n_in = 3  # Number of input features
    n_out = 42  # Number of output neurons
    learning_rate = 0.01  # Learning rate for parameter updates

    # Initialise layer
    layer = LinearLayer(n_in, n_out)

    # Forward pass
    inputs = np.random.randn(batch_size, n_in)
    outputs = layer.forward(inputs)

    # Simulate gradient from loss
    grad_loss_wrt_outputs = np.random.randn(batch_size, n_out)

    # Backward pass
    grad_loss_wrt_inputs = layer.backward(grad_loss_wrt_outputs)

    print("Gradient w.r.t Inputs Shape:", grad_loss_wrt_inputs.shape)  # Expected: (10, 3)
    print("Gradient w.r.t Weights Shape:", layer._grad_W_current.shape)  # Expected: (3, 42)
    print("Gradient w.r.t Bias Shape:", layer._grad_b_current.shape)  # Expected: (1, 42)






