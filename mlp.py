import numpy as np


class Mlp:
    """
    A simple Mlp

    Attributes
    ----------
    layer_layout : list
        a list containing the number of neurons per layer e.g (4, 15, 20, 3).
        first number (4 in the example) is the number of inputs
        last number (3 in the example) is the output layer
        the rest are the hidden layers (15 and 20 in the example)
    weights : list
        a list with all the weights for each layer/node
        each element of said list represents a layer of the mlp as a numpy array.
        The numpy array is 2d and has the weights of each node as a row vector
    Methods
    -------
    weight_initialization()
        initializes the weights for each node in each layer
    """

    def __init__(self, layer_layout):

        # np.random.seed(42) # for debugging purposes

        self.layer_layout = layer_layout
        self.weights = self.weight_initialization()

    def weight_initialization(self):
        """
        Initializes the weights for each node in the model between -0.5 and 0.5

        Returns
        -------
        list
            a list with all the weights for each layer/node
            each element of said list represents a layer of the mlp as a numpy array.
            the numpy array is 2d and has the weights of each node as a row vector
        """
        weights = []
        # starts from index 1 as the 0th element represents the input layer which has no weights
        for idx, neurons_per_layer in enumerate(self.layer_layout[1:], start=1):
            neuron_weights = []
            for _ in range(neurons_per_layer):
                neuron_weights.append(np.random.rand(self.layer_layout[idx - 1]) - 0.5)
            # numpy arrays to make dot products between layers easier
            weights.append(np.array(neuron_weights))
        return weights