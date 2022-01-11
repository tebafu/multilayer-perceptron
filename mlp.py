import numpy as np
import pandas as pd


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

    def __init__(self, layer_layout, activation_function='relu'):

        # np.random.seed(42) # for debugging purposes

        self.layer_layout = layer_layout

        if activation_function == 'sigmoid':
            self.activation_function = self.sigmoid
            self.activation_function_d = self.sigmoid_derivative
        else:
            self.activation_function = self.relu
            self.activation_function_d = self.relu_derivative

        self.weights = self.weight_initialization()

    def weight_initialization(self):
        """
        Initializes the weights for each node in the model between -0.5 and 0.5

        :return: list: a list with all the weights for each layer/node each element of said list represents a layer of
        the mlp as a numpy array. the numpy array is 2d and has the weights of each node as a row vector
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

    # noinspection PyMethodMayBeStatic
    def sigmoid(self, x):
        """
        basic sigmoid activation function
        :param x: input
        :return: sigmoid output
        """
        s = 1.0 / (1.0 + np.exp(-x))
        return s

    def sigmoid_derivative(self, x):
        """
        basic sigmoid derivative
        :param x: input
        :return: derivative of sigmoid
        """
        s = self.sigmoid(x) * (1 - self.sigmoid(x))
        return s

    # noinspection PyMethodMayBeStatic
    def relu(self, x):
        return max(0, x)

    # noinspection PyMethodMayBeStatic
    def relu_derivative(self, x):
        return (x > 0) * 1

    def forward_pass(self, x):
        """
        does a forward pass of the input through the neural network
        :param x: the input
        :return: v the outputs of each node without the activation function applied to it
        :return: y final outputs of each node
        """
        current_input = np.array(x)
        v = []
        y = []
        for layer in self.weights:
            v.append(np.dot(layer, current_input))
            y.append(self.sigmoid(v[-1]))
            current_input = y[-1]
        return v, y

    def predict(self, x):
        """
        does a forward pass of the input through the neural network but doesn't keep the outputs of each node thus
        is faster than forward_pass()
        :param x: the input
        :return: y final outputs of last layer's nodes
        """
        current_input = np.array(x)
        for layer in self.weights:
            current_input = self.sigmoid(np.dot(layer, current_input))
        return current_input

    def set_weights(self, weights):
        self.weights = weights.copy()

    def structure_visualization(self):
        """
        A method to visualize the mlp, it uses pandas since it is prettier in the console of pycharm, it has not been
        tested on any notebook or other IDE

        Returns
        -------
        nothing
        """
        print('MLP LAYOUT\n')
        length = len(self.layer_layout)
        depth = max(self.layer_layout) * 2 - 1
        center = depth // 2

        visual = []
        for i in range(depth):
            temp = [" " for _ in range(length)]
            visual.append(temp)

        for idx, layer in enumerate(self.layer_layout):
            if layer % 2 == 0:
                for i in range(layer // 2):
                    visual[center - 2 * (i + 1) + 1][idx] = "0"
                    visual[center + 2 * (i + 1) - 1][idx] = "0"
            else:
                visual[center][idx] = "0"
                for i in range(layer // 2):
                    visual[center - 2 * (i + 1)][idx] = "0"
                    visual[center + 2 * (i + 1)][idx] = "0"

        node_num = 1
        input_num = 1
        for idx, layer in enumerate(self.layer_layout):
            if idx == 0:
                for i in range(depth):
                    if visual[i][idx] == '0':
                        visual[i][idx] = f'input {input_num}'
                        input_num += 1
            if idx != 0:
                for i in range(depth):
                    if visual[i][idx] == '0':
                        visual[i][idx] = f'node {node_num}'
                        node_num += 1

        df = pd.DataFrame(visual)
        print(df.to_string(index=False, header=[f'------Layer-{x + 1}' for x in range(length)]))

    def weight_visualization(self):
        """
        Prints out the weights of the neural network in the following format
        Layer 1
            node 1
            .
            .
            node n
        .
        .
        Layer m
            .
            .
            node k
        Returns
        -------
        nothing
        """
        print('WEIGHT VISUALIZATION\n')
        node_num = 1
        for idx, layer in enumerate(self.weights, start=1):
            print(f'for layer: {idx}')
            for node in layer:
                print(f'    for node {node_num} the weights are: ')
                print(f'        {node}\n\n')
                node_num += 1
