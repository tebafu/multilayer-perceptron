import copy
import random

import numpy as np
import pandas as pd


class Mlp:
    """
    A multilayer perceptron

    Attributes
    ----------
    layer_layout : list
        a list containing the number of neurons per layer e.g (4, 15, 20, 3).
        first number (4 in the example) is the number of inputs
        last number (3) is the output layer
        the rest are the hidden layers (15 and 20)
    weights : list
        a list with all the weights for each layer/node
        each element of said list represents a layer of the mlp as a numpy array.
        The numpy array is 2d and has the weights of each node as a row vector
    learning_rate : float
        the learning rate of the mlp that affects how much the gradients returned affect the weights
    activation_function : pointer to method
        points to the function applied to each node to determine its output
    epochs
        how many iterations the model with do while training

    Methods
    -------
    predict(x)
        produces the prediction of the model according to the input x
    fit(X, Y)
        trains the model on the data according to the labels Y
    """

    def __init__(self, layer_layout, learning_rate=0.01, activation_function='relu', epochs=100, bias=False,
                 weight_initialization='default'):
        """
        Constructor method.

        Parameters
        ----------
        layer_layout : list or tuple
            defines the size of the mlp with the first element being the input layer and the last the output.
            ex. [4, 450, 200, 1] 4 input nodes, 2 hidden layers with 450 and 200 nodes eah and one output node
        learning_rate : float
            the learning rate of the mlp that affects how much the gradients returned affect the weights
        activation_function : string
            the function applied to each node to determine it's output
        epochs: int
            how many iterations the model with do while training
        weight_initialization: string
            the type of initialization for the weights of the model
        """
        # np.random.seed(42) # for debugging purposes

        self.bias = bias
        self.layer_layout = layer_layout
        self.learning_rate = learning_rate
        self.epochs = epochs
        if activation_function == 'sigmoid':
            self.activation_function = self.sigmoid
            self.activation_function_d = self.sigmoid_derivative
        else:
            self.activation_function = self.relu
            self.activation_function_d = self.relu_derivative

        if weight_initialization == 'default':
            if activation_function == 'relu':
                self.weights = self.weight_initialization_he()
            elif activation_function == 'sigmoid':
                self.weights = self.weight_initialization_xavier()
            else:
                self.weights = self.weight_initialization()
        elif weight_initialization == 'he':
            self.weights = self.weight_initialization_he()
        elif weight_initialization == 'xavier':
            self.weights = self.weight_initialization_xavier()
        else:
            self.weights = self.weight_initialization()

    def weight_initialization(self):
        """
        Basic weight initialization with random values from range [-0.5, 0.5) for each node's weight

        Returns
        -------
        List
            the list contains 2d numpy arrays (as many as layers in the model -1), each numpy array contains a row
            vector with each node's weights. Each weight is within -0.5 and 0.5
        """
        weights = []
        # starts from index 1 as the 0th element represents the input layer which has no weights
        for idx, neurons_per_layer in enumerate(self.layer_layout[1:], start=1):
            neuron_weights = []
            for _ in range(neurons_per_layer):
                neuron_weights.append(np.random.rand(self.layer_layout[idx - 1] + self.bias) - 0.5)
            # numpy arrays to make dot products between layers easier
            weights.append(np.array(neuron_weights))
        return weights

    def weight_initialization_xavier(self):
        """
        Weight initialization according to Xavier method

        Returns
        -------
        List
            the list contains 2d numpy arrays (as many as layers in the model -1), each numpy array contains a row
            vector with each node's weights, each weight is between -1/sqrt(incoming_nodes) and 1/sqrt(incoming_nodes)
        """
        weights = []
        for idx, neurons_per_layer in enumerate(self.layer_layout[1:], start=1):
            neuron_weights = []
            for neuron in range(neurons_per_layer):
                incoming_nodes = self.layer_layout[idx - 1] + self.bias
                r = 1 / np.sqrt(incoming_nodes)
                neuron_weights.append(np.random.uniform(-r, r, incoming_nodes))
            weights.append(np.array(neuron_weights))
        return weights

    def weight_initialization_he(self):
        """
        Weight initialization according to Kaiming He method

        Returns
        -------
        List
            the list contains 2d numpy arrays (as many as layers in the model -1), each numpy array contains a row
            vector with each node's weights, each weight is between -sqrt(6/incoming_nodes) and sqrt(6/incoming_nodes)
        """
        weights = []
        for idx, neurons_per_layer in enumerate(self.layer_layout[1:], start=1):
            neuron_weights = []
            for neuron in range(neurons_per_layer):
                incoming_nodes = self.layer_layout[idx - 1] + self.bias
                r = np.sqrt(6 / incoming_nodes)
                neuron_weights.append(np.random.uniform(-r, r, incoming_nodes))
            weights.append(np.array(neuron_weights))
        return weights

    # noinspection PyMethodMayBeStatic
    def sigmoid(self, x):
        """
        Sigmoid function : 1/(1+e^-x)

        Parameters
        ----------
        x : float or nd.array
            input

        Returns
        -------
        float or nd.array
            the result of the function applied to the input
        """
        s = 1.0 / (1.0 + np.exp(-x))
        return s

    def sigmoid_derivative(self, x):
        """
        Sigmoid derivative function : sigmoid(x) * (1 - sigmoid(x))

        Parameters
        ----------
        x : float or nd.array
            input

        Returns
        -------
        float or nd.array
            the result of the function applied to the input
        """
        s = self.sigmoid(x) * (1 - self.sigmoid(x))
        return s

    # noinspection PyMethodMayBeStatic
    def relu(self, x):
        """
        relu function : max(0, x)

        Parameters
        ----------
        x : float or nd.array
            input

        Returns
        -------
        float or nd.array
            the result of the function applied to the input
        """
        return np.maximum(0, x)

    # noinspection PyMethodMayBeStatic
    def relu_derivative(self, x):
        """
        relu function : 1 if x > 0 else 0

        Parameters
        ----------
        x : float or nd.array
            input

        Returns
        -------
        float or nd.array
            the result of the function applied to the input
        """
        return (x > 0) * 1

    def forward_pass(self, x):
        """
        Does a forward pass of the input through the neural network

        Parameters
        ----------
        x : list
            contains the parameters for each input
        Returns
        -------
        u : list
            the activation signal of each node
        y : list
            the outputs of each node
        """
        current_input = np.array(x)
        if self.bias:
            current_input = np.append(current_input, 1)
        u = []
        y = []
        for layer in self.weights:
            u.append(np.dot(layer, current_input))
            y.append(self.activation_function(u[-1]))
            current_input = y[-1]
            if self.bias:
                current_input = np.append(current_input, 1)
        return u, y

    def predict(self, x):
        """
        Does a forward pass of the input through the neural network but doesn't keep the activation signals or outputs
        of each node thus is faster than forward_pass()

        Parameters
        ----------
        x : list
            contains the parameters for each input

        Returns
        -------
        list
            the final output of the model
        """
        current_input = np.array(x)
        if self.bias:
            current_input = np.append(current_input, 1)
        for layer in self.weights:
            current_input = self.activation_function(np.dot(layer, current_input))
            if self.bias:
                current_input = np.append(current_input, 1)
        return current_input

    def back_propagation(self, x, target):
        """
        Has 3 phases, forward pass, backward phase and update weights.
            1) calls forward_pass() to get the activation signals and outputs of each node
            2) calculates the loss of the model (output layer delta) and then passes the loss backwards to split the
                delta between the nodes on the hidden layer
            3) after finding the deltas it updates the weights all the weights according to the deltas and the input
                of each node
        Parameters
        ----------
        x : list
            input of the model
        target : list
            the label corresponding to the input
        """
        x = np.array(x)
        target = np.array(target)

        # Forward phase
        v, y = self.forward_pass(x)
        # Backward phase
        layer_delta = (target - y[-1]) * self.activation_function_d(v[-1])
        deltas = [layer_delta]
        for idx, layer in enumerate(reversed(self.weights[1:]), start=2):
            if self.bias:
                layer = np.delete(layer, -1, 1)
            layer_delta = np.dot(layer.T, layer_delta) * self.activation_function_d(v[-idx])
            deltas.append(layer_delta)
        deltas = list(reversed(deltas))
        # Update phase
        # noinspection PyTypeChecker
        y.insert(0, x)
        if self.bias:
            for idx, layer in enumerate(y):
                y[idx] = np.append(layer, 1)
        new_weights = []
        for prev_y, delta_layer, weight_layer in zip(y, deltas, self.weights):
            node_weights = []
            for delta, weights in zip(delta_layer, weight_layer):
                node_weights.append(weights + self.learning_rate * delta * prev_y)
            new_weights.append(np.array(node_weights))
        self.weights = new_weights

    def fit(self, X, Y):
        """
        Trains the model according to the data X and the labels Y
            does as many iterations over the data as the epochs specified, also shuffles the data each time

        Parameters
        ----------
        X : list, tuple or nd.array

        Y : list, tuple or nd.array

        """
        for epoch in range(self.epochs):
            shuffled = list(zip(X, Y))
            random.shuffle(shuffled)
            for x, y in shuffled:
                self.back_propagation(x, y)

    def set_weights(self, weights):
        """
        Sets the specified weights as the models weights. Doesn't just use the given list but instead it does a
        deepcopy

        Parameters
        ----------
        weights : list of nd.arrays
            list with as many nd.arrays as layers -1, each having as row vectors the weights of each node
        """
        self.weights = copy.deepcopy(weights)

    def structure_visualization(self):
        """
        A method to visualize the mlp, it uses pandas since it is prettier in the console of pycharm, it has not been
        tested on any notebook or other IDE

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

        """
        print('WEIGHT VISUALIZATION\n')
        node_num = 1
        for idx, layer in enumerate(self.weights, start=1):
            print(f'for layer: {idx}')
            for node in layer:
                print(f'    for node {node_num} the weights are: ')
                print(f'        {node}\n\n')
                node_num += 1
