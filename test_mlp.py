from mlp import Mlp
import numpy as np


class TestMlp:

    def test_back_propagation(self):
        model = Mlp(layer_layout=(2, 3, 1), learning_rate=2, activation_function='sigmoid')
        weights = [np.array([[0.5, 0.3], [0.3, -0.5], [-0.4, -0.1]]), np.array([[0.4, -0.2, 0.2]])]
        model.set_weights(weights)
        sd = model.sigmoid_derivative

        u, y = model.forward_pass([1, 2])

        d4 = (1 - y[-1][0]) * sd(u[-1][0])
        d1 = d4 * weights[1][0][0] * sd(u[0][0])
        d2 = d4 * weights[1][0][1] * sd(u[0][1])
        d3 = d4 * weights[1][0][2] * sd(u[0][2])

        w_i1_n1 = 0.5 + 2 * d1 * 1
        w_i2_n1 = 0.3 + 2 * d1 * 2
        w_i1_n2 = 0.3 + 2 * d2 * 1
        w_i2_n2 = -0.5 + 2 * d2 * 2
        w_i1_n3 = -0.4 + 2 * d3 * 1
        w_i2_n3 = -0.1 + 2 * d3 * 2
        w_n1_n4 = 0.4 + 2 * d4 * y[0][0]
        w_n2_n4 = -0.2 + 2 * d4 * y[0][1]
        w_n3_n4 = 0.2 + 2 * d4 * y[0][2]

        model.back_propagation([1, 2], 1)

        assert round(model.weights[0][0][0], 6) == round(w_i1_n1, 6)
        assert round(model.weights[0][0][1], 6) == round(w_i2_n1, 6)

        assert round(model.weights[0][1][0], 6) == round(w_i1_n2, 6)
        assert round(model.weights[0][1][1], 6) == round(w_i2_n2, 6)

        assert round(model.weights[0][2][0], 6) == round(w_i1_n3, 6)
        assert round(model.weights[0][2][1], 6) == round(w_i2_n3, 6)

        assert round(model.weights[1][0][0], 6) == round(w_n1_n4, 6)
        assert round(model.weights[1][0][1], 6) == round(w_n2_n4, 6)
        assert round(model.weights[1][0][2], 6) == round(w_n3_n4, 6)

    def test_activation_fucntion_d(self):
        m = Mlp(layer_layout=(2, 2))
        assert m.activation_function_d(2) == 1
        assert m.activation_function_d(0.1) == 1
        assert m.activation_function_d(-0.3) == 0
        assert m.activation_function_d(0) == 0
        assert m.activation_function_d(-15) == 0
        m = Mlp(layer_layout=(2, 2), activation_function='sigmoid')
        assert m.activation_function_d(2) == m.sigmoid_derivative(2)
        assert m.activation_function_d(12) == m.sigmoid_derivative(12)
        assert m.activation_function_d(0.2) == m.sigmoid_derivative(0.2)
        assert m.activation_function_d(-5) == m.sigmoid_derivative(-5)
        assert m.activation_function_d(-0.4) == m.sigmoid_derivative(-0.4)

    def test_activation_function(self):
        m = Mlp(layer_layout=(2, 2))
        assert m.activation_function(3) == 3
        assert m.activation_function(15) == 15
        assert m.activation_function(-3) == 0
        m = Mlp(layer_layout=(2, 2), activation_function='sigmoid')
        assert m.activation_function(3) == m.sigmoid(3)
        assert m.activation_function(0.3) == m.sigmoid(0.3)
        assert m.activation_function(-0.9) == m.sigmoid(-0.9)

    def test_relu(self):
        m = Mlp(layer_layout=(2, 2, 2))
        assert 0 == m.relu(-5)
        assert 0 == m.relu(0)
        assert 0 == m.relu(-0.2)
        assert 0 == m.relu(-3423423)
        assert 15 == m.relu(15)
        assert 1.2 == m.relu(1.2)

    def test_relu_derivative(self):
        m = Mlp(layer_layout=(2, 2, 2))
        assert 0 == m.relu_derivative(-5)
        assert 0 == m.relu_derivative(0)
        assert 0 == m.relu_derivative(-0.2)
        assert 0 == m.relu_derivative(-3423423)
        assert 1 == m.relu_derivative(15)
        assert 1 == m.relu_derivative(1.2)

    def test_sigmoid(self):
        m = Mlp(layer_layout=(2, 2, 2))
        assert round(0.3775406687981454353611, 8) == round(m.sigmoid(-0.5), 8)
        assert round(0.6681877721681661065308, 8) == round(m.sigmoid(0.7), 8)
        assert round(0.9975273768433652256659, 8) == round(m.sigmoid(6), 8)

    def test_sigmoid_derivative(self):
        m = Mlp(layer_layout=(2, 2))
        assert round(0.2350037122015944890693, 8) == round(m.sigmoid_derivative(0.5), 8)
        assert round(0.1049935854035065173486, 8) == round(m.sigmoid_derivative(2), 8)
        assert round(0.2493760401928919678215, 8) == round(m.sigmoid_derivative(-0.1), 8)

    def test_forward_pass(self):
        model = Mlp(layer_layout=(2, 3, 2, 1), activation_function='sigmoid')
        weights = [np.array([[0.5, 0.3], [0.3, -0.5], [-0.4, -0.1]]), np.array([[0.5, 0.3, -0.4], [-0.4, -0.1, -0.2]]),
                   np.array([[0.4, -0.2]])]
        model.set_weights(weights)
        u, y = model.forward_pass([0, 1])
        # layer 1
        node1 = model.sigmoid(0.5 * 0 + 0.3 * 1)
        node2 = model.sigmoid(0.3 * 0 - 0.5 * 1)
        node3 = model.sigmoid(-0.4 * 0 - 0.1 * 1)
        # layer 2
        node4 = model.sigmoid(0.5 * node1 + 0.3 * node2 - 0.4 * node3)
        node5 = model.sigmoid(-0.4 * node1 - 0.1 * node2 - 0.2 * node3)
        # output layer
        node6 = model.sigmoid(0.4 * node4 - 0.2 * node5)
        assert node6 == y[-1][0]

    def test_set_weights(self):
        model = Mlp(layer_layout=(2, 3, 2, 1))
        weights = [np.array([[0.5, 0.3], [0.3, -0.5], [-0.4, -0.1]]), np.array([[0.5, 0.3, -0.4], [-0.4, -0.1, -0.2]]),
                   np.array([[0.4, -0.2]])]
        model.set_weights(weights)
        weights = None
        assert model.weights is not None

    def test_predict(self):
        model = model = Mlp(layer_layout=(2, 3, 2, 1), activation_function='sigmoid')
        u, result = model.forward_pass([2, 4])
        result = result[-1]
        assert result == model.predict([2, 4])


if __name__ == '__main__':
    pass
