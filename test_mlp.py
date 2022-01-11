from mlp import Mlp
import numpy as np


class TestMlp:

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
        model = Mlp(layer_layout=(2, 3, 2, 1))
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
        model = model = Mlp(layer_layout=(2, 3, 2, 1))
        u, result = model.forward_pass([2, 4])
        result = result[-1]
        assert result == model.predict([2, 4])


if __name__ == '__main__':
    pass
