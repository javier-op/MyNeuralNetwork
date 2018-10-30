from NeuralNetwork import NeuralNetwork, NeuronLayer
from Neurons import Sigmoid
import unittest


class TestNeuralNetwork(unittest.TestCase):

    @staticmethod
    def train_xor_gate(network, n):
        for i in range(n):
            network.train([0, 0], 0)
            network.train([0, 1], 1)
            network.train([1, 0], 1)
            network.train([1, 1], 0)

    def test_xor_gate(self):
        network = NeuralNetwork(Sigmoid, 0.1, [2, 8, 1])
        self.train_xor_gate(network, 1000)

        network.feed([0, 0])
        network.feed([0, 1])
        network.feed([1, 0])
        network.feed([1, 1])

    @staticmethod
    def create_neural_network():
        network = NeuralNetwork(Sigmoid, 0.2, [2, 2, 1])

        first_layer = NeuronLayer(Sigmoid, 2, 2, 0.2)
        first_layer.neurons[0] = Sigmoid(2, 0.2, weights=[0.5, 0.5], bias=0.5)
        first_layer.neurons[1] = Sigmoid(2, 0.2, weights=[0.6, 0.6], bias=0.6)

        inner_layer = NeuronLayer(Sigmoid, 2, 2, 0.2)
        inner_layer.neurons[0] = Sigmoid(2, 0.2, weights=[0.7, 0.7], bias=0.7)
        inner_layer.neurons[1] = Sigmoid(2, 0.2, weights=[0.8, 0.8], bias=0.8)

        last_layer = NeuronLayer(Sigmoid, 2, 1, 0.2)
        last_layer.neurons[0] = Sigmoid(2, 0.2, weights=[0.9, 0.9], bias=0.9)

        network.first_layer = first_layer
        network.inner_layers.append(inner_layer)
        network.last_layer = last_layer

        return network

    def test_feed(self):
        network = self.create_neural_network()
        network.feed([1, 2])
        output1 = network.first_layer.get_outputs()
        output2 = network.inner_layers[0].get_outputs()
        output3 = network.last_layer.get_outputs()

        self.assertAlmostEqual(output1[0], 0.88079, places=4)
        self.assertAlmostEqual(output1[1], 0.91682, places=4)
        self.assertAlmostEqual(output2[0], 0.87635, places=4)
        self.assertAlmostEqual(output2[1], 0.90361, places=4)
        self.assertAlmostEqual(output3[0], 0.92428, places=4)

    def test_backpropagate(self):
        network = self.create_neural_network()
        network.feed([1, 2])  # 0.9242
        network.backpropagate(0.5)
        errors1 = network.first_layer.get_errors()
        deltas1 = network.first_layer.get_deltas()
        errors2 = network.inner_layers[0].get_errors()
        deltas2 = network.inner_layers[0].get_deltas()
        errors3 = network.last_layer.get_errors()
        deltas3 = network.last_layer.get_deltas()

        self.assertAlmostEqual(errors1[0], -0.00388, places=4)
        self.assertAlmostEqual(errors1[1], -0.00388, places=4)
        self.assertAlmostEqual(deltas1[0], -0.00040, places=4)
        self.assertAlmostEqual(deltas1[1], -0.00029, places=4)

        self.assertAlmostEqual(errors2[0], -0.02672, places=4)
        self.assertAlmostEqual(errors2[1], -0.02672, places=4)
        self.assertAlmostEqual(deltas2[0], -0.00289, places=4)
        self.assertAlmostEqual(deltas2[1], -0.00232, places=4)

        self.assertAlmostEqual(errors3[0], -0.42428, places=4)
        self.assertAlmostEqual(deltas3[0], -0.02969, places=4)

    def test_train(self):
        network = self.create_neural_network()
        network.train([1, 2], 0.5)
        weights1 = network.first_layer.get_weights()
        bias1 = network.first_layer.get_bias()
        weights2 = network.inner_layers[0].get_weights()
        bias2 = network.inner_layers[0].get_bias()
        weights3 = network.last_layer.get_weights()
        bias3 = network.last_layer.get_bias()

        self.assertAlmostEqual(weights1[0][0], 0.49991, places=4)
        self.assertAlmostEqual(weights1[0][1], 0.49983, places=4)
        self.assertAlmostEqual(weights1[1][0], 0.59994, places=4)
        self.assertAlmostEqual(weights1[1][1], 0.59988, places=4)
        self.assertAlmostEqual(bias1[0], 0.49991, places=4)
        self.assertAlmostEqual(bias1[1], 0.59994, places=4)

        self.assertAlmostEqual(weights2[0][0], 0.69948, places=4)
        self.assertAlmostEqual(weights2[0][1], 0.69946, places=4)
        self.assertAlmostEqual(weights2[1][0], 0.79958, places=4)
        self.assertAlmostEqual(weights2[1][1], 0.79957, places=4)
        self.assertAlmostEqual(bias2[0], 0.69942, places=4)
        self.assertAlmostEqual(bias2[1], 0.79953, places=4)

        self.assertAlmostEqual(weights3[0][0], 0.89479, places=4)
        self.assertAlmostEqual(weights3[0][1], 0.89463, places=4)
        self.assertAlmostEqual(bias3[0], 0.89406, places=4)
