from NeuralNetwork import NeuralNetwork
from NeuronLayer import NeuronLayer
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
        print(network.get_output())
        network.feed([0, 1])
        print(network.get_output())
        network.feed([1, 0])
        print(network.get_output())
        network.feed([1, 1])
        print(network.get_output())

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

        print(errors1)
        print(deltas1)
        print(errors2)
        print(deltas2)
        print(errors3)
        print(deltas3)

    def test_train(self):
        network = self.create_neural_network()
        network.train([1, 2], 0.5)
        weights1 = network.first_layer.get_weights()
        bias1 = network.first_layer.get_bias()
        weights2 = network.inner_layers[0].get_weights()
        bias2 = network.inner_layers[0].get_bias()
        weights3 = network.last_layer.get_weights()
        bias3 = network.last_layer.get_bias()

        print(weights1)
        print(bias1)
        print(weights2)
        print(bias2)
        print(weights3)
        print(bias3)


