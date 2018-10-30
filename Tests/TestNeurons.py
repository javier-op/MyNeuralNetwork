from Neurons import Perceptron, Sigmoid
import unittest


class TestPerceptron(unittest.TestCase):

    @staticmethod
    def train_and_gate(neuron, n):
        for i in range(n):
            neuron.train([0, 0], 0)
            neuron.train([0, 1], 0)
            neuron.train([1, 0], 0)
            neuron.train([1, 1], 1)

    @staticmethod
    def train_or_gate(neuron, n):
        for i in range(n):
            neuron.train([0, 0], 0)
            neuron.train([0, 1], 1)
            neuron.train([1, 0], 1)
            neuron.train([1, 1], 1)

    def test_and_gate(self):
        neuron = Perceptron(2, lr=0.2)
        self.train_and_gate(neuron, 30)

        neuron.evaluate([0, 0])
        self.assertEqual(neuron.get_output(), 0)
        neuron.evaluate([0, 1])
        self.assertEqual(neuron.get_output(), 0)
        neuron.evaluate([1, 0])
        self.assertEqual(neuron.get_output(), 0)
        neuron.evaluate([1, 1])
        self.assertEqual(neuron.get_output(), 1)

    def test_or_gate(self):
        neuron = Perceptron(2, lr=0.2)
        self.train_or_gate(neuron, 30)

        neuron.evaluate([0, 0])
        self.assertEqual(neuron.get_output(), 0)
        neuron.evaluate([0, 1])
        self.assertEqual(neuron.get_output(), 1)
        neuron.evaluate([1, 0])
        self.assertEqual(neuron.get_output(), 1)
        neuron.evaluate([1, 1])
        self.assertEqual(neuron.get_output(), 1)


class TestSigmoid(TestPerceptron):

    def test_and_gate(self):
        neuron = Sigmoid(2, lr=0.2)
        self.train_and_gate(neuron, 30)

        neuron.evaluate([0, 0])
        self.assertLess(neuron.get_output(), 0.5)
        neuron.evaluate([0, 1])
        self.assertLess(neuron.get_output(), 0.5)
        neuron.evaluate([1, 0])
        self.assertLess(neuron.get_output(), 0.5)
        neuron.evaluate([1, 1])
        self.assertLessEqual(0.5, neuron.get_output())

    def test_or_gate(self):
        neuron = Sigmoid(2, lr=0.2)
        self.train_or_gate(neuron, 30)

        neuron.evaluate([0, 0])
        self.assertLess(neuron.get_output(), 0.5)
        neuron.evaluate([0, 1])
        self.assertLessEqual(0.5, neuron.get_output())
        neuron.evaluate([1, 0])
        self.assertLessEqual(0.5, neuron.get_output())
        neuron.evaluate([1, 1])
        self.assertLessEqual(0.5, neuron.get_output())
