import numpy as np
from Neurons import Perceptron


class Sigmoid(Perceptron):

    def evaluate(self, input_data):
        assert len(input_data) == self.size
        input_data = np.array(input_data)
        result = np.sum(self.weights * input_data) + self.bias
        self.output = self.sigmoid_function(result)

    @staticmethod
    def sigmoid_function(value):
        return 1/(1 + np.exp(-value))
