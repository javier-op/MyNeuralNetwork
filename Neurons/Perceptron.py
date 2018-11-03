import numpy as np
import random


class Perceptron:

    def __init__(self, input_size, lr, weights=None, bias=None):
        self.size = input_size
        self.lr = lr
        if weights:
            self.weights = np.array(weights)
        else:
            self.weights = np.random.uniform(-2, 2, input_size)
        if bias:
            self.bias = bias
        else:
            self.bias = random.uniform(-2, 2)
        self.output = None
        self.error = None
        self.delta = None

    def get_output(self):
        return self.output

    def evaluate(self, input_data):
        assert len(input_data) == self.size
        result = np.sum(input_data * self.weights)
        output = 0
        if (result + self.bias) > 0:
            output = 1
        self.output = output

    def train(self, sample_data, expected):
        sample_data = np.array(sample_data)
        assert len(sample_data) == self.size
        self.evaluate(sample_data)
        result = self.output
        diff = expected - result
        self.weights = self.weights + (self.lr * diff * sample_data)
        self.bias = self.bias + (self.lr * diff)

    def set_error(self, error):
        self.error = error

    def update_delta(self):
        self.delta = self.error * self.output * (1.0 - self.output)

    def update_weights(self, input_data):
        self.weights = self.weights + self.lr * self.delta * input_data

    def update_bias(self):
        self.bias += self.lr * self.delta
