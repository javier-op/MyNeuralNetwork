import numpy as np


class NeuronLayer:

    def __init__(self, neuron, input_size, layer_size, lr):
        self.size = layer_size
        self.neurons = []
        for i in range(layer_size):
            self.neurons.append(neuron(input_size, lr))

    def get_outputs(self):
        result = np.zeros(self.size)
        for i in range(self.size):
            result[i] = self.neurons[i].output
        return result

    def get_errors(self):
        result = np.zeros(self.size)
        for i in range(self.size):
            result[i] = self.neurons[i].error
        return result

    def get_bias(self):
        result = np.zeros(self.size)
        for i in range(self.size):
            result[i] = self.neurons[i].bias
        return result

    def get_weights(self):
        result = []
        for i in range(self.size):
            result.append(self.neurons[i].weights)
        return result

    def get_deltas(self):
        result = np.zeros(self.size)
        for i in range(self.size):
            result[i] = self.neurons[i].delta
        return result

    def feed(self, input_data):
        for i in range(self.size):
            self.neurons[i].evaluate(input_data)

    def set_last_layer_errors(self, expected):
        new_error = expected - self.get_outputs()
        for i in range(self.size):
            self.neurons[i].set_error(new_error[i])

    def update_errors(self, next_weights, next_deltas):
        next_size = len(next_weights)
        for i in range(self.size):
            new_error = 0
            for j in range(next_size):
                a = next_weights[j]
                b = a[i]
                new_error += next_weights[j][i] * next_deltas[j]
            self.neurons[i].set_error(new_error)

    def update_deltas(self):
        for i in range(self.size):
            self.neurons[i].update_delta()

    def update_neurons(self, input_data):
        input_data = np.array(input_data)
        for neuron in self.neurons:
            neuron.update_weights(input_data)
            neuron.update_bias()
