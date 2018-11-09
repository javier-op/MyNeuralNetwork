from NeuralNetwork.NeuronLayer import NeuronLayer


class NeuralNetwork:

    def __init__(self, neuron, lr, shape):
        assert len(shape) >= 3
        self.first_layer = NeuronLayer(neuron, shape[0], shape[1], lr)
        self.inner_layers = []
        for i in range(1, len(shape)-2):
            self.inner_layers.append(NeuronLayer(neuron, shape[i], shape[i+1], lr))
        self.last_layer = NeuronLayer(neuron, shape[-2], shape[-1], lr)

    def get_output(self):
        return self.last_layer.get_outputs()

    def feed(self, input_data):
        self.first_layer.feed(input_data)
        previous_output = self.first_layer.get_outputs()
        for layer in self.inner_layers:
            layer.feed(previous_output)
            previous_output = layer.get_outputs()
        self.last_layer.feed(previous_output)

    def backpropagate(self, expected):
        self.last_layer.set_last_layer_errors(expected)
        self.last_layer.update_deltas()
        weights_list = self.last_layer.get_weights()
        delta_list = self.last_layer.get_deltas()
        for layer in reversed(self.inner_layers):
            layer.update_errors(weights_list, delta_list)
            layer.update_deltas()
            weights_list = layer.get_weights()
            delta_list = layer.get_deltas()
        self.first_layer.update_errors(weights_list, delta_list)
        self.first_layer.update_deltas()

    def train(self, input_data, expected):
        self.feed(input_data)
        self.backpropagate(expected)
        self.first_layer.update_neurons(input_data)
        previous_output = self.first_layer.get_outputs()
        for layer in self.inner_layers:
            layer.update_neurons(previous_output)
            previous_output = layer.get_outputs()
        self.last_layer.update_neurons(previous_output)
