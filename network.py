import random
import math
from copy import copy


class NeuralNetwork():
    t = 1

    def __init__(self, neurals_counts, learning_rate=0.2, momentum=0, verbosity=True, weights=None):
        self.weights = []
        self.learning_rate = learning_rate
        self.neurals_counts = neurals_counts
        self.node_values = []
        self.momentum = momentum
        self.prev_weight_correction = []
        self._layers_count = len(neurals_counts) - 1
        self.verbosity = verbosity
        self.input = []

        self._init_weights(neurals_counts, weights)

    def log(self, msg):
        if self.verbosity:
            print(msg)

    def activate(self, value):
        return 1.0 / (1.0 + math.exp(-self.t * value))

    def activate_derivative(self, value):
        return self.t * value * (1.0 - value)

    def _init_weights(self, neurals_counts, weights=None):
        if not weights:
            for i in range(1, len(neurals_counts)):
                weights = []
                weights_count = neurals_counts[i - 1] + 1

                for _ in range(neurals_counts[i]):
                    weights.append([random.random() - 0.5 for _ in range(weights_count)])

                self.weights.append(weights)
        else:
            self.weights = weights

        for i in range(1, len(neurals_counts)):
            weights_count = neurals_counts[i - 1] + 1
            prew_weights_corrections = []

            for _ in range(neurals_counts[i]):
                prew_weights_corrections.append([0] * weights_count)

            self.prev_weight_correction.append(prew_weights_corrections)

    def calculate(self, input_array):
        self.node_values = []
        self.input = copy(input_array)

        for layer in self.weights:
            values = []

            for weights in layer:
                value = 0
                for i, _ in enumerate(input_array):
                    value += weights[i] * input_array[i]
                value += weights[-1]
                values.append(self.activate(value))

            self.node_values.append(copy(values))

            input_array = values

        return input_array

    def backpropagate(self, output, desired_output):
        error_gradient = [0] * self._layers_count

        layer = -1
        errors = []

        for node in range(len(self.weights[layer])):
            error = desired_output[node] - output[node]
            errors.append(self.activate_derivative(output[node]) * error)

        error_gradient[-1] = errors

        for layer in range(self._layers_count - 2, -1, -1):
            errors = []
            next_layer = layer + 1

            for node, _ in enumerate(self.weights[layer]):
                sum_value = 0

                for next_node in range(len(self.weights[next_layer])):
                    gradient = error_gradient[next_layer][next_node]
                    weight = self.weights[next_layer][next_node][node]
                    sum_value += gradient * weight

                value = self.node_values[layer][node]
                errors.append(self.activate_derivative(value) * sum_value)

            error_gradient[layer] = errors

        for layer in range(self._layers_count):
            for node, _ in enumerate(self.weights[layer]):
                er_gr = error_gradient[layer][node]
                self.weights[layer][node][-1] += self.learning_rate * er_gr

                if layer == 0:
                    input_value = self.input
                else:
                    input_value = self.node_values[layer - 1]

                for i, _ in enumerate(self.weights[layer][node][:-1]):
                    weight_correction = self.learning_rate * er_gr * input_value[i]
                    prev_correction = self.prev_weight_correction[layer][node][i]
                    self.weights[layer][node][i] += weight_correction + prev_correction * self.momentum
                    self.prev_weight_correction[layer][node][i] = weight_correction

    def teach(self, data, max_retries=1000):
        for i in range(max_retries):
            errors = []
            for item in data:
                desired_output = item['output']

                output = self.calculate(item['input'])

                errors.append(self.least_mean_square(output, desired_output))

                self.backpropagate(output, desired_output)

            if i % 100 == 0:
                self.log('step: {:5} error: {}'.format(i, self.root_mean_squared_error(errors)))

    @staticmethod
    def root_mean_squared_error(errors):
        sum_value = 0
        for error in errors:
            sum_value += error ** 2

        return math.sqrt(sum_value / len(errors))

    @staticmethod
    def least_mean_square(output, desired_output):
        err = 0

        for i, output_value in enumerate(output):
            err += (desired_output[i] - output_value) ** 2

        return err / 2
