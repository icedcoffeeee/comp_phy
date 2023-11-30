from __future__ import annotations
import numpy as np
from scipy.misc import derivative as dv


def ReLU(x):
    return x if x > 0 else 0


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class NeuralNetwork:
    np.random.seed(100)
    sig = np.vectorize(sigmoid)

    def __init__(self, learning_rate):
        self.weights = np.random.random(2)
        self.bias = np.random.random()
        self.learning_rate = learning_rate

    def _deriv(self, x, dx=1e-9):
        return (self.sig(x + dx / 2) - self.sig(x - dx / 2)) / dx

    def predict(self, input_vector):
        return self.sig(np.dot(input_vector, self.weights) + self.bias)

    def _compute_gradients(self, input_vector, target):
        layer_1 = np.dot(input_vector, self.weights) + self.bias
        prediction = self.sig(layer_1)

        derror_dprediction = 2 * (prediction - target)
        dprediction_dlayer1 = self._deriv(layer_1)
        dlayer1_dbias = 1
        dlayer1_dweights = (0 * self.weights) + (1 * input_vector)

        derror_dbias = derror_dprediction * dprediction_dlayer1 * dlayer1_dbias
        derror_dweights = derror_dprediction * dprediction_dlayer1 * dlayer1_dweights

        return derror_dbias, derror_dweights

    def _update_parameters(self, derror_dbias, derror_dweights):
        self.bias = self.bias - (derror_dbias * self.learning_rate)
        self.weights = self.weights - (derror_dweights * self.learning_rate)

    def train(self, input_vectors, targets, iterations, output_err=False):
        cumulative_errors = []
        for current_iteration in range(iterations):
            # Pick a data instance at random
            random_data_index = np.random.randint(len(input_vectors))

            input_vector = input_vectors[random_data_index]
            target = targets[random_data_index]

            # Compute the gradients and update the weights
            derror_dbias, derror_dweights = self._compute_gradients(
                input_vector, target
            )

            self._update_parameters(derror_dbias, derror_dweights)

            if output_err:
                # Measure the cumulative error for all the instances
                if current_iteration % 100 == 0:
                    cumulative_error = 0
                    # Loop through all the instances to measure the error
                    for data_instance_index in range(len(input_vectors)):
                        data_point = input_vectors[data_instance_index]
                        target = targets[data_instance_index]

                        prediction = self.predict(data_point)
                        error = (prediction - target) ** 2

                        cumulative_error = cumulative_error + error
                    cumulative_errors.append(cumulative_error)

        if output_err:
            return cumulative_errors


# nn = NeuralNetwork(2)
# nn.train([[0, 0], [0, 1], [1, 0], [1, 1]], [0, 1, 1, 0], 2000)
# print(nn.predict([0, 1]))
# print(nn.weights)
# print(nn.bias)


class NN:
    np.random.seed(100)
    sig = np.vectorize(sigmoid)

    def __init__(self, inputs, outputs, layers, rate=1, saved=False):
        self.inputs = inputs
        self.outputs = outputs
        self.layers = layers
        self.rate = rate

        if not saved:
            pairs = (len(inputs[0]), *layers, len(outputs[0]))
            self.weights = [np.random.random((i, j)) for i, j in zip(pairs[1:], pairs)]
            self.biases = [np.random.random(i) for i in pairs[1:]]
        else:
            with open(f"WB_{self.layers}.txt") as f:
                file = f.read()
                W, B = file.replace(
                    "array",
                    "np.array",
                ).split("\n\n")
                self.weights, self.biases = eval(W), eval(B)

    def save(self):
        with open(f"WB_{self.layers}.txt", "w") as f:
            f.write(repr(self.weights))
            f.write("\n\n")
            f.write(repr(self.biases))

    def forward(self, A, x, b):
        return self.sig(A @ x + b)

    def Z(self, input, index=None):
        Z = input
        for n in range(index or len(self.weights)):
            Z = self.forward(self.weights[n], Z, self.biases[n])
        return Z

    def get_derivatives(self, n, input, output):
        W, B, Z, dt = self.weights[n], self.biases[n], self.Z(input, n), 1e-2
        target = output if n == len(self.weights) - 1 else self.Z(input, n + 1)
        CW = lambda a: (self.forward(a, Z, B) - target) ** 2
        CB = lambda b: (self.forward(W, Z, b) - target) ** 2

        X, Y = dv(CW, W, dt, 2), dv(CW, W, dt, 1)
        dW = Y.copy()
        dW[abs(Y) > dt] /= X[abs(Y) > dt]
        dW[abs(Y) <= dt] = 0

        X, Y = dv(CB, B, dt, 2), dv(CB, B, dt, 1)
        dB = Y.copy()
        dB[abs(Y) > dt] /= X[abs(Y) > dt]
        dB[abs(Y) <= dt] = 0
        print(Y, X)

        return dW, dB

    def update(self, input, output):
        N = len(self.weights) - 1
        for n in range(N):
            dW, dB = self.get_derivatives(N - n, input, output)
            self.weights[N - n] -= dW * self.rate
            self.biases[N - n] -= dB * self.rate

    def train(self, n=100):
        for i, o in zip(self.inputs, self.outputs):
            for _ in range(n):
                self.update(i, o)


nn = NN([[0, 0], [1, 0], [0, 1], [1, 1]], [[0], [1], [1], [0]], (3, 3, 3), 5, saved=0)
nn.train(10)
nn.save()
print(nn.Z([0, 0]))
print(nn.Z([1, 0]))
print(nn.Z([1, 1]))
