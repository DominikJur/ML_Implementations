import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def reLU(x):
    return np.maximum(0, x)


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


def reLU_derivative(x):
    return np.where(0<x, 1, 0)

class Softmax:
    def forward(self, X):
        exp_X = np.exp(X - np.max(X, axis=1, keepdims=True)) 
        self.output = exp_X / (np.sum(exp_X, axis=1, keepdims=True) + 1e-12) 
        return self.output
    
    def backward(self, dE_dy):
        return dE_dy

    def step(self, eta):
        return None

def cross_entropy_loss(y_true, y_pred):
    epsilon = 1e-12
    y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
    N = y_pred.shape[0]
    return -np.sum(y_true * np.log(y_pred)) / N

def cross_entropy_loss_derivative(y_true, y_pred):
    return y_pred - y_true

class FullyConnected:

    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.passes = 0
        self.weights = np.random.rand(self.input_size, self.output_size) - 0.5
        self.bias = np.random.rand(1, self.output_size) - 0.5
        self.delta_w = np.zeros(self.weights.shape)
        self.delta_b = np.zeros(self.bias.shape)
        
    def clip_gradients(self, grad, clip_value=1.0):
        return np.clip(grad, -clip_value, clip_value)

    def forward(self, X):
        self.X = X
        return np.dot(X, self.weights) + self.bias

    def backward(self, dE_dy):
        dE_dx = np.dot(dE_dy, self.weights.T)
        dE_dW = np.dot(self.X.T, dE_dy)
        dE_db = dE_dy
        dE_dx = self.clip_gradients(dE_dx)
        dE_dW = self.clip_gradients(dE_dW)
        dE_db = self.clip_gradients(dE_db)
        self.delta_w += dE_dW
        self.delta_b += dE_db
        self.passes += 1
        return dE_dx

    def step(self, eta):
        self.weights -= eta * self.delta_w / self.passes
        self.bias -= eta * self.delta_b / self.passes
        self.delta_w = np.zeros(self.weights.shape)
        self.delta_b = np.zeros(self.bias.shape)
        self.passes = 0
        return None


class Sigmoid:

    def forward(self, X):
        self.X = X
        return sigmoid(X)

    def backward(self, dE_dy):
        return sigmoid_derivative(self.X) * dE_dy

    def step(self, eta):
        return None


class ReLU:

    def forward(self, X):
        self.X = X
        return reLU(X)

    def backward(self, dE_dy):
        return reLU_derivative(self.X) * dE_dy

    def step(self, eta):
        return None


class NeuralNet:

    def __init__(self, layers, verbose):
        self.layers = layers
        self.verbose = verbose

    def predict(self, X):
        result = []
        for i in range(X.shape[0]):
            prediction = X[i]
            for layer in self.layers:
                prediction = layer.forward(prediction)
            result.append(prediction)
        return np.array(result)

    def fit(self, X, y, epochs=1, minibatches=None, eta=0.1, batch_size=64):

        if minibatches == None:
            minibatches = len(X) // 64 
        for epoch in range(epochs):
            for i in range(minibatches):
                mean_error = 0
                idx = np.random.choice(X.shape[0], batch_size, replace=False)
                X_batch = X[idx]
                y_batch = y[idx]
                for j in range(batch_size):
                    prediction = X_batch[j]
                    for layer in self.layers:
                        prediction = layer.forward(prediction)
                        
                    mean_error += cross_entropy_loss(y_batch[j], prediction)

                    dE_dy = cross_entropy_loss_derivative(y_batch[j], prediction)
                    for layer in reversed(self.layers):
                        dE_dy = layer.backward(dE_dy)

                for layer in self.layers:
                    layer.step(eta)
                if self.verbose and i % 10 == 0:
                    mean_error = mean_error / batch_size
                    print(f"Epoch: {epoch+1}/{epochs}, Minibatch: {i}/{minibatches},\nError: {mean_error}.\n")
