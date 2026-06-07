import numpy as np

class Optimizer:
    def step(self, layers):
        raise NotImplementedError
    
class SGD(Optimizer):
    def step(self, layers, learningRate=0.01):
        for layer in layers:
            layer.weights -= learningRate * layer.dw
            layer.biases -= learningRate * layer.db

class Momentum(Optimizer):

    def __init__(self, beta=0.9):
        self.beta = beta
        self.vw = []
        self.vb = []
        self.initialized = False

    def step(self, layers, learningRate=0.01):

        if not self.initialized:
            for layer in layers:
                self.vw.append(np.zeros_like(layer.weights))
                self.vb.append(np.zeros_like(layer.biases))
            self.initialized = True

        for i, layer in enumerate(layers):

            self.vw[i] = self.beta * self.vw[i] + (1 - self.beta) * layer.dw
            self.vb[i] = self.beta * self.vb[i] + (1 - self.beta) * layer.db

            layer.weights -= learningRate * self.vw[i]
            layer.biases -= learningRate * self.vb[i]

class Adam(Optimizer):

    def __init__(self, beta1=0.9, beta2=0.999, eps=1e-8):

        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.mw = []
        self.vw = []

        self.mb = []
        self.vb = []

        self.t = 0
        self.initialized = False


    def step(self, layers, learningRate=0.01):

        if not self.initialized:
            for layer in layers:
                self.mw.append(np.zeros_like(layer.weights))
                self.vw.append(np.zeros_like(layer.weights))

                self.mb.append(np.zeros_like(layer.biases))
                self.vb.append(np.zeros_like(layer.biases))

            self.initialized = True

        self.t += 1

        for i, layer in enumerate(layers):

            # first moment
            self.mw[i] = self.beta1 * self.mw[i] + (1 - self.beta1) * layer.dw
            self.mb[i] = self.beta1 * self.mb[i] + (1 - self.beta1) * layer.db

            # second moment
            self.vw[i] = self.beta2 * self.vw[i] + (1 - self.beta2) * (layer.dw ** 2)
            self.vb[i] = self.beta2 * self.vb[i] + (1 - self.beta2) * (layer.db ** 2)

            # bias correction
            mw_hat = self.mw[i] / (1 - self.beta1 ** self.t)
            vw_hat = self.vw[i] / (1 - self.beta2 ** self.t)

            mb_hat = self.mb[i] / (1 - self.beta1 ** self.t)
            vb_hat = self.vb[i] / (1 - self.beta2 ** self.t)

            layer.weights -= learningRate * mw_hat / (np.sqrt(vw_hat) + self.eps)
            layer.biases -= learningRate * mb_hat / (np.sqrt(vb_hat) + self.eps)