import numpy as np

#===Activation Functions===
def Linear(X):
    return X

def ReLU(x):
    return np.maximum(0, x)

def Sigmoid(x):
    return 1 / (1 + np.exp(-x))

def Tanh(x):
    return np.tanh(x)

def Softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    exp = np.exp(x)
    return exp / np.sum(exp, axis=1, keepdims=True)

activationFunctionsDict = {
    "linear": Linear,
    "relu": ReLU,
    "sigmoid": Sigmoid,
    "tanh": Tanh,
    "softmax": Softmax,
}

#===Activation Functions Derivatives===
def LinearDerivative(x):
    return np.ones_like(x)

def ReluDerivative(x):
    return (x > 0).astype(float)

def SigmoidDerivative(x):
    s = Sigmoid(x)
    return s * (1 - s)

def TanhDerivative(x):
    return 1 - np.tanh(x) ** 2

activationFunctionsDerivativeDict = {
    "linear": LinearDerivative,
    "relu": ReluDerivative,
    "sigmoid": SigmoidDerivative,
    "tanh": TanhDerivative,
    "softmax": None
    #Softmax is none because the differentiation is handled
    #when combined with Categorical cross entropy as the derivative
    # is much simpler
}