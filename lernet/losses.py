import numpy as np

#===Loss Functions===
def CrossEntropyLoss(y_true, y_pred):
    return -np.mean(np.sum(y_true * np.log(y_pred + 1e-12), axis=1))

def MSELoss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

lossFunctionsDict = {
    "crossentropy": CrossEntropyLoss,
    "mse": MSELoss
}

#===Loss Functions Derivatives===
def CrossEntropyDerivative(y_true, y_pred):
    return y_pred - y_true

def MSELossDerivative(y_true, y_pred):
    return 2 * (y_pred - y_true)

lossFunctionsDerivativeDict = {
    "crossentropy": CrossEntropyDerivative,
    "mse": MSELossDerivative
}