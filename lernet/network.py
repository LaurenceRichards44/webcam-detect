import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
from lernet.layer import *
from lernet.activations import *
from lernet.losses import *
from lernet.optimizers import SGD, Momentum, Adam
from lernet.utils import *

class Network():
    """
    lerler network

    A fully connected feedforward neural network implemented using NumPy.

    Features
    --------
    - Multiple hidden layers
    - Custom activation functions
    - Cross-entropy and MSE loss
    - Mini-batch gradient descent training
    - Model saving and loading
    - Training visualization tools

    Designed for educational purposes and experimentation.
    """
    def __init__(self, layerSizes=None, activations=None, lossName=None, optimizer=None, lambda_regularization=0.001):
        """
        Initialize a neural network.

        Parameters
        ----------
        layerSizes : list[int]
            List of integers defining the number of neurons in each layer,
            including input and output layers.

        activations : list[str]
            List of activation function names for each layer except the input layer.
            Must have length len(layerSizes) - 1.

        lossName : str
            Name of the loss function used for training.
            Supported values are defined in `lossFunctionsDict`.

        optimizer : str, optional
            Name of the optimizer used during training.
            (Currently unused but reserved for future expansion.)

        Raises
        ------
        ValueError
            If required parameters are missing or invalid.
        """
        
        self.lossName = lossName.lower() if lossName != None else lossName
        self.lossFunction = None
        self.lossFunctionDerivative = None

        self.layers = []
        self.layerSizes = layerSizes
        self.activations = [activation.lower() for activation in activations]
        self.optimizer = optimizer.lower() if optimizer != None else optimizer
        self.lambda_regularization = lambda_regularization

        self.InitializeLoss(self.lossName)
        self.InitializeLayers(self.layerSizes, self.activations, self.lambda_regularization)
        self.InitializeOptimizer(self.optimizer)
        

        self.learningRate = []
        self.epochs = []
        self.batchSize = []
        
        self.accuracyArray = []
        self.lossesArray = []

        #Raise error if any parameters are not provided
        if self.layerSizes is None:
            raise ValueError("layerSizes must be provided")
        if self.activations is None:
            raise ValueError("activations must be provided")
        if self.lossName is None:
            raise ValueError("lossName must be provided")
        
        #Raise error if activation list has incorrect length or lossName is not supported
        if len(self.activations) != len(self.layerSizes) - 1:
            raise ValueError(f"activations list must have exactly {len(self.layerSizes) - 1} elements")
        if self.lossName not in lossFunctionsDict:
            raise ValueError(f"Loss '{self.lossName}' not supported. Available: {list(lossFunctionsDict.keys())}")

        #Raise error if final activation is not supported with loss
        if (self.activations[-1] == "softmax") != (self.lossName == "crossentropy"):
            raise ValueError("Softmax must be used with CrossEntropy, and other activations cannot use CrossEntropy")

    def InitializeLoss(self, lossName):
        """
        Initialize the loss function and its derivative.

        Parameters
        ----------
        lossName : str
            Name of the loss function to use.

        Notes
        -----
        The function and its derivative are retrieved from
        `lossFunctionsDict` and `lossFunctionsDerivativeDict`.
        """     
        self.lossFunction = lossFunctionsDict[lossName]
        self.lossFunctionDerivative = lossFunctionsDerivativeDict[lossName]

    def InitializeLayers(self, layerSizes, activations, lambda_regularization):
        """
        Create and initialize all layers of the neural network.

        Parameters
        ----------
        layerSizes : list[int]
            List of integers specifying the number of neurons in each layer.

        activations : list[str]
            List of activation function names corresponding to each layer.

        Raises
        ------
        ValueError
            If an activation function name is not supported.

        Notes
        -----
        Each layer contains:
        - Weight matrix
        - Bias vector
        - Activation function
        - Activation derivative
        """
        self.layers = []
        
        for i in range(len(layerSizes) - 1):
            # If user gave an empty string "", set default activation depending on the layer and the loss function
            if activations[i]:
                activationName = activations[i]
            elif i == len(layerSizes) - 2:
                activationName = "linear" if self.lossName == "mse" else "softmax"
            else:
                activationName = "relu"

            #Raise error if activation name is not supported
            if activationName not in activationFunctionsDict:
                raise ValueError(f"Activation '{activationName}' not supported. Available: {list(activationFunctionsDict.keys())}")

            #Get activation and corresponding derivative function according to activation name
            activationFunction = activationFunctionsDict[activationName]
            activationFunctionDerivative = activationFunctionsDerivativeDict.get(activationName, None)

            #Add layer
            self.layers.append(Layer(
                shape=(layerSizes[i], layerSizes[i + 1]),
                activation=activationFunction,
                activationDerivative=activationFunctionDerivative,
                lambda_regularization=lambda_regularization
            ))

            print(f"Layer {i + 1}: {layerSizes[i]} → {layerSizes[i + 1]} ({activationName})")

    def InitializeOptimizer(self, optimizerName):
        if optimizerName == "sgd":
            self.optimizer = SGD()
        elif optimizerName == "momentum":
            self.optimizer = Momentum()
        elif optimizerName == "adam":
            self.optimizer = Adam()
        else:
            print(f"Optimizer '{optimizerName}' not supported. Defaulting to 'sgd'")
            self.optimizer = SGD()

    def Save(self, fileName, folderDir="models/"):
        """
        Save the trained model to disk.

        Parameters
        ----------
        filename : str
            Path to the file where the model should be saved.

        Notes
        -----
        The model is saved as a `.npz` file containing:

        - layer sizes
        - activation functions
        - loss function name
        - weights and biases for each layer
        - training history (learning rate, epochs, batch size)
        - loss history
        - accuracy history
        """
        modelData = {}

        # architecture
        modelData["layerSizes"] = np.array(self.layerSizes)
        modelData["activations"] = np.array(self.activations)
        modelData["lossName"] = self.lossName
        modelData["optimizer"] = self.optimizer.__class__.__name__.lower()

        # weights
        for i, layer in enumerate(self.layers):
            modelData[f"W{i}"] = layer.weights
            modelData[f"B{i}"] = layer.biases

        # training history
        modelData["learningRate"] = np.array(self.learningRate, dtype=object)
        modelData["epochs"] = np.array(self.epochs, dtype=object)
        modelData["batchSize"] = np.array(self.batchSize, dtype=object)

        modelData["lossHistory"] = np.array(self.lossesArray, dtype=object)
        modelData["accuracyHistory"] = np.array(self.accuracyArray, dtype=object)

        os.makedirs(folderDir, exist_ok=True)
        np.savez_compressed(folderDir + fileName, **modelData)

        print(f"Model saved to '{folderDir + fileName}'")

    def Load(self, filename):
        """
        Load a saved model from disk.

        Parameters
        ----------
        filename : str
            Path to the `.npz` file containing the model.

        Notes
        -----
        This restores:
        - network architecture
        - weights and biases
        - loss function
        - training history
        """
        data = np.load(filename, allow_pickle=True)

        # architecture
        self.layerSizes = data["layerSizes"].tolist()
        self.activations = data["activations"].tolist()
        self.lossName = str(data["lossName"])
        self.optimizer = str(data["optimizer"])

        # rebuild network structure
        self.InitializeLoss(self.lossName)
        self.InitializeLayers(self.layerSizes, self.activations, self.lambda_regularization)
        self.InitializeOptimizer(self.optimizer)

        # load weights
        print(len(self.layers))
        for i, layer in enumerate(self.layers):
            layer.weights = data[f"W{i}"]
            layer.biases = data[f"B{i}"]

        # restore history
        self.learningRate = data["learningRate"].tolist()
        self.epochs = data["epochs"].tolist()
        self.batchSize = data["batchSize"].tolist()

        self.lossesArray = data["lossHistory"].tolist()
        self.accuracyArray = data["accuracyHistory"].tolist()

        print(f"Model loaded from '{filename}'")

    @classmethod
    def FromFile(cls, filename, folderDir="models/"):

        model = cls.__new__(cls)
        model.Load(folderDir + filename + ".npz")

        return model

    def Forward(self, X):
        """
        Perform a forward pass through the neural network.

        Parameters
        ----------
        X : np.ndarray
            Input data with shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Output of the final layer.

        Notes
        -----
        This method returns the raw network outputs.

        - For **regression tasks**, this method should be used to obtain
        predicted numerical values.
        - For **classification tasks**, this method returns the predicted
        class probabilities (e.g. Softmax outputs).

        In classification problems, `Predict()` should usually be used
        instead to obtain the final class labels.
        """
        for layer in self.layers:
            X = layer.Forward(X)
        return X
    
    def Predict(self, X):
        """
        Generate predicted class labels for input data.

        Parameters
        ----------
        X : np.ndarray
            Input features.

        returnLabels : bool, optional
            If True, returns class labels instead of indices.

        Returns
        -------
        np.ndarray
            Array containing the predicted class index for each sample.

        Notes
        -----
        This method should be used for **classification tasks**.

        It performs a forward pass through the network and then returns
        the index/label of the highest output probability for each sample
        (using `argmax`).

        For **regression tasks**, use `Forward()` instead to obtain the
        predicted numerical outputs.
        """
        pred_indices = np.argmax(self.Forward(X)[0])

        return pred_indices 
    
    def Train(self, X_train, Y_train, X_test, Y_test, learningRate=0.01, epochs=10, batchSize=32):
        """
        Train the neural network using mini-batch gradient descent.

        Parameters
        ----------
        X : np.ndarray
            Training inputs of shape (n_samples, n_features).

        Y : np.ndarray
            Target labels of shape (n_samples, n_classes).

        learningRate : float, default=0.01
            Learning rate used to update weights.

        epochs : int, default=10
            Number of training iterations over the dataset.

        batchSize : int, default=32
            Number of samples per training batch.

        Notes
        -----
        Training process:
        1. Shuffle training samples
        2. Perform forward pass
        3. Compute loss
        4. Apply optimizer

        Training history (loss and accuracy) is stored in:

        - `self.lossesArray`
        - `self.accuracyArray`
        """

        #Store hyperparameters
        self.learningRate.append(learningRate)
        self.epochs.append(epochs)
        self.batchSize.append(batchSize)

        #Initialize arrays for loss and accuracy
        self.accuracyArray.append([])
        self.lossesArray.append([])
        n = len(X_train)

        for epoch in range(epochs):
            #intitalize total loss and randomize batches
            totalLoss = 0
            indices = np.random.permutation(n)

            #Loop through batches for each epoch
            for i in tqdm(range(0, n, batchSize), desc=f"Epoch {epoch+1}/{epochs}", unit="batch"):
                #Get batch X and Y
                batchIdx = indices[i:i+batchSize]
                x = X_train[batchIdx]
                y = Y_train[batchIdx]
                
                #Forward pass
                yPred = self.Forward(x)
                
                #Calculate loss and loss function gradient to prepare for Vanilla SGD
                lossValue = self.lossFunction(y, yPred)
                grad = self.lossFunctionDerivative(y, yPred)
                
                #Calculate gradients for each layer
                for layer in reversed(self.layers):
                    grad = layer.ComputeGradients(grad)

                #Apply optimizer
                self.optimizer.step(self.layers, learningRate)

                #Update total loss
                totalLoss += lossValue * len(x)

            #Calculate average loss for epoch and append to the list
            loss = totalLoss / n

            #Calculate accuracy if test data is provided
            accuracy = calculate_accuracy(self.Forward(X_test), Y_test, self.lossName)
            #accuracies.append(accuracy)
            
            #Append losses and accuracies
            self.accuracyArray[-1].append(accuracy)
            self.lossesArray[-1].append(loss)

    def TrainSummary(self, epochsPerPrint=1):
        """
        Print a summary of all completed training runs.
        """
        print("Training Summary")
        print("=" * 60)

        for run_idx, losses in enumerate(self.lossesArray):
            accuracies = self.accuracyArray[run_idx] if run_idx < len(self.accuracyArray) else None

            print(f"Run {run_idx + 1}")
            print("-" * 60)

            for epoch, loss in enumerate(losses):
                if (epoch + 1) % epochsPerPrint != 0:
                    continue

                if accuracies is None:
                    print(f"Epoch {epoch+1:<5} Loss={loss:.4f}, No accuracy available")
                    continue
                
                acc = accuracies[epoch]

                if self.lossName == "crossentropy":
                    print(f"Epoch {epoch+1:<5} Loss={loss:.4f}, Test Accuracy={acc*100:.2f}%")
                elif self.lossName == "mse":
                    print(f"Epoch {epoch+1:<5} Loss={loss:.4f}, Test R²={acc:.4f}")

            print()

        print("=" * 60)

    def ModelSummary(self):
        """
        Print a summary of the neural network architecture.

        Displays:
        - layer index
        - input → output size
        - activation function
        - number of parameters

        Also prints the total parameter count.
        """
        print("Model Summary")
        print("=" * 60)
        print(f"{'Layer':<10}{'Input→Output':<20}{'Activation':<15}{'Params':<15}")
        print("-" * 60)

        total_params = 0

        for i, layer in enumerate(self.layers):
            input_size, output_size = layer.weights.shape
            params = input_size * output_size + output_size  # weights + biases
            total_params += params

            activation_name = layer.activation.__name__ if layer.activation else "None"
            io = f"{input_size}→{output_size}"

            print(f"{i:<10}{io:<20}{activation_name:<15}{params:<15}")

        print("-" * 60)
        print(f"{'Total parameters:':<45}{total_params}")
        print("=" * 60)

    def PlotLossAccuracy(self, splitters : bool = True, linearFit : bool = True, legend=True):
        """
        Plot training loss and accuracy over epochs.

        Parameters
        ----------
        splitters : bool, default=True
            If True, draw vertical lines separating multiple training runs.

        linearFit : bool, default=True
            If True, fit a linear regression to the later portion of the
            accuracy curve to estimate improvement rate.

        Notes
        -----
        Two plots are produced:

        1. Loss vs Epoch
        2. Accuracy vs Epoch

        Linear regression is applied to the last portion of each run
        to estimate learning trends.
        """
        losses = np.concatenate(self.lossesArray)
        accuracy = np.concatenate(self.accuracyArray)

        fig, ax = plt.subplots(1, 2, figsize=(12,6))

        midPercent = 0.8

        #===Loss===
        ax[0].plot(losses, color='red', linestyle='-', label='Model Loss')

        ax[0].set_ylim(0, max(losses) * 1.1)
        ax[0].set_yticks(np.arange(0, max(losses) * 1.1, max(losses) * 0.1))
        ax[0].set_xlabel("Epoch")
        ax[0].set_ylabel("Average Loss")

        if legend:
            ax[0].legend()

        if splitters:
            offset = 0
            for i, run in enumerate(self.lossesArray):
                ax[0].axvline(offset, color='grey', linestyle='--') if i > 0 else None
                offset += len(run)

        #===Accuracy===
        ax[1].plot(accuracy, color='green', linestyle='-', label='Model Accuracy')

        ax[1].set_ylim(0, 1)
        ax[1].set_yticks(np.arange(0, 1.1, 0.1))
        ax[1].set_xlabel("Epoch")
        ax[1].set_ylabel("Percentage Accuracy")

        if legend:
            ax[1].legend()

        if splitters:
            offset = 0
            for i, run in enumerate(self.lossesArray):
                ax[1].axvline(offset, color='grey', linestyle='--') if i > 0 else None
                offset += len(run)

        if linearFit:
            offset = 0
            for i, run in enumerate(self.accuracyArray):
                run = np.array(run)
                runLength = len(run)
            
                startLocal = int(runLength * midPercent)
                xFit = np.arange(offset + startLocal, offset + runLength).reshape(-1,1)
                yFit = run[startLocal:]
                
                fit = LinearRegression().fit(xFit, yFit)
                
                xFull = np.arange(offset, offset + runLength).reshape(-1,1)
                yPredFull = fit.predict(xFull)
                
                ax[1].plot(xFull, yPredFull, color='black', linestyle='--', label=f'Fit {i+1} ({fit.coef_[0]*10000:.2f}% per 100 epochs)')

                offset += runLength

        plt.tight_layout()
        plt.show()

def ListSavedModels(path="models"):
    """
    List all saved model files in the default models directory.

    Parameters
    ----------
    path : str
        Directory containing saved models.

    Returns
    -------
    list[str]
        List of model file paths.
    """
    if not os.path.exists(path):
        os.mkdir(path)

    models = [f for f in os.listdir(path) if f.endswith(".npz")]

    if not models:
        print("No saved models found.")
        return []

    print("Saved Models:")
    print("=" * 40)

    for i, model in enumerate(models):
        print(f"{i+1:<3} {model}")

    print("=" * 40)

    return models