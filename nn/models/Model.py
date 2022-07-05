from asyncio import current_task
import datetime
import numpy as np
import matplotlib.pyplot as plt 
import pickle
import os

class Model:
    error_arr = []

    def __init__(self, layers=None):
        if layers is None: self.layers = []
        else: self.layers = layers
        self.error = None
        self.loss = None
        self.epochs = None
        self.output = None
        self.grad = None

    def fit(self, X, Y, epochs:int=1, learning_rate=0.01):
        self.epochs = epochs
        X: np.ndarray = X[..., np.newaxis]
        Y: np.ndarray = Y[..., np.newaxis]

        for i in range(epochs + 1):
            self.error = 0

            for x, y in zip(X, Y):
                output = x

                for layer in self.layers:
                    output = layer.forward(output)

                self.grad = self.loss.loss_prime(output, y)
       
                for error in reversed(self.layers):
                    self.grad = error.backwards(self.grad, learning_rate)

            error = self.loss.loss(output, y)
            print(error)
            Model.error_arr.append(self.loss.loss(output, y))

    def graph(self):
        plt.xlabel("EPOCHS")
        plt.ylabel("ERROR")
        plt.scatter(np.arange(0, self.epochs + 1), np.squeeze(self.error_arr))
        plt.plot(np.squeeze(self.error_arr), c="RED")
        plt.show()

    def build(self, loss):
        self.loss = loss

    def predict(self, X):
        X = np.array(X)[..., np.newaxis]
        output = X

        for layer in self.layers:
            output = layer.forward(output)

        return output

    def save(self, filename, default_file_dir="nn\saved_models", behaviour=0):

        '''
        behaviour: int
            behaviour == 0: if the value of behaviour is 0 (default), then any existing "<filename>.pickle" files will be overwritten, unless the filename is changed.
            behaviour == 1: the filname that you have passed as an argument will end with the time it was created. 
            behaviour != 0 or 1: if the value of behaviour is a value other than 0 or 1, behaviour will default to 0
        '''  

        SAVED_MODELS_FILE_PATH = default_file_dir

        if os.path.exists(SAVED_MODELS_FILE_PATH) is not True: os.makedirs(SAVED_MODELS_FILE_PATH)
        if os.path.exists(os.path.join(SAVED_MODELS_FILE_PATH, filename) + ".pickle") and behaviour == 1:
            now = datetime.datetime.now()
            current_time = now.strftime("-%H-%M-%S")
            filename += current_time

        filename = os.path.join(SAVED_MODELS_FILE_PATH, filename) + ".pickle" if ".pickle" not in filename else None

        store = open(filename, "wb")
        pickle.dump(self, store)
        store.close()
