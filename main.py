from matplotlib import pyplot as plt
from nn.activations.ActivationBlock import ActivationBlock
from nn.activations.Sigmoid import Sigmoid
from nn.layers.Softmax import Softmax
from nn.layers.Dense import Dense
from nn.activations.Relu import Relu
from nn.activations.Tanh import Tanh
from nn.losses.CategoricalCrossEntropy import CategoricalCrossEntropy
from nn.models.Model import Model
from nn.optimisers.Adam import Adam
from nn.optimisers.Momentum import Momentum
from nn.optimisers.RMSProp import RMSProp
from nn.optimisers.SGD import SGD
from nn.utils import one_hot_array

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from random import randint


# preprocessing
df = pd.read_csv("dataset\mnist_train.csv")

numpydf = df.to_numpy()
np.random.shuffle(numpydf)

batch = numpydf.T[1:].T[0:100]
labels = numpydf.T[0].T[0:100]

one_hot_labels = one_hot_array(labels)

# training
layers = [Dense(784, 64, activation=Tanh()),
          Dense(64, 64, activation=Tanh()),
          Dense(64, 10),
          Softmax()]

model = Model(layers=layers)

model.build(loss=CategoricalCrossEntropy(), optimiser=Adam())

model.fit(batch, one_hot_labels, epochs=1000)

model.save("model-adam", behaviour=0)

# ~ Graphs the error at each epoch
model.graph()

# predictions
randidx = randint(0, 99)

input_value = np.reshape(batch[randidx], newshape=(28, 28))
predicted_values = model.predict(batch[randidx])
argmax_predicted = np.argmax(predicted_values)
classes = np.unique(labels)

print(np.squeeze(predicted_values), np.squeeze(one_hot_labels[randidx]))

fig, axes = plt.subplots(nrows=2, ncols=1)

print(argmax_predicted == labels[randidx])

# change the colour depending on the predicted output
colour = np.repeat("b", predicted_values.size)

if argmax_predicted == labels[randidx]: colour[argmax_predicted] = "g"
if argmax_predicted != labels[randidx]: 
        colour[argmax_predicted] = "r"
        colour[labels[randidx]] = "g"

# plot the data
axes[0].imshow(np.squeeze(input_value))
axes[1].bar(classes, np.squeeze(predicted_values), color=colour)

plt.xticks(classes)
plt.show()

# print(model.predict(batch[randidx]), np.argmax(labels[randidx]), np.argmax(model.predict(batch[randidx])))

