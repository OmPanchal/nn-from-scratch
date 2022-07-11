from pickletools import optimize
from random import randint
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from nn.losses.CategoricalCrossEntropy import CategoricalCrossEntropy
from nn.optimisers.Momentum import Momentum
from nn.utils import load_model, one_hot_array


model = load_model("nn\saved_models\model-momentum.pickle")

df = pd.read_csv("dataset\mnist_train.csv")


numpydf = df.to_numpy().T[1:].T[0]
batch = df.to_numpy().T[1:].T[0:100]
labels = df.to_numpy().T[0].T[0:100]
one_hot_labels = one_hot_array(labels)

model.build(loss=CategoricalCrossEntropy(), optimiser=Momentum())

model.fit(batch, one_hot_labels, epochs=500)


model.save("model-momentum", behaviour=0)