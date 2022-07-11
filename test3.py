import random
import numpy as np
from nn.utils import load_model, one_hot_array
import pandas as pd
from nn.models.Model import Model


model: Model = load_model("nn\saved_models\model-adam.pickle")

df = pd.read_csv("dataset\mnist_train.csv")

numpydf = df.to_numpy().T[1:].T[0]
batch = df.to_numpy().T[1:].T[0:1000]
labels = df.to_numpy().T[0].T[0:1000]
one_hot_labels = one_hot_array(labels)

print(batch.shape)

tests = int(input("how many tests would you like to do?: "))

correct = 0
count = 0


for i in range(tests):
	randidx = random.randint(0, 99)

	predictions = np.squeeze(model.predict(batch[randidx]))
	pred_val = np.argmax(predictions)

	actual = np.argmax(one_hot_labels[randidx])

	if actual == pred_val: 
		count += 1
		correct += 1
		print(f"{(correct / count) * 100}% accuracy")
	else: 
		count += 1
		print(f"{(correct / count) * 100}% accuracy")

	# print(np.squeeze(predictions), one_hot_labels[randidx])

	