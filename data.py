import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("dataset\mnist_train.csv")

plt.imshow(np.reshape(df.to_numpy()[0]), [28, 28])
plt.show()