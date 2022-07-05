import numpy as np
import pickle


# for a single value only
def one_hot(value, num_classes: int):
    arr = np.zeros(shape=(num_classes))
    arr[value - 1] = 1
    return arr


# one hot encodes a batch of arrays
def one_hot_array(arr: np.ndarray):

    assert len(arr.shape) == 1

    classes = np.unique(arr)
    output = []
    arr = np.squeeze(arr)

    for i in arr:
        array = np.zeros(shape=(classes.size))

        for idx, _class in enumerate(classes):
            if i == _class: array[idx] = 1
            else: pass
        
        output.append(array)

    return np.array(output)


# loads a saved model pickle file
def load_model(filename):
	if ".pickle" in filename: pass
	if ".pickle" not in filename: filename += ".pickle" 

	model_file = open(filename, "rb")
	model = (pickle.load(model_file))
	model_file.close

	return model


if __name__ == "__main__":
    arr = np.array([1, 2, 2, 3, 1, 5, 2, 3, 3, 1, 2, 1, 3, 1, 2])
    arr2 = np.array(["nice", "this", "nice", "nice", "no", "this"])

    print(one_hot_array(arr))