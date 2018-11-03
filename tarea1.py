import pandas as pd
from NeuralNetwork import NeuralNetwork
from Neurons import Sigmoid
import time


def current_milli_time():
    return int(round(time.time() * 1000))


def normalize(dL, dH, nL, nH):
    return lambda x: (float(x) - dL) * (nH - nL) / (dH - dL) + nL


file = "Datasets/iris.data"
data = pd.read_csv(file)
data.columns = ["sepal length", "sepal width", "petal length", "petal width", "class"]

nL = 0
nH = 1

dL = data["sepal length"].min()
dH = data["sepal length"].max()
data["sepal length"] = data["sepal length"].apply(normalize(dL, dH, nL, nH))

dL = data["sepal width"].min()
dH = data["sepal width"].max()
data["sepal width"] = data["sepal width"].apply(normalize(dL, dH, nL, nH))

dL = data["petal length"].min()
dH = data["petal length"].max()
data["petal length"] = data["petal length"].apply(normalize(dL, dH, nL, nH))

dL = data["petal width"].min()
dH = data["petal width"].max()
data["petal width"] = data["petal width"].apply(normalize(dL, dH, nL, nH))

class_mapping = {
    "Iris-setosa": [1, 0, 0],
    "Iris-versicolor": [0, 1, 0],
    "Iris-virginica": [0, 0, 1]
}


def train_dataset(network, n):
    for i in range(n):
        for j in range(data.shape[0]):
            input_data = data.iloc[j].values[0:4]
            expected = class_mapping[data.iloc[j].values[4]]
            network.train(input_data, expected)
