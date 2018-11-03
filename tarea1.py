import pandas as pd
import numpy as np
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


def approx_class(result):
    if result[0] > result[1] and result[0] > result[2]:
        return [1, 0, 0]
    elif result[1] > result[2]:
        return [0, 1, 0]
    else:
        return [0, 0, 1]


def get_f1_score(network):
    tp = np.zeros(3)
    fp = np.zeros(3)
    fn = np.zeros(3)
    index = {
        (1, 0, 0): 0,
        (0, 1, 0): 1,
        (0, 0, 1): 2
    }
    for i in range(data.shape[0]):
        input_data = data.iloc[i].values[0:4]
        expected = class_mapping[data.iloc[i].values[4]]
        network.feed(input_data)
        result = approx_class(network.get_output())
        if expected == result:
            tp[index[tuple(expected)]] += 1
        else:
            fp[index[tuple(result)]] += 1
            fn[index[tuple(expected)]] += 1
    precision = np.mean(tp / (tp + fp + 0.001))
    recall = np.mean(tp / (tp + fn + 0.001))
    return (precision + recall) / 2


network = NeuralNetwork(Sigmoid, 0.2, [4, 4, 3])
train_dataset(network, 10)
print(get_f1_score(network))
train_dataset(network, 10)
print(get_f1_score(network))
train_dataset(network, 10)
print(get_f1_score(network))
train_dataset(network, 10)
print(get_f1_score(network))
train_dataset(network, 10)
print(get_f1_score(network))
train_dataset(network, 10)
print(get_f1_score(network))
train_dataset(network, 10)
print(get_f1_score(network))
train_dataset(network, 10)
print(get_f1_score(network))
