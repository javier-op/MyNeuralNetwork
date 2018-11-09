import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
    "Iris-setosa": (1, 0, 0),
    "Iris-versicolor": (0, 1, 0),
    "Iris-virginica": (0, 0, 1)
}


def train_dataset(network, n=1):
    for i in range(n):
        for j in range(data.shape[0]):
            input_data = data.iloc[j].values[0:4]
            expected = class_mapping[data.iloc[j].values[4]]
            network.train(input_data, expected)


def train_shuffled_dataset(network, n=1):
    shuffled = data.sample(frac=1)
    for i in range(n):
        for j in range(shuffled.shape[0]):
            input_data = shuffled.iloc[j].values[0:4]
            expected = class_mapping[shuffled.iloc[j].values[4]]
            network.train(input_data, expected)


def approx_class(result):
    if result[2] > result[0] and result[2] > result[1]:
        return 0, 0, 1
    elif result[1] > result[0]:
        return 0, 1, 0
    else:
        return 1, 0, 0


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
            tp[index[expected]] += 1
        else:
            fp[index[result]] += 1
            fn[index[expected]] += 1
    precision = np.mean(tp / (tp + fp + 0.0001))
    recall = np.mean(tp / (tp + fn + 0.0001))
    return (precision + recall) / 2


def f1_score_plot():
    n_samples = 5
    x_len = 10
    x_increment = 5
    f1_scores = []
    for i in range(n_samples):
        network = NeuralNetwork(Sigmoid, 0.2, [4, 4, 4, 3])
        current_scores = []
        for j in range(x_len):
            train_dataset(network, x_increment)
            current_scores.append(get_f1_score(network))
        f1_scores.append(current_scores)
    x = np.linspace(x_increment, x_increment*x_len, x_len)
    y = np.mean(f1_scores, axis=0)
    fig = plt.figure(figsize=(4, 4))
    plt.plot(x, y)
    plt.xlabel('epochs')
    plt.ylabel('f1_score')
    fig.savefig('f1_score', bbox_inches='tight')


def speed_plot():
    n_samples = 3
    x_len = 10
    x_increment = 1
    time_samples = []
    for i in range(n_samples):
        network = NeuralNetwork(Sigmoid, 0.2, [4, 4, 4, 3])
        current_sample = [0]
        start_time = current_milli_time()
        for j in range(x_len):
            train_dataset(network, x_increment)
            current_time = current_milli_time()
            current_sample.append(current_time - start_time)
        time_samples.append(current_sample)
    x = np.linspace(0, x_increment*x_len, x_len+1)
    y = np.mean(time_samples, axis=0)
    fig = plt.figure(figsize=(4, 4))
    plt.plot(x, y)
    plt.xlabel('epochs')
    plt.ylabel('milisegundos')
    fig.savefig('tiempo', bbox_inches='tight')


def hidden_layers_plot():
    n_samples = 10
    x_len = 10
    x_increment = 5
    shape = [4, 4]
    f1_scores = []

    for i in range(3):
        shape.append(3)
        samples = []
        for j in range(10):
            network = NeuralNetwork(Sigmoid, 0.2, shape)
            current_score = []
            for k in range(x_len):
                train_dataset(network, x_increment)
                current_score.append(get_f1_score(network))
            samples.append(current_score)
        samples = np.array(samples)
        f1_scores.append(np.mean(samples, axis=0))
    fig = plt.figure(figsize=(9, 3))
    x = np.linspace(x_increment, x_increment*x_len, x_len)
    plt.subplot(1, 3, 1)
    plt.plot(x, f1_scores[0])
    plt.xlabel('epochs')
    plt.ylabel('f1_score')
    plt.title('0 hidden layers')
    plt.subplot(1, 3, 2)
    plt.plot(x, f1_scores[1])
    plt.xlabel('epochs')
    plt.ylabel('f1_score')
    plt.title('1 hidden layer')
    plt.subplot(1, 3, 3)
    plt.plot(x, f1_scores[2])
    plt.xlabel('epochs')
    plt.ylabel('f1_score')
    plt.title('2 hidden layers')
    fig.savefig('hidden_layers', bbox_inches='tight')


def learning_rate_plot():
    lrs = [0.001, 0.1, 0.2, 0.5, 1, 5]
    n_samples = 5
    x_len = 10
    x_increment = 5
    f1_scores = []
    for lr in lrs:
        samples = []
        for i in range(n_samples):
            network = NeuralNetwork(Sigmoid, lr, [4, 4, 4, 3])
            current_scores = []
            for j in range(x_len):
                train_dataset(network, x_increment)
                current_scores.append(get_f1_score(network))
            samples.append(current_scores)
        f1_scores.append(np.mean(samples, axis=0))
    x = np.linspace(x_increment, x_increment*x_len, x_len)
    fig = plt.figure(figsize=(18, 12))
    for i in range(6):
        ax = plt.subplot(2, 3, i+1)
        ax.set_ylim([0, 1])
        plt.plot(x, f1_scores[i])
        plt.xlabel('epochs')
        plt.ylabel('f1_score')
        plt.title('lr = ' + str(lrs[i]))
    fig.savefig('learning_rate', bbox_inches='tight')


learning_rate_plot()

