import numpy as np
from numpy.random import shuffle
from enum import Enum
import matplotlib.pyplot as plt
path = "../phishing.data"
params_count = 30

def load_file(name, train_frac = 0.6, valid_frac = 0.2):
    return categorize_and_split(open_file(name), train_frac, valid_frac)

def open_file(name):
    data = []
    with open(name, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            X = list(map(int, parts[:params_count]))
            Y = int(parts[params_count])
            data.append((X, Y))
    shuffle(data)
    return data

class Dataset(Enum):
    Valid = 1,
    Test = 2

def sign(x):
    if x >= 0:
        return 1
    return -1

def plot_data(X, Y, label, xlabel, ylabel, log = True, dim = (12, 5), ticks = False, color='cornflowerblue'):
    plt.figure(figsize=dim)
    plt.plot(X, Y, marker='o', color=color)
    plt.title(label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if log:
        plt.xscale('log')
    if ticks:
        plt.xticks(X, [x for x in X])
    plt.grid(True, which="both", linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

def categorize_and_split(data, train = 0.6, valid = 0.2):
    negative = [x[0] for x in data if x[1] == -1]
    positive = [x[0] for x in data if x[1] == 1]
    shuffle(positive)
    shuffle(negative)
    neg_train = int(len(negative) * train)
    pos_train = int(len(positive) * train)
    neg_valid = int(len(negative) * (train + valid))
    pos_valid = int(len(positive) * (train + valid))
    train_set = [(x, -1) for x in negative[:neg_train]] + [(x, 1) for x in positive[:pos_train]]
    valid_set = [(x, -1) for x in negative[neg_train:neg_valid]] + [(x, 1) for x in positive[pos_train:pos_valid]]
    test_set = [(x, -1) for x in negative[neg_valid:]] + [(x, 1) for x in positive[pos_valid:]]

    shuffle(train_set)
    shuffle(valid_set)
    shuffle(test_set)
    return train_set, valid_set, test_set


def accuracy(Y_pred, Y):
    total = 0
    true_positive = 0
    false_negative = 0
    false_positive = 0
    true_negative = 0
    for i in range(len(Y_pred)):
        if Y_pred[i] == -1 and Y[i] == 1:
            false_negative += 1
        elif Y_pred[i] == 1 and Y[i] == -1:
            false_positive += 1
        elif Y_pred[i] == -1 and Y[i] == -1:
            true_negative += 1
        else:
            true_positive += 1

        total += 1

    accuracy = (true_positive + true_negative ) / total
    precision = true_positive / (true_positive + false_positive)
    sensitivity = true_positive / (true_positive + false_negative)
    specificity = (true_negative) / (true_negative + false_positive)
    return accuracy, precision, sensitivity, specificity