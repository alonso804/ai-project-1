import csv
import numpy as np
from sklearn import preprocessing
from variables import month, day


def passData(fileName):
    x = []
    y = []
    with open(fileName) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:

            x.append(list(map(float, [row['X'], row['Y'], month[row['month']], day[row['day']], row['FFMC'], row['DMC'],
                     row['DC'], row['ISI'], row['temp'], row['RH'], row['wind'], row['rain']])))
            y.append(float(row['area']))

    return np.array(x), np.array(y)


def normalizeColumn(data):
    minimum = min(data)
    maximum = max(data)
    for i in range(len(data)):
        data[i] = (float(data[i]) - minimum) / (maximum - minimum)


def normalize(x, k):
    for i in range(k):
        normalizeColumn(x[:, i])

    # print(x)


def normalize2(x):
    x = preprocessing.normalize(x, norm='max', axis=0)
    print(x)


def percentage(length, fraction):
    return int(length * fraction / 100)
