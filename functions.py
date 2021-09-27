import csv
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from variables import month, day


def passData(fileName):
    x = []
    y = []

    with open(fileName) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            x.append(list(map(float, [row['X'], row['Y'], month[row['month']], day[row['day']], row['FFMC'],
                     row['DMC'], row['DC'], row['ISI'], row['temp'], row['RH'], row['wind'], row['rain']])))

            y.append(float(row['area']))

    return np.array(x), np.array(y).reshape(-1, 1)


def normalize(data):
    scaler = MinMaxScaler()
    normalizeData = scaler.fit_transform(data)

    return normalizeData


def percentage(length, fraction):
    return int(length * fraction / 100)


def shuffle(x, y):
    p = np.random.permutation(len(x))
    return x[p], y[p]
