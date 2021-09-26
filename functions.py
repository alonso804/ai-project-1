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

    return np.array(x), np.array(y)


def normalize(x):
    scaler = MinMaxScaler()
    # x[:, :] = scaler.fit_transform(x)
    x[:, [0, 1, 4, 5, 6, 7, 8, 9, 10, 11]] = scaler.fit_transform(
        x[:, [0, 1, 4, 5, 6, 7, 8, 9, 10, 11]])


def percentage(length, fraction):
    return int(length * fraction / 100)


def shuffle(x, y):
    p = np.random.permutation(len(x))
    return x[p], y[p]
