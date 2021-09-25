import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn import preprocessing

month = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12
}

day = {
    "mon": 1,
    "tue": 2,
    "wed": 3,
    "thu": 4,
    "fri": 5,
    "sat": 6,
    "sun": 7
}


def passData():
    x = []
    y = []
    with open('forestfires.csv') as csvfile:
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


class MultivariateRegression:
    def __init__(self, k, x, y, epoch, alpha):
        self.x = x
        self.y = y
        self.epoch = epoch
        self.alpha = alpha
        self.k = k

    def hypothesis(self, w, b, x):
        """
        h = 0
        for i in range(len(w)):
            h += w[i] * x[i]
        """
        h = np.dot(w, x)

        h += b

        return h

    def derivate(self, w, b, x):
        m = len(x)

        dw = [0] * self.k
        db = 0

        for i in range(m):
            db += (y[i] - self.hypothesis(w, b, x[i])) * (-1)

            for j in range(self.k):
                dw[j] += (y[i] - self.hypothesis(w, b, x[i])) * \
                    (-self.x[i][j])

            for j in range(self.k):
                dw[j] /= m

        db /= m
        return db, dw

    def error(self, w, b, x):
        err = 0
        m = len(x)
        # print(type(x[0]))

        for i in range(m):
            err += y[i] - self.hypothesis(w, b, x[i])
            #err += y[i] - (np.dot(w, x[i]) + b)

        err /= (2 * m)

        return err

    def update(self, b, db, w, dw):
        for i in range(len(w)):
            w[i] -= self.alpha * dw[i]

        b -= self.alpha * db
        return b, w

    def train(self):
        w = [np.random.rand() for i in range(self.k)]
        b = np.random.rand()

        err = self.error(w, b, self.x)
        errorList = [err]

        for i in range(self.epoch):
            db, dw = self.derivate(w, b, self.x)

            b, w = self.update(b, db, w, dw)

            err = self.error(w, b, self.x)
            errorList.append(err)

        plt.plot(errorList)
        plt.show()

    def plotError(self, errorList):
        plt.plot(errorList)
        plt.show()


if __name__ == "__main__":

    x, y = passData()
    # print(x)
    # print()
    normalize(x, 12)

    e1 = MultivariateRegression(12, x, y, 1000, 0.01)
    e1.train()
