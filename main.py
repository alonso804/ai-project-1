import numpy as np
import csv
import matplotlib.pyplot as plt

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


def passData(x, y):
    with open('forestfires.csv') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            x.append([float(row['X']), float(row['Y']), float(month[row['month']]), float(day[row['day']]), float(row['FFMC']), float(row['DMC']),
                     float(row['DC']), float(row['ISI']), float(row['temp']), float(row['RH']), float(row['wind']), float(row['rain'])])
            y.append(float(row['area']))
        x = np.array(x)
        y = np.array(y)

def normalize(data):
    data = (np.amax(data) - data) / (np.amax(data) - np.amin(data))

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
        # h = np.dot(w, x)
        h = 0
        for i in range(len(w)):
            h += w[i] * x[i]
        h += b

        return h

    def derivate(self, w, b, x):
        m = len(x)

        dw = [0] * self.k
        db = 0

        for i in range(m):
            db += (y[i] - self.hypothesis(w, b, self.x[i])) * (-1)

            for j in range(self.k):
                dw[j] += (y[i] - self.hypothesis(w, b, self.x[i])) * \
                    (-self.x[i][j])

        db /= m
        dw /= m
        return db, dw

    def error(self, w, b, x):
        err = 0
        m = len(x)

        for i in range(m):
            err += y[i] - self.hypothesis(w, b, x[i])

        err /= (2 * m)

        return err

    def train(self):
        w = [np.random.rand() for i in range(self.k)]
        b = np.random.rand()

        normalize(x)
        err = self.error(w, b, self.x)
        errorList = []

        for i in range(self.epoch):
            db, dw = self.derivative(w, b, self.x)

            b, w = self.update(b, db, w, dw)

            err = self.error(w, b, self.x)
            errorList.append(err)
            self.plotError(errorList)

    def plotError(self, errorList):
        plt.plot(errorList)
        plt.show


if __name__ == "__main__":

    x = []
    y = []

    passData(x, y)

    e1 = MultivariateRegression(12, x, y, 1000, 0.01)
    e1.train()
