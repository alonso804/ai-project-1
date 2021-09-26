import numpy as np
import matplotlib.pyplot as plt
from functions import percentage


class MultivariateRegression:
    def __init__(self, x, y, epoch, alpha):
        self.k = len(x[0])
        self.x = x
        self.y = y
        self.epoch = epoch
        self.alpha = alpha

        rowsAmount = len(x)

        self.xTrain = x[:percentage(rowsAmount, 70)]

        self.xValidation = x[percentage(
            rowsAmount, 70):percentage(rowsAmount, 90)]

        self.xTest = x[percentage(rowsAmount, 90):]

    def hypothesis(self, w, b, x):
        return np.dot(w, x) + b

    def derivate(self, w, b, x):
        m = len(x)

        dw = [0] * self.k
        db = 0

        for i in range(m):
            db += (self.y[i] - self.hypothesis(w, b, x[i])) * (-1)

            for j in range(self.k):
                dw[j] += (self.y[i] - self.hypothesis(w, b, x[i])) * \
                    (-self.x[i][j])

            for j in range(self.k):
                dw[j] /= m

        db /= m
        return db, dw

    def error(self, w, b, x):
        err = 0
        m = len(x)

        for i in range(m):
            err += self.y[i] - self.hypothesis(w, b, x[i])

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

        errTrain = self.error(w, b, self.xTrain)
        errValidation = self.error(w, b, self.xValidation)

        errorListTrain = [errTrain]
        errorListValidation = [errValidation]
        errorListTest = []

        for i in range(self.epoch):
            print(i)
            db, dw = self.derivate(w, b, self.x)

            b, w = self.update(b, db, w, dw)

            errTrain = self.error(w, b, self.xTrain)
            errValidation = self.error(w, b, self.xValidation)
            errTest = self.error(w, b, self.xTest)

            errorListTrain.append(errTrain)
            errorListValidation.append(errValidation)
            errorListTest.append(errTest)

        plt.plot(errorListTrain, label="Training")
        plt.plot(errorListValidation, label="Validation")
        plt.plot(errorListTest, label="Testing")

        plt.legend()
        plt.show()
