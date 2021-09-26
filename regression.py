import matplotlib.pyplot as plt
import numpy as np


class MultivariateRegression:
    def __init__(self, k, x, y, epoch, alpha):
        self.x = x
        self.y = y
        self.epoch = epoch
        self.alpha = alpha
        self.k = k

    def hypothesis(self, w, b, x):
        h = np.dot(w, x)

        h += b

        return h

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
