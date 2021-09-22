import numpy as np


def percentage(length, fraction):
    return int(length * fraction / 100)


class MultivariateRegression:
    def __init__(self, dimensions, x, y, epoch, alpha):
        self.dimensions = dimensions
        self.x = x
        self.y = y
        self.epoch = epoch
        self.alpha = alpha

    def hypothesis(self, w, b, x):
        h = np.dot(w, x)
        h += b

        return h

    def derivate(self):
        pass

    def error(self):
        pass

    def train(self):
        w = [np.random.rand() for i in range(self.dimensions)]
        b = np.random.rand()

        err = self.error(w, b, self.x)

        for i in range(self.epoch):
            db, dw = self.derivative(w, b, self.x)

            b, w = self.update(b, db, w, dw)

            err = self.error(w, b, self.x)

    def plotError(self, errorList):
        pass


if __name__ == "__main__":
    pass


m = 4
k = 100

x = []

for i in range(m):
    temp = []
    for j in range(k):
        temp.append(np.random.normal(0, 0.))
    x.append(temp)

y = np.array([j + np.random.normal(0, 0.5) for j in range(k)])


def derivada(var_depen, b):
    deva = [0] * m
    db = 0.0
    for i in range(k):
        sum_a = b
        sum_b = b
        for j in range(m):
            sum_a = sum_a+var_depen[j]*x[j][i]
            sum_b = sum_b+var_depen[j]*x[j][i]
        for j in range(m):
            deva[j] = deva[j]-((y[j]-sum_a)*x[j][i])
        db = db-(y[i]-sum_b)

    for i in range(m):
        deva[i] = deva[i]/k
    db = db/k
    return deva, db


def err(var_depen, b):
    error = 0.0
    for i in range(k):
        sum_a = b
        for j in range(m):
            sum_a = sum_a+var_depen[j]*x[j][i]

        for j in range(m):
            error = error+(y[j]-sum_a)**2

    error = error / (2 * k)
    return error


def train(alfa, umbral):
    # a ind
    # bias
    var_depen = []
    for i in range(m):
        var_depen.append(np.random.rand())
    b = np.random.rand()
    error = err(var_depen, b)
    print(error)
    ciclos = 0
    while(error > umbral and ciclos < 15):
        deva, db = derivada(var_depen, b)
        b = b - db * alfa
        for i in range(m):
            var_depen[i] = var_depen[i] - deva[i] * alfa
        error = err(var_depen, b)
        print("error", error)
        ciclos += 1
    for i in var_depen:
        print(i)
    # (var_depen, b)


train(0.0005, 0.1)
