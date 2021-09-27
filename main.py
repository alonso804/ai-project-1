import numpy as np
from functions import passData, normalize, shuffle
from regression import MultivariateRegression


if __name__ == "__main__":
    x, y = passData('forestfires.csv')
    # x[:, :] = normalize(x)
    x[:, [0, 1, 4, 5, 6, 7, 8, 9, 10, 11]] = normalize(
        x[:, [0, 1, 4, 5, 6, 7, 8, 9, 10, 11]])

    y = normalize(y)

    np.random.seed(0)
    shuffleX, shuffleY = shuffle(x, y)
    epoch = 3000
    alpha = 0.01

    e1 = MultivariateRegression(shuffleX, shuffleY, epoch, alpha)
    e1.train()
