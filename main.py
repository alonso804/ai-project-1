from functions import passData, normalize, normalizeY, shuffle
from regression import MultivariateRegression


if __name__ == "__main__":
    x, y = passData('forestfires.csv')
    normalize(x)
    y = normalizeY(y)

    shuffleX, shuffleY = shuffle(x, y)
    epoch = 1000
    alpha = 0.0024

    e1 = MultivariateRegression(shuffleX, shuffleY, epoch, alpha)
    e1.train()
