from functions import passData, normalize, shuffle
from regression import MultivariateRegression


if __name__ == "__main__":
    x, y = passData('forestfires.csv')
    normalize(x)

    shuffleX, shuffleY = shuffle(x, y)
    epoch = 1000
    alpha = 0.01

    e1 = MultivariateRegression(shuffleX, shuffleY, epoch, alpha)
    e1.train()
