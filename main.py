from functions import passData, normalize
from regression import MultivariateRegression


if __name__ == "__main__":

    x, y = passData('forestfires.csv')
    normalize(x, 12)

    e1 = MultivariateRegression(12, x, y, 1000, 0.01)
    e1.train()
