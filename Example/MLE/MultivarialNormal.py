import numpy as np

class MultivarialNormal:
    def __init__(self, X = np.array([[[1], [2], [3]], [[3], [4], [6]]])):
        self.X = X

    def kyVong(self):
        return self.X.mean(0)

    def MTHiepPSai(self):
        u = self.kyVong()
        sum = np.zeros((u.shape[0], u.shape[0]), dtype=np.float64)
        for xi in self.X:
            sum = sum + np.dot((xi - u), (xi - u).T)
        return sum / self.X.shape[0]


mul = MultivarialNormal()
print(mul.kyVong())
print(mul.MTHiepPSai())