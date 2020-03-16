class Beta:
    def __init__(self, a, b, n, N):
        self.a = a
        self.b = b
        self.n = n
        self.N = N

    def lamda(self):
        return float(self.n + self.a -1)/(self.N + self.a + self.b -2)

beta = Beta(11, 10, 1, 5)
print(beta.lamda())