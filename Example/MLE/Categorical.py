class Categorical:
    def __init__(self, n = [1,2,1,1,1,1]):
        self.n = n


    def lamda(self, iface):
        N = 0
        for i in self.n:
            N += i
        return 1.0*self.n[iface-1]/N

cat = Categorical()
print(cat.lamda(2))