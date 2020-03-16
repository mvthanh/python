class UnivarialNormal:
    def __init__(self, x = [1,2,3,4,5,6]):
        self.x = x

    def kyVong(self):
        sum = 0
        for xi in self.x:
            sum += xi
        return float(sum)/self.x.__len__()

    def phuongSai(self):
        u = self.kyVong()
        sum = 0
        for xi in self.x:
            sum += (xi - u)**2
        return float(sum)/self.x.__len__()

uni = UnivarialNormal()
print(uni.kyVong())
print(uni.phuongSai())