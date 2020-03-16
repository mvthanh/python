class Bernoulli:
    def __init__(self, head = 1, sum = 1):
        self.head = head
        self.sum = sum
    
    def lamda(self):
        return 1.0*self.head/self.sum


ber = Bernoulli(7,9)
print(ber.lamda())
