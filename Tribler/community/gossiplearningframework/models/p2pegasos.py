import numpy as np
from models import GossipLearningModel

if __debug__:
    from Tribler.dispersy.dprint import dprint

class P2PegasosModel(GossipLearningModel):

    def __init__(self):
        super(P2PegasosModel, self).__init__()

        # Initial model
        self.w = np.array([0])
        self.age = 0

    def update(self, x, y):
        """Update the model with a new training example."""
        # Set up some variables.
        label = -1.0 if y == 0 else 1.0
        
        self.age = self.age + 1
        lam = 0.0001
        rate = 1.0 / (self.age * lam)
        
        import sys
        print >> sys.stderr, "w", type(self.w), len(self.w), self.w
        print >> sys.stderr, "x", type(x), len(x), x
        
        _sum = sum([self.w[i] * x[i] for i in range(min(len(self.w), len(x)))])
        is_sv = label * _sum < 1.0
        self.w = self.w * (1.0 - 1.0 / self.age)
        if is_sv:
            self.w = self.w + (rate * label * x)

    def predict(self, x):
        """
        Compute the inner product of the hyperplane and the instance as a
        prediction.
        """
        wx = sum([self.w[i] * x[i] for i in range(min(len(self.w), len(x)))])
        return 1.0 if wx >= 0.0 else 0.0

    def merge(self, model):
        self.age = max(self.age, model.age)
        self.w = (self.w + model.w) / 2.0

