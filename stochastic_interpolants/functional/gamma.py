import torch
import math
import torch.nn as nn

EPS=1e-4
class Gamma(nn.Module):
    def __init__(self):
        super(Gamma, self).__init__()
        pass

    def gamma(self, t):
        return torch.zeros_like(t)

    def gamma_(self, t):
        return torch.zeros_like(t)

    def gamma_gamma(self, t):
        return self.gamma(t) * self.gamma_(t)

class SqrtGamma(Gamma):
    def __init__(self):
        super(SqrtGamma, self).__init__()
        self.a = 0.4

    def gamma(self, t):
        return torch.sqrt(self.a * t * (1. - t))

    def gamma_(self, t):
        return self.gamma_gamma(t) / (self.gamma(t) + EPS)

    def gamma_gamma(self, t):
        return 0.5 * self.a * (1. - 2 * t)

class QuadGamma(Gamma):
    def __init__(self):
        super(QuadGamma, self).__init__()
        self.a = 0.8

    def gamma(self, t):
        return self.a * t * (1-t)

    def gamma_(self, t):
        return self.a * (1. - 2 * t)

class TrigGamma(Gamma):
    def __init__(self):
        super(TrigGamma, self).__init__()
        self.a = 0.4

    def gamma(self, t):
        return self.a * torch.sin(math.pi * t) ** 2
    
    def gamma_(self, t):
        return 2 * self.a * math.pi * torch.sin(math.pi * t) * torch.cos(math.pi * t)
