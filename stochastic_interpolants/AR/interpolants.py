import torch
import math

class Interpolant():
    def __init__(self):
        super(Interpolant, self).__init__()
        pass

    def interpolate(self, x_0, x_1, t):
        return self.alpha(t) * x_0 + self.beta(t) * x_1
    
    def noisy_interpolate(self, x_0, x_1, t):
        z = torch.randn_like(x_0)
        return self.interpolate(x_0, x_1, t) + self.gamma(t) * z, z

    def dt_interpolate(self, x_0, x_1, t):
        return self.alpha_(t) * x_0 + self.beta_(t) * x_1

class TrigonometricODEInterpolants(Interpolant):
    def __init__(self):
        super(TrigonometricODEInterpolants, self).__init__()
        pass

    def gamma(self, t):
        return torch.zeros_like(t)

    def gamma_gamma(self, t):
        return torch.zeros_like(t)

    def gamma_(self, t):
        return torch.zeros_like(t)

    def alpha(self, t):
        return torch.cos(0.5 * math.pi * t)
    
    def alpha_(self, t):
        return -0.5 * math.pi * torch.sin(0.5 * math.pi * t)
    
    def beta(self, t):
        return torch.sin(0.5 * math.pi * t)
    
    def beta_(self, t):
        return 0.5 * math.pi * torch.cos(0.5 * math.pi * t)
    
class EncDecInterpolant(Interpolant):
    def __init__(self):
        super(EncDecInterpolant, self).__init__()
        pass

    def gamma(self, t):
        return torch.sin(math.pi * t) ** 2

    def gamma_gamma(self, t):
        return 2 * math.pi * (torch.sin(math.pi * t) ** 3) * torch.cos(math.pi * t)    
    
    def gamma_(self, t):
        return 2 * math.pi * torch.sin(math.pi * t) * torch.cos(math.pi * t)

    def alpha(self, t):
        indicator = (t <= 0.5).float()
        return (torch.cos(math.pi * t) ** 2) * indicator
    
    def alpha_(self, t):
        indicator = (t <= 0.5).float()
        return -(2 * math.pi * torch.cos(math.pi * t) * torch.sin(math.pi * t)) * indicator
    
    def beta(self, t):
        indicator = (t > 0.5).float()
        return (torch.cos(math.pi * t) ** 2) * indicator
    
    def beta_(self, t):
        indicator = (t > 0.5).float()
        return -(2 * math.pi * torch.cos(math.pi * t) * torch.sin(math.pi * t)) * indicator