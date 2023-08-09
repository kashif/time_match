# ---------------------------------------------------------------
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for I2SB. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import torch
import torch.nn as nn
import math
from .gamma import Gamma
EPS = 1e-4

# +
class Interpolant(nn.Module):
    def __init__(self, gamma_fn):
        super(Interpolant, self).__init__()
        self.gamma_fn = gamma_fn

    def interpolate(self, x_0, x_1, t):
        return self.alpha(t) * x_0 + self.beta(t) * x_1

    def noisy_interpolate(self, x_0, x_1, t):
        z = torch.randn_like(x_0)
        return self.interpolate(x_0, x_1, t) + self.gamma(t) * z, z

    def dt_interpolate(self, x_0, x_1, t):
        return self.alpha_(t) * x_0 + self.beta_(t) * x_1
    
    def gamma(self, t):
        return self.gamma_fn.gamma(t)

    def gamma_(self, t):
        return self.gamma_fn.gamma_(t)

    def gamma_gamma(self, t):
        return self.gamma_fn.gamma_gamma(t)

class Linear(Interpolant):
    def __init__(self, gamma=Gamma()):
        super(Linear, self).__init__(gamma)
        pass

    def alpha(self, t):
        return 1. - t
    
    def alpha_(self, t):
        return -torch.ones_like(t)
    
    def beta(self, t):
        return t
    
    def beta_(self, t):
        return torch.ones_like(t)

class Trigonometric(Interpolant):
    def __init__(self, gamma=Gamma()):
        super(Trigonometric, self).__init__(gamma)
        pass

    def alpha(self, t):
        return torch.sqrt(1 - self.gamma(t) ** 2) * torch.cos(0.5 * math.pi * t)
    
    def alpha_(self, t):
        udv = - 0.5 * math.pi * torch.sqrt(1 - self.gamma(t) ** 2) * torch.sin(0.5 * math.pi * t)
        vdu = - self.gamma_gamma(t) * torch.cos(0.5 * math.pi * t) / (EPS + torch.sqrt(1 - self.gamma(t) ** 2))
        return udv + vdu
    
    def beta(self, t):
        return torch.sqrt(1 - self.gamma(t) ** 2) * torch.sin(0.5 * math.pi * t)
    
    def beta_(self, t):
        udv = 0.5 * math.pi * torch.sqrt(1 - self.gamma(t) ** 2) * torch.cos(0.5 * math.pi * t)
        vdu = - self.gamma_gamma(t) * torch.sin(0.5 * math.pi * t) / (EPS + torch.sqrt(1 - self.gamma(t) ** 2))
        return udv + vdu
    
class EncDec(Interpolant):
    def __init__(self, gamma=Gamma()):
        super(EncDec, self).__init__(gamma)
        pass

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
