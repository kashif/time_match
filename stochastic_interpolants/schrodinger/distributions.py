import torch
import torch.nn as nn

class Gaussian(nn.Module):
    def __init__(self, dim):
        super(Gaussian, self).__init__()
        self.dim = dim
        self.mean = nn.Parameter(torch.zeros(dim))
        self.logstd = nn.Parameter(torch.zeros(dim))
    
    @property
    def var(self):
        return (2 * self.logstd).exp()

    def sample(self, batch_size):
        z = torch.randn(batch_size, self.dim).to(self.mean)
        return self.mean.unsqueeze(0) + self.logstd.exp().unsqueeze(0) * z