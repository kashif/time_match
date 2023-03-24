import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal

from torchdyn.core import NeuralODE
from torchcubicspline import NaturalCubicSpline, natural_cubic_spline_coeffs


class CFM(nn.Module):
    """Conditional Flow Matching Module."""

    def __init__(
        self,
        unet,
        dim,
        sigma_min: float = 0.1,
        avg_size: int = -1,
        leaveout_timepoint: int = -1,
    ):
        super().__init__()
        self.dim = dim
        self.net = unet
        self.sigma_min = sigma_min

    def loss(self, past_target, hidden_state, future_target, loc=None, scale=None):
        """Compute the loss for the given batch conditioned on hidden_state."""
        scaled_past_target = (past_target - loc) / scale
        scaled_future_target = (future_target - loc) / scale

        batch, time, _ = scaled_past_target.shape
        random_time = torch.rand((batch, time, 1), device=scaled_past_target.device)

        ut = scaled_future_target - scaled_past_target
        mu_t = (
            random_time * scaled_future_target + (1 - random_time) * scaled_past_target
        )
        x = mu_t + self.sigma_min * torch.randn_like(scaled_past_target)

        outputs = self.net(
            inputs=x.reshape(batch * time, 1, -1),
            time=random_time.reshape(batch * time),
            cond=hidden_state.reshape(batch * time, 1, -1),
        )

        return F.mse_loss(outputs.reshape(batch, time, -1), ut, reduction="none")

    def sample(self, scaled_past_target, hidden_state, loc=None, scale=None):
        """Sample from the model for the given batch conditioned on the hidden_state."""
        # TODO: implement
        # return dummy output
        return (scaled_past_target + loc) * scale


class SBCFM(nn.Module):
    """Implements a Schrodinger Bridge based conditional flow matching model."""


class FM(nn.Module):
    """Implements a Lipman et al. 2023 style flow matching loss."""


class SplineCFM(nn.Module):
    """Implements cubic spline version of OT-CFM."""


class CNF(nn.Module):
    """Implements a conditional normalizing flow."""
