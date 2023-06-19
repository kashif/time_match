import torch
import torch.nn as nn
import torch.nn.functional as F

from .optimal_transport import OTPlanSampler


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
        # TODO double check this
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

    def sample(
        self, scaled_past_target, hidden_state, loc=None, scale=None, sample_size=()
    ):
        """Sample from the model for the given batch conditioned on the hidden_state."""
        # TODO: implement
        # return dummy output

        return (scaled_past_target + loc) * scale


class SBCFM(nn.Module):
    """Implements a Schrodinger Bridge based conditional flow matching model."""

    def __init__(self, net=None, num_steps=1000, sig=0, eps=1e-3):
        super().__init__()
        self.net = net

        self.N = num_steps
        self.sig = sig
        self.eps = eps

        self.ot_sampler = OTPlanSampler(method="sinkhorn", reg=2 * sig**2)

    def loss(self, past_target, hidden_state, future_target, loc=None, scale=None):
        """Compute the loss for the given batch conditioned on hidden_state."""
        # TODO double check this
        scaled_past_target = (past_target - loc) / scale
        scaled_future_target = (future_target - loc) / scale
        z0, z1 = self.ot_sampler.sample_plan(scaled_past_target, scaled_future_target)

        batch, time, _ = scaled_past_target.shape
        random_time = (
            torch.rand((batch, time, 1), device=scaled_past_target.device)
            * (1 - 2 * self.eps)
            + self.eps
        )

        z_t = random_time * z1 + (1.0 - random_time) * z0
        z = torch.randn_like(z_t)
        z_t = z_t + self.sig * torch.sqrt(random_time * (1.0 - random_time)) * z
        target = z1 - z0
        target = (
            target
            - self.sig
            * (
                torch.sqrt(random_time) / torch.sqrt(1.0 - random_time)
                - 0.5 / torch.sqrt(random_time * (1.0 - random_time))
            )
            * z
        )

        outputs = self.net(
            inputs=z_t.reshape(batch * time, 1, -1),
            time=random_time.reshape(batch * time),
            cond=hidden_state.reshape(batch * time, 1, -1),
        )

        return F.mse_loss(outputs.reshape(batch, time, -1), target, reduction="none")

    def sample(
        self, scaled_past_target, hidden_state, loc=None, scale=None, sample_size=()
    ):
        """Sample from the model for the given batch conditioned on the hidden_state."""
        # TODO: Use Euler method to sample from the learned flow
        # return dummy output

        return (scaled_past_target + loc) * scale


class FM(nn.Module):
    """Implements a Lipman et al. 2023 style flow matching loss."""


class SplineCFM(nn.Module):
    """Implements cubic spline version of OT-CFM."""


class CNF(nn.Module):
    """Implements a conditional normalizing flow."""
