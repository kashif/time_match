import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal


class AlphaBlend(nn.Module):
    """Conditional AlphaBlend Module."""

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
        alpha = torch.rand((batch, time), device=scaled_past_target.device)
        x_alpha = (
            alpha.view(-1, time, 1) * scaled_future_target
            + (1 - alpha).view(-1, time, 1) * scaled_past_target
        )

        loc, scale = self.net(
            inputs=x_alpha.reshape(batch * time, 1, -1),
            time=alpha.reshape(batch * time),
            cond=hidden_state.reshape(batch * time, 1, -1),
        )

        target = scaled_future_target - scaled_past_target

        normal = Normal(
            loc=loc.view(batch, time, -1), scale=scale.view(batch, time, -1)
        )
        return -normal.log_prob(target)

    def sample(
        self, scaled_past_target, hidden_state, loc=None, scale=None, sample_size=()
    ):
        """Sample from the model for the given batch conditioned on the hidden_state."""
        # TODO: implement
        # return dummy output

        return (scaled_past_target + loc) * scale
