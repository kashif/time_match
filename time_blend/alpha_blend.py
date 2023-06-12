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
        nb_step: int = 1000,
    ):
        super().__init__()
        self.dim = dim
        self.net = unet
        self.nb_step = nb_step

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

        mu, sigma = self.net(
            inputs=x_alpha.reshape(batch * time, 1, -1),
            time=alpha.reshape(batch * time),
            cond=hidden_state.reshape(batch * time, 1, -1),
        )

        target = scaled_future_target - scaled_past_target

        normal = Normal(loc=mu.view(batch, time, -1), scale=sigma.view(batch, time, -1))
        return -normal.log_prob(target)

    @torch.no_grad()
    def sample(self, scaled_past_target, hidden_state, loc=None, scale=None):
        batch, _, _ = scaled_past_target.shape
        x_alpha = scaled_past_target
        for t in range(self.nb_step):
            alpha_start = t / self.nb_step
            alpha_end = (t + 1) / self.nb_step

            mu, sigma = self.net(
                inputs=x_alpha,
                time=torch.tensor([alpha_start], device=x_alpha.device).expand(batch),
                cond=hidden_state,
            )

            d = Normal(loc=mu, scale=sigma).sample()
            x_alpha = x_alpha + (alpha_end - alpha_start) * d.unsqueeze(1)

        return (x_alpha * scale) + loc
