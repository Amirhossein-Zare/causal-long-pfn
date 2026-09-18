import math

import torch

from clpfn.models.causal_long_pfn import predictive_mean_from_gmm


def gaussian_mixture_loss(log_pi, mu, sigma, target_y):
    """Gaussian-mixture negative log likelihood."""
    log_pi = log_pi.float()
    mu = mu.float()
    sigma = sigma.float()
    target_y = target_y.float()

    target = target_y.unsqueeze(-1)
    log_probs = (
        -0.5 * ((target - mu) / sigma) ** 2
        - torch.log(sigma)
        - 0.5 * math.log(2 * math.pi)
    )
    per_example_nll = -torch.logsumexp(log_pi + log_probs, dim=-1)
    loss = per_example_nll.mean()

    mixture_weights = log_pi.exp()
    pred_mean = predictive_mean_from_gmm(log_pi, mu)
    predictive_variance = (
        mixture_weights * (sigma.square() + mu.square())
    ).sum(dim=-1) - pred_mean.square()

    aux = {
        "loss_total": loss.detach(),
        "loss_nll": loss.detach(),
        "pred_mean": pred_mean.detach(),
        "predictive_sigma_mean": sigma.mean().detach(),
        "predictive_std_mean": predictive_variance.clamp_min(0.0).sqrt().mean().detach(),
    }
    return loss, aux
