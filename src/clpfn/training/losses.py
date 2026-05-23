import math

import torch
import torch.nn.functional as F

from clpfn.config.defaults import (
    CONC_PEN_MAX_PI,
    CONC_PEN_WEIGHT,
    MEAN_HUBER_DELTA,
    MEAN_LOSS_WEIGHT,
    NLL_HUBER_SLOPE,
    NLL_HUBER_THRESHOLD,
)
from clpfn.models.causal_long_pfn import predictive_mean_from_gmm


def gaussian_mixture_loss(log_pi, mu, sigma, target_y):
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

    robust_nll = torch.where(
        per_example_nll > NLL_HUBER_THRESHOLD,
        NLL_HUBER_THRESHOLD + NLL_HUBER_SLOPE * (per_example_nll - NLL_HUBER_THRESHOLD),
        per_example_nll,
    ).mean()

    pred_mean = predictive_mean_from_gmm(log_pi, mu)
    mean_loss = F.huber_loss(pred_mean, target_y, delta=MEAN_HUBER_DELTA)

    loss = robust_nll + MEAN_LOSS_WEIGHT * mean_loss

    mixture_weights = log_pi.exp()
    max_weight = mixture_weights.max(dim=-1).values
    concentration_penalty = F.relu(max_weight - CONC_PEN_MAX_PI).mean() * CONC_PEN_WEIGHT

    loss = loss + concentration_penalty

    aux = {
        "pred_mean": pred_mean.detach(),
        "sigma_mean": sigma.mean().detach(),
        "concentration_penalty": concentration_penalty.detach(),
    }

    return loss, aux
