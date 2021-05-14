import math

import torch
import torch.nn.functional as F


def log_normal(x, mean, logvar):
    """Log-pdf for factorized Normal distributions."""
    return -0.5 * ((math.log(2 * math.pi) + logvar).sum(1) + ((x - mean).pow(2) / torch.exp(logvar)).sum(1))


def log_bernoulli(logit, target):
    """Numerically stable variant of log-pmf for Bernoulli using ReLU."""
    loss = -F.relu(logit) + torch.mul(target, logit) - torch.log(1. + torch.exp(-logit.abs()))
    loss = torch.sum(loss, 1)
    return loss


def logmeanexp(x, dim=1):
    max_, _ = torch.max(x, dim=dim, keepdim=True)
    return torch.log(torch.mean(torch.exp(x - max_), dim=dim)) + max_.squeeze(dim=dim)


def safe_repeat(x, n):
    return x.repeat(n, *[1 for _ in range(len(x.size()) - 1)])
