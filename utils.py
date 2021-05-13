from __future__ import print_function

import math

import torch
import torch.nn.functional as F


def log_normal(x, mean, logvar):
  return -0.5 * ((math.log(2 * math.pi) + logvar).sum(1) + ((x - mean).pow(2) / torch.exp(logvar)).sum(1))


def log_bernoulli(logit, target):
  loss = -F.relu(logit) + torch.mul(target, logit) - torch.log(1. + torch.exp(-logit.abs()))
  loss = torch.sum(loss, 1)
  return loss


def log_mean_exp(x):
  max_, _ = torch.max(x, 1, keepdim=True)
  return torch.log(torch.mean(torch.exp(x - max_), 1)) + torch.squeeze(max_)


def safe_repeat(x, n):
  return x.repeat(n, *[1 for _ in range(len(x.size()) - 1)])
