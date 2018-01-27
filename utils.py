import numpy as np
import numpy.linalg as linalg

import torch
from torch.autograd import Variable
import torch.nn.functional as F


def log_normal(x, mean, logvar):
    """Implementation WITHOUT constant, since the constants in p(z) 
    and q(z|x) cancels out.
    Args:
        x: [B,Z]
        mean,logvar: [B,Z]

    Returns:
        output: [B]
    """

    return -0.5 * (logvar.sum(1) + ((x - mean).pow(2) / torch.exp(logvar)).sum(1))


def log_normal_full_cov(x, mean, L):
    """Log density of full covariance multivariate Gaussian.
    Note: results are off by the constant log(), since this 
    quantity cancels out in p(z) and q(z|x)."""

    def batch_diag(M):
        diag = [t.diag() for t in torch.functional.unbind(M)]
        diag = torch.functional.stack(diag)
        return diag

    def batch_inverse(M, damp=False, eps=1e-6):
        damp_matrix = Variable(torch.eye(M[0].size(0)).type(M.data.type())).mul_(eps)
        inverse = []
        for t in torch.functional.unbind(M):
            # damping to ensure invertible due to float inaccuracy
            # this problem is very UNLIKELY when using double
            m = t if not damp else t + damp_matrix
            inverse.append(m.inverse())
        inverse = torch.functional.stack(inverse)
        return inverse

    L_diag = batch_diag(L)
    term1 = -torch.log(L_diag).sum(1)

    L_inverse = batch_inverse(L)
    scaled_diff = L_inverse.matmul((x - mean).unsqueeze(2)).squeeze()
    term2 = -0.5 * (scaled_diff ** 2).sum(1)

    return term1 + term2


def log_bernoulli(logit, target):
    """
    Args:
        logit:  [B, X]
        target: [B, X]
    
    Returns:
        output:      [B]
    """

    loss = -F.relu(logit) + torch.mul(target, logit) - torch.log(1. + torch.exp( -logit.abs() ))
    loss = torch.sum(loss, 1)

    return loss


def mean_squared_error(prediction, target):

    prediction, target = flatten(prediction), flatten(target)
    diff = prediction - target

    return -torch.sum(torch.mul(diff, diff), 1)


def discretized_logistic(mu, logs, x):
    """Probability mass follow discretized logistic. 
    https://arxiv.org/pdf/1606.04934.pdf. Assuming pixel values scaled to be
    within [0,1]. """

    sigmoid = torch.nn.Sigmoid()

    s = torch.exp(logs).unsqueeze(-1).unsqueeze(-1)
    logp = torch.log(sigmoid((x + 1./256. - mu) / s) - sigmoid((x - mu) / s) + 1e-7)

    return logp.sum(-1).sum(-1).sum(-1)


def log_mean_exp(x):

    max_, _ = torch.max(x, 1, keepdim=True)
    return torch.log(torch.mean(torch.exp(x - max_), 1)) + torch.squeeze(max_)


def numpy_nan_guard(arr):
    return np.all(arr == arr)


def safe_repeat(x, n):
    return x.repeat(n, *[1 for _ in range(len(x.size()) - 1)])
