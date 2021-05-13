import numpy as np
import torch
from tqdm import tqdm

import hmc
import utils


@torch.no_grad()
def ais_trajectory(
    model,
    loader,
    forward=True,
    schedule=np.linspace(0., 1., 500),
    n_sample=100,
    device=None,
):
    """Compute annealed importance sampling trajectories for a batch of data.

    Could be used for *both* forward and reverse chain in BDMC.

    Args:
      model (vae.VAE): VAE model
      loader (iterator): iterator that returns pairs, with first component
        being `x`, second would be `z` or label (will not be used)
      forward (boolean): indicate forward/backward chain
      schedule (list or 1D np.ndarray): temperature schedule, i.e. `p(z)p(x|z)^t`
      n_sample (int): number of importance samples
      device (torch.device): device to run all computation on

    Returns:
        A list where each element is a torch.autograd.Variable that contains the
        log importance weights for a single batch of data
    """

    def log_f_i(z, data, t, log_likelihood_fn=utils.log_bernoulli):
        """Unnormalized density for intermediate distribution `f_i`:
            f_i = p(z)^(1-t) p(x,z)^(t) = p(z) p(x|z)^t
        =>  log f_i = log p(z) + t * log p(x|z)
        """
        zeros = torch.zeros(size=(B, model.latent_dim), device=device)
        log_prior = utils.log_normal(z, zeros, zeros)
        log_likelihood = log_likelihood_fn(model.decode(z), data)

        return log_prior + log_likelihood.mul_(t)

    logws = []
    for i, (batch, post_z) in enumerate(loader):
        B = batch.size(0) * n_sample
        batch = batch.to(device)
        batch = utils.safe_repeat(batch, n_sample)

        epsilon = torch.ones(size=(B,), device=device).mul_(0.01)
        accept_hist = torch.zeros(size=(B,), device=device)
        logw = torch.zeros(size=(B,), device=device)

        # initial sample of z
        if forward:
            current_z = torch.randn(size=(B, model.latent_dim), device=device)
        else:
            current_z = utils.safe_repeat(post_z, n_sample).to(device)

        for j, (t0, t1) in tqdm(enumerate(zip(schedule[:-1], schedule[1:]), 1)):
            # update log importance weight
            log_int_1 = log_f_i(current_z, batch, t0)
            log_int_2 = log_f_i(current_z, batch, t1)
            logw += log_int_2 - log_int_1

            def U(z):
                return -log_f_i(z, batch, t1)

            @torch.enable_grad()
            def grad_U(z):
                z = z.clone().requires_grad_(True)
                grad, = torch.autograd.grad(U(z).sum(), z)
                max_ = B * model.latent_dim * 100.
                grad = torch.clamp(grad, -max_, max_)
                return grad

            def normalized_kinetic(v):
                zeros = torch.zeros(size=(B, model.latent_dim), device=device)
                return -utils.log_normal(v, zeros, zeros)

            # resample velocity
            current_v = torch.randn_like(current_z)
            z, v = hmc.hmc_trajectory(current_z, current_v, U, grad_U, epsilon)
            current_z, epsilon, accept_hist = hmc.accept_reject(
                current_z, current_v,
                z, v,
                epsilon,
                accept_hist, j,
                U, K=normalized_kinetic,
            )

        logw = utils.log_mean_exp(logw.view(n_sample, -1).transpose(0, 1))
        if not forward:
            logw = -logw
        logws.append(logw.data)
        print('Last batch stats %.4f' % (logw.mean().cpu().data.numpy()))

    return logws
