from __future__ import print_function

import numpy as np
from tqdm import tqdm

import torch
from torch.autograd import Variable
from torch.autograd import grad as torchgrad
import hmc
import utils


def ais_trajectory(model,
                   loader,
                   forward=True,
                   schedule=np.linspace(0., 1., 500),
                   n_sample=100):
  """Compute annealed importance sampling trajectories for a batch of data. 
  Could be used for *both* forward and reverse chain in BDMC.

  Args:
    model (vae.VAE): VAE model
    loader (iterator): iterator that returns pairs, with first component
      being `x`, second would be `z` or label (will not be used)
    forward (boolean): indicate forward/backward chain
    schedule (list or 1D np.ndarray): temperature schedule, i.e. `p(z)p(x|z)^t`
    n_sample (int): number of importance samples

  Returns:
      A list where each element is a torch.autograd.Variable that contains the 
      log importance weights for a single batch of data
  """

  def log_f_i(z, data, t, log_likelihood_fn=utils.log_bernoulli):
    """Unnormalized density for intermediate distribution `f_i`:
        f_i = p(z)^(1-t) p(x,z)^(t) = p(z) p(x|z)^t
    =>  log f_i = log p(z) + t * log p(x|z)
    """
    zeros = Variable(torch.zeros(B, z_size).cuda())
    log_prior = utils.log_normal(z, zeros, zeros)
    log_likelihood = log_likelihood_fn(model.decode(z), data)

    return log_prior + log_likelihood.mul_(t)

  # shorter aliases
  z_size = model.hps.z_size

  logws = []
  for i, (batch, post_z) in enumerate(loader):

    B = batch.size(0) * n_sample
    batch = Variable(batch.cuda())
    batch = utils.safe_repeat(batch, n_sample)

    # batch of step sizes, one for each chain
    epsilon = Variable(torch.ones(B).cuda()).mul_(0.01)
    # accept/reject history for tuning step size
    accept_hist = Variable(torch.zeros(B).cuda())
    with torch.no_grad():  # Avoid OOM for long chains
      logw = Variable(torch.zeros(B).cuda())

    # initial sample of z
    if forward:
      current_z = Variable(
          torch.randn(B, z_size).cuda(), requires_grad=True)
    else:
      current_z = Variable(
          utils.safe_repeat(post_z, n_sample).cuda(), requires_grad=True)
    
    for j, (t0, t1) in tqdm(enumerate(zip(schedule[:-1], schedule[1:]), 1)):
      # update log importance weight
      log_int_1 = log_f_i(current_z, batch, t0)
      log_int_2 = log_f_i(current_z, batch, t1)
      logw.add_(log_int_2 - log_int_1)

      # resample speed
      current_v = Variable(torch.randn(current_z.size()).cuda())

      def U(z):
        return -log_f_i(z, batch, t1)

      def grad_U(z):
        # grad w.r.t. outputs; mandatory in this case
        grad_outputs = torch.ones(B).cuda()
        # torch.autograd.grad default returns volatile
        grad = torchgrad(U(z), z, grad_outputs=grad_outputs)[0]
        # clip by norm
        grad = torch.clamp(grad, -B * z_size * 100, B * z_size * 100)
        # needs variable wrapper to make differentiable
        grad = Variable(grad.data, requires_grad=True)
        return grad

      def normalized_kinetic(v):
        zeros = Variable(torch.zeros(B, z_size).cuda())
        # this is superior to the unnormalized version
        return -utils.log_normal(v, zeros, zeros)

      z, v = hmc.hmc_trajectory(current_z, current_v, U, grad_U, epsilon)

      # accept-reject step
      current_z, epsilon, accept_hist = hmc.accept_reject(
          current_z, current_v,
          z, v,
          epsilon,
          accept_hist, j,
          U, K=normalized_kinetic)

    # IWAE lower bound
    logw = utils.log_mean_exp(logw.view(n_sample, -1).transpose(0, 1))
    if not forward:
      logw = -logw

    logws.append(logw.data)

    print('Last batch stats %.4f' % (logw.mean().cpu().data.numpy()))

  return logws
