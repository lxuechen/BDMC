from __future__ import print_function

import torch


def hmc_trajectory(current_z, current_v, U, grad_U, epsilon, L=10):
  """This version of HMC follows https://arxiv.org/pdf/1206.1901.pdf.

  Args:
      U: function to compute potential energy/minus log-density
      grad_U: function to compute gradients w.r.t. U
      epsilon: (adaptive) step size
      L: number of leap-frog steps
      current_z: current position
  """
  eps = epsilon.view(-1, 1)
  z = current_z
  v = current_v - grad_U(z).mul(eps).mul_(.5)

  for i in range(1, L + 1):
    z = z + v.mul(eps)
    if i < L:
      v = v - grad_U(z).mul(eps)

  v = v - grad_U(z).mul(eps).mul_(.5)
  v = -v

  return z.detach(), v.detach()


def accept_reject(current_z, current_v,
                  z, v,
                  epsilon,
                  accept_hist, hist_len,
                  U, K=lambda v: torch.sum(v * v, 1)):
  """Accept/reject based on Hamiltonians for current and propose.

  Args:
      current_z: position *before* leap-frog steps
      current_v: speed *before* leap-frog steps
      z: position *after* leap-frog steps
      v: speed *after* leap-frog steps
      epsilon: step size of leap-frog.
      U: function to compute potential energy
      K: function to compute kinetic energy
  """
  current_Hamil = K(current_v) + U(current_z)
  propose_Hamil = K(v) + U(z)

  prob = torch.clamp_max(torch.exp(current_Hamil - propose_Hamil), 1.)

  with torch.no_grad():
    uniform_sample = torch.rand(prob.size()).cuda()
    accept = (prob > uniform_sample).float().cuda()
    z = z.mul(accept.view(-1, 1)) + current_z.mul(1. - accept.view(-1, 1))

    accept_hist = accept_hist.add(accept)
    criteria = (accept_hist / hist_len > 0.65).float().cuda()
    adapt = 1.02 * criteria + 0.98 * (1. - criteria)
    epsilon = epsilon.mul(adapt).clamp(1e-4, .5)

  z.requires_grad_()

  return z, epsilon, accept_hist
