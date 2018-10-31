from __future__ import print_function
import numpy as np
import itertools
import time

import torch
from torch.autograd import Variable
from torch.autograd import grad as torchgrad
from ais import ais_trajectory
from simulate import simulate_data
from vae import VAE
from hparams import get_default_hparams


def bdmc(model, loader, forward_schedule=np.linspace(0., 1., 500), n_sample=100):
  """Bidirectional Monte Carlo. Integrate forward and backward AIS.
  The backward schedule is the reverse of the forward.

  Args:
      model (vae.VAE): VAE model
      loader (iterator): iterator to loop over pairs of Variables; the first 
          entry being `x`, the second being `z` sampled from the true 
          posterior `p(z|x)`
      forward_schedule (list or numpy.ndarray): forward temperature schedule;
          backward schedule is used as its reverse
  Returns:
      Two lists for forward and backward bounds on batchs of data
  """

  # iterator is exhaustable in py3, so need duplicate
  load, load_ = itertools.tee(loader, 2)

  # forward chain
  forward_logws = ais_trajectory(
      model,
      load,
      mode='forward',
      schedule=forward_schedule,
      n_sample=n_sample)

  # backward chain
  backward_schedule = np.flip(forward_schedule, axis=0)
  backward_logws = ais_trajectory(
      model,
      load_,
      mode='backward',
      schedule=backward_schedule,
      n_sample=n_sample)

  upper_bounds = []
  lower_bounds = []

  for i, (forward, backward) in enumerate(zip(forward_logws, backward_logws)):
    lower_bounds.append(forward.mean())
    upper_bounds.append(backward.mean())

  upper_bounds = np.mean(upper_bounds)
  lower_bounds = np.mean(lower_bounds)

  print('Average bounds on simulated data: lower %.4f, upper %.4f' %
        (lower_bounds, upper_bounds))

  return forward_logws, backward_logws


def main(f='checkpoints/model.pth'):

  hps = get_default_hparams()
  model = VAE(hps)
  model.cuda()
  model.load_state_dict(torch.load(f)['state_dict'])
  model.eval()

  loader = simulate_data(model, batch_size=100, n_batch=10)
  bdmc(model, loader, forward_schedule=np.linspace(0., 1., 500), n_sample=100)


if __name__ == '__main__':
  main()
