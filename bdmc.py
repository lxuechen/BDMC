from __future__ import print_function

import numpy as np
import itertools

import torch
from torch.autograd import Variable
from torch.autograd import grad as torchgrad
import torch.nn.functional as F

import ais
import simulate
import vae
import hparams


def bdmc(model,
         loader,
         forward_schedule=np.linspace(0., 1., 500),
         n_sample=100):
  """Bidirectional Monte Carlo. Backward schedule is set to be the reverse of
  the forward schedule.

  Args:
    model (vae.VAE): VAE model
    loader (iterator): iterator to loop over pairs of Variables; the first 
      entry being `x`, the second being `z` sampled from the *true* 
      posterior `p(z|x)`
    forward_schedule (list or numpy.ndarray): forward temperature schedule

  Returns:
      Two lists for forward and backward bounds on batchs of data
  """

  # iterator is exhaustable in py3, so need duplicate
  load, load_ = itertools.tee(loader, 2)

  # forward chain
  forward_logws = ais.ais_trajectory(
      model,
      load,
      forward=True,
      schedule=forward_schedule,
      n_sample=n_sample)

  # backward chain
  backward_schedule = np.flip(forward_schedule, axis=0)
  backward_logws = ais.ais_trajectory(
      model,
      load_,
      forward=False,
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


def get_default_hparams():
  return hparams.HParams(
      z_size=50,
      act_func=F.elu,
      has_flow=False,
      large_encoder=False,
      wide_encoder=False,
      cuda=True)


def main(f='checkpoints/model.pth'):
  hps = get_default_hparams()
  model = vae.VAE(hps)
  model.cuda()
  model.load_state_dict(torch.load(f)['state_dict'])
  model.eval()

  loader = simulate.simulate_data(model, batch_size=100, n_batch=10)
  bdmc(model, loader, forward_schedule=np.linspace(0., 1., 500), n_sample=100)


if __name__ == '__main__':
  main()
