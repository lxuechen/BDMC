from __future__ import print_function

import argparse
import numpy as np
import itertools

import torch
import torch.nn.functional as F

import ais
import simulate
import vae


parser = argparse.ArgumentParser(description='BDMC')
parser.add_argument('--latent-dim', type=int, default=50, metavar='D',
                    help='number of latent variables (default: 50)')
parser.add_argument('--batch-size', type=int, default=10, metavar='N',
                    help='number of examples to eval at once (default: 10)')
parser.add_argument('--n-batch', type=int, default=10, metavar='B',
                    help='number of batches to eval in total (default: 10)')
parser.add_argument('--chain-length', type=int, default=500, metavar='L',
                    help='length of ais chain (default: 500)')
parser.add_argument('--iwae-samples', type=int, default=100, metavar='I',
                    help='number of iwae samples (default: 100)')
parser.add_argument('--ckpt-path', type=str, default='checkpoints/model.pth',
                    metavar='C', help='path to checkpoint')
args = parser.parse_args()


def bdmc(model, loader, forward_schedule, n_sample):
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


def main():
  model = vae.VAE(latent_dim=args.latent_dim)
  model.cuda()
  model.load_state_dict(torch.load(args.ckpt_path)['state_dict'])
  model.eval()

  # bdmc uses simulated data from the model
  loader = simulate.simulate_data(
      model,
      batch_size=args.batch_size,
      n_batch=args.n_batch)
  # run bdmc
  forward_schedule = np.linspace(0., 1., args.chain_length)
  bdmc(
      model,
      loader,
      forward_schedule=forward_schedule,
      n_sample=args.iwae_samples)


if __name__ == '__main__':
  main()
